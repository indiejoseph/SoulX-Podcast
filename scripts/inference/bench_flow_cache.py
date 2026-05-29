"""Benchmark cached vs uncached chunked flow on TTFA and total wall.

Runs the same chunked-streaming loop twice on the same speech tokens:

  * **uncached**: each chunk calls ``flow(prompt+all_tokens, streaming=True)``
    and HiFT on the full accumulated mel. This is what the production path
    does today when ``FLOW_CHUNK_CACHE`` is unset.
  * **cached**: each chunk calls ``flow.forward_chunk_cached(new_tokens)``
    and HiFT on the accumulated mel. This is the experimental path.

Reports TTFA (time of first chunk's audio), per-chunk wall, total wall, and
RTF. The token sequence is generated once by driving the LLM so the inputs
are realistic.

Usage:
    python scripts/inference/bench_flow_cache.py [chunk=50] [target_tokens=400]
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os
import sys
import time

import torch

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input


def _prompt_state(model, prepared):
    prompt_mels = prepared["prompt_mels_for_llm"]
    prompt_mels_lens = prepared["prompt_mels_lens_for_llm"]
    spk_tok, spk_len = model.audio_tokenizer.quantize(
        prompt_mels.cuda(), prompt_mels_lens.cuda()
    )
    plen = spk_len[0].item()
    pmel = prepared["prompt_mels_for_flow_ori"][0][: plen * 2].cuda()
    return {
        "prompt_tokens": spk_tok[0, :plen].tolist(),
        "prompt_mel": pmel[None],
        "prompt_mel_len": torch.tensor([pmel.shape[0]], device="cuda"),
        "spk_emb": prepared["spk_emb_for_flow"][0:1].cuda(),
    }


def _generate_tokens(model, prepared, n_target):
    from soulxpodcast.utils.streaming import SpeechTokenStreamer, run_llm_in_thread

    eos_id = model.config.hf_config.eos_token_id
    offset = model.config.hf_config.speech_token_offset
    pmels = prepared["prompt_mels_for_llm"]
    plens = prepared["prompt_mels_lens_for_llm"]
    spk_tok, spk_len = model.audio_tokenizer.quantize(pmels.cuda(), plens.cuda())
    spk = spk_tok[0, : spk_len[0].item()].tolist()
    spk = [t + offset for t in spk] + [eos_id]
    inputs = prepared["prompt_text_tokens_for_llm"][0] + spk + prepared["text_tokens_for_llm"][0]

    streamer = SpeechTokenStreamer(eos_token_id=eos_id)
    thread = run_llm_in_thread(
        model.llm, list(inputs), prepared["sampling_params"], streamer=streamer
    )
    tokens = []
    try:
        for chunk in streamer.iter_chunks(chunk_size=25, first_chunk_size=25):
            tokens.extend(t - offset for t in chunk)
            if len(tokens) >= n_target:
                break
    finally:
        streamer.cancel()
        thread.join(timeout=2.0)
    if thread.exc:
        raise thread.exc
    return tokens[:n_target]


def _bench_uncached(model, state, tokens, chunk_size, flow_steps):
    """Per-chunk: flow(prompt + all_accumulated_tokens) then HiFT on full mel."""
    pre_lookahead = model.flow.pre_lookahead_len
    accumulated = []
    prev_audio_len = 0
    timings = []
    n = len(tokens)
    cursor = 0
    ttfa = None

    while cursor < n:
        cursor = min(cursor + chunk_size, n)
        new = tokens[len(accumulated) : cursor]
        accumulated.extend(new)
        finalize = cursor >= n

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        flow_input = torch.tensor([state["prompt_tokens"] + accumulated], device="cuda")
        flow_input_len = torch.tensor([flow_input.shape[1]], device="cuda")
        with torch.amp.autocast("cuda",
                dtype=torch.float16 if model.config.hf_config.fp16_flow else torch.float32):
            mels, mel_lens = model.flow(
                flow_input, flow_input_len,
                state["prompt_mel"], state["prompt_mel_len"], state["spk_emb"],
                streaming=True, finalize=finalize, n_timesteps=flow_steps,
            )
            mel = mels[:, :, state["prompt_mel_len"][0].item() : mel_lens[0].item()]
            audio, _ = model.hift(speech_feat=mel)
        new_audio = audio[:, prev_audio_len:].detach().cpu()
        prev_audio_len = audio.shape[-1]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        timings.append(dt)
        if ttfa is None:
            ttfa = sum(timings)

    total_audio_s = prev_audio_len / 24000.0
    return {
        "ttfa": ttfa,
        "wall": sum(timings),
        "per_chunk": timings,
        "audio_s": total_audio_s,
        "rtf": sum(timings) / total_audio_s if total_audio_s else float("nan"),
    }


def _bench_cached(model, state, tokens, chunk_size, flow_steps):
    """Per-chunk: forward_chunk_cached(new_tokens) then HiFT on accumulated mel."""
    pre_lookahead = model.flow.pre_lookahead_len
    cache = None
    mel_chunks = []
    prev_audio_len = 0
    timings = []
    n = len(tokens)
    cursor = 0
    processed = 0
    ttfa = None

    while cursor < n:
        cursor = min(cursor + chunk_size, n)
        finalize = cursor >= n
        if finalize:
            new_tokens = tokens[processed:]
            ctx_tokens = []
        else:
            process_until = max(0, cursor - pre_lookahead)
            new_tokens = tokens[processed:process_until]
            ctx_tokens = tokens[process_until:cursor]
            if not new_tokens:
                continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first = cache is None or not cache.get("started", False)
        block = (state["prompt_tokens"] + new_tokens) if first else new_tokens
        flow_input = torch.tensor([block], device="cuda")
        flow_input_len = torch.tensor([flow_input.shape[1]], device="cuda")
        ctx = torch.tensor([ctx_tokens], device="cuda")
        with torch.amp.autocast("cuda",
                dtype=torch.float16 if model.config.hf_config.fp16_flow else torch.float32):
            mel_chunk, _h, cache = model.flow.forward_chunk_cached(
                flow_input, flow_input_len, ctx,
                state["prompt_mel"], state["prompt_mel_len"], state["spk_emb"],
                cache=cache, n_timesteps=flow_steps,
            )
            mel_chunks.append(mel_chunk)
            mel = torch.cat(mel_chunks, dim=-1)
            audio, _ = model.hift(speech_feat=mel)
        new_audio = audio[:, prev_audio_len:].detach().cpu()
        prev_audio_len = audio.shape[-1]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        timings.append(dt)
        if ttfa is None:
            ttfa = sum(timings)
        processed = cursor if finalize else max(0, cursor - pre_lookahead)

    total_audio_s = prev_audio_len / 24000.0
    return {
        "ttfa": ttfa,
        "wall": sum(timings),
        "per_chunk": timings,
        "audio_s": total_audio_s,
        "rtf": sum(timings) / total_audio_s if total_audio_s else float("nan"),
    }


def _report(label, r):
    pc = "  ".join(f"{t:.3f}" for t in r["per_chunk"])
    print(f"[{label}]  ttfa={r['ttfa']:.3f}s  wall={r['wall']:.3f}s  "
          f"audio={r['audio_s']:.2f}s  rtf={r['rtf']:.3f}")
    print(f"[{label}]  per-chunk: {pc}")


def main():
    chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    target_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    flow_steps = int(os.environ.get("FLOW_STEPS", "4"))

    model_path = os.environ.get(
        "MODEL_PATH",
        "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/pretrained_models/SoulX-Podcast-1.7B-dialect",
    )
    print(f"[init] loading {model_path} (hf engine)")
    t0 = time.perf_counter()
    model, dataset = initiate_model(
        seed=198964, model_path=model_path, llm_engine="hf", fp16_flow=True
    )
    print(f"[init] load: {time.perf_counter()-t0:.1f}s")

    prepared = process_single_input(
        dataset,
        target_text_list=[
            "[S1]Hello everyone, welcome to today's podcast. Today we are going to "
            "talk about a fascinating topic — the history and future of artificial "
            "intelligence and how it has shaped modern computing over the past several "
            "decades. From the earliest symbolic systems all the way through deep "
            "learning, large language models, and now multimodal foundation models, "
            "the field has gone through dramatic transformations. We will explore "
            "where things started, where they are today, and where leading researchers "
            "believe the next breakthroughs will come from."
        ],
        prompt_wav_list=["example/audios/female_mandarin.wav"],
        prompt_text_list=[
            "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。"
        ],
        use_dialect_prompt=False,
        dialect_prompt_text_list=None,
    )

    state = _prompt_state(model, prepared)
    print(f"[gen] driving LLM for ~{target_tokens} speech tokens")
    tokens = _generate_tokens(model, prepared, target_tokens)
    print(f"[gen] got {len(tokens)} tokens "
          f"(~{len(tokens) * 2 / 25:.1f}s audio at 25Hz with token_mel_ratio=2)")

    n_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    print(f"[bench] chunk_size={chunk_size}  flow_steps={flow_steps}  n_chunks={n_chunks}")

    # warm-up to avoid first-call cuBLAS/cudnn allocation overhead in the bench
    _ = _bench_cached(model, state, tokens[: min(chunk_size * 2, len(tokens))], chunk_size, flow_steps)
    _ = _bench_uncached(model, state, tokens[: min(chunk_size * 2, len(tokens))], chunk_size, flow_steps)
    torch.cuda.synchronize()

    print("[run] uncached streaming (production default)")
    r_un = _bench_uncached(model, state, tokens, chunk_size, flow_steps)
    print("[run] cached streaming (FLOW_CHUNK_CACHE=1)")
    r_ca = _bench_cached(model, state, tokens, chunk_size, flow_steps)

    print()
    _report("uncached", r_un)
    _report("cached  ", r_ca)
    print()
    ttfa_delta = (r_ca["ttfa"] - r_un["ttfa"]) / r_un["ttfa"] * 100
    wall_delta = (r_ca["wall"] - r_un["wall"]) / r_un["wall"] * 100
    rtf_delta = (r_ca["rtf"] - r_un["rtf"]) / r_un["rtf"] * 100
    print(f"[delta] ttfa: {ttfa_delta:+.1f}%   wall: {wall_delta:+.1f}%   rtf: {rtf_delta:+.1f}%")


if __name__ == "__main__":
    main()
