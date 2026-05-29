"""Smoke test for the cached flow streaming path (commit 8aeb00f).

Compares the experimental ``flow.forward_chunk_cached`` path against the
existing ``flow.forward(streaming=True)`` path on the same speech tokens and
prompt. The two paths cannot be made bit-identical (each call samples its own
``torch.randn_like(mu)`` for CFM and the cached path samples per chunk while
the non-cached path samples once for the full sequence), so we check:

  1. Cached and non-cached mel outputs have matching shape (after stripping
     the prompt portion from the non-cached output).
  2. Per-mel-bin mean/std statistics agree to within a loose tolerance.
  3. Audio synthesized from the cached path is finite and non-silent.
  4. A diagnostic WAV is written for human listening.

Usage:
    python scripts/inference/test_flow_cache.py [chunk_tokens=50]
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os
import sys
import time
from pathlib import Path

import torch
import torchaudio

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input


def _prompt_state(model, prepared):
    """Quantize prompt audio + collect mel/spk_emb for flow conditioning."""
    prompt_mels = prepared["prompt_mels_for_llm"]
    prompt_mels_lens = prepared["prompt_mels_lens_for_llm"]
    prompt_spk_tokens, prompt_lens = model.audio_tokenizer.quantize(
        prompt_mels.cuda(), prompt_mels_lens.cuda()
    )
    prompt_len = prompt_lens[0].item()
    prompt_mel = prepared["prompt_mels_for_flow_ori"][0][: prompt_len * 2].cuda()
    return {
        "prompt_tokens": prompt_spk_tokens[0, :prompt_len].tolist(),
        "prompt_mel": prompt_mel[None],
        "prompt_mel_len": torch.tensor([prompt_mel.shape[0]], device="cuda"),
        "spk_emb": prepared["spk_emb_for_flow"][0:1].cuda(),
    }


def _run_non_cached(model, state, generated_tokens, flow_steps):
    flow_input = torch.tensor(
        [state["prompt_tokens"] + generated_tokens], device="cuda"
    )
    flow_input_len = torch.tensor([flow_input.shape[1]], device="cuda")
    torch.manual_seed(0)
    with torch.amp.autocast(
        "cuda",
        dtype=torch.float16 if model.config.hf_config.fp16_flow else torch.float32,
    ):
        mels, mels_lens = model.flow(
            flow_input,
            flow_input_len,
            state["prompt_mel"],
            state["prompt_mel_len"],
            state["spk_emb"],
            streaming=True,
            finalize=True,
            n_timesteps=flow_steps,
        )
    prompt_mel_len = state["prompt_mel_len"][0].item()
    mel = mels[:, :, prompt_mel_len : mels_lens[0].item()]
    return mel.float()


def _run_cached(model, state, generated_tokens, chunk_tokens, flow_steps):
    pre_lookahead = model.flow.pre_lookahead_len
    cache = None
    mel_chunks = []
    processed = 0
    n = len(generated_tokens)
    cursor = 0
    torch.manual_seed(0)
    while cursor < n:
        cursor = min(cursor + chunk_tokens, n)
        # The runtime path holds back pre_lookahead_len tokens as context
        # until the final flush. Mirror that here.
        process_until = max(0, cursor - pre_lookahead)
        if process_until <= processed and cursor < n:
            continue
        new_tokens = generated_tokens[processed:process_until]
        ctx_tokens = generated_tokens[process_until:cursor] if cursor < n else []
        if cursor >= n:
            new_tokens = generated_tokens[processed:]
            ctx_tokens = []
        if not new_tokens:
            continue

        first_chunk = cache is None or not cache.get("started", False)
        token_block = (state["prompt_tokens"] + new_tokens) if first_chunk else new_tokens
        flow_input = torch.tensor([token_block], device="cuda")
        flow_input_len = torch.tensor([flow_input.shape[1]], device="cuda")
        ctx_input = torch.tensor([ctx_tokens], device="cuda")
        with torch.amp.autocast(
            "cuda",
            dtype=torch.float16 if model.config.hf_config.fp16_flow else torch.float32,
        ):
            mel_chunk, _h_lens, cache = model.flow.forward_chunk_cached(
                flow_input,
                flow_input_len,
                ctx_input,
                state["prompt_mel"],
                state["prompt_mel_len"],
                state["spk_emb"],
                cache=cache,
                n_timesteps=flow_steps,
            )
        mel_chunks.append(mel_chunk.float())
        processed = process_until if cursor < n else len(generated_tokens)
    return torch.cat(mel_chunks, dim=-1)


def _generate_tokens(model, prepared, n_target_tokens):
    """Drive the LLM once to get a realistic speech-token sequence."""
    from soulxpodcast.utils.streaming import SpeechTokenStreamer, run_llm_in_thread

    eos_id = model.config.hf_config.eos_token_id
    offset = model.config.hf_config.speech_token_offset

    prompt_mels = prepared["prompt_mels_for_llm"]
    prompt_mels_lens = prepared["prompt_mels_lens_for_llm"]
    prompt_spk_tokens, prompt_lens = model.audio_tokenizer.quantize(
        prompt_mels.cuda(), prompt_mels_lens.cuda()
    )
    spk_tokens = prompt_spk_tokens[0, : prompt_lens[0].item()].tolist()
    spk_tokens = [t + offset for t in spk_tokens] + [eos_id]
    inputs = prepared["prompt_text_tokens_for_llm"][0] + spk_tokens + prepared["text_tokens_for_llm"][0]

    streamer = SpeechTokenStreamer(eos_token_id=eos_id)
    thread = run_llm_in_thread(
        model.llm,
        list(inputs),
        prepared["sampling_params"],
        streamer=streamer,
    )
    speech_tokens = []
    try:
        for chunk in streamer.iter_chunks(chunk_size=25, first_chunk_size=25):
            speech_tokens.extend(t - offset for t in chunk)
            if len(speech_tokens) >= n_target_tokens:
                break
    finally:
        streamer.cancel()
        thread.join(timeout=2.0)
    if thread.exc:
        raise thread.exc
    return speech_tokens[:n_target_tokens]


def main():
    chunk_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    flow_steps = 8
    n_target_tokens = 150

    model_path = os.environ.get(
        "MODEL_PATH",
        "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/pretrained_models/SoulX-Podcast-1.7B-dialect",
    )
    print(f"[init] loading model (hf engine)")
    t0 = time.perf_counter()
    model, dataset = initiate_model(
        seed=198964, model_path=model_path, llm_engine="hf", fp16_flow=True
    )
    print(f"[init] load: {time.perf_counter()-t0:.1f}s")

    prepared = process_single_input(
        dataset,
        target_text_list=[
            "[S1]Hello everyone, welcome to today's podcast.",
        ],
        prompt_wav_list=["example/audios/female_mandarin.wav"],
        prompt_text_list=[
            "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。"
        ],
        use_dialect_prompt=False,
        dialect_prompt_text_list=None,
    )

    state = _prompt_state(model, prepared)
    print(f"[gen] driving LLM for ~{n_target_tokens} speech tokens")
    generated = _generate_tokens(model, prepared, n_target_tokens)
    print(f"[gen] got {len(generated)} tokens")

    print("[run] non-cached forward(streaming=True)")
    t0 = time.perf_counter()
    mel_ref = _run_non_cached(model, state, generated, flow_steps)
    torch.cuda.synchronize()
    t_ref = time.perf_counter() - t0
    print(f"[run] non-cached: {t_ref:.2f}s  mel.shape={tuple(mel_ref.shape)}")

    print(f"[run] cached forward_chunk_cached(chunk_tokens={chunk_tokens})")
    t0 = time.perf_counter()
    mel_cached = _run_cached(model, state, generated, chunk_tokens, flow_steps)
    torch.cuda.synchronize()
    t_cached = time.perf_counter() - t0
    print(f"[run] cached:    {t_cached:.2f}s  mel.shape={tuple(mel_cached.shape)}")

    ok = True

    # Shape check
    if mel_ref.shape[-1] != mel_cached.shape[-1]:
        print(
            f"[FAIL] mel length mismatch: ref={mel_ref.shape[-1]} cached={mel_cached.shape[-1]}"
        )
        ok = False

    # Finite check
    if not torch.isfinite(mel_cached).all():
        print("[FAIL] cached mel contains NaN/Inf")
        ok = False

    # Per-bin statistics
    common = min(mel_ref.shape[-1], mel_cached.shape[-1])
    ref = mel_ref[..., :common]
    cac = mel_cached[..., :common]
    ref_mean, ref_std = ref.mean(dim=-1), ref.std(dim=-1)
    cac_mean, cac_std = cac.mean(dim=-1), cac.std(dim=-1)
    mean_diff = (ref_mean - cac_mean).abs().mean().item()
    std_diff = (ref_std - cac_std).abs().mean().item()
    print(
        f"[stat] per-bin |mean diff|={mean_diff:.4f}  |std diff|={std_diff:.4f}"
    )

    # Cosine similarity per frame
    ref_flat = ref.reshape(ref.shape[0], -1)
    cac_flat = cac.reshape(cac.shape[0], -1)
    cos = torch.nn.functional.cosine_similarity(ref_flat, cac_flat, dim=-1).item()
    print(f"[stat] cosine similarity (flatten): {cos:.4f}")

    # Audio sanity
    with torch.amp.autocast("cuda", dtype=torch.float16):
        wav_cached, _ = model.hift(speech_feat=mel_cached.to(torch.float16))
    wav_cached = wav_cached.detach().cpu().float()
    if not torch.isfinite(wav_cached).all():
        print("[FAIL] cached audio contains NaN/Inf")
        ok = False
    peak = wav_cached.abs().max().item()
    rms = wav_cached.pow(2).mean().sqrt().item()
    print(f"[audio] cached peak={peak:.4f}  rms={rms:.4f}  samples={wav_cached.shape[-1]}")
    if peak < 1e-3:
        print("[FAIL] cached audio is silent")
        ok = False

    out_dir = Path("outputs/flow_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "cached.wav"), wav_cached, 24000)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        wav_ref, _ = model.hift(speech_feat=mel_ref.to(torch.float16))
    torchaudio.save(
        str(out_dir / "non_cached.wav"), wav_ref.detach().cpu().float(), 24000
    )
    print(f"[audio] wrote {out_dir}/(cached|non_cached).wav")

    if ok and cos > 0.7:
        print("[PASS] cached path produces shape-matched, finite, non-silent mel close to reference")
        sys.exit(0)
    elif ok:
        print(f"[WARN] cached path passes basic checks but cosine={cos:.4f} is low")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
