"""Direct flow+HiFT A/B: same speech tokens, sync vs stream.

Isolates whether per-turn loudness inconsistency lives in the flow/HiFT
stage by replaying the SAME LLM-generated speech token sequence through
two flow paths and comparing per-turn audio RMS.

Pipeline:
  1. Run LLM once on a 4-turn Cantonese dialect dialogue (HF engine, fixed
     seed) via `model.forward_longform`. This captures `per_turn_speech_tokens`
     AND produces the "SYNC" audio (flow `streaming=False, finalize=True`,
     single flow call per turn, matches `/generate`).
  2. Replay the captured tokens through a streaming-emulated flow loop
     (`streaming=True`, chunked at chunk_size=150, finalize=True flush,
     matches `/generate-stream`). Produces "STREAM" audio.
  3. Save both wavs and report per-turn peak/RMS spread.

If SYNC and STREAM show the same per-turn loudness profile, the variation
is in the LLM's token sequence (prosody). If they differ, the variation is
introduced by the streaming flow attention pattern + chunked finalize=False
calls — the flow stage itself, not the tokens.

Usage:
    python scripts/inference/flow_direct_sync_vs_stream.py [--chunk N] [--steps N]

Outputs under outputs/flow_direct/.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser


MODEL_PATH = "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged"

# 4-turn Cantonese dialect dialogue (same as multi_turn_bistream_test.py).
S1_PROMPT_WAV = Path("example/audios/female_mandarin.wav")
S2_PROMPT_WAV = Path("example/audios/male_mandarin.wav")
DATA = {
    "speakers": {
        "S1": {
            "prompt_audio": S1_PROMPT_WAV,
            "prompt_text": "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
            "dialect_prompt": "<|Yue|>真係冇讲错啊！攀山滑雪嘅语言专家几巴闭，都唔及我听日拖成副身家去景德镇玩泥巴，呢铺真系发哂白日梦咯！",
        },
        "S2": {
            "prompt_audio": S2_PROMPT_WAV,
            "prompt_text": "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
            "dialect_prompt": "<|Yue|>咪搞错啊！陕北民谣响度唱咗几十年，黑神话边有咁大面啊？你估佢哋抄游戏咩！",
        },
    },
    "text": [
        ["S1", "<|Yue|>哈囉大家好啊，歡迎收聽我哋嘅節目。喂，我今日想問你樣嘢啊，你覺唔覺得，嗯，而家揸電動車，最煩，最煩嘅一樣嘢係咩啊？"],
        ["S2", "<|Yue|>梗係充電啦。大佬啊，搵個位都已經好煩，搵到個位仲要喺度等，你話快極都要半個鐘一個鐘，真係，有時諗起都覺得好冇癮。"],
        ["S1", "<|Yue|>係咪先。如果我而家同你講，充電可以快到同入油差唔多時間，你信唔信先？喂你平時喺油站入滿一缸油，要幾耐啊？五六分鐘？"],
        ["S2", "<|Yue|>差唔多啦，七八分鐘，點都走得啦。電車喎，可以做到咁快？你咪玩啦。"],
    ],
}


def prepare_flow_state(model, prepared):
    """Same prompt-state precomputation as forward_longform.

    Returns per-speaker dict with prompt_speech_tokens, prompt_mel,
    prompt_mel_len_t, spk_emb. Indexed by spk_id (matches spk_ids[i]).
    """
    prompt_mels = prepared["prompt_mels_for_llm"]
    prompt_mels_lens = prepared["prompt_mels_lens_for_llm"]
    prompt_mels_for_flow_ori = prepared["prompt_mels_for_flow_ori"]
    spk_emb_for_flow = prepared["spk_emb_for_flow"]

    prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = model.audio_tokenizer.quantize(
        prompt_mels.cuda(), prompt_mels_lens.cuda()
    )

    states = []
    for i in range(len(prompt_mels)):
        L = prompt_speech_tokens_lens_ori[i].item()
        spk_tokens = prompt_speech_tokens_ori[i, :L]
        mel = prompt_mels_for_flow_ori[i]
        mel_len = mel.shape[0]
        if L * 2 > mel_len:
            spk_tokens = spk_tokens[: mel_len // 2]
            mel = mel.detach().clone().cuda()
            mel_len_t = torch.tensor([mel_len], device="cuda")
        else:
            mel = mel.detach().clone()[: L * 2].cuda()
            mel_len_t = torch.tensor([L * 2], device="cuda")
        states.append({
            "prompt_tokens": spk_tokens.tolist(),
            "prompt_mel": mel[None],
            "prompt_mel_len_t": mel_len_t,
            "spk_emb": spk_emb_for_flow[i : i + 1].cuda(),
        })
    return states


def flow_call(model, state, all_tokens, *, streaming: bool, finalize: bool, flow_steps: int):
    """Single flow + HiFT call. Returns wav [1, T]."""
    flow_input = torch.tensor([state["prompt_tokens"] + all_tokens], device="cuda")
    flow_input_len = torch.tensor([flow_input.shape[1]], device="cuda")
    with torch.amp.autocast("cuda",
            dtype=torch.float16 if model.config.hf_config.fp16_flow else torch.float32):
        mels, mels_lens = model.flow(
            flow_input, flow_input_len,
            state["prompt_mel"], state["prompt_mel_len_t"], state["spk_emb"],
            streaming=streaming, finalize=finalize,
            n_timesteps=flow_steps,
        )
    prompt_mel_len = state["prompt_mel_len_t"][0].item()
    mel = mels[:, :, prompt_mel_len : mels_lens[0].item()]
    wav, _ = model.hift(speech_feat=mel)
    return wav


def synth_sync(model, state, tokens, flow_steps):
    """One flow call, streaming=False, finalize=True. Matches /generate."""
    return flow_call(model, state, tokens, streaming=False, finalize=True, flow_steps=flow_steps)


def synth_stream(model, state, tokens, flow_steps, chunk_size, first_chunk_size,
                 return_pieces: bool = False):
    """Chunked flow calls, streaming=True. Matches /generate-stream.

    Mirrors forward_longform_streaming logic:
      - For chunks 0..N-2: flow(streaming=True, finalize=False) on accumulated tokens
      - For the trailing partial: skip finalize=False, go straight to finalize=True
      - Final flush: flow(streaming=True, finalize=True) on the full sequence

    Returns concatenated wav by default. With return_pieces=True returns
    (wav, pieces) where pieces is a list of (label, audio_tensor) tuples
    so we can analyse the per-chunk-piece loudness profile.
    """
    accumulated = []
    pieces = []
    labelled_pieces = []
    prev_len = 0

    chunks = []
    i = 0
    first = True
    while i < len(tokens):
        target = first_chunk_size if first else chunk_size
        end = min(i + target, len(tokens))
        is_final = (end == len(tokens) and (end - i) < target)
        chunks.append((tokens[i:end], is_final))
        i = end
        first = False

    for idx, (chunk_tokens, is_final_partial) in enumerate(chunks):
        accumulated.extend(chunk_tokens)
        if is_final_partial:
            break  # finalize=True flush below handles this in one call
        wav = flow_call(model, state, accumulated,
                        streaming=True, finalize=False, flow_steps=flow_steps)
        new_audio = wav[:, prev_len:].detach()
        pieces.append(new_audio)
        labelled_pieces.append((f"c{idx}_nf{len(chunk_tokens)}", new_audio))
        prev_len = wav.shape[-1]

    # Final flush.
    wav_full = flow_call(model, state, accumulated,
                         streaming=True, finalize=True, flow_steps=flow_steps)
    new_audio = wav_full[:, prev_len:].detach()
    pieces.append(new_audio)
    labelled_pieces.append((f"flush", new_audio))
    full = torch.cat(pieces, dim=-1)
    if return_pieces:
        return full, labelled_pieces
    return full


def rms_db(x: torch.Tensor) -> float:
    y = x.float().cpu().numpy().reshape(-1)
    r = float(np.sqrt((y ** 2).mean()) + 1e-12)
    return 20 * np.log10(r)


def report(label, wavs):
    print(f"\n=== {label} ===")
    print(f"  {'turn':>4} {'samples':>8} {'dur':>6} {'peak':>7} {'rms':>7} {'rms_dB':>8}")
    rmss = []
    peaks = []
    for i, w in enumerate(wavs):
        y = w.float().cpu().numpy().reshape(-1)
        n = len(y)
        dur = n / 24000.0
        peak = float(np.abs(y).max())
        r = float(np.sqrt((y ** 2).mean()) + 1e-12)
        rmss.append(r)
        peaks.append(peak)
        print(f"  {i:>4} {n:>8d} {dur:>5.2f}s {peak:>7.3f} {r:>7.4f} {20*np.log10(r):>8.2f}")
    r = np.array(rmss); p = np.array(peaks)
    print(f"  rms σ/μ={r.std()/r.mean():.3f}  max/min rms={20*np.log10(r.max()/r.min()):.2f} dB  "
          f"peak σ/μ={p.std()/p.mean():.3f}  max/min peak={20*np.log10(p.max()/p.min()):.2f} dB")
    return rmss, peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--first-chunk", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=198964)
    ap.add_argument("--sweep", type=str, default=None,
                    help="Comma-separated chunk sizes to sweep, e.g. 50,100,150,250. "
                         "Captures tokens once and replays through each. Overrides --chunk.")
    args = ap.parse_args()

    os.chdir(_Path(__file__).resolve().parents[2])

    print(f"[init] loading model from {MODEL_PATH} (hf engine, fp16_flow, seed={args.seed})")
    t0 = time.perf_counter()
    model, dataset = initiate_model(seed=args.seed, model_path=MODEL_PATH,
                                    llm_engine="hf", fp16_flow=True)
    print(f"[init] load: {time.perf_counter()-t0:.1f}s")

    inputs = podcast_format_parser(DATA)
    prepared = process_single_input(
        dataset, inputs["text"], inputs["prompt_wav"], inputs["prompt_text"],
        inputs["use_dialect_prompt"], inputs["dialect_prompt_text"],
    )

    # ---- Step 1: Run forward_longform to capture per-turn speech tokens. -----
    # forward_longform uses flow(streaming=False, finalize=True) internally —
    # this gives us the SYNC wavs for free.
    print("\n[step 1/2] forward_longform (LLM + sync flow, streaming=False finalize=True)")
    t0 = time.perf_counter()
    results = model.forward_longform(
        prompt_mels_for_llm=prepared["prompt_mels_for_llm"],
        prompt_mels_lens_for_llm=prepared["prompt_mels_lens_for_llm"],
        prompt_text_tokens_for_llm=prepared["prompt_text_tokens_for_llm"],
        text_tokens_for_llm=prepared["text_tokens_for_llm"],
        prompt_mels_for_flow_ori=prepared["prompt_mels_for_flow_ori"],
        spk_emb_for_flow=prepared["spk_emb_for_flow"],
        sampling_params=prepared["sampling_params"],
        spk_ids=prepared["spk_ids"],
        use_dialect_prompt=inputs["use_dialect_prompt"],
        dialect_prompt_text_tokens_for_llm=prepared.get("dialect_prompt_text_tokens_for_llm"),
        dialect_prefix=prepared.get("dialect_prefix"),
    )
    print(f"[step 1/2] done in {time.perf_counter()-t0:.1f}s — "
          f"{len(results['generated_speech_tokens'])} turns captured")

    sync_wavs = results["generated_wavs"]
    per_turn_tokens = results["generated_speech_tokens"]

    # ---- Step 2: Replay the SAME tokens through chunked streaming flow. ------
    flow_states = prepare_flow_state(model, prepared)
    spk_ids = prepared["spk_ids"]
    out_dir = _Path("outputs/flow_direct")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always save sync.
    for i, w in enumerate(sync_wavs):
        torchaudio.save(str(out_dir / f"turn{i}_sync.wav"), w.float().cpu(), 24000)
    sync_full = torch.cat([w.cpu() for w in sync_wavs], dim=-1).float()
    torchaudio.save(str(out_dir / "full_sync.wav"), sync_full, 24000)
    sync_rms, sync_peak = report("SYNC   (streaming=False, single flow call per turn)", sync_wavs)

    chunk_sweep = ([int(x) for x in args.sweep.split(",")]
                   if args.sweep else [args.chunk])

    summary = []  # (chunk_size, rms_max_min_dB, peak_max_min_dB, per_turn_deltas_max_dB)
    for chunk in chunk_sweep:
        fcs_default = args.first_chunk if args.first_chunk <= chunk else chunk
        print(f"\n[step 2/2] replay tokens through streaming flow "
              f"(chunk_size={chunk}, first_chunk={fcs_default}, steps={args.steps})")
        stream_wavs = []
        t0 = time.perf_counter()
        for turn_i, (tokens, spk_id) in enumerate(zip(per_turn_tokens, spk_ids)):
            state = flow_states[spk_id]
            fcs = fcs_default if turn_i == 0 else chunk
            wav = synth_stream(model, state, tokens,
                               flow_steps=args.steps,
                               chunk_size=chunk, first_chunk_size=fcs)
            stream_wavs.append(wav)
        print(f"  done in {time.perf_counter()-t0:.1f}s")

        # Save per-chunk-size full wav
        for i, w in enumerate(stream_wavs):
            torchaudio.save(str(out_dir / f"turn{i}_stream_chunk{chunk}.wav"),
                            w.float().cpu(), 24000)
        full = torch.cat([w.cpu() for w in stream_wavs], dim=-1).float()
        torchaudio.save(str(out_dir / f"full_stream_chunk{chunk}.wav"), full, 24000)

        stream_rms, stream_peak = report(
            f"STREAM chunk={chunk} (streaming=True, same tokens)", stream_wavs)

        # Per-turn delta vs sync
        print(f"\n  Per-turn delta vs SYNC (chunk={chunk}):")
        print(f"  {'turn':>4} {'sync_rms_dB':>12} {'stream_rms_dB':>14} {'Δ dB':>8}  "
              f"{'sync_peak':>9} {'stream_peak':>11} {'peak Δ dB':>10}")
        max_rms_delta = 0.0
        max_peak_delta = 0.0
        for i, (sa, sb, pa, pb) in enumerate(zip(sync_rms, stream_rms, sync_peak, stream_peak)):
            rms_d = 20*np.log10(sb/sa); peak_d = 20*np.log10(pb/pa)
            max_rms_delta = max(max_rms_delta, abs(rms_d))
            max_peak_delta = max(max_peak_delta, abs(peak_d))
            print(f"  {i:>4} {20*np.log10(sa):>12.2f} {20*np.log10(sb):>14.2f} "
                  f"{rms_d:>+8.2f}  {pa:>9.3f} {pb:>11.3f} {peak_d:>+10.2f}")

        r = np.array(stream_rms); p = np.array(stream_peak)
        summary.append({
            "chunk": chunk,
            "stream_rms_range_dB": 20*np.log10(r.max()/r.min()),
            "stream_peak_range_dB": 20*np.log10(p.max()/p.min()),
            "max_abs_rms_delta_vs_sync_dB": max_rms_delta,
            "max_abs_peak_delta_vs_sync_dB": max_peak_delta,
        })

    # ---- Final sweep summary -------------------------------------------------
    print("\n" + "=" * 80)
    print("CHUNK-SIZE SWEEP SUMMARY (sync per-turn RMS spread: "
          f"{20*np.log10(np.array(sync_rms).max()/np.array(sync_rms).min()):.2f} dB, "
          f"peak spread: {20*np.log10(np.array(sync_peak).max()/np.array(sync_peak).min()):.2f} dB)")
    print("=" * 80)
    print(f"  {'chunk':>6} {'rms range':>10} {'peak range':>11} "
          f"{'max |Δrms|':>11} {'max |Δpeak|':>12}  (vs SYNC, per-turn)")
    for s in summary:
        print(f"  {s['chunk']:>6} {s['stream_rms_range_dB']:>9.2f}dB {s['stream_peak_range_dB']:>10.2f}dB "
              f"{s['max_abs_rms_delta_vs_sync_dB']:>10.2f}dB {s['max_abs_peak_delta_vs_sync_dB']:>11.2f}dB")
    print(f"\nWAVs saved under {out_dir}/")


if __name__ == "__main__":
    main()
