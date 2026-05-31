"""Per-piece loudness diagnostic for streaming flow.

Targets two specific artifacts heard in listening A/B:
  (1) turn 0 is louder than later turns in stream output
  (2) the END of each streamed turn is louder than the body

Mechanism candidates:
  (1) `first_chunk_size=4` for turn 0 only — emits a tiny first chunk, then
      accumulates. Hypothesis: the 4-token-only flow call renders with very
      little context and produces hotter audio. Test: rerun turn 0 with
      first_chunk_size = chunk_size (no special first chunk) and compare.
  (2) The final `finalize=True` flush re-runs flow on the full sequence with
      different attention context than the chunked `finalize=False` body.
      Hypothesis: the trailing audio (~last N samples) emitted from the flush
      is hotter than the rest. Test: measure per-piece RMS within each turn.

Usage:
    python scripts/inference/flow_piece_diagnostic.py [--chunk N] [--steps N]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import os
import time

import numpy as np
import torch
import torchaudio

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser

from flow_direct_sync_vs_stream import (
    DATA, MODEL_PATH, prepare_flow_state, synth_sync, synth_stream,
)


def rms_db_arr(x: torch.Tensor) -> float:
    y = x.float().cpu().numpy().reshape(-1)
    r = float(np.sqrt((y ** 2).mean()) + 1e-12)
    return 20 * np.log10(r)


def peak_arr(x: torch.Tensor) -> float:
    return float(x.float().abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=198964)
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

    print("\n[step 1/4] forward_longform → capture per-turn speech tokens")
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
    per_turn_tokens = results["generated_speech_tokens"]
    flow_states = prepare_flow_state(model, prepared)
    spk_ids = prepared["spk_ids"]

    # ---- Test 2: per-piece RMS profile across all turns ---------------------
    print(f"\n[step 2/4] STREAM with first_chunk=4 (current default) — "
          f"per-piece RMS within each turn (chunk={args.chunk})")
    print(f"  {'turn':>4} {'piece':>10} {'samples':>8} {'dur':>6} {'peak':>7} {'rms_dB':>8}")
    for turn_i, (tokens, spk_id) in enumerate(zip(per_turn_tokens, spk_ids)):
        state = flow_states[spk_id]
        fcs = 4 if turn_i == 0 else args.chunk
        _, pieces = synth_stream(model, state, tokens,
                                 flow_steps=args.steps,
                                 chunk_size=args.chunk, first_chunk_size=fcs,
                                 return_pieces=True)
        for label, piece in pieces:
            n = piece.shape[-1]
            print(f"  {turn_i:>4} {label:>10} {n:>8d} {n/24000:>5.2f}s "
                  f"{peak_arr(piece):>7.3f} {rms_db_arr(piece):>8.2f}")

    # ---- Test 1: turn 0 with first_chunk = chunk (no tiny first chunk) ------
    print(f"\n[step 3/4] TURN 0 ONLY — first_chunk_size sweep "
          f"(turn 0 has {len(per_turn_tokens[0])} tokens, chunk={args.chunk})")
    print(f"  Hypothesis: tiny first_chunk_size=4 makes turn 0 hot. "
          f"Larger first_chunk should match later turns.")
    print(f"  {'first_chunk':>12} {'peak':>7} {'rms_dB':>8} {'piece0_peak':>12} {'piece0_rms':>11}")
    tokens0 = per_turn_tokens[0]
    state0 = flow_states[spk_ids[0]]
    sync0 = synth_sync(model, state0, tokens0, flow_steps=args.steps)
    print(f"  {'(sync)':>12} {peak_arr(sync0):>7.3f} {rms_db_arr(sync0):>8.2f}  "
          f"{'(single flow call, finalize=True)':>30}")
    for fcs in [4, 16, 50, 100, args.chunk]:
        if fcs > len(tokens0):
            continue
        full, pieces = synth_stream(model, state0, tokens0,
                                    flow_steps=args.steps,
                                    chunk_size=args.chunk, first_chunk_size=fcs,
                                    return_pieces=True)
        p0_label, p0_audio = pieces[0]
        print(f"  {fcs:>12} {peak_arr(full):>7.3f} {rms_db_arr(full):>8.2f}  "
              f"{peak_arr(p0_audio):>11.3f} {rms_db_arr(p0_audio):>10.2f}  ({p0_label})")

    # ---- Test 3: isolate the flush piece ------------------------------------
    print(f"\n[step 4/4] FLUSH PIECE — compare to body of same turn "
          f"(chunk={args.chunk}, first_chunk=4)")
    print(f"  Hypothesis: trailing audio from finalize=True flush is hotter "
          f"than body of finalize=False chunks.")
    print(f"  {'turn':>4} {'body_rms_dB':>12} {'flush_rms_dB':>14} {'Δ dB':>8}  "
          f"{'body_peak':>10} {'flush_peak':>11} {'peak Δ dB':>10}")
    for turn_i, (tokens, spk_id) in enumerate(zip(per_turn_tokens, spk_ids)):
        state = flow_states[spk_id]
        fcs = 4 if turn_i == 0 else args.chunk
        _, pieces = synth_stream(model, state, tokens,
                                 flow_steps=args.steps,
                                 chunk_size=args.chunk, first_chunk_size=fcs,
                                 return_pieces=True)
        # Body = concat of all finalize=False pieces; flush = last piece
        body = torch.cat([p for lbl, p in pieces if lbl != "flush"], dim=-1)
        flush = pieces[-1][1]
        body_rms = rms_db_arr(body); flush_rms = rms_db_arr(flush)
        body_peak = peak_arr(body); flush_peak = peak_arr(flush)
        print(f"  {turn_i:>4} {body_rms:>12.2f} {flush_rms:>14.2f} "
              f"{flush_rms - body_rms:>+8.2f}  "
              f"{body_peak:>10.3f} {flush_peak:>11.3f} "
              f"{20*np.log10(flush_peak/body_peak):>+10.2f}")


if __name__ == "__main__":
    main()
