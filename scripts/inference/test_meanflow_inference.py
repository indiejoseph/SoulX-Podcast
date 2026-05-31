"""End-to-end inference test with the Chatterbox-derived MeanFlow flow weights.

Runs the same 4-turn Cantonese dialect dialogue used by
`flow_direct_sync_vs_stream.py` through a model whose flow.pt has been
replaced with the converted Chatterbox `s3gen_meanflow.safetensors` weights
(see `convert_chatterbox_meanflow.py`). Compares the resulting audio to the
original CFM-based output and measures per-turn loudness.

The two key questions this answers:
  (1) Does inference run at all (architecture + weight compat)?
  (2) Is the audio intelligible on Cantonese — the language Chatterbox was
      NOT trained on (their model is English-only per the README)?

Usage:
    python scripts/inference/test_meanflow_inference.py [--steps 1] [--model-path PATH]
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

from flow_direct_sync_vs_stream import DATA  # reuse the 4-turn Cantonese dialogue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="tmp/meanflow_model",
                    help="path with flow.pt = chatterbox meanflow weights")
    ap.add_argument("--steps", type=int, default=1,
                    help="n_timesteps; 1 is the distilled 1-step inference target")
    ap.add_argument("--seed", type=int, default=198964)
    args = ap.parse_args()

    os.chdir(_Path(__file__).resolve().parents[2])

    print(f"[init] model_path={args.model_path}  flow_steps={args.steps}  seed={args.seed}")
    t0 = time.perf_counter()
    model, dataset = initiate_model(seed=args.seed, model_path=args.model_path,
                                    llm_engine="hf", fp16_flow=True)
    print(f"[init] load: {time.perf_counter()-t0:.1f}s")
    print(f"[init] flow.meanflow = {model.flow.meanflow}")
    print(f"[init] decoder.meanflow = {model.flow.decoder.meanflow}")
    assert model.flow.meanflow, "expected meanflow=True after auto-detect; check flow.pt"

    inputs = podcast_format_parser(DATA)
    prepared = process_single_input(
        dataset, inputs["text"], inputs["prompt_wav"], inputs["prompt_text"],
        inputs["use_dialect_prompt"], inputs["dialect_prompt_text"],
    )

    print("\n[run] forward_longform — LLM + MeanFlow flow + HiFT")
    t0 = time.perf_counter()
    # Override flow_steps via the decoder default — forward_longform calls
    # flow(... n_timesteps=15) by default. Patch the flow.forward to use args.steps.
    orig_forward = model.flow.forward
    def _forward_with_steps(*a, **kw):
        kw["n_timesteps"] = args.steps
        return orig_forward(*a, **kw)
    model.flow.forward = _forward_with_steps

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
    wall = time.perf_counter() - t0
    print(f"[run] wall: {wall:.1f}s")

    # Save outputs
    out_dir = _Path("outputs/meanflow_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = results["generated_wavs"]
    total_samples = sum(w.shape[-1] for w in wavs)
    total_dur = total_samples / 24000.0
    print(f"\n[result] {len(wavs)} turns, {total_dur:.1f}s audio, RTF={wall/total_dur:.3f}")

    for i, w in enumerate(wavs):
        torchaudio.save(str(out_dir / f"turn{i}_meanflow.wav"),
                        w.float().cpu(), 24000)
        y = w.float().cpu().numpy().reshape(-1)
        n = len(y); dur = n / 24000.0
        peak = float(np.abs(y).max())
        rms = float(np.sqrt((y ** 2).mean()) + 1e-12)
        finite = bool(np.isfinite(y).all())
        print(f"  turn {i}: {n:>7d} samples, {dur:.2f}s, peak={peak:.3f}, "
              f"rms_dB={20*np.log10(rms):+.2f}, finite={finite}")

    full = torch.cat([w.cpu() for w in wavs], dim=-1).float()
    torchaudio.save(str(out_dir / "full_meanflow.wav"), full, 24000)
    print(f"\n[done] outputs under {out_dir}/")


if __name__ == "__main__":
    main()
