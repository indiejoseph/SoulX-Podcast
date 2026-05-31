"""Convert Chatterbox `s3gen_meanflow.safetensors` into a SoulX-format flow.pt.

The Chatterbox `s3gen_meanflow.safetensors` checkpoint packs the full
S3Token2Wav stack (flow + mel2wav HiFiGAN + speaker_encoder + tokenizer).
Our model only needs the `flow.*` slice; the rest of the components
(HiFTGenerator, CAMPPlus, s3tokenizer) are already loaded from our own
checkpoints. This script extracts just the flow tensors, strips the
`flow.` prefix to match our state_dict layout, and writes a torch
checkpoint that can be dropped in as `flow.pt`.

Key compatibility (verified against runs/merged/flow.pt):
  - 1121 / 1121 keys match exactly, zero shape mismatches.
  - One additional key in theirs: `decoder.estimator.time_embed_mixer.weight`
    of shape (1024, 2048). Our MeanFlowTimeMixer must be the matching
    bias-free Linear (handled in estimator.py).

Usage:
    python scripts/inference/convert_chatterbox_meanflow.py \
        --src tmp/chatterbox/s3gen_meanflow.safetensors \
        --dst pretrained_models/SoulX-Podcast-1.7B-dialect-meanflow/flow.pt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to s3gen_meanflow.safetensors")
    ap.add_argument("--dst", required=True, help="output .pt path")
    ap.add_argument("--reference", default=None,
                    help="optional existing flow.pt to validate key + shape compat against")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {src} ({src.stat().st_size/1e9:.2f} GB)")
    extracted = {}
    with safe_open(str(src), framework="pt") as f:
        for k in f.keys():
            if not k.startswith("flow."):
                continue
            extracted[k[len("flow."):]] = f.get_tensor(k)
    print(f"[extract] kept {len(extracted)} flow.* keys "
          f"({sum(v.numel() for v in extracted.values()):,} params, "
          f"{sum(v.numel()*v.element_size() for v in extracted.values())/1e6:.1f} MB)")

    # Sanity check the meanflow marker is present.
    mixer_key = "decoder.estimator.time_embed_mixer.weight"
    assert mixer_key in extracted, (
        f"expected meanflow marker {mixer_key} not found in source — is this "
        f"really the MeanFlow checkpoint and not the regular s3gen one?"
    )
    print(f"[check] meanflow marker {mixer_key} shape={tuple(extracted[mixer_key].shape)}")

    # Optional cross-check against an existing CFM flow.pt.
    if args.reference:
        ref = torch.load(args.reference, map_location="cpu", weights_only=True)
        only_ref = set(ref) - set(extracted)
        only_ours = set(extracted) - set(ref)
        if only_ref:
            print(f"[warn] {len(only_ref)} keys in reference but missing here: "
                  f"{sorted(only_ref)[:5]} ...")
        if only_ours - {mixer_key}:
            print(f"[warn] {len(only_ours - {mixer_key})} unexpected new keys: "
                  f"{sorted(only_ours - {mixer_key})[:5]} ...")
        shape_mism = [k for k in (set(ref) & set(extracted))
                      if ref[k].shape != extracted[k].shape]
        if shape_mism:
            print(f"[fatal] {len(shape_mism)} shape mismatches: {shape_mism[:5]}")
            raise SystemExit(2)
        print(f"[check] reference compat OK ({len(set(ref) & set(extracted))} shared keys, "
              f"no shape mismatches)")

    torch.save(extracted, str(dst))
    print(f"[done] wrote {dst} ({dst.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
