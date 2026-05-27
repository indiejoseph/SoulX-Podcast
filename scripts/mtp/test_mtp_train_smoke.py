"""Smoke test: end-to-end MTP training step on a synthetic 4-sample dataset.

Verifies the full training pipeline works (model load → trunk forward → MTP
forward → loss → backward → step) without needing the real dataset to be
saved first. Catches integration bugs before the H100 run.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset

from soulxpodcast.training.train_mtp import TrainConfig, train


MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect"


def make_fake_dataset(out_path: Path, n: int = 8):
    """Build a tiny HF dataset with realistic-shape synthetic speech tokens."""
    rng = torch.Generator().manual_seed(0)
    rows = []
    for i in range(n):
        # Realistic ranges: 50-200 speech tokens per sample, 0-based.
        n_tok = int(torch.randint(50, 200, (1,), generator=rng).item())
        speech_ids = torch.randint(0, 6500, (n_tok,), generator=rng).tolist()
        speech_str = " ".join(str(x) for x in speech_ids)
        rows.append({
            "id": f"FAKE_{i:04d}",
            "text": f"This is fake sample number {i}.",
            "speech_tokens": speech_str,
            "lang": ["en", "zh", "yue"][i % 3],
        })
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(out_path))
    print(f"[fake] saved {n} synthetic samples → {out_path}")
    return out_path


def main():
    fake_dir = Path("/tmp/mtp_fake_ds")
    if fake_dir.exists():
        import shutil
        shutil.rmtree(fake_dir)
    make_fake_dataset(fake_dir, n=8)

    out_dir = Path("/tmp/mtp_smoke_run")
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    cfg = TrainConfig(
        dataset_path=str(fake_dir),
        model_path=MODEL_PATH,
        output_dir=str(out_dir),
        batch_size=2,
        num_epochs=1,
        max_steps=3,           # just enough to verify multiple optim steps
        lr=1e-4,
        warmup_steps=1,
        log_every=1,
        save_every=0,
        num_workers=0,
        num_mtp_layers=2,      # small for smoke
    )
    train(cfg)

    # Verify checkpoint exists & shape sanity.
    ckpt = out_dir / "mtp_final.pt"
    assert ckpt.exists(), f"checkpoint not saved: {ckpt}"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    n_params = sum(v.numel() for k, v in state["mtp_state"].items())
    print(f"\n[verify] checkpoint OK — mtp_state has {n_params/1e6:.1f}M params")
    print(f"[verify] mtp_config: {json.dumps(state['mtp_config'], indent=2)}")
    print(f"[verify] final step: {state['step']}")
    print(f"\n[smoke] PASSED")


if __name__ == "__main__":
    main()
