"""Bundle a trained PhonemeComposer into a SoulX model directory.

Makes a model dir "inpaint-capable" by dropping a slim, inference-only
``composer.pt`` next to ``flow.pt`` / ``hift.pt`` / ``model.safetensors`` and
recording an ``inpaint`` block in ``soulxpodcast_config.json``. Mirrors how the
flow variant (MeanFlow) is detected by checkpoint contents — presence of the
bundled composer is the capability signal (see ``inpaint_capability``).

Why slim: a training ``composer.pt`` is ~247 MB (composer 86 MB + AdamW
optimizer state 172 MB). Inference only needs ``config`` + ``composer``, so we
strip ``optimizer`` / ``scheduler`` → ~86 MB.

Why a sibling file (not merged into model.safetensors): the LLM is loaded by
HF ``AutoModelForCausalLM`` (rejects unknown keys) and is AWQ-quantized in
production (can't add fp16 tensors to an int4 checkpoint). A sibling keeps the
composer in native fp16 and ties it to the exact trunk it was trained on.

Usage::

    python scripts/inpaint/bundle_composer.py \
        --composer outputs/inpaint_final/step_0030000/composer.pt \
        --model_dir /path/to/SoulX-Podcast-1.7B-dialect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.training.inpaint_dataset import LANG_TO_ALPHABET


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--composer", required=True, help="trained composer.pt (with optimizer state)")
    ap.add_argument("--model_dir", required=True, help="SoulX model dir to bundle into")
    ap.add_argument("--config_name", default="soulxpodcast_config.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"model_dir not found: {model_dir}")

    ckpt = torch.load(args.composer, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    # Slim, inference-only bundle: config + composer weights (+ provenance).
    slim = {
        "step": ckpt.get("step"),
        "config": cfg,
        "composer": ckpt["composer"],
    }
    out = model_dir / "composer.pt"
    torch.save(slim, out)
    n_params = sum(v.numel() for v in ckpt["composer"].values() if torch.is_tensor(v))
    print(f"wrote {out}  ({out.stat().st_size/1e6:.0f} MB, {n_params/1e6:.1f}M params, "
          f"d_model={cfg['d_model']} K={cfg['slots_per_token']})")

    # Record capability + config in soulxpodcast_config.json so it travels with
    # the model and is readable without loading the composer.
    cfg_path = model_dir / args.config_name
    meta = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    meta["inpaint"] = {
        "composer": "composer.pt",
        "slots_per_token": cfg["slots_per_token"],
        "vocab_size": cfg["vocab_size"],
        "d_model": cfg["d_model"],
        "alphabets": sorted(set(LANG_TO_ALPHABET.values())),
        "trained_step": ckpt.get("step"),
    }
    cfg_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"updated {cfg_path}  → inpaint block: {meta['inpaint']}")
    print("model dir is now inpaint-capable (composer.pt present + config marked).")


if __name__ == "__main__":
    main()
