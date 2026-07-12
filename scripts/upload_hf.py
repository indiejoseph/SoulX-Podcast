"""Upload the AWQ + MeanFlow + InPaint model to HuggingFace.

Files sourced from:
  - tmp/meanflow_awq_model/  — AWQ-INT4 LLM + MeanFlow flow + vocoder
  - runs/merged/composer.pt  — v12 inpaint composer (added to upload)
  - runs/merged/soulxpodcast_config.json — has the inpaint config section

Usage:
    source .venv/bin/activate
    huggingface-cli login          # one-time
    python scripts/upload_hf.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_TMP = ROOT / "tmp" / "meanflow_awq_model"
MERGED_DIR = Path("/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged")

REPO_ID = "indiejoseph/SoulX-Podcast-1.7B-AWQ-MeanFlow-InPaint"

# Files to include from meanflow_awq_model (skip broken symlinks)
INCLUDE = [
    "model.safetensors",    # AWQ-INT4 LLM  ~2.0 GB
    "flow.pt",              # MeanFlow flow ~438 MB
    "hift.pt",              # HiFT vocoder   ~80 MB
    "campplus.onnx",        # Speaker encoder ~27 MB
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "README.md",
]

# Skip: flow.cache.pt (CFM cache, not needed for MeanFlow)
#       flow.decoder.estimator.fp32.onnx (TRT only, disabled)


def resolve(path: Path) -> Path:
    """Follow symlinks to the real file."""
    return path.resolve()


def main():
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        sys.exit("huggingface_hub not installed — pip install huggingface-hub")

    api = HfApi()

    # Check login
    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        sys.exit("Not logged in — run: huggingface-cli login")

    # Create repo (private, ok if already exists)
    print(f"\nCreating repo: {REPO_ID} (private=True) ...")
    create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)

    # Build merged soulxpodcast_config.json with inpaint section
    awq_cfg_path  = resolve(MODEL_TMP / "soulxpodcast_config.json")
    merged_cfg_path = MERGED_DIR / "soulxpodcast_config.json"
    awq_cfg   = json.loads(awq_cfg_path.read_text())
    merged_cfg = json.loads(merged_cfg_path.read_text())
    if "inpaint" in merged_cfg:
        awq_cfg["inpaint"] = merged_cfg["inpaint"]
        print(f"  Merged inpaint config: {awq_cfg['inpaint']}")

    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)

        # Write merged config
        cfg_out = staging / "soulxpodcast_config.json"
        cfg_out.write_text(json.dumps(awq_cfg, indent=2, ensure_ascii=False))
        print(f"\nUploading soulxpodcast_config.json (merged with inpaint section) ...")
        api.upload_file(
            path_or_fileobj=str(cfg_out),
            path_in_repo="soulxpodcast_config.json",
            repo_id=REPO_ID,
            repo_type="model",
        )

    # Upload files from meanflow_awq_model
    for fname in INCLUDE:
        src = MODEL_TMP / fname
        if not src.exists():
            print(f"  ⚠️  skip {fname} (not found / broken symlink)")
            continue
        real = resolve(src)
        size_mb = real.stat().st_size / 1e6
        print(f"\nUploading {fname} ({size_mb:.0f} MB) from {real} ...")
        api.upload_file(
            path_or_fileobj=str(real),
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"  ✅ {fname}")

    # Upload composer.pt from runs/merged
    composer_src = MERGED_DIR / "composer.pt"
    if composer_src.exists():
        size_mb = composer_src.stat().st_size / 1e6
        print(f"\nUploading composer.pt ({size_mb:.0f} MB) from {composer_src} ...")
        api.upload_file(
            path_or_fileobj=str(composer_src),
            path_in_repo="composer.pt",
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"  ✅ composer.pt")
    else:
        print(f"  ⚠️  composer.pt not found at {composer_src}")

    print(f"\n🎉 Upload complete!")
    print(f"   https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
