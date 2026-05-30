#!/usr/bin/env python3
"""Apply small compatibility patches to a local vllm-project/speculators checkout."""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "# SoulX compatibility: DataLoader samplers may pass numpy integer scalars."
OLD = """    def __getitem__(self, index) -> BatchType | None:
        data = self._get_raw_data(index)
"""
NEW = f"""    def __getitem__(self, index) -> BatchType | None:
        {PATCH_MARKER}
        if hasattr(index, "item"):
            index = int(index.item())
        else:
            index = int(index)
        data = self._get_raw_data(index)
"""


def patch_data_py(speculators_root: Path) -> bool:
    path = speculators_root / "src" / "speculators" / "train" / "data.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return False
    if OLD not in text:
        raise RuntimeError(f"Expected BaseDataset.__getitem__ block not found in {path}")

    path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speculators-root",
        default="third_party/speculators",
        help="Path to the vllm-project/speculators source checkout.",
    )
    args = parser.parse_args()

    changed = patch_data_py(Path(args.speculators_root))
    print(
        "patched speculators checkout"
        if changed
        else "speculators checkout already patched"
    )


if __name__ == "__main__":
    main()
