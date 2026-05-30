#!/usr/bin/env python3
"""Apply small compatibility patches to a local vllm-project/speculators checkout."""

from __future__ import annotations

import argparse
from pathlib import Path


INDEX_PATCH_MARKER = "# SoulX compatibility: DataLoader samplers may pass numpy integer scalars."
ONLINE_FILE_PATCH_MARKER = "# SoulX compatibility: wait for online generated hidden-state files."
PEAGLE_POSITION_PATCH_MARKER = "# SoulX compatibility: preserve packed-sample position_ids for P-EAGLE."

OLD_INDEX = """    def __getitem__(self, index) -> BatchType | None:
        data = self._get_raw_data(index)
"""
NEW_INDEX = f"""    def __getitem__(self, index) -> BatchType | None:
        {INDEX_PATCH_MARKER}
        if hasattr(index, "item"):
            index = int(index.item())
        else:
            index = int(index)
        data = self._get_raw_data(index)
"""

OLD_IMPORTS = """import shutil
import warnings
"""
NEW_IMPORTS = """import shutil
import time
import warnings
"""

OLD_ONLINE_LOAD_VARIANTS = [
    """            loaded_hs = load_file(hs_filepath)

            match self.on_generate:
                case "cache":
                    file_idx = self._map_to_file_idx(index)
                    target_path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
                    shutil.move(hs_filepath, target_path)
                case "delete":
                    Path(hs_filepath).unlink()
""",
    """            loaded_hs = load_file(hs_filepath)
            match self.on_generate:
                case "cache":
                    file_idx = self._map_to_file_idx(index)
                    target_path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
                    shutil.move(hs_filepath, target_path)
                case "delete":
                    Path(hs_filepath).unlink()
""",
]
NEW_ONLINE_LOAD = f"""            {ONLINE_FILE_PATCH_MARKER}
            hs_path = Path(hs_filepath)
            hs_lock_path = str(hs_path) + ".lock"
            load_error = None
            loaded_hs = None
            for _ in range(100):
                if Path(hs_lock_path).exists():
                    wait_for_lock(hs_lock_path)
                if hs_path.exists():
                    try:
                        loaded_hs = load_file(hs_path)
                        break
                    except Exception as exc:  # noqa: BLE001
                        load_error = exc
                time.sleep(0.1)
            if loaded_hs is None:
                if load_error is not None:
                    raise load_error
                raise FileNotFoundError(hs_path)
            match self.on_generate:
                case "cache":
                    file_idx = self._map_to_file_idx(index)
                    target_path = self.hidden_states_path / f"hs_{{file_idx}}.safetensors"
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if hs_path.exists():
                        shutil.move(str(hs_path), target_path)
                case "delete":
                    hs_path.unlink(missing_ok=True)
"""


def patch_data_py(speculators_root: Path) -> bool:
    path = speculators_root / "src" / "speculators" / "train" / "data.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    text = path.read_text(encoding="utf-8")
    changed = False
    if INDEX_PATCH_MARKER not in text:
        if OLD_INDEX not in text:
            raise RuntimeError(
                f"Expected BaseDataset.__getitem__ block not found in {path}"
            )
        text = text.replace(OLD_INDEX, NEW_INDEX)
        changed = True

    if ONLINE_FILE_PATCH_MARKER not in text:
        if "import time\n" not in text:
            if OLD_IMPORTS not in text:
                raise RuntimeError(f"Expected import block not found in {path}")
            text = text.replace(OLD_IMPORTS, NEW_IMPORTS)
        old_online_load = next(
            (variant for variant in OLD_ONLINE_LOAD_VARIANTS if variant in text),
            None,
        )
        if old_online_load is None:
            raise RuntimeError(
                f"Expected ArrowDataset._maybe_generate_hs load block not found in {path}"
            )
        text = text.replace(old_online_load, NEW_ONLINE_LOAD)
        changed = True

    if not changed:
        return False

    path.write_text(text, encoding="utf-8")
    return True


def patch_peagle_core_py(speculators_root: Path) -> bool:
    path = speculators_root / "src" / "speculators" / "models" / "peagle" / "core.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    text = path.read_text(encoding="utf-8")
    if PEAGLE_POSITION_PATCH_MARKER in text:
        if (
            "sampled_position_ids = position_ids[:, orig_positions]" not in text
            or "position_ids=sampled_position_ids" not in text
        ):
            raise RuntimeError(
                f"{path} contains the P-EAGLE position patch marker, but the "
                "patched position_ids wiring is incomplete."
            )
        return False

    old = """        position_ids = orig_positions.unsqueeze(0)  # [1, total_sampled]\n\n        position_embeddings = self.rotary_emb(layer_input, position_ids)\n"""
    new = f"""        {PEAGLE_POSITION_PATCH_MARKER}\n        if position_ids is None:\n            sampled_position_ids = orig_positions.unsqueeze(0)\n        else:\n            sampled_position_ids = position_ids[:, orig_positions]\n\n        position_embeddings = self.rotary_emb(layer_input, sampled_position_ids)\n"""
    if old not in text:
        raise RuntimeError(f"Expected P-EAGLE position_ids block not found in {path}")
    text = text.replace(old, new)

    old = """                position_ids=position_ids,\n                position_embeddings=position_embeddings,\n"""
    new = """                position_ids=sampled_position_ids,\n                position_embeddings=position_embeddings,\n"""
    if old not in text:
        raise RuntimeError(f"Expected P-EAGLE decoder position_ids call not found in {path}")
    text = text.replace(old, new, 1)

    path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speculators-root",
        default="third_party/speculators",
        help="Path to the vllm-project/speculators source checkout.",
    )
    args = parser.parse_args()

    speculators_root = Path(args.speculators_root)
    changed = patch_data_py(speculators_root)
    changed = patch_peagle_core_py(speculators_root) or changed
    print(
        "patched speculators checkout"
        if changed
        else "speculators checkout already patched"
    )


if __name__ == "__main__":
    main()
