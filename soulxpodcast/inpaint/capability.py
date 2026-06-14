"""Inpaint capability detection for a SoulX model directory.

A model dir is "inpaint-capable" if it carries a bundled ``composer.pt`` next
to ``flow.pt`` / ``hift.pt`` (see ``scripts/inpaint/bundle_composer.py``). This
mirrors the MeanFlow flow-variant detection (presence of marker → enable mode)
and lets the engine / API auto-discover support without an explicit
``--composer_ckpt``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

COMPOSER_FILENAME = "composer.pt"
_CONFIG_NAMES = ("soulxpodcast_config.json", "config.json")


def composer_path(model_path: str | Path) -> Optional[Path]:
    """Return the bundled composer path if present in the model dir, else None."""
    p = Path(model_path) / COMPOSER_FILENAME
    return p if p.exists() else None


def inpaint_capability(model_path: str | Path) -> Optional[dict]:
    """Inspect a model dir and report inpaint support.

    Returns ``None`` if the dir is not inpaint-capable (no bundled composer),
    else a dict ``{"composer": <abs path>, "alphabets": [...], "slots_per_token":
    int, ...}`` merging the bundled file's presence with any ``inpaint`` block
    recorded in the model config by ``bundle_composer.py``.
    """
    cp = composer_path(model_path)
    if cp is None:
        return None
    info: dict = {"composer": str(cp)}
    model_dir = Path(model_path)
    for name in _CONFIG_NAMES:
        cfg = model_dir / name
        if cfg.exists():
            try:
                block = json.loads(cfg.read_text()).get("inpaint")
            except (json.JSONDecodeError, OSError):
                block = None
            if isinstance(block, dict):
                info.update(block)
                info["composer"] = str(model_dir / block.get("composer", COMPOSER_FILENAME))
                break
    return info
