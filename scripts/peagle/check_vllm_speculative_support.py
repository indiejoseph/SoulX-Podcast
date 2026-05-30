#!/usr/bin/env python3
"""Check whether the installed vLLM exposes offline speculative_config support."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import vllm
        from vllm import EngineArgs
    except ImportError as exc:
        print(f"vLLM import failed: {exc}", file=sys.stderr)
        return 1

    version = getattr(vllm, "__version__", "unknown")
    signature = inspect.signature(EngineArgs)
    params = set(signature.parameters)
    has_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    has_speculative_config = has_var_kwargs or "speculative_config" in params

    print(f"vLLM version: {version}")
    print(f"EngineArgs.speculative_config: {has_speculative_config}")
    try:
        from soulxpodcast.engine.vllm_ras import install_soulx_vllm_ras_sampler

        install_soulx_vllm_ras_sampler()
        print("SoulX Qwen3 model-owned RAS sampler: available")
    except Exception as exc:
        print(f"SoulX Qwen3 model-owned RAS sampler: unavailable ({exc})")
        if has_speculative_config:
            print(
                "P-EAGLE can start in this runtime, but SoulX RAS will not be "
                "preserved unless the vLLM V1 model sampler hook is available."
            )
    if not has_speculative_config:
        print(
            "P-EAGLE via VLLM_SPECULATIVE_CONFIG requires a newer "
            "vLLM/speculators runtime than the default patched vLLM 0.10.1 image."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
