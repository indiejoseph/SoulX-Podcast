#!/usr/bin/env python3
"""Write a vLLM speculative_config JSON file for a trained P-EAGLE model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _json_object(value: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speculator-model",
        required=True,
        help="Path or Hugging Face id for the trained speculator checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="exports/peagle/speculative_config.json",
        help="Where to write the JSON file.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=3,
        help="Draft tokens requested from vLLM per verifier step.",
    )
    parser.add_argument(
        "--method",
        default="peagle",
        help="vLLM speculative decoding method string for the installed runtime.",
    )
    parser.add_argument(
        "--extra-json",
        type=_json_object,
        default={},
        help="Extra JSON object to merge into the config for runtime-specific keys.",
    )
    args = parser.parse_args()

    config = {
        "model": args.speculator_model,
        "num_speculative_tokens": args.num_speculative_tokens,
        "method": args.method,
    }
    config.update(args.extra_json)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
