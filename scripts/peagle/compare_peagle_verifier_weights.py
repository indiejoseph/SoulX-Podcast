#!/usr/bin/env python3
"""Compare verifier tensors used by a P-EAGLE checkpoint and runtime model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_TENSORS = [
    "lm_head.weight",
    "model.norm.weight",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verifier_path_from_checkpoint(checkpoint: Path) -> str | None:
    config = read_json(checkpoint / "config.json")
    if not config:
        return None
    speculators_config = config.get("speculators_config") or {}
    verifier = speculators_config.get("verifier") or {}
    value = verifier.get("name_or_path")
    return str(value) if value else None


def config_summary(model_path: Path) -> dict[str, Any]:
    config = read_json(model_path / "config.json") or {}
    keys = [
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "speech_token_offset",
        "torch_dtype",
    ]
    return {key: config.get(key) for key in keys if key in config}


def tensor_file_for_name(model_path: Path, tensor_name: str) -> Path | None:
    single = model_path / "model.safetensors"
    if single.exists():
        return single

    index = read_json(model_path / "model.safetensors.index.json")
    if index:
        weight_map = index.get("weight_map") or {}
        filename = weight_map.get(tensor_name)
        if filename:
            return model_path / filename

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(f"safetensors import failed: {exc}") from exc

    for candidate in sorted(model_path.glob("*.safetensors")):
        with safe_open(candidate, framework="pt", device="cpu") as handle:
            if tensor_name in handle.keys():
                return candidate
    return None


def tensor_fingerprint(model_path: Path, tensor_name: str) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        return {"name": tensor_name, "error": f"import failed: {exc}"}

    tensor_file = tensor_file_for_name(model_path, tensor_name)
    if tensor_file is None:
        return {"name": tensor_name, "error": "tensor not found"}

    with safe_open(tensor_file, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(tensor_name).detach().cpu().contiguous()

    digest = hashlib.sha256()
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {
        "name": tensor_name,
        "file": str(tensor_file),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": digest.hexdigest(),
    }


def compare(
    *,
    training_model: Path,
    runtime_model: Path,
    tensor_names: list[str],
) -> tuple[dict[str, Any], list[str]]:
    report: dict[str, Any] = {
        "training_model": str(training_model),
        "runtime_model": str(runtime_model),
        "training_config": config_summary(training_model),
        "runtime_config": config_summary(runtime_model),
        "tensors": [],
    }
    errors: list[str] = []

    if report["training_config"] != report["runtime_config"]:
        errors.append("config.json selected fields differ")

    for tensor_name in tensor_names:
        training = tensor_fingerprint(training_model, tensor_name)
        runtime = tensor_fingerprint(runtime_model, tensor_name)
        same = (
            training.get("shape") == runtime.get("shape")
            and training.get("dtype") == runtime.get("dtype")
            and training.get("sha256") == runtime.get("sha256")
            and "error" not in training
            and "error" not in runtime
        )
        report["tensors"].append(
            {
                "name": tensor_name,
                "match": same,
                "training": training,
                "runtime": runtime,
            }
        )
        if not same:
            errors.append(f"tensor mismatch: {tensor_name}")

    return report, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "P-EAGLE checkpoint directory. If --training-model is omitted, "
            "the verifier path is read from checkpoint/config.json."
        ),
    )
    parser.add_argument(
        "--training-model",
        type=Path,
        help="Verifier path used by Speculators training/validation.",
    )
    parser.add_argument(
        "--runtime-model",
        type=Path,
        required=True,
        help="Verifier path used by deployed runtime.",
    )
    parser.add_argument(
        "--tensor",
        action="append",
        dest="tensors",
        default=[],
        help=(
            "Tensor name to compare. Can be repeated. Defaults to lm_head.weight "
            "and model.norm.weight."
        ),
    )
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    training_model = args.training_model
    checkpoint_verifier = None
    if args.checkpoint:
        checkpoint_verifier = verifier_path_from_checkpoint(args.checkpoint)
        if training_model is None and checkpoint_verifier:
            training_model = Path(checkpoint_verifier)

    if training_model is None:
        raise SystemExit(
            "Provide --training-model, or provide --checkpoint whose config.json "
            "contains speculators_config.verifier.name_or_path."
        )

    report, errors = compare(
        training_model=training_model,
        runtime_model=args.runtime_model,
        tensor_names=args.tensors or DEFAULT_TENSORS,
    )
    if checkpoint_verifier is not None:
        report["checkpoint_verifier_name_or_path"] = checkpoint_verifier
    report["errors"] = errors
    print(json.dumps(report, indent=2, sort_keys=True))

    if errors and args.fail_on_mismatch:
        raise SystemExit("P-EAGLE verifier weights differ")


if __name__ == "__main__":
    main()
