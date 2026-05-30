#!/usr/bin/env python3
"""Inspect the runtime contract for a trained SoulX P-EAGLE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SMALL_HASH_LIMIT = 64 * 1024 * 1024
LARGE_HASH_WINDOW = 1024 * 1024


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path: Path, *, hash_large_files: bool = False) -> dict[str, Any]:
    stat = path.stat()
    report: dict[str, Any] = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if stat.st_size <= SMALL_HASH_LIMIT or hash_large_files:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        report["sha256"] = h.hexdigest()
        return report

    h = hashlib.sha256()
    with path.open("rb") as handle:
        h.update(handle.read(LARGE_HASH_WINDOW))
        handle.seek(max(0, stat.st_size - LARGE_HASH_WINDOW))
        h.update(handle.read(LARGE_HASH_WINDOW))
    report["sha256_head_tail_1mib"] = h.hexdigest()
    return report


def fingerprint_dir(path: Path, *, hash_large_files: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "model.safetensors.index.json",
    ]
    out = []
    for name in names:
        candidate = path / name
        if candidate.exists():
            out.append(fingerprint(candidate, hash_large_files=hash_large_files))
    if not out:
        for candidate in sorted(path.glob("*.safetensors"))[:8]:
            out.append(fingerprint(candidate, hash_large_files=hash_large_files))
    return out


def safetensor_meta(path: Path, *, wanted: set[str] | None = None) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        return {"path": str(path), "error": f"safetensors import failed: {exc}"}

    errors = []
    for framework_kwargs in (
        {"framework": "numpy"},
        {"framework": "np"},
        {"framework": "pt", "device": "cpu"},
    ):
        try:
            with safe_open(path, **framework_kwargs) as handle:
                keys = list(handle.keys())
                selected = keys
                if wanted is not None:
                    selected = [key for key in keys if key in wanted]
                elif len(keys) > 32:
                    selected = keys[:32]

                tensors: dict[str, Any] = {}
                for key in selected:
                    try:
                        tensor_slice = handle.get_slice(key)
                        shape = list(tensor_slice.get_shape())
                        dtype = (
                            str(tensor_slice.get_dtype())
                            if hasattr(tensor_slice, "get_dtype")
                            else None
                        )
                    except Exception:
                        tensor = handle.get_tensor(key)
                        shape = list(tensor.shape)
                        dtype = str(tensor.dtype)
                    tensors[key] = {"shape": shape, "dtype": dtype}
                return {
                    "path": str(path),
                    "framework": framework_kwargs["framework"],
                    "num_tensors": len(keys),
                    "tensors": tensors,
                }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{framework_kwargs}: {exc}")
    return {"path": str(path), "error": errors}


def first_safetensors(path: Path, *, limit: int) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file() and path.suffix == ".safetensors":
        return [path]
    preferred = path / "model.safetensors"
    if preferred.exists():
        return [preferred]
    return sorted(path.glob("*.safetensors"))[:limit]


def checkpoint_config_summary(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    tl_config = config.get("transformer_layer_config") or {}
    spec_config = config.get("speculators_config") or {}
    verifier = spec_config.get("verifier") or {}
    return {
        "architectures": config.get("architectures"),
        "speculators_model_type": config.get("speculators_model_type"),
        "draft_vocab_size": config.get("draft_vocab_size"),
        "num_depths": config.get("num_depths"),
        "mask_token_id": config.get("mask_token_id"),
        "norm_before_residual": config.get("norm_before_residual"),
        "eagle_aux_hidden_state_layer_ids": config.get(
            "eagle_aux_hidden_state_layer_ids"
        ),
        "transformer_layer_model_type": tl_config.get("model_type"),
        "transformer_layer_vocab_size": tl_config.get("vocab_size"),
        "transformer_layer_hidden_size": tl_config.get("hidden_size"),
        "verifier_name_or_path": verifier.get("name_or_path"),
    }


def parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part) for part in value.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-dir", default=None)
    parser.add_argument("--hidden-states-dir", default=None)
    parser.add_argument("--speculative-config", default=None)
    parser.add_argument("--expected-draft-vocab-size", type=int, default=None)
    parser.add_argument("--expected-num-depths", type=int, default=None)
    parser.add_argument("--expected-target-layer-ids", default=None)
    parser.add_argument("--expected-method", default="eagle3")
    parser.add_argument(
        "--expect-parallel-drafting",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-hidden-state-files", type=int, default=2)
    parser.add_argument("--hash-large-files", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    checkpoint = Path(args.checkpoint)
    preprocessed_dir = Path(args.preprocessed_dir) if args.preprocessed_dir else None
    hidden_states_dir = Path(args.hidden_states_dir) if args.hidden_states_dir else None
    spec_config_path = Path(args.speculative_config) if args.speculative_config else None
    expected_layers = parse_int_list(args.expected_target_layer_ids)

    ckpt_config = read_json(checkpoint / "config.json")
    ckpt_summary = checkpoint_config_summary(ckpt_config)
    spec_config = read_json(spec_config_path) if spec_config_path else None

    errors: list[str] = []
    if args.expected_draft_vocab_size is not None:
        if ckpt_summary.get("draft_vocab_size") != args.expected_draft_vocab_size:
            errors.append(
                "checkpoint draft_vocab_size "
                f"{ckpt_summary.get('draft_vocab_size')} != "
                f"{args.expected_draft_vocab_size}"
            )
    if args.expected_num_depths is not None:
        if ckpt_summary.get("num_depths") != args.expected_num_depths:
            errors.append(
                f"checkpoint num_depths {ckpt_summary.get('num_depths')} != "
                f"{args.expected_num_depths}"
            )
        if (
            spec_config is not None
            and spec_config.get("num_speculative_tokens") != args.expected_num_depths
        ):
            errors.append(
                "speculative_config num_speculative_tokens "
                f"{spec_config.get('num_speculative_tokens')} != {args.expected_num_depths}"
            )
    if spec_config is not None:
        if args.expected_method and spec_config.get("method") != args.expected_method:
            errors.append(
                f"speculative_config method {spec_config.get('method')!r} != "
                f"{args.expected_method!r}"
            )
        if (
            args.expect_parallel_drafting
            and spec_config.get("parallel_drafting") is not True
        ):
            errors.append("speculative_config parallel_drafting is not true")
    if expected_layers is not None:
        actual_layers = ckpt_summary.get("eagle_aux_hidden_state_layer_ids")
        if actual_layers != expected_layers:
            errors.append(
                f"checkpoint target layers {actual_layers} != {expected_layers}"
            )

    wanted_checkpoint_tensors = {
        "d2t",
        "t2d",
        "fc.weight",
        "fc.bias",
        "lm_head.weight",
        "embed_tokens.weight",
        "midlayer.weight",
        "midlayer.bias",
    }
    checkpoint_tensors = [
        safetensor_meta(path, wanted=wanted_checkpoint_tensors)
        for path in first_safetensors(checkpoint, limit=4)
    ]
    hidden_state_tensors = []
    if hidden_states_dir is not None:
        wanted_hs = {"hidden_states", "token_ids"}
        hidden_state_tensors = [
            safetensor_meta(path, wanted=wanted_hs)
            for path in first_safetensors(
                hidden_states_dir, limit=args.max_hidden_state_files
            )
        ]

    prepare_summary = None
    if preprocessed_dir is not None:
        prepare_summary = read_json(preprocessed_dir / "soulx_peagle_prepare_summary.json")

    report = {
        "model_path": str(model_path),
        "checkpoint": str(checkpoint),
        "model_fingerprints": fingerprint_dir(
            model_path, hash_large_files=args.hash_large_files
        ),
        "checkpoint_fingerprints": fingerprint_dir(
            checkpoint, hash_large_files=args.hash_large_files
        ),
        "checkpoint_config": ckpt_summary,
        "speculative_config": spec_config,
        "prepare_summary": prepare_summary,
        "checkpoint_tensors": checkpoint_tensors,
        "hidden_state_tensors": hidden_state_tensors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors and args.fail_on_error:
        raise SystemExit("Invalid P-EAGLE runtime contract")


if __name__ == "__main__":
    main()
