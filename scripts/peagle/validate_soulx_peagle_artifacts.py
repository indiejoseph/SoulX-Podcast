#!/usr/bin/env python3
"""Validate SoulX P-EAGLE vocab artifacts before and after training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_token_id(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Tokenizer did not encode {token!r} as one token: {ids}")
    return int(ids[0])


def load_target_vocab_size(model_path: str) -> int:
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, "text_config"):
        config = config.text_config
    return int(config.vocab_size)


def load_token_freq(path: Path) -> dict[int, int]:
    raw = torch_load(path)
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a dict[token_id, count], got {type(raw)}")
    return {int(k): int(v) for k, v in raw.items()}


def build_vocab_mapping(
    *,
    token_freq: dict[int, int],
    draft_vocab_size: int,
    target_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Match vllm-project/speculators train.vocab_mapping implementation."""
    sorted_tokens = sorted(token_freq, key=lambda tid: (-token_freq[tid], tid))
    selected_token_ids = sorted_tokens[: min(draft_vocab_size, len(sorted_tokens))]
    if len(selected_token_ids) < draft_vocab_size:
        current_ids = set(selected_token_ids)
        for tid in range(draft_vocab_size):
            if tid not in current_ids:
                selected_token_ids.append(tid)
            if len(selected_token_ids) >= draft_vocab_size:
                break
    selected_token_ids.sort()

    selected = torch.tensor(selected_token_ids, dtype=torch.long)
    d2t = selected - torch.arange(draft_vocab_size, dtype=torch.long)
    t2d = torch.zeros(target_vocab_size, dtype=torch.bool)
    valid = selected[(selected >= 0) & (selected < target_vocab_size)]
    t2d[valid] = True
    return d2t, t2d, selected_token_ids


def classify_ids(
    token_ids: list[int],
    *,
    speech_start: int,
    speech_vocab_size: int,
    allowed_control_ids: set[int],
) -> dict[str, int]:
    speech_end = speech_start + speech_vocab_size
    speech = 0
    control = 0
    non_speech = 0
    for token_id in token_ids:
        if speech_start <= token_id < speech_end:
            speech += 1
        elif token_id in allowed_control_ids:
            control += 1
        else:
            non_speech += 1
    return {
        "speech": speech,
        "control": control,
        "non_speech": non_speech,
    }


def validate_counts(
    *,
    name: str,
    counts: dict[str, int],
    min_speech_tokens: int,
    max_non_speech_tokens: int,
) -> list[str]:
    errors = []
    if counts["speech"] < min_speech_tokens:
        errors.append(
            f"{name}: only {counts['speech']} speech tokens; "
            f"expected at least {min_speech_tokens}"
        )
    if counts["non_speech"] > max_non_speech_tokens:
        errors.append(
            f"{name}: {counts['non_speech']} non-speech draft tokens; "
            f"allowed at most {max_non_speech_tokens}"
        )
    return errors


def find_safetensors(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    preferred = path / "model.safetensors"
    if preferred.exists():
        return [preferred]
    return sorted(path.glob("*.safetensors"))


def load_safetensor(path: Path, name: str) -> torch.Tensor | None:
    try:
        from safetensors.torch import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to inspect checkpoint d2t/t2d") from exc

    with safe_open(path, framework="pt", device="cpu") as handle:
        if name not in handle.keys():
            return None
        return handle.get_tensor(name).cpu()


def load_checkpoint_tensor(checkpoint: Path, name: str) -> torch.Tensor | None:
    for path in find_safetensors(checkpoint):
        tensor = load_safetensor(path, name)
        if tensor is not None:
            return tensor
    return None


def effective_d2t_targets(
    d2t: torch.Tensor,
    *,
    speech_start: int,
    speech_vocab_size: int,
    allowed_control_ids: set[int],
) -> tuple[str, torch.Tensor, dict[str, int]]:
    raw = d2t.to(torch.long).view(-1)
    offset_targets = raw + torch.arange(raw.numel(), dtype=torch.long)
    candidates = [
        ("offset", offset_targets),
        ("direct", raw),
    ]
    scored = []
    for mode, targets in candidates:
        counts = classify_ids(
            targets.tolist(),
            speech_start=speech_start,
            speech_vocab_size=speech_vocab_size,
            allowed_control_ids=allowed_control_ids,
        )
        scored.append((counts["speech"] + counts["control"], mode, targets, counts))
    _, mode, targets, counts = max(scored, key=lambda item: item[0])
    return mode, targets, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-vocab-size", type=int, required=True)
    parser.add_argument("--speech-vocab-size", type=int, default=6561)
    parser.add_argument("--min-speech-draft-tokens", type=int, default=6000)
    parser.add_argument("--max-non-speech-draft-tokens", type=int, default=16)
    parser.add_argument("--mapping-output-dir", default=None)
    parser.add_argument("--write-vocab-mapping", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    token_freq_path = preprocessed_dir / "token_freq.pt"
    summary_path = preprocessed_dir / "soulx_peagle_prepare_summary.json"
    if not token_freq_path.exists():
        raise FileNotFoundError(token_freq_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    speech_start = resolve_token_id(tokenizer, "<|0|>")
    semantic_token_end = resolve_token_id(tokenizer, "<|semantic_token_end|>")
    allowed_control_ids = {semantic_token_end}
    target_vocab_size = load_target_vocab_size(args.model_path)

    token_freq = load_token_freq(token_freq_path)
    token_ids = sorted(token_freq)
    token_freq_counts = classify_ids(
        token_ids,
        speech_start=speech_start,
        speech_vocab_size=args.speech_vocab_size,
        allowed_control_ids=allowed_control_ids,
    )

    d2t, t2d, selected_ids = build_vocab_mapping(
        token_freq=token_freq,
        draft_vocab_size=args.draft_vocab_size,
        target_vocab_size=target_vocab_size,
    )
    selected_counts = classify_ids(
        selected_ids,
        speech_start=speech_start,
        speech_vocab_size=args.speech_vocab_size,
        allowed_control_ids=allowed_control_ids,
    )

    errors = []
    errors.extend(
        validate_counts(
            name="token_freq",
            counts=token_freq_counts,
            min_speech_tokens=args.min_speech_draft_tokens,
            max_non_speech_tokens=args.max_non_speech_draft_tokens,
        )
    )
    errors.extend(
        validate_counts(
            name="draft_vocab_mapping",
            counts=selected_counts,
            min_speech_tokens=args.min_speech_draft_tokens,
            max_non_speech_tokens=args.max_non_speech_draft_tokens,
        )
    )

    checkpoint_stats: dict[str, Any] | None = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        ckpt_d2t = load_checkpoint_tensor(checkpoint, "d2t")
        if ckpt_d2t is None:
            errors.append(f"checkpoint {checkpoint} does not contain d2t")
        else:
            mode, target_ids, ckpt_counts = effective_d2t_targets(
                ckpt_d2t,
                speech_start=speech_start,
                speech_vocab_size=args.speech_vocab_size,
                allowed_control_ids=allowed_control_ids,
            )
            checkpoint_stats = {
                "d2t_mode": mode,
                "d2t_shape": list(ckpt_d2t.shape),
                "d2t_unique_raw_values": int(torch.unique(ckpt_d2t).numel()),
                "effective_unique_target_ids": int(torch.unique(target_ids).numel()),
                "effective_target_counts": ckpt_counts,
            }
            errors.extend(
                validate_counts(
                    name="checkpoint_d2t",
                    counts=ckpt_counts,
                    min_speech_tokens=args.min_speech_draft_tokens,
                    max_non_speech_tokens=args.max_non_speech_draft_tokens,
                )
            )

        ckpt_t2d = load_checkpoint_tensor(checkpoint, "t2d")
        if ckpt_t2d is not None and checkpoint_stats is not None:
            true_ids = torch.nonzero(ckpt_t2d.to(torch.bool).view(-1), as_tuple=False)
            checkpoint_stats["t2d_true_counts"] = classify_ids(
                true_ids.view(-1).tolist(),
                speech_start=speech_start,
                speech_vocab_size=args.speech_vocab_size,
                allowed_control_ids=allowed_control_ids,
            )
            checkpoint_stats["t2d_dtype"] = str(ckpt_t2d.dtype)

    if args.write_vocab_mapping:
        if not args.mapping_output_dir:
            raise ValueError("--write-vocab-mapping requires --mapping-output-dir")
        mapping_dir = Path(args.mapping_output_dir)
        mapping_dir.mkdir(parents=True, exist_ok=True)
        np.save(mapping_dir / "d2t.npy", d2t.numpy())
        np.save(mapping_dir / "t2d.npy", t2d.numpy())

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    report = {
        "preprocessed_dir": str(preprocessed_dir),
        "model_path": args.model_path,
        "speech_start": speech_start,
        "speech_end_exclusive": speech_start + args.speech_vocab_size,
        "semantic_token_end": semantic_token_end,
        "target_vocab_size": target_vocab_size,
        "draft_vocab_size": args.draft_vocab_size,
        "token_freq_unique_tokens": len(token_freq),
        "token_freq_counts": token_freq_counts,
        "selected_mapping_counts": selected_counts,
        "mapping_output_dir": args.mapping_output_dir if args.write_vocab_mapping else None,
        "prepare_summary": summary,
        "checkpoint": checkpoint_stats,
        "errors": errors,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if errors:
        raise SystemExit("Invalid SoulX P-EAGLE vocab artifacts")


if __name__ == "__main__":
    main()
