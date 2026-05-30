#!/usr/bin/env python3
"""Compare cached P-EAGLE hidden states against a live vLLM extractor."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any


def resolve_model_id(endpoint: str, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    url = endpoint.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models returned by {url}")
    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError(f"First model entry from {url} has no id: {models[0]}")
    return str(model_id)


def wait_for_lock(lock_path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while lock_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for lock file: {lock_path}")
        time.sleep(0.1)


def extract_hidden_states_path(response: Any, expected_token_ids: list[int]) -> str:
    choices = getattr(response, "choices", None)
    prompt_token_ids = None
    if choices:
        prompt_token_ids = getattr(choices[0], "prompt_token_ids", None)
    if prompt_token_ids is None:
        prompt_token_ids = getattr(response, "prompt_token_ids", None)
    if prompt_token_ids is not None and list(prompt_token_ids) != expected_token_ids:
        raise RuntimeError(
            "Response prompt_token_ids mismatch: "
            f"expected {expected_token_ids}, got {prompt_token_ids}"
        )

    kv_transfer_params = getattr(response, "kv_transfer_params", None)
    if kv_transfer_params is None and hasattr(response, "model_dump"):
        kv_transfer_params = response.model_dump().get("kv_transfer_params")
    if kv_transfer_params is None and isinstance(response, dict):
        kv_transfer_params = response.get("kv_transfer_params")
    if not kv_transfer_params:
        raise RuntimeError(f"Response missing kv_transfer_params: {response}")
    hidden_states_path = kv_transfer_params.get("hidden_states_path")
    if not hidden_states_path:
        raise RuntimeError(f"Response missing hidden_states_path: {kv_transfer_params}")
    return str(hidden_states_path)


def request_live_hidden_states(
    *,
    endpoint: str,
    model: str,
    token_ids: list[int],
    timeout: float,
) -> Path:
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. Run this inside the speculators venv."
        ) from exc

    client = openai.Client(
        base_url=endpoint.rstrip("/"),
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
    )
    response = client.completions.create(
        model=model,
        prompt=token_ids,
        max_tokens=1,
        extra_body={"return_token_ids": True},
        timeout=timeout,
    )
    path = Path(extract_hidden_states_path(response, token_ids))
    wait_for_lock(Path(str(path) + ".lock"), timeout=timeout)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_preprocessed_item(preprocessed_dir: Path, index: int) -> dict[str, Any]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required. Run this inside the speculators venv."
        ) from exc

    dataset = load_from_disk(str(preprocessed_dir))
    item = dataset[int(index)]
    input_ids = item["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    return {"input_ids": [int(token) for token in input_ids]}


def cached_indices(hidden_states_dir: Path, limit: int) -> list[int]:
    indices = []
    for path in sorted(hidden_states_dir.glob("hs_*.safetensors")):
        stem = path.stem
        try:
            indices.append(int(stem.removeprefix("hs_")))
        except ValueError:
            continue
        if len(indices) >= limit:
            break
    return indices


def compare_tensors(cached_path: Path, live_path: Path) -> dict[str, Any]:
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "torch and safetensors are required. Run this inside the speculators venv."
        ) from exc

    cached = load_file(cached_path)
    live = load_file(live_path)
    out: dict[str, Any] = {
        "cached_path": str(cached_path),
        "live_path": str(live_path),
        "cached_keys": sorted(cached.keys()),
        "live_keys": sorted(live.keys()),
    }

    cached_tokens = cached["token_ids"].to(torch.long).cpu()
    live_tokens = live["token_ids"].to(torch.long).cpu()
    out["token_ids_match"] = torch.equal(cached_tokens, live_tokens)
    out["token_len"] = int(cached_tokens.numel())
    if cached_tokens.shape != live_tokens.shape:
        out["token_shape_match"] = False
        out["cached_token_shape"] = list(cached_tokens.shape)
        out["live_token_shape"] = list(live_tokens.shape)
    else:
        out["token_shape_match"] = True
    if not out["token_ids_match"] and out["token_shape_match"]:
        mismatch = (cached_tokens != live_tokens).nonzero(as_tuple=False).flatten()
        out["token_mismatch_count"] = int(mismatch.numel())
        if mismatch.numel() > 0:
            first = int(mismatch[0].item())
            out["first_token_mismatch"] = {
                "position": first,
                "cached": int(cached_tokens[first].item()),
                "live": int(live_tokens[first].item()),
            }

    cached_hs = cached["hidden_states"].float().cpu()
    live_hs = live["hidden_states"].float().cpu()
    out["cached_hidden_shape"] = list(cached_hs.shape)
    out["live_hidden_shape"] = list(live_hs.shape)
    if cached_hs.shape != live_hs.shape:
        out["hidden_shape_match"] = False
        return out
    out["hidden_shape_match"] = True

    diff = (cached_hs - live_hs).abs()
    out["max_abs"] = float(diff.max().item())
    out["mean_abs"] = float(diff.mean().item())
    out["rms"] = float(torch.sqrt(torch.mean((cached_hs - live_hs) ** 2)).item())

    flat_cached = cached_hs.reshape(-1, cached_hs.shape[-1])
    flat_live = live_hs.reshape(-1, live_hs.shape[-1])
    cos = torch.nn.functional.cosine_similarity(flat_cached, flat_live, dim=-1)
    finite_cos = cos[torch.isfinite(cos)]
    if finite_cos.numel() == 0:
        out["cos_mean"] = math.nan
        out["cos_min"] = math.nan
    else:
        out["cos_mean"] = float(finite_cos.mean().item())
        out["cos_min"] = float(finite_cos.min().item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--hidden-states-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--index", type=int, action="append", default=[])
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--keep-live-dir", type=Path, default=None)
    parser.add_argument("--min-cos", type=float, default=0.999)
    parser.add_argument("--max-mean-abs", type=float, default=1e-3)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    model = resolve_model_id(args.endpoint, args.model)
    indices = args.index or cached_indices(args.hidden_states_dir, args.num_samples)
    if not indices:
        raise SystemExit(f"No cached hs_*.safetensors files found in {args.hidden_states_dir}")

    reports = []
    errors = []
    for index in indices:
        cached_path = args.hidden_states_dir / f"hs_{index}.safetensors"
        if not cached_path.exists():
            errors.append(f"missing cached hidden-state file for index {index}: {cached_path}")
            continue

        item = load_preprocessed_item(args.preprocessed_dir, index)
        live_path = request_live_hidden_states(
            endpoint=args.endpoint,
            model=model,
            token_ids=item["input_ids"],
            timeout=args.request_timeout,
        )
        try:
            report = compare_tensors(cached_path, live_path)
            report["index"] = index
            reports.append(report)
            if not report.get("token_ids_match"):
                errors.append(f"index {index}: token_ids mismatch")
            if not report.get("hidden_shape_match"):
                errors.append(f"index {index}: hidden_states shape mismatch")
            if report.get("cos_min", -1.0) < args.min_cos:
                errors.append(
                    f"index {index}: cos_min {report.get('cos_min')} < {args.min_cos}"
                )
            if report.get("mean_abs", float("inf")) > args.max_mean_abs:
                errors.append(
                    f"index {index}: mean_abs {report.get('mean_abs')} > "
                    f"{args.max_mean_abs}"
                )
        finally:
            if args.keep_live_dir is not None:
                args.keep_live_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(
                    str(live_path),
                    str(args.keep_live_dir / f"live_hs_{index}.safetensors"),
                )
            else:
                live_path.unlink(missing_ok=True)

    output = {
        "endpoint": args.endpoint,
        "model": model,
        "indices": indices,
        "reports": reports,
        "errors": errors,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if errors and args.fail_on_mismatch:
        raise SystemExit("Cached hidden states differ from live vLLM hidden states")


if __name__ == "__main__":
    main()
