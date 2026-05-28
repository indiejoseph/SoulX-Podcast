#!/usr/bin/env python3
"""Measure client-visible TTFA for the OpenAI-style speech endpoint.

This script intentionally goes through HTTP so it measures the production
serving path, including Docker, API parsing, prompt cache, MTP streaming, flow,
and vocoder. Use format=pcm for TTFA measurements: streamed WAV sends a header
before synthesized audio, which makes first-byte timing misleading.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


PCM_SAMPLE_RATE = 24_000
PCM_BYTES_PER_SAMPLE = 2


@dataclass
class PromptSpec:
    prompt_audio: Optional[str]
    prompt_text: Optional[str]
    prompt_cache_id: Optional[str] = None


@dataclass
class Case:
    label: str
    cache_state: str
    prompt: PromptSpec


@dataclass
class Measurement:
    label: str
    cache_state: str
    status: int
    ttfa_s: Optional[float]
    wall_s: float
    bytes_received: int
    audio_s: Optional[float]
    rtf: Optional[float]
    first_llm_s: Optional[float]
    first_flow_s: Optional[float]
    prompt_cache_id: Optional[str]
    output_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure TTFA and wall time for /v1/audio/speech.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=os.getenv("SPEECH_URL", "http://localhost:8000/v1/audio/speech"),
        help="Speech endpoint URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY", ""),
        help="Bearer token. Pass an empty string for unauthenticated local servers.",
    )
    parser.add_argument("--model", default="tts")
    parser.add_argument(
        "--input",
        "--text",
        dest="text",
        default="Maple est le meilleur golden retriever du monde entier.",
        help="Text to synthesize.",
    )
    parser.add_argument("--language", default=None, help="Optional BCP-47 language code.")
    parser.add_argument(
        "--format",
        choices=("pcm", "wav"),
        default="pcm",
        help="Use pcm for valid TTFA. WAV streams a header before synthesized audio.",
    )
    parser.add_argument(
        "--prompt-audio",
        default=None,
        required=True,
        help="Top-level prompt_audio value, e.g. file:///app/example/audios/female_mandarin.wav.",
    )
    parser.add_argument(
        "--prompt-text",
        default=None,
        required=True,
        help="Prompt transcript. Required with --prompt-audio.",
    )
    parser.add_argument(
        "--other-prompt-audio",
        default=None,
        help="Optional second prompt_audio value for the third case.",
    )
    parser.add_argument(
        "--other-prompt-text",
        default=None,
        help="Prompt transcript for --other-prompt-audio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for comparable runs. Ignored when --random-seed is set.",
    )
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="Omit seed from the request and let the server choose.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--first-chunk-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument(
        "--flow-streaming",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override server flow_streaming for this request.",
    )
    parser.add_argument(
        "--read-size",
        type=int,
        default=64 * 1024,
        help="Read size after the first byte has arrived.",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save received audio bytes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print raw measurement JSON.",
    )
    parser.add_argument(
        "--no-reuse-prompt-cache-id",
        action="store_true",
        help="Send prompt_audio/prompt_text again on run 2 instead of reusing Prompt-Cache-Id.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.other_prompt_audio and not args.other_prompt_text:
        raise SystemExit("--other-prompt-text is required with --other-prompt-audio")
    if args.format != "pcm":
        print(
            "warning: streamed WAV includes an immediate header; TTFA will be first-byte response time, "
            "not first synthesized audio time",
            file=sys.stderr,
        )
    if args.read_size < 1:
        raise SystemExit("--read-size must be >= 1")


def make_cases(args: argparse.Namespace) -> list[Case]:
    primary = PromptSpec(
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
    )

    cases = [
        Case(
            label="run1_primary",
            cache_state="current-process first request",
            prompt=primary,
        ),
        Case(
            label="run2_primary",
            cache_state="same prompt cache warm",
            prompt=primary,
        ),
    ]

    if args.other_prompt_audio:
        other = PromptSpec(
            prompt_audio=args.other_prompt_audio,
            prompt_text=args.other_prompt_text,
        )
    else:
        other = None

    if other is not None:
        cases.append(
            Case(
                label="run3_other",
                cache_state="different prompt",
                prompt=other,
            )
        )
    return cases


def build_payload(args: argparse.Namespace, prompt: PromptSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "input": args.text,
        "format": args.format,
        "stream": True,
    }
    if args.language:
        payload["language"] = args.language
    if not args.random_seed:
        payload["seed"] = args.seed

    optional_fields = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "first_chunk_size": args.first_chunk_size,
        "chunk_size": args.chunk_size,
        "flow_steps": args.flow_steps,
        "flow_streaming": args.flow_streaming,
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value

    if prompt.prompt_audio:
        payload["prompt_audio"] = prompt.prompt_audio
        payload["prompt_text"] = prompt.prompt_text
    elif prompt.prompt_cache_id:
        payload["prompt_cache_id"] = prompt.prompt_cache_id
    return payload


def timing_header(headers: urllib.response.addinfourl, names: Iterable[str]) -> Optional[float]:
    for name in names:
        value = headers.get(name)
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def request_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "audio/pcm" if args.format == "pcm" else "audio/wav",
    }
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    return headers


def measure_case(args: argparse.Namespace, case: Case) -> Measurement:
    payload = build_payload(args, case.prompt)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=body,
        method="POST",
        headers=request_headers(args),
    )

    output_path = None
    output_file = None
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".pcm" if args.format == "pcm" else ".wav"
        output_path = args.save_dir / f"{safe_label(case.label)}{suffix}"
        output_file = output_path.open("wb")

    start_s = time.perf_counter()
    first_byte_s: Optional[float] = None
    total_bytes = 0
    try:
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                status = getattr(response, "status", 200)
                first_llm_s = timing_header(
                    response.headers,
                    (
                        "X-First-LLM-Seconds",
                        "X-Time-To-First-LLM-Token",
                        "X-TTFT",
                    ),
                )
                first_flow_s = timing_header(
                    response.headers,
                    (
                        "X-First-Flow-Seconds",
                        "X-Time-To-First-Flow",
                    ),
                )
                prompt_cache_id = response.headers.get("Prompt-Cache-Id")

                first = response.read(1)
                if first:
                    first_byte_s = time.perf_counter()
                    total_bytes += len(first)
                    if output_file is not None:
                        output_file.write(first)

                while True:
                    chunk = response.read(args.read_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if output_file is not None:
                        output_file.write(chunk)
        except urllib.error.HTTPError as e:
            error_body = e.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {error_body}") from e
    finally:
        if output_file is not None:
            output_file.close()

    end_s = time.perf_counter()
    wall_s = end_s - start_s
    ttfa_s = first_byte_s - start_s if first_byte_s is not None else None
    audio_s = None
    rtf = None
    if args.format == "pcm" and total_bytes > 0:
        audio_s = total_bytes / (PCM_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE)
        rtf = wall_s / audio_s if audio_s > 0 else None

    return Measurement(
        label=case.label,
        cache_state=case.cache_state,
        status=status,
        ttfa_s=ttfa_s,
        wall_s=wall_s,
        bytes_received=total_bytes,
        audio_s=audio_s,
        rtf=rtf,
        first_llm_s=first_llm_s,
        first_flow_s=first_flow_s,
        prompt_cache_id=prompt_cache_id,
        output_path=output_path,
    )


def fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_table(measurements: list[Measurement]) -> None:
    headers = [
        "Run",
        "Cache state",
        "TTFA",
        "First LLM",
        "First flow",
        "Wall",
        "Audio",
        "RTF",
        "Bytes",
        "Prompt cache",
        "Output",
    ]
    rows = []
    for item in measurements:
        rows.append(
            [
                item.label,
                item.cache_state,
                fmt_float(item.ttfa_s),
                fmt_float(item.first_llm_s),
                fmt_float(item.first_flow_s),
                fmt_float(item.wall_s),
                fmt_float(item.audio_s),
                fmt_float(item.rtf),
                str(item.bytes_received),
                item.prompt_cache_id or "",
                str(item.output_path) if item.output_path else "",
            ]
        )

    widths = [
        max(len(str(row[col])) for row in ([headers] + rows))
        for col in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-|-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))


def measurement_to_dict(item: Measurement) -> dict[str, Any]:
    return {
        "label": item.label,
        "cache_state": item.cache_state,
        "status": item.status,
        "ttfa_s": item.ttfa_s,
        "first_llm_s": item.first_llm_s,
        "first_flow_s": item.first_flow_s,
        "prompt_cache_id": item.prompt_cache_id,
        "wall_s": item.wall_s,
        "bytes_received": item.bytes_received,
        "audio_s": item.audio_s,
        "rtf": item.rtf,
        "output_path": str(item.output_path) if item.output_path else None,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)

    cases = make_cases(args)
    print(f"Endpoint: {args.url}")
    print("Note: restart the API process immediately before this script for a true cold run.")
    if args.format == "pcm":
        print("Format: pcm s16le mono at 24 kHz; first byte corresponds to streamed audio.")
    print()

    measurements = []
    for index, case in enumerate(cases):
        print(f"Running {case.label} ({case.cache_state})...", flush=True)
        measurement = measure_case(args, case)
        measurements.append(measurement)
        if (
            index == 0
            and len(cases) > 1
            and not args.no_reuse_prompt_cache_id
            and measurement.prompt_cache_id
        ):
            cases[1].prompt = PromptSpec(
                prompt_audio=None,
                prompt_text=None,
                prompt_cache_id=measurement.prompt_cache_id,
            )
            cases[1].cache_state = "Prompt-Cache-Id reuse"

    print()
    print_table(measurements)
    if args.json:
        print()
        print(json.dumps([measurement_to_dict(m) for m in measurements], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
