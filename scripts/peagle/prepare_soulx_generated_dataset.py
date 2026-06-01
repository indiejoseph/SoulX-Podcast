#!/usr/bin/env python3
"""Prepare verifier-generated SoulX trajectories for P-EAGLE training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from soulxpodcast.training.mtp_dataset import DIALECT_PREFIX, SPECIAL_TOKENS


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_soulx_generated_peagle")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

_thread_local = threading.local()


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


def resolve_speech_token_offset(tokenizer) -> int:
    ids = tokenizer.encode("<|0|>", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Tokenizer did not encode '<|0|>' as one token: {ids}")
    return int(ids[0])


def resolve_special_tokens(tokenizer) -> dict[str, int]:
    resolved = {}
    missing = []
    for key, sym in SPECIAL_TOKENS.items():
        ids = tokenizer.encode(sym, add_special_tokens=False)
        if len(ids) != 1:
            missing.append(f"{sym} (encoded to {ids})")
            continue
        resolved[key] = int(ids[0])
    if missing:
        raise ValueError("Tokenizer is missing required special tokens: " + ", ".join(missing))
    return resolved


def load_split(dataset_path: Path, split: str | None):
    dataset = load_from_disk(str(dataset_path))
    if isinstance(dataset, DatasetDict):
        if split:
            return dataset[split]
        if "train" in dataset:
            return dataset["train"]
        first = next(iter(dataset.keys()))
        log.warning("No --split passed; using DatasetDict split %r", first)
        return dataset[first]
    return dataset


def make_features() -> Features:
    return Features(
        {
            "input_ids": Sequence(Value("int64")),
            "loss_mask": Sequence(Value("int8")),
            "seq_len": Value("int32"),
        }
    )


def apply_dialect_prefix(text: str, lang: str, *, skip_dialect_prefix: bool) -> str:
    if skip_dialect_prefix:
        return text
    prefix = DIALECT_PREFIX.get(lang, "")
    if prefix and not text.startswith(prefix):
        return prefix + text
    return text


def parse_speech_tokens(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(x) for x in value.split()] if value else []
    return [int(x) for x in value]


def count_speech_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.split()) if value else 0
    return len(value)


def get_openai_client(endpoint: str):
    client = getattr(_thread_local, "client", None)
    if client is not None:
        return client
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
    _thread_local.client = client
    return client


def extract_choice_payload(response: Any) -> tuple[list[int], str | None, str | None]:
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, dict):
        data = response
    else:
        data = json.loads(response.model_dump_json())

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Completion response has no choices: {data}")
    choice = choices[0]
    token_ids = (
        choice.get("token_ids")
        or choice.get("text_token_ids")
        or choice.get("output_token_ids")
    )
    if token_ids is None:
        logprobs = choice.get("logprobs") or {}
        token_ids = logprobs.get("token_ids")
    if token_ids is None:
        raise RuntimeError(
            "vLLM completion response did not include generated token IDs. "
            "Ensure extra_body.return_token_ids is supported by this vLLM build."
        )

    hidden_states_path = None
    kv_transfer_params = data.get("kv_transfer_params")
    if isinstance(kv_transfer_params, dict):
        hidden_states_path = kv_transfer_params.get("hidden_states_path")
    return [int(tok) for tok in token_ids], choice.get("finish_reason"), hidden_states_path


def cleanup_hidden_state_side_effect(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.unlink(missing_ok=True)
    Path(str(path) + ".lock").unlink(missing_ok=True)


def normalize_generated_tokens(
    *,
    token_ids: list[int],
    prefix_ids: list[int],
    finish_reason: str | None,
    speech_start: int,
    speech_end: int,
    eos_id: int,
) -> tuple[list[int], str | None]:
    if token_ids[: len(prefix_ids)] == prefix_ids:
        token_ids = token_ids[len(prefix_ids) :]

    generated: list[int] = []
    for tok in token_ids:
        if tok == eos_id:
            generated.append(tok)
            return generated, None
        if speech_start <= tok < speech_end:
            generated.append(tok)
            continue
        return generated, f"non_speech_token:{tok}"

    if finish_reason != "length":
        generated.append(eos_id)
    return generated, None


def request_generated_tokens(
    *,
    endpoint: str,
    model: str,
    prefix_ids: list[int],
    args: argparse.Namespace,
    eos_id: int,
    seed: int,
) -> tuple[list[int], str | None]:
    client = get_openai_client(endpoint)
    extra_body: dict[str, Any] = {
        "return_token_ids": True,
        "stop_token_ids": [eos_id],
        "top_k": args.top_k,
        "min_tokens": args.min_speech_tokens,
        "repetition_penalty": args.repetition_penalty,
        "seed": seed,
    }
    response = client.completions.create(
        model=model,
        prompt=prefix_ids,
        max_tokens=args.generation_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        extra_body=extra_body,
        timeout=args.request_timeout,
    )
    token_ids, finish_reason, hidden_states_path = extract_choice_payload(response)
    cleanup_hidden_state_side_effect(hidden_states_path)
    return token_ids, finish_reason


def generate_one(
    *,
    source_index: int,
    prefix_ids: list[int],
    endpoint: str,
    model: str,
    args: argparse.Namespace,
    speech_start: int,
    speech_end: int,
    eos_id: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            token_ids, finish_reason = request_generated_tokens(
                endpoint=endpoint,
                model=model,
                prefix_ids=prefix_ids,
                args=args,
                eos_id=eos_id,
                seed=args.seed + source_index + attempt,
            )
            generated, invalid_reason = normalize_generated_tokens(
                token_ids=token_ids,
                prefix_ids=prefix_ids,
                finish_reason=finish_reason,
                speech_start=speech_start,
                speech_end=speech_end,
                eos_id=eos_id,
            )
            speech_count = sum(1 for tok in generated if speech_start <= tok < speech_end)
            if invalid_reason is not None:
                return {
                    "ok": False,
                    "source_index": source_index,
                    "reason": invalid_reason,
                    "speech_count": speech_count,
                }
            if speech_count < args.min_speech_tokens:
                return {
                    "ok": False,
                    "source_index": source_index,
                    "reason": "too_short",
                    "speech_count": speech_count,
                }
            if speech_count > args.max_speech_tokens:
                return {
                    "ok": False,
                    "source_index": source_index,
                    "reason": "too_long",
                    "speech_count": speech_count,
                }
            input_ids = prefix_ids + generated
            if len(input_ids) > args.seq_length:
                return {
                    "ok": False,
                    "source_index": source_index,
                    "reason": "seq_length",
                    "seq_len": len(input_ids),
                    "speech_count": speech_count,
                }

            mask_generated_len = len(generated)
            if args.no_eos_loss and generated and generated[-1] == eos_id:
                mask_generated_len -= 1
            loss_mask = [0] * len(prefix_ids) + [1] * mask_generated_len
            loss_mask.extend([0] * (len(input_ids) - len(loss_mask)))
            return {
                "ok": True,
                "source_index": source_index,
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "seq_len": len(input_ids),
                "speech_count": speech_count,
                "finish_reason": finish_reason,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= args.max_retries:
                break
            time.sleep(min(2.0 * (attempt + 1), 10.0))

    return {
        "ok": False,
        "source_index": source_index,
        "reason": f"request_error:{last_error}",
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_resume_config(args: argparse.Namespace, *, served_model: str) -> dict[str, Any]:
    return {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "served_model": served_model,
        "split": args.split,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "seq_length": args.seq_length,
        "min_speech_tokens": args.min_speech_tokens,
        "max_speech_tokens": args.max_speech_tokens,
        "generation_max_tokens": args.generation_max_tokens,
        "prefix_reserve_tokens": args.prefix_reserve_tokens,
        "max_prompt_speech_tokens": args.max_prompt_speech_tokens,
        "max_prompt_text_chars": args.max_prompt_text_chars,
        "max_prompt_text_tokens": args.max_prompt_text_tokens,
        "max_turn_text_chars": args.max_turn_text_chars,
        "max_turn_text_tokens": args.max_turn_text_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "skip_dialect_prefix": args.skip_dialect_prefix,
        "no_eos_loss": args.no_eos_loss,
        "multi_speaker": bool(args.multi_speaker),
        "num_speakers": int(args.num_speakers) if args.multi_speaker else None,
        "num_dialogues": int(args.num_dialogues) if (args.multi_speaker and args.num_dialogues) else None,
        "lang_mix": args.lang_mix if args.multi_speaker else None,
    }


def validate_resume_config(meta_path: Path, expected: dict[str, Any]) -> None:
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} is missing for --resume. Delete the stale records JSONL "
            "or rerun without --resume so generated trajectories cannot be mixed "
            "across incompatible settings."
        )
    actual = json.loads(meta_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in sorted(set(expected) | set(actual))
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(
            "Generated trajectory resume metadata does not match current settings:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )


def iter_existing_records(
    path: Path,
    *,
    allowed_source_indices: set[int] | None = None,
) -> tuple[list[dict[str, Any]], set[int], Counter[str]]:
    records = []
    processed: set[int] = set()
    reasons: Counter[str] = Counter()
    if not path.exists():
        return records, processed, reasons
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source_index = int(row["source_index"])
            if allowed_source_indices is not None and source_index not in allowed_source_indices:
                continue
            processed.add(source_index)
            if row.get("ok"):
                records.append(
                    {
                        "input_ids": [int(tok) for tok in row["input_ids"]],
                        "loss_mask": [int(mask) for mask in row["loss_mask"]],
                        "seq_len": int(row["seq_len"]),
                    }
                )
            else:
                reasons[str(row.get("reason", "unknown"))] += 1
    return records, processed, reasons


def build_prefixes(
    *,
    hf_ds,
    tokenizer,
    special: dict[str, int],
    args: argparse.Namespace,
) -> list[tuple[int, list[int]]]:
    task_prefix_ids = [
        special["task_podcast"],
        special["speaker_0"],
        special["text_start"],
    ]
    text_to_speech_ids = [
        special["text_end"],
        special["semantic_token_start"],
    ]
    prefixes: list[tuple[int, list[int]]] = []
    for source_index, row in enumerate(tqdm(hf_ds, desc="Tokenizing SoulX prompts")):
        text = apply_dialect_prefix(
            str(row["text"]),
            str(row["lang"]),
            skip_dialect_prefix=args.skip_dialect_prefix,
        )
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        prefix_ids = task_prefix_ids + list(text_ids) + text_to_speech_ids
        if len(prefix_ids) >= args.seq_length - args.min_speech_tokens:
            continue
        prefixes.append((source_index, [int(tok) for tok in prefix_ids]))
    return prefixes


def parse_speaker_id(row_id: str, lang: str) -> str | None:
    """Extract a stable speaker tag from the row's id.

    Returns None when speaker identity cannot be parsed (e.g. yue rows
    use anonymous ``yue_NNN`` ids). Callers should treat None as
    'unknown speaker' and avoid speaker-distinct constraints.
    """
    if not row_id:
        return None
    if lang == "en":
        parts = row_id.split("-")
        if parts and parts[0].isdigit():
            return f"en:{parts[0]}"
    elif lang == "zh":
        if row_id.startswith("SSB") and len(row_id) >= 7:
            return f"zh:{row_id[:7]}"
    return None


def resolve_speaker_token_ids(tokenizer, num_speakers: int) -> list[int]:
    """Resolve <|SPEAKER_K|> token ids for K in [0, num_speakers)."""
    out: list[int] = []
    for k in range(num_speakers):
        sym = f"<|SPEAKER_{k}|>"
        ids = tokenizer.encode(sym, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Tokenizer did not encode {sym} as one token: {ids}")
        out.append(int(ids[0]))
    return out


def parse_lang_mix(spec: str, available_langs: list[str]) -> dict[str, float]:
    """Parse '--lang-mix' spec like 'yue:1,zh:1,en:1' into weight dict.

    Empty spec → equal weights across all available_langs.
    Items without ':<weight>' default to weight 1.0.
    """
    if not spec.strip():
        return {lang: 1.0 for lang in available_langs}
    weights: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            lang, w = item.split(":", 1)
            weights[lang.strip()] = float(w)
        else:
            weights[item] = 1.0
    return weights


def choose_unblocked_row(
    candidates: list[int],
    blocked: set[int],
    rng: random.Random,
) -> int | None:
    """Pick a row not in blocked without scanning large language pools.

    Some zh/en corpora have many single-utterance speakers. Falling back to
    ``[r for r in candidates if r not in blocked]`` inside every dialogue
    attempt turns prefix construction into an O(num_dialogues * lang_rows)
    CPU loop. With only a handful of blocked prompt rows, rejection sampling
    succeeds almost immediately.
    """
    for _ in range(32):
        row = rng.choice(candidates)
        if row not in blocked:
            return row
    for row in candidates:
        if row not in blocked:
            return row
    return None


def build_multispeaker_prefixes(
    *,
    hf_ds,
    tokenizer,
    special: dict[str, int],
    args: argparse.Namespace,
    speech_token_offset: int,
) -> list[tuple[int, list[int]]]:
    """Pair rows into multi-speaker podcast-format prefixes that mirror runtime.

    Each output prefix represents one dialogue sample with shape:

        [task_podcast]
          [SPK_0][text_start]<spk0_prompt_text>[text_end][semantic_token_start]
              <spk0_prompt_speech_tokens+offset>[semantic_token_end]
          [SPK_1][text_start]<spk1_prompt_text>[text_end][semantic_token_start]
              <spk1_prompt_speech_tokens+offset>[semantic_token_end]
          [SPK_t][text_start]<turn_text>[text_end][semantic_token_start]
              <-- generation starts here -->

    This matches the runtime LLM input layout produced by
    ``soulxpodcast/models/soulxpodcast.py:_run_llm_prompt`` so the verifier
    sees the same multi-speaker conditioning during data generation as it
    does at deployment.
    """
    if args.num_speakers < 2:
        raise ValueError("--num-speakers must be >= 2 for multi-speaker mode")

    speaker_token_ids = resolve_speaker_token_ids(tokenizer, args.num_speakers)
    semantic_token_end_id = special["semantic_token_end"]
    prefix_budget = args.seq_length - max(args.min_speech_tokens, args.prefix_reserve_tokens)
    if prefix_budget <= 0:
        raise ValueError(
            f"Invalid prefix budget: seq_length={args.seq_length}, "
            f"prefix_reserve_tokens={args.prefix_reserve_tokens}"
        )

    log.info("Scanning dataset for language + speaker grouping (%d rows)", len(hf_ds))
    rows_lang: list[str] = []
    rows_speaker: list[str | None] = []
    rows_text: list[str] = []
    by_lang: dict[str, list[int]] = defaultdict(list)
    by_lang_speaker: dict[tuple[str, str], list[int]] = defaultdict(list)
    prompt_by_lang: dict[str, list[int]] = defaultdict(list)
    prompt_by_lang_speaker: dict[tuple[str, str], list[int]] = defaultdict(list)
    prompt_speech_by_row: dict[int, list[int]] = {}
    prompt_prefilter_reasons: Counter[str] = Counter()
    for i in tqdm(range(len(hf_ds)), desc="Indexing rows"):
        row = hf_ds[i]
        lang = str(row.get("lang", ""))
        text_value = str(row.get("text", ""))
        rows_text.append(text_value)
        if not lang:
            rows_lang.append("")
            rows_speaker.append(None)
            continue
        sid = parse_speaker_id(str(row.get("id", "")), lang)
        rows_lang.append(lang)
        rows_speaker.append(sid)
        by_lang[lang].append(i)
        if sid:
            by_lang_speaker[(lang, sid)].append(i)
        speech_value = row.get("speech_tokens")
        speech_count = count_speech_tokens(speech_value)
        if speech_count <= 0:
            prompt_prefilter_reasons["empty_prompt_speech"] += 1
            continue
        if speech_count > args.max_prompt_speech_tokens:
            prompt_prefilter_reasons["prompt_speech_too_long"] += 1
            continue
        text_chars = len(text_value)
        if args.max_prompt_text_chars and text_chars > args.max_prompt_text_chars:
            prompt_prefilter_reasons["prompt_text_too_long"] += 1
            continue
        prompt_speech_by_row[i] = parse_speech_tokens(speech_value)
        prompt_by_lang[lang].append(i)
        if sid:
            prompt_by_lang_speaker[(lang, sid)].append(i)

    if prompt_prefilter_reasons:
        log.info(
            "Prompt prefilter skipped rows: %s",
            dict(prompt_prefilter_reasons.most_common()),
        )

    min_rows_per_lang = args.num_speakers + 1
    available_langs = sorted(
        lang
        for lang, idxs in by_lang.items()
        if len(idxs) >= min_rows_per_lang and len(prompt_by_lang.get(lang, [])) >= args.num_speakers
    )
    if not available_langs:
        raise RuntimeError(
            f"No language has >={min_rows_per_lang} rows; cannot build any "
            "multi-speaker dialogue"
        )

    lang_weights = parse_lang_mix(args.lang_mix, available_langs)
    lang_weights = {
        lang: w for lang, w in lang_weights.items() if lang in available_langs and w > 0
    }
    if not lang_weights:
        raise RuntimeError(
            f"--lang-mix {args.lang_mix!r} matches no language with enough rows; "
            f"available langs: {available_langs}"
        )

    total_dialogues = int(args.num_dialogues) if args.num_dialogues else len(hf_ds)
    weight_total = sum(lang_weights.values())
    quotas = {
        lang: int(total_dialogues * w / weight_total) for lang, w in lang_weights.items()
    }
    leftover = total_dialogues - sum(quotas.values())
    if leftover:
        ranked = sorted(lang_weights.items(), key=lambda x: -x[1])
        for k in range(leftover):
            quotas[ranked[k % len(ranked)][0]] += 1

    log.info(
        "Multi-speaker plan: %d total dialogues, per-lang quotas=%s",
        total_dialogues,
        quotas,
    )

    rng = random.Random(args.seed)
    prefixes: list[tuple[int, list[int]]] = []
    dialog_idx = 0
    total_rejections: Counter[str] = Counter()
    prompt_segment_cache: dict[int, list[int] | None] = {}
    turn_text_cache: dict[int, list[int] | None] = {}

    def get_prompt_segment(row_idx: int) -> list[int] | None:
        cached = prompt_segment_cache.get(row_idx)
        if cached is not None or row_idx in prompt_segment_cache:
            return cached
        row_speech = prompt_speech_by_row.get(row_idx)
        if not row_speech:
            prompt_segment_cache[row_idx] = None
            return None
        if len(row_speech) > args.max_prompt_speech_tokens:
            prompt_segment_cache[row_idx] = None
            return None
        row_text = apply_dialect_prefix(
            rows_text[row_idx],
            rows_lang[row_idx],
            skip_dialect_prefix=args.skip_dialect_prefix,
        )
        text_ids = tokenizer(row_text, add_special_tokens=False)["input_ids"]
        if args.max_prompt_text_tokens and len(text_ids) > args.max_prompt_text_tokens:
            prompt_segment_cache[row_idx] = None
            return None
        segment = [special["text_start"]]
        segment.extend(int(t) for t in text_ids)
        segment.append(special["text_end"])
        segment.append(special["semantic_token_start"])
        segment.extend(int(t) + speech_token_offset for t in row_speech)
        segment.append(semantic_token_end_id)
        prompt_segment_cache[row_idx] = segment
        return segment

    def get_turn_text_ids(row_idx: int) -> list[int] | None:
        cached = turn_text_cache.get(row_idx)
        if cached is not None or row_idx in turn_text_cache:
            return cached
        turn_text = apply_dialect_prefix(
            rows_text[row_idx],
            rows_lang[row_idx],
            skip_dialect_prefix=args.skip_dialect_prefix,
        )
        if args.max_turn_text_chars and len(turn_text) > args.max_turn_text_chars:
            turn_text_cache[row_idx] = None
            return None
        turn_text_ids = tokenizer(turn_text, add_special_tokens=False)["input_ids"]
        if args.max_turn_text_tokens and len(turn_text_ids) > args.max_turn_text_tokens:
            turn_text_cache[row_idx] = None
            return None
        out = [int(t) for t in turn_text_ids]
        turn_text_cache[row_idx] = out
        return out

    for lang, quota in quotas.items():
        candidates = by_lang.get(lang, [])
        prompt_candidates = prompt_by_lang.get(lang, [])
        if len(candidates) < min_rows_per_lang or len(prompt_candidates) < args.num_speakers:
            log.warning(
                "Skipping lang=%s — rows=%d prompt_rows=%d need rows>=%d prompt_rows>=%d",
                lang,
                len(candidates),
                len(prompt_candidates),
                min_rows_per_lang,
                args.num_speakers,
            )
            continue
        # Pool keyed by speaker for distinct-speaker pairing
        speaker_pool: dict[str, list[int]] = {
            sid: rows for (lng, sid), rows in prompt_by_lang_speaker.items() if lng == lang
        }
        turn_speaker_pool: dict[str, list[int]] = {
            sid: rows for (lng, sid), rows in by_lang_speaker.items() if lng == lang
        }
        speakers_with_data = sorted(speaker_pool.keys())

        produced = 0
        attempts = 0
        rejections: Counter[str] = Counter()
        attempt_cap = max(quota * 3, 100)
        while produced < quota and attempts < attempt_cap:
            attempts += 1
            if attempts % 10000 == 0:
                log.info(
                    "Lang %s: attempts=%d produced=%d/%d rejects=%s",
                    lang,
                    attempts,
                    produced,
                    quota,
                    dict(rejections.most_common(5)),
                )
            prompt_rows: list[int]
            turn_spk_idx = rng.randint(0, args.num_speakers - 1)

            if len(speakers_with_data) >= args.num_speakers:
                spk_ids = rng.sample(speakers_with_data, args.num_speakers)
                prompt_rows = [rng.choice(speaker_pool[sid]) for sid in spk_ids]
                # Turn row: prefer a different utterance from the chosen turn speaker
                turn_speaker_id = spk_ids[turn_spk_idx]
                turn_row = choose_unblocked_row(
                    turn_speaker_pool.get(turn_speaker_id, []),
                    {prompt_rows[turn_spk_idx]},
                    rng,
                )
                if turn_row is None:
                    # Speaker has only one utterance — fall back to any other row
                    prompt_row_set = set(prompt_rows)
                    fallback_row = choose_unblocked_row(candidates, prompt_row_set, rng)
                    if fallback_row is None:
                        rejections["no_turn_row"] += 1
                        continue
                    turn_row = fallback_row
            else:
                # No speaker info → pick distinct rows from the same language.
                if len(prompt_candidates) < args.num_speakers or len(candidates) < min_rows_per_lang:
                    rejections["not_enough_candidates"] += 1
                    continue
                prompt_rows = rng.sample(prompt_candidates, args.num_speakers)
                prompt_row_set = set(prompt_rows)
                fallback_row = choose_unblocked_row(candidates, prompt_row_set, rng)
                if fallback_row is None:
                    rejections["no_turn_row"] += 1
                    continue
                turn_row = fallback_row

            prefix: list[int] = [special["task_podcast"]]

            ok = True
            for spk_idx, row_idx in enumerate(prompt_rows):
                segment = get_prompt_segment(row_idx)
                if segment is None:
                    rejections["invalid_prompt_row"] += 1
                    ok = False
                    break
                prefix.append(speaker_token_ids[spk_idx])
                prefix.extend(segment)
            if not ok:
                continue

            turn_text_ids = get_turn_text_ids(turn_row)
            if turn_text_ids is None:
                rejections["invalid_turn_text"] += 1
                continue
            prefix.append(speaker_token_ids[turn_spk_idx])
            prefix.append(special["text_start"])
            prefix.extend(turn_text_ids)
            prefix.append(special["text_end"])
            prefix.append(special["semantic_token_start"])

            if len(prefix) >= prefix_budget:
                rejections["prefix_overflow"] += 1
                continue

            prefixes.append((dialog_idx, [int(tok) for tok in prefix]))
            dialog_idx += 1
            produced += 1
            if produced % 10000 == 0:
                log.info(
                    "Lang %s: produced %d/%d dialogues (attempts=%d)",
                    lang,
                    produced,
                    quota,
                    attempts,
                )

        log.info(
            "Lang %s: produced %d/%d dialogues (attempts=%d rejects=%s)",
            lang,
            produced,
            quota,
            attempts,
            dict(rejections.most_common(8)),
        )
        total_rejections.update(rejections)

    if total_rejections:
        log.warning("Rejected multi-speaker prefix attempts: %s", dict(total_rejections.most_common()))

    if not prefixes:
        raise RuntimeError("No valid multi-speaker dialogues were produced")
    return prefixes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--min-speech-tokens", type=int, default=8)
    parser.add_argument("--max-speech-tokens", type=int, default=750)
    parser.add_argument(
        "--prefix-reserve-tokens",
        type=int,
        default=256,
        help=(
            "Minimum target-generation room to leave when building multi-speaker "
            "prefixes. This avoids sending near-full prompts to vLLM only to have "
            "them fail seq_length after generation."
        ),
    )
    parser.add_argument(
        "--max-prompt-speech-tokens",
        type=int,
        default=320,
        help=(
            "Maximum reference speech-token length for each speaker prompt in "
            "--multi-speaker mode. Long prompt clips make prefix construction slow "
            "and leave too little room for generated target speech."
        ),
    )
    parser.add_argument(
        "--max-prompt-text-chars",
        type=int,
        default=1024,
        help="Cheap prefilter for overlong prompt text rows before tokenization; 0 disables.",
    )
    parser.add_argument(
        "--max-prompt-text-tokens",
        type=int,
        default=384,
        help="Maximum tokenized text length for each speaker prompt; 0 disables.",
    )
    parser.add_argument(
        "--max-turn-text-chars",
        type=int,
        default=2048,
        help="Cheap prefilter for overlong target-turn text before tokenization; 0 disables.",
    )
    parser.add_argument(
        "--max-turn-text-tokens",
        type=int,
        default=768,
        help="Maximum tokenized text length for the target turn; 0 disables.",
    )
    parser.add_argument("--speech-vocab-size", type=int, default=6561)
    parser.add_argument("--generation-max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.25)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--skip-dialect-prefix", action="store_true")
    parser.add_argument("--no-eos-loss", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--multi-speaker",
        action="store_true",
        help=(
            "Build multi-speaker dialogue prefixes (matches the runtime SoulX podcast "
            "prompt layout) instead of single-utterance prefixes. Closes the "
            "single-turn-vs-multi-turn distribution gap that causes runtime acceptance "
            "to collapse despite high single-utterance validation accuracy."
        ),
    )
    parser.add_argument(
        "--num-dialogues",
        type=int,
        default=None,
        help=(
            "Number of multi-speaker dialogue samples to produce. Only used with "
            "--multi-speaker. Defaults to len(dataset)."
        ),
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Number of speaker prompts per dialogue (default 2). Only used with --multi-speaker.",
    )
    parser.add_argument(
        "--lang-mix",
        type=str,
        default="",
        help=(
            "Language mix for --multi-speaker, e.g. 'yue:1,zh:1,en:1' for equal thirds "
            "or 'yue:2,zh:1,en:1' for weighted. Empty = equal weights across all langs "
            "present in the dataset that have >=3 rows."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records_path = output_dir.parent / f"{output_dir.name}.generated_records.jsonl"
    records_meta_path = output_dir.parent / f"{output_dir.name}.generated_records.meta.json"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    if not args.resume:
        records_path.unlink(missing_ok=True)
        records_meta_path.unlink(missing_ok=True)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.generation_max_tokens = args.generation_max_tokens or (args.max_speech_tokens + 1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    special = resolve_special_tokens(tokenizer)
    speech_start = resolve_speech_token_offset(tokenizer)
    speech_end = speech_start + args.speech_vocab_size
    eos_id = special["semantic_token_end"]
    model = resolve_model_id(args.endpoint, args.served_model)
    resume_config = build_resume_config(args, served_model=model)
    if args.resume and records_path.exists():
        try:
            validate_resume_config(records_meta_path, resume_config)
        except (FileNotFoundError, ValueError) as exc:
            if not args.overwrite:
                raise
            log.warning(
                "Discarding incompatible generated-trajectory resume records "
                "because --overwrite was passed: %s",
                exc,
            )
            records_path.unlink(missing_ok=True)
            records_meta_path.unlink(missing_ok=True)
            args.resume = False
    records_meta_path.write_text(
        json.dumps(resume_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    hf_ds = load_split(Path(args.dataset_path), args.split)
    # In single-utterance mode --max-samples bounds the input rows. In multi-speaker
    # mode we keep the full row pool for sampling and use --num-dialogues to bound
    # output count; --max-samples then only acts as an explicit pool cap.
    if args.shuffle or (args.max_samples is not None and not args.multi_speaker):
        hf_ds = hf_ds.shuffle(seed=args.seed)
    if args.max_samples is not None and not args.multi_speaker:
        hf_ds = hf_ds.select(range(min(args.max_samples, len(hf_ds))))
    elif args.multi_speaker and args.max_samples is not None:
        hf_ds = hf_ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(hf_ds))))

    if args.multi_speaker:
        prefixes = build_multispeaker_prefixes(
            hf_ds=hf_ds,
            tokenizer=tokenizer,
            special=special,
            args=args,
            speech_token_offset=speech_start,
        )
    else:
        prefixes = build_prefixes(hf_ds=hf_ds, tokenizer=tokenizer, special=special, args=args)
    if not prefixes:
        raise RuntimeError("No valid generation prompts were produced")

    prefix_index_set = {idx for idx, _ in prefixes}
    records, processed_indices, filtered_reasons = (
        iter_existing_records(records_path, allowed_source_indices=prefix_index_set)
        if args.resume
        else ([], set(), Counter())
    )
    pending = [(idx, prefix) for idx, prefix in prefixes if idx not in processed_indices]
    log.info(
        "Generating SoulX P-EAGLE trajectories: model=%s prompts=%d pending=%d resume_records=%d",
        model,
        len(prefixes),
        len(pending),
        len(records),
    )

    token_freq: Counter[int] = Counter()
    total_tokens = sum(record["seq_len"] for record in records)
    total_loss_tokens = sum(sum(record["loss_mask"]) for record in records)
    max_seq_len = max((record["seq_len"] for record in records), default=0)
    for record in records:
        token_freq.update(
            tok for tok, mask in zip(record["input_ids"], record["loss_mask"], strict=True) if mask
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        in_flight: dict[Future[dict[str, Any]], None] = {}
        pending_iter = iter(pending)
        progress = tqdm(total=len(pending), desc="Generating SoulX P-EAGLE data")

        def submit_next() -> bool:
            try:
                source_index, prefix_ids = next(pending_iter)
            except StopIteration:
                return False
            future = executor.submit(
                generate_one,
                source_index=source_index,
                prefix_ids=prefix_ids,
                endpoint=args.endpoint,
                model=model,
                args=args,
                speech_start=speech_start,
                speech_end=speech_end,
                eos_id=eos_id,
            )
            in_flight[future] = None
            return True

        for _ in range(min(args.concurrency, len(pending))):
            submit_next()

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.pop(future, None)
                row = future.result()
                append_jsonl(records_path, row)
                if row.get("ok"):
                    record = {
                        "input_ids": [int(tok) for tok in row["input_ids"]],
                        "loss_mask": [int(mask) for mask in row["loss_mask"]],
                        "seq_len": int(row["seq_len"]),
                    }
                    records.append(record)
                    total_tokens += record["seq_len"]
                    loss_tokens = int(sum(record["loss_mask"]))
                    total_loss_tokens += loss_tokens
                    max_seq_len = max(max_seq_len, record["seq_len"])
                    token_freq.update(
                        tok
                        for tok, mask in zip(record["input_ids"], record["loss_mask"], strict=True)
                        if mask
                    )
                else:
                    filtered_reasons[str(row.get("reason", "unknown"))] += 1
                progress.update(1)
                submit_next()
        progress.close()

    if not records:
        raise RuntimeError("No valid generated P-EAGLE samples were produced")

    out_ds = Dataset.from_list(records, features=make_features())
    out_ds.set_format(type="torch", columns=["input_ids", "loss_mask", "seq_len"])
    out_ds.save_to_disk(str(output_dir))
    torch.save(dict(token_freq), output_dir / "token_freq.pt")

    summary = {
        "source_dataset": str(args.dataset_path),
        "model_path": args.model_path,
        "served_model": model,
        "endpoint": args.endpoint,
        "input_rows_scanned": len(hf_ds),
        "generation_prompts": len(prefixes),
        "kept_samples": len(records),
        "filtered_samples": len(prefixes) - len(records),
        "filtered_reasons": dict(filtered_reasons),
        "seq_length": args.seq_length,
        "min_speech_tokens": args.min_speech_tokens,
        "max_speech_tokens": args.max_speech_tokens,
        "generation_max_tokens": args.generation_max_tokens,
        "prefix_reserve_tokens": args.prefix_reserve_tokens,
        "max_prompt_speech_tokens": args.max_prompt_speech_tokens,
        "max_prompt_text_chars": args.max_prompt_text_chars,
        "max_prompt_text_tokens": args.max_prompt_text_tokens,
        "max_turn_text_chars": args.max_turn_text_chars,
        "max_turn_text_tokens": args.max_turn_text_tokens,
        "max_seq_len": max_seq_len,
        "avg_seq_len": total_tokens / len(records),
        "avg_loss_tokens": total_loss_tokens / len(records),
        "speech_token_offset": speech_start,
        "speech_end_exclusive": speech_end,
        "include_eos_in_loss_mask": not args.no_eos_loss,
        "token_freq_tokens": len(token_freq),
        "prepare_mode": "generated_trajectory",
        "prepare_variant": "multispeaker" if args.multi_speaker else "single_utterance",
        "multi_speaker": bool(args.multi_speaker),
        "num_speakers": int(args.num_speakers) if args.multi_speaker else None,
        "num_dialogues": int(args.num_dialogues) if (args.multi_speaker and args.num_dialogues) else None,
        "lang_mix": args.lang_mix if args.multi_speaker else None,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "concurrency": args.concurrency,
        "records_jsonl": str(records_path),
    }
    (output_dir / "soulx_peagle_prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d generated samples to %s", len(records), output_dir)
    log.info("Wrote token frequencies to %s", output_dir / "token_freq.pt")


if __name__ == "__main__":
    main()
