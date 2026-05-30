#!/usr/bin/env python3
"""Prepare SoulX speech-token data for Speculators P-EAGLE training."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from soulxpodcast.training.mtp_dataset import (
    DIALECT_PREFIX,
    SPECIAL_TOKENS,
    MtpDataset,
    MtpDatasetConfig,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_soulx_peagle")


def resolve_speech_token_offset(tokenizer) -> int:
    ids = tokenizer.encode("<|0|>", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Tokenizer did not encode '<|0|>' as one token: {ids}")
    return int(ids[0])


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


def parse_speech_tokens(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(x) for x in value.split()] if value else []
    return [int(x) for x in value]


def apply_dialect_prefix(text: str, lang: str, *, skip_dialect_prefix: bool) -> str:
    if skip_dialect_prefix:
        return text
    prefix = DIALECT_PREFIX.get(lang, "")
    if prefix and not text.startswith(prefix):
        return prefix + text
    return text


def prepare_fast(
    *,
    hf_ds,
    tokenizer,
    output_dir: Path,
    args: argparse.Namespace,
    speech_token_offset: int,
) -> None:
    special = resolve_special_tokens(tokenizer)
    task_prefix_ids = [
        special["task_podcast"],
        special["speaker_0"],
        special["text_start"],
    ]
    text_to_speech_ids = [
        special["text_end"],
        special["semantic_token_start"],
    ]
    speech_suffix_ids = [special["semantic_token_end"]]

    def convert_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        texts = [
            apply_dialect_prefix(
                str(text),
                str(lang),
                skip_dialect_prefix=args.skip_dialect_prefix,
            )
            for text, lang in zip(
                batch["text"],
                batch["lang"],
                strict=True,
            )
        ]
        text_ids_batch = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]

        out_input_ids: list[list[int]] = []
        out_loss_mask: list[list[int]] = []
        out_seq_len: list[int] = []

        for text_ids, speech_value in zip(
            text_ids_batch,
            batch["speech_tokens"],
            strict=True,
        ):
            speech_tokens_raw = parse_speech_tokens(speech_value)
            n_speech = len(speech_tokens_raw)
            if n_speech < args.min_speech_tokens or n_speech > args.max_speech_tokens:
                continue

            speech_tokens_llm = [tok + speech_token_offset for tok in speech_tokens_raw]
            ids = (
                task_prefix_ids
                + list(text_ids)
                + text_to_speech_ids
                + speech_tokens_llm
                + speech_suffix_ids
            )
            if len(ids) > args.seq_length:
                continue

            speech_start = len(task_prefix_ids) + len(text_ids) + len(text_to_speech_ids)
            speech_end = speech_start + len(speech_tokens_llm)
            mask_end = speech_end + (0 if args.no_eos_loss else 1)
            loss_mask = [0] * len(ids)
            loss_mask[speech_start:mask_end] = [1] * (mask_end - speech_start)

            out_input_ids.append(ids)
            out_loss_mask.append(loss_mask)
            out_seq_len.append(len(ids))

        return {
            "input_ids": out_input_ids,
            "loss_mask": out_loss_mask,
            "seq_len": out_seq_len,
        }

    remove_columns = list(hf_ds.column_names)
    out_ds = hf_ds.map(
        convert_batch,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=args.num_proc if args.num_proc > 1 else None,
        remove_columns=remove_columns,
        desc="Preparing SoulX P-EAGLE data",
        features=make_features(),
    )
    if len(out_ds) == 0:
        raise RuntimeError("No valid P-EAGLE samples were produced")

    out_ds.set_format(type="torch", columns=["input_ids", "loss_mask", "seq_len"])
    out_ds.save_to_disk(str(output_dir))

    token_freq: Counter[int] = Counter()
    total_tokens = 0
    total_loss_tokens = 0
    max_seq_len = 0
    for row in tqdm(out_ds, desc="Counting SoulX P-EAGLE tokens"):
        input_ids = row["input_ids"].tolist()
        loss_mask = row["loss_mask"].tolist()
        seq_len = int(row["seq_len"])
        loss_tokens = int(sum(loss_mask))
        total_tokens += seq_len
        total_loss_tokens += loss_tokens
        max_seq_len = max(max_seq_len, seq_len)
        token_freq.update(tok for tok, mask in zip(input_ids, loss_mask, strict=True) if mask)

    torch.save(dict(token_freq), output_dir / "token_freq.pt")

    summary = {
        "source_dataset": str(args.dataset_path),
        "model_path": args.model_path,
        "input_rows_scanned": len(hf_ds),
        "kept_samples": len(out_ds),
        "filtered_samples": len(hf_ds) - len(out_ds),
        "seq_length": args.seq_length,
        "max_seq_len": max_seq_len,
        "avg_seq_len": total_tokens / len(out_ds),
        "avg_loss_tokens": total_loss_tokens / len(out_ds),
        "speech_token_offset": speech_token_offset,
        "include_eos_in_loss_mask": not args.no_eos_loss,
        "token_freq_tokens": len(token_freq),
        "prepare_mode": "fast_map",
        "map_batch_size": args.map_batch_size,
        "num_proc": args.num_proc,
    }
    (output_dir / "soulx_peagle_prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d samples to %s", len(out_ds), output_dir)
    log.info("Wrote token frequencies to %s", output_dir / "token_freq.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="HF dataset saved with load_from_disk().",
    )
    parser.add_argument("--model-path", required=True, help="SoulX verifier model path.")
    parser.add_argument("--output-dir", required=True, help="Output preprocessed dataset dir.")
    parser.add_argument(
        "--split",
        default=None,
        help="DatasetDict split to use. Defaults to train if present.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max input rows to scan after shuffle.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before conversion. Full-dataset prepare skips this by default for faster Arrow reads.",
    )
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--min-speech-tokens", type=int, default=8)
    parser.add_argument("--max-speech-tokens", type=int, default=750)
    parser.add_argument(
        "--skip-dialect-prefix",
        action="store_true",
        help="Do not prepend dialect tags from the lang column.",
    )
    parser.add_argument(
        "--no-eos-loss",
        action="store_true",
        help="Do not include semantic_token_end in the P-EAGLE loss mask.",
    )
    parser.add_argument(
        "--allow-audio-tokenize-fallback",
        action="store_true",
        help="Allow slow audio->s3tokenizer fallback when speech_tokens is absent.",
    )
    parser.add_argument(
        "--legacy-loop",
        action="store_true",
        help="Use the old per-row MtpDataset loop. Mostly for debugging.",
    )
    parser.add_argument("--map-batch-size", type=int, default=2000)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    speech_token_offset = resolve_speech_token_offset(tokenizer)

    hf_ds = load_split(dataset_path, args.split)
    if (
        "speech_tokens" not in hf_ds.column_names
        and not args.allow_audio_tokenize_fallback
    ):
        raise ValueError(
            "Dataset has no 'speech_tokens' column. Precompute speech tokens first "
            "or pass --allow-audio-tokenize-fallback for slow debug runs."
        )

    if args.shuffle or args.max_samples is not None:
        hf_ds = hf_ds.shuffle(seed=args.seed)
    if args.max_samples is not None:
        hf_ds = hf_ds.select(range(min(args.max_samples, len(hf_ds))))

    if "speech_tokens" in hf_ds.column_names and not args.legacy_loop:
        prepare_fast(
            hf_ds=hf_ds,
            tokenizer=tokenizer,
            output_dir=output_dir,
            args=args,
            speech_token_offset=speech_token_offset,
        )
        return

    mtp_config = MtpDatasetConfig(
        speech_token_offset=speech_token_offset,
        max_total_tokens=args.seq_length,
        min_speech_tokens=args.min_speech_tokens,
        max_speech_tokens=args.max_speech_tokens,
        skip_dialect_prefix=args.skip_dialect_prefix,
        include_eos_in_speech_mask=not args.no_eos_loss,
        s3tokenizer_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    mtp_ds = MtpDataset(hf_ds, tokenizer, mtp_config)

    records: list[dict] = []
    token_freq: Counter[int] = Counter()
    filtered = 0
    total_tokens = 0
    total_loss_tokens = 0
    max_seq_len = 0

    for i in tqdm(range(len(mtp_ds)), desc="Preparing SoulX P-EAGLE data"):
        sample = mtp_ds[i]
        if sample is None:
            filtered += 1
            continue
        input_ids = sample["input_ids"].tolist()
        loss_mask = sample["speech_mask"].to(torch.int8).tolist()
        loss_tokens = int(sum(loss_mask))
        if loss_tokens <= 0:
            filtered += 1
            continue

        seq_len = len(input_ids)
        records.append(
            {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "seq_len": seq_len,
            }
        )
        total_tokens += seq_len
        total_loss_tokens += loss_tokens
        max_seq_len = max(max_seq_len, seq_len)
        token_freq.update(
            tok for tok, mask in zip(input_ids, loss_mask, strict=True) if mask
        )

    if not records:
        raise RuntimeError("No valid P-EAGLE samples were produced")

    out_ds = Dataset.from_list(records, features=make_features())
    out_ds.set_format(type="torch", columns=["input_ids", "loss_mask", "seq_len"])
    out_ds.save_to_disk(str(output_dir))

    torch.save(dict(token_freq), output_dir / "token_freq.pt")

    summary = {
        "source_dataset": str(dataset_path),
        "model_path": args.model_path,
        "input_rows_scanned": len(mtp_ds),
        "kept_samples": len(records),
        "filtered_samples": filtered,
        "seq_length": args.seq_length,
        "max_seq_len": max_seq_len,
        "avg_seq_len": total_tokens / len(records),
        "avg_loss_tokens": total_loss_tokens / len(records),
        "speech_token_offset": speech_token_offset,
        "include_eos_in_loss_mask": not args.no_eos_loss,
        "token_freq_tokens": len(token_freq),
    }
    (output_dir / "soulx_peagle_prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d samples to %s", len(records), output_dir)
    log.info("Wrote token frequencies to %s", output_dir / "token_freq.pt")


if __name__ == "__main__":
    main()
