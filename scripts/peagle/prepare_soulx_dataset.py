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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from soulxpodcast.training.mtp_dataset import MtpDataset, MtpDatasetConfig


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

    hf_ds = hf_ds.shuffle(seed=args.seed)
    if args.max_samples is not None:
        hf_ds = hf_ds.select(range(min(args.max_samples, len(hf_ds))))

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
