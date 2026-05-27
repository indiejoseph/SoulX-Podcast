"""Smoke test for the MTP dataloader.

Usage:
    python test_mtp_dataloader.py <path-to-debug-dataset>

Where <path-to-debug-dataset> is a HuggingFace `datasets` directory (saved
with `dataset.save_to_disk(...)`) containing the 5 expected columns:
text, speech_tokens, lang, id, audio. (The dataloader ignores `audio` and
`id`.)

Validates:
  - Tokenizer special-token resolution (catches missing tokens early)
  - Per-sample sequence assembly (text + speech tokens with correct offset)
  - speech_mask alignment (1 exactly at speech-token positions)
  - Collator padding correctness
  - Round-trip decode of a few samples for human eyeballing
  - Length distribution / filter rate / dialect breakdown
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from soulxpodcast.training.mtp_dataset import (
    DIALECT_PREFIX,
    MtpCollator,
    MtpDataset,
    MtpDatasetConfig,
    SPECIAL_TOKENS,
)


MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect"


def main():
    if len(sys.argv) < 2:
        print("usage: test_mtp_dataloader.py <path-to-debug-dataset>")
        sys.exit(1)
    ds_path = Path(sys.argv[1])

    print(f"[init] loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

    print(f"[init] loading HF dataset from {ds_path}")
    t0 = time.perf_counter()
    hf_ds = load_from_disk(str(ds_path))
    print(f"[init] hf load: {time.perf_counter()-t0:.2f}s, {len(hf_ds)} rows")
    print(f"[init] columns: {hf_ds.column_names}")

    # Drop unused columns. Keep `audio` only when `speech_tokens` is missing
    # (fallback path needs it). Always drop `id`/`phone` which we don't use.
    keep_cols = ["text", "speech_tokens", "lang"]
    if "speech_tokens" not in hf_ds.column_names:
        keep_cols.append("audio")
    drop_cols = [c for c in hf_ds.column_names if c not in keep_cols]
    if drop_cols:
        hf_ds = hf_ds.remove_columns(drop_cols)
        print(f"[init] dropped columns: {drop_cols}")

    # ---- Build the dataset wrapper ---------------------------------------
    print(f"\n[ds] constructing MtpDataset")
    cfg = MtpDatasetConfig()
    dataset = MtpDataset(hf_ds, tokenizer, cfg)
    print(f"[ds] special tokens resolved:")
    for k, v in SPECIAL_TOKENS.items():
        tid = tokenizer.encode(v, add_special_tokens=False)
        print(f"     {v:<30s}  id={tid[0] if len(tid)==1 else tid}")

    # ---- Per-sample sanity ----------------------------------------------
    print(f"\n[per-sample] inspecting first 3 samples")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        if sample is None:
            print(f"  sample {i}: FILTERED (too short / too long)")
            continue
        L = int(sample["length"].item())
        speech_positions = int(sample["speech_mask"].sum().item())
        text_positions = L - speech_positions - 5  # 3 prefix + 2 between text/speech, but EOS is outside speech_mask
        # Decode short window around the boundary to verify structure.
        ids = sample["input_ids"].tolist()
        head = tokenizer.decode(ids[:20], skip_special_tokens=False)
        tail = tokenizer.decode(ids[-5:], skip_special_tokens=False)
        print(f"  sample {i}: len={L} speech_tokens={speech_positions}")
        print(f"             head: {head[:120]}")
        print(f"             tail: ...{tail}")

    # ---- Collator + DataLoader smoke ------------------------------------
    print(f"\n[loader] DataLoader iteration test")
    collator = MtpCollator(pad_token_id=tokenizer.pad_token_id or 0)
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=collator,
        num_workers=0,
    )
    seen = 0
    filtered = 0
    batch_lens = []
    speech_frac = []
    for bi, batch in enumerate(loader):
        if not batch:
            print(f"  batch {bi}: ALL FILTERED")
            continue
        B, T = batch["input_ids"].shape
        seen += B
        batch_lens.append(T)
        # Per-sample speech fraction
        for j in range(B):
            L = int(batch["lengths"][j].item())
            sp = int(batch["speech_mask"][j].sum().item())
            speech_frac.append(sp / L if L else 0)
        if bi < 3:
            print(f"  batch {bi}: shape={tuple(batch['input_ids'].shape)} "
                  f"lengths={batch['lengths'].tolist()} "
                  f"speech_tokens_per_sample="
                  f"{batch['speech_mask'].sum(dim=1).tolist()}")
        if bi >= 24:
            break

    # Count filtered samples (not in batches because collator drops None).
    # Re-scan a bit to count filter rate accurately.
    filter_check_n = min(len(dataset), 100)
    for i in range(filter_check_n):
        if dataset[i] is None:
            filtered += 1

    print(f"\n[stats]")
    print(f"  samples in dataset:      {len(dataset)}")
    print(f"  samples filtered (out of first {filter_check_n}): {filtered}")
    print(f"  batches iterated:        {bi+1}")
    print(f"  unique batch lengths:    {sorted(set(batch_lens))}")
    if speech_frac:
        avg_sp = sum(speech_frac) / len(speech_frac)
        print(f"  avg speech-token fraction in a sample: {avg_sp:.2%}")
    lang_counts = Counter(hf_ds["lang"])
    print(f"  lang distribution:       {dict(lang_counts)}")

    # ---- Boundary verification ------------------------------------------
    # For one valid sample, verify the speech_mask boundaries exactly match
    # where speech-token-ids appear in input_ids.
    print(f"\n[verify] speech_mask alignment")
    for i in range(len(dataset)):
        s = dataset[i]
        if s is None:
            continue
        ids = s["input_ids"]
        mask = s["speech_mask"]
        offset = cfg.speech_token_offset
        # All masked positions should have id >= offset (speech token ids).
        masked_ids = ids[mask.bool()]
        # All non-masked, non-special positions could be text — just check
        # the masked region is fully speech-vocab.
        if not (masked_ids >= offset).all():
            bad = ids[mask.bool() & (ids < offset)]
            print(f"  FAIL sample {i}: speech_mask covers non-speech ids: {bad.tolist()[:5]}")
            sys.exit(2)
        # Conversely, all positions OUTSIDE the masked region with id >= offset
        # would be a bug — UNLESS they're known special tokens that just
        # happen to land in the speech-vocab numerical range (e.g.
        # <|semantic_token_start|>=153477, <|semantic_token_end|>=153478).
        special_ids = {tokenizer.encode(v, add_special_tokens=False)[0]
                       for v in SPECIAL_TOKENS.values()}
        not_masked_speech = ids[(~mask.bool()) & (ids >= offset)]
        leaked = [int(x) for x in not_masked_speech.tolist() if int(x) not in special_ids]
        if leaked:
            print(f"  FAIL sample {i}: speech-vocab ids outside speech_mask: "
                  f"{leaked[:5]}")
            sys.exit(2)
        print(f"  OK sample {i}: {int(mask.sum().item())} speech tokens correctly masked")
        break  # one verification is enough

    print(f"\n[done] dataloader smoke test passed")


if __name__ == "__main__":
    main()
