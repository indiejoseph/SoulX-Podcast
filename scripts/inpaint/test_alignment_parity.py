"""Training-vs-inference alignment parity test.

The composer is trained against the slot tensors produced by
``training/inpaint_dataset.InpaintDataset.__getitem__`` and at inference is
fed slot tensors built by ``InpaintInferenceEngine._build_text_and_phone_tokens``.
If these two paths produce different slot layouts for the same logical input,
the composer learns a distribution that doesn't match what it sees in
deployment — the v1-v10 cycle of "trains fine, sounds wrong" was largely
this contract drifting unnoticed.

This test runs every fixture through BOTH pipelines and asserts:

  1. ``text_ids`` (the LLM input token sequence over the text portion) is
     byte-identical.
  2. ``phone_token`` slot blocks at positions the inference pipeline marks
     as inpainted (``phone_mask=True``) are byte-identical to the training
     pipeline's blocks at the same logical positions.
  3. ``phone_mask`` agrees over those positions.

Train phonemes are constructed in ``fixtures_adversarial.py`` using 'X'
placeholders so non-target chars are skipped — letting us drive both pipelines
with the SAME logical override and compare apples-to-apples.

Run:

  python scripts/inpaint/test_alignment_parity.py

Exit code 0 = all parity asserts passed. Non-zero = at least one drift.
This test does NOT load the LLM or composer weights; it is fast (seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer

from scripts.inpaint.fixtures_adversarial import ALL_FIXTURES, Fixture, verify_bpe_assertions
from soulxpodcast.inpaint.inference import InpaintInferenceEngine
from soulxpodcast.inpaint.ssml import parse_ssml
from soulxpodcast.inpaint.tokenizer import PhonemeTokenizer
from soulxpodcast.training.inpaint_dataset import (
    DIALECT_PREFIX, LANG_TO_ALPHABET, _align_chinese, _align_english,
    normalise_text,
)


MODEL_PATH = "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged"
SLOTS_PER_TOKEN = 8  # K — must match training cfg


# ---------------------------------------------------------------- helpers


def _build_inference_side(
    tokenizer, phone_tok: PhonemeTokenizer, ssml: str, lang: str, K: int
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Mirror InpaintInferenceEngine._build_text_and_phone_tokens *without*
    loading the full engine (which would load 1.7B weights). Re-implements
    the alignment logic inline by calling the same _align_spans_to_tokens
    method we want to test.
    """
    # We need a tiny stand-in for the engine that has `.K`, `.tokenizer`,
    # `.phone_tok`, and the alignment methods on InpaintInferenceEngine
    # (which are pure functions of those three attributes).
    class _Stub:
        pass
    stub = _Stub()
    stub.K = K
    stub.tokenizer = tokenizer
    stub.phone_tok = phone_tok

    # Borrow all the alignment-related bound methods from the real class —
    # `_build_text_and_phone_tokens` delegates to `_align_spans_to_tokens`
    # and `_write_unit`. They depend only on the stubbed attrs above.
    for name in ("_build_text_and_phone_tokens", "_align_spans_to_tokens", "_write_unit"):
        attr = getattr(InpaintInferenceEngine, name)
        # `_write_unit` is a @staticmethod — bind directly.
        if isinstance(InpaintInferenceEngine.__dict__.get(name), staticmethod):
            setattr(stub, name, attr)
        else:
            setattr(stub, name, attr.__get__(stub))
    _build = stub._build_text_and_phone_tokens

    surface_text, spans = parse_ssml(ssml)
    prefix = DIALECT_PREFIX.get(lang, "")
    _, text_ids, _, phone_token, phone_mask, _ = _build(
        surface_text=surface_text, spans=spans, prefix=prefix, lang=lang,
        disable_inpaint=False,
    )
    return text_ids, phone_token, phone_mask


def _build_training_side(
    tokenizer, phone_tok: PhonemeTokenizer,
    text: str, train_phonemes: list[str], lang: str, K: int,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Run the training-side alignment over the same logical input.

    This mirrors the relevant slice of InpaintDataset.__getitem__ — just the
    text → input_ids + phone_token portion, without the speech-tokens append
    and the silence stripping (none of which affect the text-portion slot
    tensors we want to check).
    """
    text = normalise_text(text, lang)
    prefix = DIALECT_PREFIX.get(lang, "")
    text_with_prefix = prefix + text
    enc = tokenizer(text_with_prefix, add_special_tokens=False, return_offsets_mapping=True)
    text_ids = enc["input_ids"]
    offsets = [tuple(o) for o in enc["offset_mapping"]]

    # Adjust offsets to text-portion-relative (skip dialect-prefix tokens).
    prefix_len = len(prefix)
    text_part_offsets: list[tuple[int, int]] = []
    text_part_token_index: list[int] = []
    for ti, (cs, ce) in enumerate(offsets):
        if ce <= prefix_len:
            continue
        adj = (max(0, cs - prefix_len), max(0, ce - prefix_len))
        text_part_offsets.append(adj)
        text_part_token_index.append(ti)

    alphabet = LANG_TO_ALPHABET[lang]
    if alphabet == "cmu":
        per_token_text, _, _ = _align_english(
            text, text_part_offsets, train_phonemes, phone_tok, K, keep_prob=1.0,
        )
    else:
        per_token_text, _, _ = _align_chinese(
            text, text_part_offsets, train_phonemes, alphabet, phone_tok, K, keep_prob=1.0,
        )

    # Write per_token_text into a full phone_token aligned to text_ids
    # (zero-padded for prefix-only tokens).
    T = len(text_ids)
    phone_token = torch.zeros(K * T, dtype=torch.long)
    for local_ti, global_ti in enumerate(text_part_token_index):
        block = per_token_text[local_ti]
        phone_token[global_ti * K : global_ti * K + K] = torch.tensor(block, dtype=torch.long)

    phone_mask = (phone_token.view(T, K) != 0).any(dim=-1)
    return text_ids, phone_token, phone_mask


# ---------------------------------------------------------------- checks


def _compare_one_pole(
    fx: Fixture,
    pole: str,  # "correct" or "wrong"
    tokenizer,
    phone_tok: PhonemeTokenizer,
    K: int,
) -> list[str]:
    """Run a single SSML pole through both paths; return list of failure messages
    (empty list = parity passes)."""
    ssml = fx.ssml_correct if pole == "correct" else fx.ssml_wrong
    train_ph = fx.train_phonemes_correct if pole == "correct" else fx.train_phonemes_wrong

    inf_ids, inf_phone, inf_mask = _build_inference_side(
        tokenizer, phone_tok, ssml, fx.lang, K
    )
    tr_ids, tr_phone, tr_mask = _build_training_side(
        tokenizer, phone_tok, fx.text, train_ph, fx.lang, K
    )

    fails: list[str] = []

    # 1. text_ids byte-identical
    if inf_ids != tr_ids:
        fails.append(
            f"text_ids drift: inference={inf_ids} training={tr_ids}"
        )
        # If text_ids differ the slot comparison is meaningless — return early.
        return fails

    # 2. phone_mask agrees at positions inference marks as inpainted.
    if not torch.equal(inf_mask, tr_mask):
        # Allow training to mark MORE positions (full annotation) than
        # inference (sparse SSML) — that's expected. Fail only if inference
        # marks a position training does NOT.
        only_in_inf = (inf_mask & ~tr_mask).nonzero(as_tuple=False).flatten().tolist()
        if only_in_inf:
            fails.append(
                f"phone_mask drift: inference marks positions {only_in_inf} "
                f"as inpainted but training does not"
            )

    # 3. slot blocks at masked positions are byte-identical.
    masked_positions = inf_mask.nonzero(as_tuple=False).flatten().tolist()
    for pos in masked_positions:
        inf_block = inf_phone[pos * K : (pos + 1) * K].tolist()
        tr_block = tr_phone[pos * K : (pos + 1) * K].tolist()
        if inf_block != tr_block:
            fails.append(
                f"slot drift at BPE pos {pos}: "
                f"inference={inf_block} training={tr_block}"
            )

    return fails


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    phone_tok = PhonemeTokenizer()

    # Sanity: tokenizer hasn't drifted from what fixtures assume.
    verify_bpe_assertions(tokenizer)

    total = 0
    failed = 0
    for fx in ALL_FIXTURES:
        for pole in ("correct", "wrong"):
            total += 1
            fails = _compare_one_pole(fx, pole, tokenizer, phone_tok, SLOTS_PER_TOKEN)
            tag = f"{fx.name}.{pole}"
            if fails:
                failed += 1
                print(f"[FAIL] {tag}")
                for f in fails:
                    print(f"         {f}")
            else:
                print(f"[ok]   {tag}")

    print()
    print(f"{total - failed}/{total} parity checks passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
