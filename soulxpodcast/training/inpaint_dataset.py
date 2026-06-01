"""Inpaint training dataset for SoulX-Podcast.

Reads the JSONL corpus at ``tmp/dataset.jsonl`` (schema:
``{id, lang, text, phonemes, phoneme_ids, speech_tokens}``) and emits
per-sample tensor dicts ready for the inpaint trainer.

Per-sample assembly mirrors :mod:`soulxpodcast.training.mtp_dataset` so
the input_ids match what the LLM sees at inference:

    <|task_podcast|><|SPEAKER_0|>
        <|text_start|>{dialect_prefix}{text}<|text_end|>
        <|semantic_token_start|>{speech_tokens + offset}<|semantic_token_end|>

Inpaint additions on top of MTP layout:

- ``phone_token`` — LongTensor ``(K * T,)`` with phoneme ids aligned to the
  full input_ids buffer (K slots per token). Non-text positions are zero.
- ``phone_mask``  — BoolTensor ``(T,)`` True where at least one slot is set.

Alignment rule (universal — "broadcast within alignment unit"):

- The **alignment unit** is one syllable for Chinese and one word for
  English. Each unit produces one slot block of phoneme ids, and that
  same slot block is written to **every BPE text token whose char range
  overlaps the unit**. This means:

  Chinese ``yue`` / ``zh`` (spaces stripped — production text is space-free):

  - One Chinese character = one syllable = one ``(initial, final+tone)``
    pair (2 slots).
  - When two characters merge into one BPE (e.g. ``好似``) their slot
    pairs stack in the same BPE's slot block (cursor advances).
  - When one character byte-falls-back into multiple BPE tokens, the
    same pair is broadcast into all of them.

  English ``en`` (CMUdict ARPAbet, position-tagged via syllabification):

  - The dataset's ``|`` markers segment the phoneme stream into words.
  - Each word's ARPAbet stream is **syllabified** (maximal-onset) and
    encoded into position-tagged ids (``L_on`` vs ``L_co`` are
    distinct). See ``PhonemeTokenizer.encode_span``.
  - The resulting id sequence is broadcast across all BPE text tokens
    covering the word.

If a unit's id sequence overflows K slots, only the first K are kept
(the tail is dropped); the model then falls back to text embedding
for the dropped phonemes. From the dataset scan K=8 covers 97.3% of
English words and ~100% of CJK syllables.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from soulxpodcast.inpaint.tokenizer import PhonemeTokenizer

logger = logging.getLogger(__name__)


SPECIAL_TOKENS = {
    "task_podcast": "<|task_podcast|>",
    "speaker_0": "<|SPEAKER_0|>",
    "text_start": "<|text_start|>",
    "text_end": "<|text_end|>",
    "semantic_token_start": "<|semantic_token_start|>",
    "semantic_token_end": "<|semantic_token_end|>",
}
DIALECT_PREFIX = {
    "en": "",
    "zh": "",
    "yue": "<|Yue|>",
    "sichuan": "<|Sichuan|>",
    "henan": "<|Henan|>",
}
# Per-language alphabet name expected by PhonemeTokenizer.
LANG_TO_ALPHABET = {"en": "cmu", "zh": "pinyin", "yue": "jyutping"}
# Tokens that appear in the corpus's ``phonemes`` field but are not phonemes
# (punctuation pass-through).
NON_PHONEME_TOKENS = frozenset({".", ",", "!", "?", "|"})


@dataclass
class InpaintDatasetConfig:
    speech_token_offset: int = 153595
    max_total_tokens: int = 2048
    max_speech_tokens: int = 750
    min_speech_tokens: int = 8
    slots_per_token: int = 8  # must match the composer
    strict: bool = False  # if True, raise on alignment overflow; else drop
    # Unit-level random masking — mirrors CosyVoice-Inpaint upstream.
    # Each alignment unit (one CJK char or one English word) is *kept*
    # with probability `phoneme_keep_prob` and dropped to text-only
    # otherwise. Default 0.25 matches the upstream "sparse hint" recipe
    # and approximates the inference distribution where the user only
    # annotates a few words per sentence.
    phoneme_keep_prob: float = 0.25
    # If True, the alignment functions consume their internal RNG via
    # `random.random()` so behaviour is reproducible per-process with the
    # caller's global seed (useful for eval). When False, each
    # __getitem__ uses a fresh `random.random()` so dropout differs every
    # epoch and across DataLoader workers.
    deterministic_dropout: bool = False


def _is_cjk(ch: str) -> bool:
    """True for the CJK Unified Ideographs ranges (incl. extensions A-F)."""
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0x20000 <= o <= 0x2A6DF
        or 0x2A700 <= o <= 0x2EBEF
    )


def normalise_text(text: str, lang: str) -> str:
    """Strip spaces for zh/yue. Leave English alone."""
    if lang in {"zh", "yue", "sichuan", "henan"}:
        return text.replace(" ", "").replace("　", "")
    return text


def _resolve_special_tokens(tokenizer) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, surface in SPECIAL_TOKENS.items():
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"special token {surface!r} did not encode to a single id "
                f"(got {ids}); tokenizer mismatch?"
            )
        out[name] = ids[0]
    return out


def _build_char_to_tokens(
    text_offsets: list[tuple[int, int]],
) -> list[list[int]]:
    """Return char_index → list of text-token indices whose offsets cover it.

    Used by the broadcast rule: a single char may be covered by N>=1 BPE
    tokens (byte fallback), so every covering token receives the same
    composed embedding.
    """
    max_char = max((ce for _, ce in text_offsets), default=0)
    out: list[list[int]] = [[] for _ in range(max_char)]
    for ti, (cs, ce) in enumerate(text_offsets):
        for ci in range(cs, ce):
            out[ci].append(ti)
    return out


def _write_unit_into_tokens(
    per_token: list[list[int]],
    cursor: list[int],
    token_indices: list[int],
    ids: list[int],
    K: int,
) -> bool:
    """Write the same `ids` into the slot blocks of all `token_indices`.

    Returns True on success, False if any of the tokens runs out of room
    (mismatched cursor). On failure the partial writes are left in place;
    the caller decides whether to roll the sample back.
    """
    if not token_indices:
        return False
    # All target tokens must have the same number of free slots — otherwise
    # the broadcast can't write identical blocks.
    if any(cursor[t] != cursor[token_indices[0]] for t in token_indices):
        # If callers previously wrote into one of these tokens but not the
        # others (e.g. two chars share BPE A but only one shares BPE B),
        # broadcast can't honour identical blocks. Fall back to a single
        # write into the first token only.
        target = token_indices[0]
        free = K - cursor[target]
        n_write = min(free, len(ids))
        per_token[target][cursor[target] : cursor[target] + n_write] = ids[:n_write]
        cursor[target] += n_write
        return n_write > 0

    cur = cursor[token_indices[0]]
    free = K - cur
    if free <= 0:
        return False
    n_write = min(free, len(ids))
    head = ids[:n_write]
    for t in token_indices:
        per_token[t][cur : cur + n_write] = head
        cursor[t] += n_write
    return True


def _align_chinese(
    text: str,
    text_offsets: list[tuple[int, int]],
    phonemes: list[str],
    alphabet: str,
    tokenizer_obj: PhonemeTokenizer,
    K: int,
    keep_prob: float = 1.0,
    rng: Optional[random.Random] = None,
) -> tuple[list[list[int]], int, int]:
    """One syllable per CJK char; broadcast each pair to all BPE tokens covering that char.

    Returns ``(per_token_slots, n_units_kept, n_units_seen)``. With
    ``keep_prob < 1`` each unit is independently kept with probability
    ``keep_prob``; otherwise it's dropped (zeroed) — mimics the sparse-
    annotation inference distribution.
    """
    T_text = len(text_offsets)
    per_token = [[0] * K for _ in range(T_text)]
    cursor = [0] * T_text
    char_to_tokens = _build_char_to_tokens(text_offsets)
    queue = [p for p in phonemes if p and p not in NON_PHONEME_TOKENS]
    rng_ = rng or random
    qi = 0
    kept = 0
    seen = 0
    for ci, ch in enumerate(text):
        if qi >= len(queue):
            break
        if not _is_cjk(ch):
            continue
        syl = queue[qi]
        if not syl[-1].isdigit():
            qi += 1
            continue
        seen += 1
        qi += 1
        # Unit-level random drop — sparse-annotation training (auxiliary
        # info: model must learn to use phonemes when present AND fall
        # back to text when absent).
        if rng_.random() >= keep_prob:
            continue
        ini, fin = tokenizer_obj.split_whole_syllable(alphabet, syl)
        try:
            ids = tokenizer_obj.encode_span(alphabet, [ini, fin])
        except ValueError:
            continue
        toks = char_to_tokens[ci] if ci < len(char_to_tokens) else []
        if not toks:
            continue
        if _write_unit_into_tokens(per_token, cursor, toks, ids, K):
            kept += 1
    return per_token, kept, seen


def _align_english(
    text: str,
    text_offsets: list[tuple[int, int]],
    phonemes: list[str],
    tokenizer_obj: PhonemeTokenizer,
    K: int,
    keep_prob: float = 1.0,
    rng: Optional[random.Random] = None,
) -> tuple[list[list[int]], int, int]:
    """One word = one unit. Each word's ARPAbet stream is syllabified and
    position-tagged via ``PhonemeTokenizer``, then broadcast across every
    BPE token covering the word's char range. Tail beyond K is dropped.

    With ``keep_prob < 1`` each word is independently kept with
    probability ``keep_prob`` (sparse-annotation training).
    Returns ``(per_token_slots, n_words_kept, n_words_seen)``.
    """
    T_text = len(text_offsets)
    per_token = [[0] * K for _ in range(T_text)]
    cursor = [0] * T_text
    char_to_tokens = _build_char_to_tokens(text_offsets)
    rng_ = rng or random

    # Split phonemes by '|' into per-word lists.
    pwords: list[list[str]] = [[]]
    for p in phonemes:
        if p == "|":
            pwords.append([])
        else:
            pwords[-1].append(p)
    pwords = [w for w in pwords if w]

    # Find text words (runs of alpha + apostrophe).
    text_words: list[tuple[int, int]] = []
    in_word = False
    start = 0
    for ci, ch in enumerate(text):
        if ch.isalpha() or ch == "'":
            if not in_word:
                start = ci
                in_word = True
        else:
            if in_word:
                text_words.append((start, ci))
                in_word = False
    if in_word:
        text_words.append((start, len(text)))

    kept = 0
    seen = 0
    for word_idx, (cs, ce) in enumerate(text_words):
        if word_idx >= len(pwords):
            break
        seen += 1
        if rng_.random() >= keep_prob:
            continue
        ids = tokenizer_obj.encode_span("cmu", pwords[word_idx])
        token_set: list[int] = []
        already: set[int] = set()
        for ci in range(cs, ce):
            if ci >= len(char_to_tokens):
                continue
            for t in char_to_tokens[ci]:
                if t not in already:
                    already.add(t)
                    token_set.append(t)
        if not token_set:
            continue
        if _write_unit_into_tokens(per_token, cursor, token_set, ids, K):
            kept += 1
    return per_token, kept, seen


@dataclass
class InpaintSample:
    input_ids: torch.Tensor      # (T,)
    attention_mask: torch.Tensor  # (T,)
    speech_mask: torch.Tensor    # (T,) — 1 at speech-token positions (loss target)
    phone_token: torch.Tensor    # (K * T,)
    phone_mask: torch.Tensor     # (T,) bool — True where composed embed replaces text emb


class InpaintDataset(Dataset):
    """JSONL-backed inpaint training dataset."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        config: Optional[InpaintDatasetConfig] = None,
        lang_filter: Optional[set[str]] = None,
        max_samples: int = 0,
    ):
        self.cfg = config or InpaintDatasetConfig()
        self.tokenizer = tokenizer
        self.phone_tok = PhonemeTokenizer()
        self.special = _resolve_special_tokens(tokenizer)

        # Materialise the rows we want into memory. The smoke dataset is small;
        # for a full training run a streaming reader would be better.
        self.rows: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if lang_filter and r["lang"] not in lang_filter:
                    continue
                self.rows.append(r)
                if max_samples and len(self.rows) >= max_samples:
                    break
        logger.info(f"InpaintDataset loaded {len(self.rows)} rows from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Optional[dict]:
        row = self.rows[idx]
        lang = row["lang"]
        text = normalise_text(row["text"], lang)
        phonemes: list[str] = row["phonemes"]
        speech_raw: list[int] = row["speech_tokens"]

        if not (self.cfg.min_speech_tokens <= len(speech_raw) <= self.cfg.max_speech_tokens):
            return None

        # Text → tokens with offsets.
        prefix = DIALECT_PREFIX.get(lang, "")
        text_with_prefix = prefix + text
        enc = self.tokenizer(
            text_with_prefix, add_special_tokens=False, return_offsets_mapping=True
        )
        text_ids: list[int] = enc["input_ids"]
        offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]

        # Phoneme alignment to the **text portion only** (relative offsets).
        K = self.cfg.slots_per_token
        alphabet = LANG_TO_ALPHABET[lang]
        # Shift offsets so they index into ``text_with_prefix``; for alignment
        # we treat the dialect-prefix tokens as having no phoneme. We use the
        # raw text portion of the string (after the dialect prefix) for
        # CJK/word checks — so subtract len(prefix) from offsets and skip
        # tokens whose offsets fall inside the prefix.
        prefix_len = len(prefix)
        text_part_offsets: list[tuple[int, int]] = []
        text_part_token_index: list[int] = []  # index into text_ids
        for ti, (cs, ce) in enumerate(offsets):
            if ce <= prefix_len:
                continue  # entirely inside the prefix
            adj = (max(0, cs - prefix_len), max(0, ce - prefix_len))
            text_part_offsets.append(adj)
            text_part_token_index.append(ti)

        # Per-call RNG so DataLoader workers see different draws each step.
        # deterministic_dropout=True uses the row index as the seed for
        # reproducible eval.
        if self.cfg.deterministic_dropout:
            rng = random.Random(idx)
        else:
            rng = random
        keep_prob = self.cfg.phoneme_keep_prob

        if alphabet == "cmu":
            per_token_text, kept, seen = _align_english(
                text, text_part_offsets, phonemes, self.phone_tok, K,
                keep_prob=keep_prob, rng=rng,
            )
        else:
            per_token_text, kept, seen = _align_chinese(
                text, text_part_offsets, phonemes, alphabet, self.phone_tok, K,
                keep_prob=keep_prob, rng=rng,
            )

        # We keep the sample even when `kept == 0` so the LLM also trains on
        # text-only inputs (composer is fully off for those). `seen == 0`
        # means the row has no phoneme info at all — drop it.
        if seen == 0:
            return None

        # Speech tokens with offset.
        speech_ids = [t + self.cfg.speech_token_offset for t in speech_raw]

        # Assemble the full sequence.
        task_prefix = [
            self.special["task_podcast"],
            self.special["speaker_0"],
            self.special["text_start"],
        ]
        bridge = [self.special["text_end"], self.special["semantic_token_start"]]
        suffix = [self.special["semantic_token_end"]]

        input_ids = task_prefix + text_ids + bridge + speech_ids + suffix
        if len(input_ids) > self.cfg.max_total_tokens:
            return None

        T = len(input_ids)
        text_token_global_start = len(task_prefix)
        speech_global_start = text_token_global_start + len(text_ids) + len(bridge)
        speech_global_end = speech_global_start + len(speech_ids)

        phone_token = torch.zeros(K * T, dtype=torch.long)
        # write per_token_text into the global buffer
        for local_ti, global_ti in enumerate(text_part_token_index):
            tgt = text_token_global_start + global_ti
            slot_block = per_token_text[local_ti]
            phone_token[tgt * K : tgt * K + K] = torch.tensor(slot_block, dtype=torch.long)

        speech_mask = torch.zeros(T, dtype=torch.long)
        speech_mask[speech_global_start : speech_global_end + 1] = 1  # include EOS

        # phone_mask derived directly from any non-zero slot in each block
        phone_mask_2d = phone_token.view(T, K) != 0
        phone_mask = phone_mask_2d.any(dim=-1)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(T, dtype=torch.long),
            "speech_mask": speech_mask,
            "phone_token": phone_token,
            "phone_mask": phone_mask,
            "n_phonemes_kept": kept,
            "n_phonemes_seen": seen,
        }


def collate(batch: list[Optional[dict]], pad_token_id: int = 0) -> dict:
    """Pad-and-stack collator. Drops Nones (filter failures)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    K = batch[0]["phone_token"].numel() // batch[0]["input_ids"].numel()
    Tmax = max(b["input_ids"].numel() for b in batch)
    B = len(batch)

    input_ids = torch.full((B, Tmax), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, Tmax), dtype=torch.long)
    speech_mask = torch.zeros((B, Tmax), dtype=torch.long)
    phone_token = torch.zeros((B, K * Tmax), dtype=torch.long)
    phone_mask = torch.zeros((B, Tmax), dtype=torch.bool)

    for i, b in enumerate(batch):
        T = b["input_ids"].numel()
        input_ids[i, :T] = b["input_ids"]
        attention_mask[i, :T] = b["attention_mask"]
        speech_mask[i, :T] = b["speech_mask"]
        phone_token[i, : K * T] = b["phone_token"]
        phone_mask[i, :T] = b["phone_mask"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "speech_mask": speech_mask,
        "phone_token": phone_token,
        "phone_mask": phone_mask,
        "n_phonemes_kept": sum(b["n_phonemes_kept"] for b in batch),
        "n_phonemes_seen": sum(b["n_phonemes_seen"] for b in batch),
    }
