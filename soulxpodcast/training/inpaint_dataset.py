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

# Per-language speech-token IDs identified as "silence padding" via audit of
# dataset position distributions. These are s3tokenizer codebook entries that
# decode to flat / low-energy audio and are heavily over-represented at the
# start/end of utterances. Used to mask silence-target positions out of the
# speech-token CE loss so the composer can't shortcut to "predict silence".
SILENCE_TOKEN_IDS: dict[str, frozenset[int]] = {
    # Boundary-share ≥0.1% AND ≥2× global unigram frequency at position 0 or -1
    # in tmp/dataset.jsonl. zh has ~66 silence-coding ids clustered around
    # 3700-3900 (intake-breath) and 4200-4300 (fade-out).
    "zh": frozenset({
        3969, 6402, 4227, 3972, 6405, 3975, 4105, 4106, 6159, 3729,
        6162, 3732, 3863, 1944, 1947, 1950, 5919, 4131, 1701, 4134,
        1704, 4137, 1707, 3888, 3891, 6324, 3892, 3894, 3895, 4024,
        4025, 1594, 1595, 2109, 6078, 6079, 2112, 3648, 6082, 6081,
        3651, 3781, 3782, 1734, 4296, 4299, 5835, 1869, 5838, 3704,
        4308, 6486, 4056, 2025, 1514, 2028, 2031, 3700, 3701, 2037,
        4215, 2040, 1785, 4218, 4212, 1788,
    }),
    # yue has minimal silence padding — just 4 tokens at ~1% each
    "yue": frozenset({2031, 1950, 4218, 4137}),
    # en (audiobook) silence vocabulary
    "en": frozenset({
        4218, 2031, 4299, 2112, 4137, 1950, 3648, 3651, 5838, 5835,
        1707, 3975, 3894, 1788, 3645, 6405, 6486, 3732, 1704, 3891,
        5919, 2922, 5109, 3888, 1461, 5832,
    }),
}


@dataclass
class InpaintDatasetConfig:
    speech_token_offset: int = 153595
    max_total_tokens: int = 2048
    max_speech_tokens: int = 750
    min_speech_tokens: int = 8
    # Storage K — must equal PhonemeComposer.K_STORAGE (6 in v11).
    slots_per_token: int = 6
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
    # v12 — grapheme substitution. When True (the v12 default), a kept
    # alignment unit's covering BPE token(s) are REMOVED from input_ids and
    # replaced by the phoneme-carrying pad(s). This makes the composer the
    # ONLY signal at the annotated position, so the LM cannot read the
    # natural reading off the surviving grapheme. When False, the v11
    # behaviour is used (pad INSERTED after the grapheme; grapheme kept —
    # which left the phoneme redundant and the composer collapsed onto a
    # per-alphabet marker; see docs/inpaint_v12_spec.md). Dropout is also
    # made BPE-group-coherent under substitution so merged BPEs (银行, 中国)
    # are kept/dropped as a whole and actually get masked at rate keep_prob.
    grapheme_subst: bool = True


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


def _block(ids: list[int], K: int) -> list[int]:
    """Zero-pad an id list out to K slots."""
    if len(ids) > K:
        return list(ids[:K])
    return list(ids) + [0] * (K - len(ids))


def _splice_pads(
    text_ids: list[int],
    offsets: list[tuple[int, int]],
    inserts: list[tuple[int, list[list[int]]]],
    pad_id: int,
    K: int,
) -> tuple[list[int], list[list[int]], list[bool]]:
    """Insert pad tokens after the BPE positions covering each char.

    Args:
        text_ids: BPE ids of the natural (unmarked) text.
        offsets: per-BPE (char_start, char_end) offsets, parallel to text_ids.
        inserts: list of (char_pos, [K-block, K-block, ...]) — each entry
            requests one pad token per K-block, inserted after the BPE that
            covers ``char_pos``.
        pad_id: tokenizer's pad token id (also acts as a marker the LLM
            sees at composer-injected positions).
        K: storage K (constant across alphabets).

    Returns:
        new_text_ids: text_ids with pads inserted.
        new_slots: per-position K-slot block, parallel to new_text_ids.
            Zero blocks at non-pad positions.
        new_mask: per-position bool, True at pad positions only.

    If a char appears as N consecutive byte-fallback BPEs (e.g. CJK chars
    that tokenize to 2-3 byte tokens), the pad is appended after the LAST
    of those BPEs. That keeps the composer's fire-position invariant to
    the char's byte-length, which is what unblocks the "broadcast stutter"
    on byte-fallback CJK.
    """
    # char_pos → index of the LAST BPE that covers it (largest-bi wins on tie)
    char_to_last_bpe: dict[int, int] = {}
    for bi, (cs, ce) in enumerate(offsets):
        for cp in range(cs, ce):
            char_to_last_bpe[cp] = bi

    # bpe_idx → list of K-blocks to insert immediately after that BPE
    by_bpe: dict[int, list[list[int]]] = {}
    for cp, blocks in inserts:
        if cp not in char_to_last_bpe:
            continue
        bi = char_to_last_bpe[cp]
        by_bpe.setdefault(bi, []).extend(blocks)

    new_text_ids: list[int] = []
    new_slots: list[list[int]] = []
    new_mask: list[bool] = []
    for bi, tid in enumerate(text_ids):
        new_text_ids.append(tid)
        new_slots.append([0] * K)
        new_mask.append(False)
        if bi in by_bpe:
            for block in by_bpe[bi]:
                new_text_ids.append(pad_id)
                new_slots.append(_block(block, K))
                new_mask.append(True)
    return new_text_ids, new_slots, new_mask


def _align_padsub_chinese(
    text: str,
    text_offsets: list[tuple[int, int]],
    text_ids: list[int],
    phonemes: list[str],
    alphabet: str,
    tokenizer_obj: PhonemeTokenizer,
    K: int,
    pad_id: int,
    prefix_len: int = 0,
    keep_prob: float = 1.0,
    rng: Optional[random.Random] = None,
) -> tuple[list[int], list[list[int]], list[bool], int, int]:
    """Pad-substitution alignment for jyutping / pinyin.

    For each CJK char in ``text``, with probability ``keep_prob``, append a
    pad token after that char's last BPE position; write the syllable's
    ``[initial, final-with-tone]`` ids into slots 0..1 of the pad's K-block.

    Returns (new_text_ids, new_slots, new_mask, n_kept, n_seen). Lengths of
    the first three are equal: prefix BPEs + text BPEs + N inserted pads.
    """
    rng_ = rng or random
    queue = [p for p in phonemes if p and p not in NON_PHONEME_TOKENS]
    qi = 0
    n_kept = 0
    n_seen = 0
    inserts: list[tuple[int, list[list[int]]]] = []
    for ci, ch in enumerate(text):
        if qi >= len(queue):
            break
        if not _is_cjk(ch):
            continue
        syl = queue[qi]
        if not syl[-1].isdigit():
            qi += 1
            continue
        n_seen += 1
        qi += 1
        if rng_.random() >= keep_prob:
            continue
        ini, fin = tokenizer_obj.split_whole_syllable(alphabet, syl)
        try:
            ids = tokenizer_obj.encode_span(alphabet, [ini, fin])
        except ValueError:
            continue
        # char_pos is measured in the prefix+text string for offsets lookup
        inserts.append((ci + prefix_len, [list(ids)]))
        n_kept += 1
    new_text_ids, new_slots, new_mask = _splice_pads(
        text_ids, text_offsets, inserts, pad_id, K
    )
    return new_text_ids, new_slots, new_mask, n_kept, n_seen


def _align_padsub_english(
    text: str,
    text_offsets: list[tuple[int, int]],
    text_ids: list[int],
    phonemes: list[str],
    tokenizer_obj: PhonemeTokenizer,
    K: int,
    pad_id: int,
    prefix_len: int = 0,
    keep_prob: float = 1.0,
    rng: Optional[random.Random] = None,
) -> tuple[list[int], list[list[int]], list[bool], int, int]:
    """Pad-substitution alignment for English (ARPAbet).

    For each word in ``text``, with probability ``keep_prob``, syllabify
    the word's ARPAbet via ``encode_arpabet_per_syllable`` and append ONE
    pad token per syllable after the word's last BPE. Each pad's K-block
    holds the phone ids for its syllable (zero-padded out to K).

    Returns (new_text_ids, new_slots, new_mask, n_kept, n_seen).
    """
    from soulxpodcast.inpaint.tokenizer import encode_arpabet_per_syllable

    rng_ = rng or random

    pwords: list[list[str]] = [[]]
    for p in phonemes:
        if p == "|":
            pwords.append([])
        else:
            pwords[-1].append(p)
    pwords = [w for w in pwords if w]

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

    inserts: list[tuple[int, list[list[int]]]] = []
    n_kept = 0
    n_seen = 0
    for word_idx, (cs, ce) in enumerate(text_words):
        if word_idx >= len(pwords):
            break
        n_seen += 1
        if rng_.random() >= keep_prob:
            continue
        syll_groups = encode_arpabet_per_syllable(pwords[word_idx])
        if not syll_groups:
            continue
        # Insert one pad per syllable after the last char of the word
        # (which routes to the word's last BPE via _splice_pads).
        last_char = ce - 1 + prefix_len
        inserts.append((last_char, [list(g) for g in syll_groups]))
        n_kept += 1
    new_text_ids, new_slots, new_mask = _splice_pads(
        text_ids, text_offsets, inserts, pad_id, K
    )
    return new_text_ids, new_slots, new_mask, n_kept, n_seen


# --------------------------------------------------------------------- #
# v12 — grapheme substitution
# --------------------------------------------------------------------- #
#
# v11 (above) INSERTED a pad after the covering BPE and kept the grapheme,
# so at an annotated position the trunk saw both the natural reading (the
# grapheme) and the composer's phoneme. The phoneme was therefore redundant
# and the composer collapsed onto a per-alphabet "marker" direction (the LM
# read the answer off the surviving grapheme). v12 SUBSTITUTES: it removes
# the unit's grapheme BPE(s) and replaces them with the phoneme pad(s), so
# the composed embedding is the only signal for that position. The shared
# splicer below is used by BOTH training and inference so the
# train/inference contract (guarded by test_alignment_parity.py) holds.


def _bpe_coherence_groups(
    offsets: list[tuple[int, int]],
) -> list[list[int]]:
    """Group char positions that share a BPE token into excisable units.

    Two char positions belong to the same group iff some single BPE token
    covers both (a merged BPE like ``银行`` → chars {2,3}) OR a char is
    covered by several byte-fallback BPEs that cover no other char (``哋``
    → 2 BPEs, both over char 8 → group {8}). A group is the smallest set of
    chars that cannot be split without slicing a BPE, i.e. the smallest unit
    whose grapheme can be cleanly removed from ``input_ids``.

    Returns a list of char-position lists, in ascending char order. Zero-width
    offsets (special tokens with empty char spans) are ignored.
    """
    char_bpes: dict[int, set[int]] = {}
    max_c = 0
    for bi, (s, e) in enumerate(offsets):
        for c in range(s, e):
            char_bpes.setdefault(c, set()).add(bi)
        max_c = max(max_c, e)

    groups: list[list[int]] = []
    cur: list[int] = []
    cur_bpes: set[int] = set()
    for c in range(max_c):
        cb = char_bpes.get(c)
        if cb is None:
            # Gap with no covering BPE — flush the running group.
            if cur:
                groups.append(cur)
                cur, cur_bpes = [], set()
            continue
        if cur and (cb & cur_bpes):
            cur.append(c)
            cur_bpes |= cb
        else:
            if cur:
                groups.append(cur)
            cur, cur_bpes = [c], set(cb)
    if cur:
        groups.append(cur)
    return groups


def _substitute_units(
    text_ids: list[int],
    offsets: list[tuple[int, int]],
    units: list[tuple[int, int, list[list[int]]]],
    pad_id: int,
    K: int,
    on_fallback=None,
) -> tuple[list[int], list[list[int]], list[bool]]:
    """Replace each unit's covering BPE tokens with its phoneme pad(s).

    Shared by training (:func:`_align_grapheme_subst_chinese` /
    ``_english``) and inference
    (``InpaintInferenceEngine._build_text_and_phone_tokens``) so both build
    byte-identical ``input_ids`` — this is the contract guarded by
    ``scripts/inpaint/test_alignment_parity.py``.

    Args:
        text_ids: BPE ids of the natural (unmarked) prefix+text.
        offsets:  per-BPE ``(char_start, char_end)``, parallel to ``text_ids``.
        units:    list of ``(char_start, char_end, blocks)``. ``blocks`` is a
                  list of phoneme-id lists (one pad per block) to splice in
                  place of the unit's graphemes. Char ranges are in the same
                  (prefix+text) coordinate space as ``offsets`` and must be
                  non-overlapping.
        pad_id:   tokenizer pad id (what the LM sees at composed positions if
                  the composer is off — e.g. the text-only eval baseline).
        K:        storage K.

    Substitution vs. insertion fallback: a unit's covering BPEs are removed
    (true substitution) only when every char those BPEs cover lies inside the
    substituted char set — i.e. removing them cannot delete a neighbouring,
    non-substituted grapheme. If a covering BPE bleeds onto an unsubstituted
    char (the English ``" B"`` leading-space case), that unit falls back to
    INSERTION (grapheme kept, pads appended after its last covering BPE) so we
    never corrupt a neighbour. For zh/yue every BPE covers whole CJK chars, so
    substitution is always clean there.

    Returns ``(new_text_ids, new_slots, new_mask)`` — equal length; pad
    positions carry the unit's block in ``new_slots`` and ``True`` in
    ``new_mask``.
    """
    covered_chars: set[int] = set()
    for cs, ce, _ in units:
        covered_chars.update(range(cs, ce))

    def _covering_bpes(cs: int, ce: int) -> list[int]:
        return [bi for bi, (bcs, bce) in enumerate(offsets)
                if not (bce <= cs or bcs >= ce) and bce > bcs]

    emit_before: dict[int, list[list[int]]] = {}  # at first covering BPE
    emit_after: dict[int, list[list[int]]] = {}    # insertion fallback
    remove: set[int] = set()
    for cs, ce, blocks in units:
        cov = _covering_bpes(cs, ce)
        if not cov:
            continue
        # Clean substitution only if no covering BPE bleeds onto a char that
        # is not part of any substituted unit.
        clean = all(
            all(c in covered_chars for c in range(offsets[bi][0], offsets[bi][1]))
            for bi in cov
        )
        if clean:
            emit_before.setdefault(cov[0], []).extend(blocks)
            remove.update(cov)
        else:
            # A covering BPE bleeds onto an un-substituted char (e.g. the
            # English " B" leading-space token, or a span covering only PART
            # of a merged CJK BPE). Keep the grapheme and INSERT the pads
            # instead — never corrupt a neighbour. This is a train/inference
            # divergence for zh/yue (training only ever substitutes whole
            # BPE-coherence groups), so let callers warn.
            emit_after.setdefault(cov[-1], []).extend(blocks)
            if on_fallback is not None:
                on_fallback((cs, ce, blocks))

    new_text_ids: list[int] = []
    new_slots: list[list[int]] = []
    new_mask: list[bool] = []

    def _push_pad(block: list[int]) -> None:
        new_text_ids.append(pad_id)
        new_slots.append(_block(block, K))
        new_mask.append(True)

    for bi, tid in enumerate(text_ids):
        for block in emit_before.get(bi, []):
            _push_pad(block)
        if bi not in remove:
            new_text_ids.append(tid)
            new_slots.append([0] * K)
            new_mask.append(False)
        for block in emit_after.get(bi, []):
            _push_pad(block)
    return new_text_ids, new_slots, new_mask


def _align_grapheme_subst_chinese(
    text: str,
    text_offsets: list[tuple[int, int]],
    text_ids: list[int],
    phonemes: list[str],
    alphabet: str,
    tokenizer_obj: PhonemeTokenizer,
    K: int,
    pad_id: int,
    prefix_len: int = 0,
    keep_prob: float = 1.0,
    rng: Optional[random.Random] = None,
) -> tuple[list[int], list[list[int]], list[bool], int, int]:
    """Grapheme-substitution alignment for jyutping / pinyin (v12).

    Per CJK char, pull its ``[initial, final-with-tone]`` syllable from the
    phoneme queue (skipping non-syllable placeholders, exactly as the v11
    path does — keeps the queue↔char alignment the fixtures rely on). Then,
    with **BPE-group-coherent** dropout (one keep/drop decision per
    :func:`_bpe_coherence_groups` group, so merged BPEs like ``银行`` are
    kept/dropped whole), SUBSTITUTE each kept group's grapheme BPE(s) with one
    phoneme pad per CJK char in the group.

    Returns ``(new_text_ids, new_slots, new_mask, n_kept, n_seen)`` where
    ``n_seen`` counts CJK chars with a valid syllable and ``n_kept`` counts
    chars actually substituted.
    """
    rng_ = rng or random
    queue = [p for p in phonemes if p and p not in NON_PHONEME_TOKENS]

    # Map each CJK char (in prefix+text coords) to its syllable, advancing the
    # queue per CJK char so placeholders ('X') stay aligned.
    char_syl: dict[int, str] = {}
    qi = 0
    n_seen = 0
    for ci, ch in enumerate(text):
        if qi >= len(queue):
            break
        if not _is_cjk(ch):
            continue
        syl = queue[qi]
        qi += 1
        if not syl[-1].isdigit():
            continue
        char_syl[ci + prefix_len] = syl
        n_seen += 1

    def _is_cjk_at(p: int) -> bool:
        return p >= prefix_len and _is_cjk(text[p - prefix_len])

    units: list[tuple[int, int, list[list[int]]]] = []
    n_kept = 0
    for group in _bpe_coherence_groups(text_offsets):
        cjk_chars = [p for p in group if _is_cjk_at(p)]
        if not cjk_chars:
            continue
        # Substitute only when EVERY CJK char in the group has a valid
        # syllable — never partially mask a merged BPE.
        if any(p not in char_syl for p in cjk_chars):
            continue
        if rng_.random() >= keep_prob:
            continue
        blocks: list[list[int]] = []
        ok = True
        for p in cjk_chars:
            ini, fin = tokenizer_obj.split_whole_syllable(alphabet, char_syl[p])
            try:
                ids = tokenizer_obj.encode_span(alphabet, [ini, fin])
            except ValueError:
                ok = False
                break
            blocks.append(list(ids))
        if not ok or not blocks:
            continue
        units.append((cjk_chars[0], cjk_chars[-1] + 1, blocks))
        n_kept += len(cjk_chars)

    new_text_ids, new_slots, new_mask = _substitute_units(
        text_ids, text_offsets, units, pad_id, K
    )
    return new_text_ids, new_slots, new_mask, n_kept, n_seen


def _align_grapheme_subst_english(
    text: str,
    text_offsets: list[tuple[int, int]],
    text_ids: list[int],
    phonemes: list[str],
    tokenizer_obj: PhonemeTokenizer,
    K: int,
    pad_id: int,
    prefix_len: int = 0,
    keep_prob: float = 1.0,
    rng: Optional[random.Random] = None,
) -> tuple[list[int], list[list[int]], list[bool], int, int]:
    """Grapheme-substitution alignment for English (ARPAbet, v12).

    Per word (kept with prob ``keep_prob``), syllabify its ARPAbet and
    SUBSTITUTE the word's grapheme BPE(s) with one pad per syllable. A word
    whose covering BPE bleeds onto an adjacent space (e.g. the ``" B"`` token)
    falls back to insertion inside :func:`_substitute_units` — English is not
    parity- or behavioral-gated (the SoulX en LLM baseline loops), so this is
    best-effort and kept structurally consistent with the zh/yue path.
    """
    from soulxpodcast.inpaint.tokenizer import encode_arpabet_per_syllable

    rng_ = rng or random

    pwords: list[list[str]] = [[]]
    for p in phonemes:
        if p == "|":
            pwords.append([])
        else:
            pwords[-1].append(p)
    pwords = [w for w in pwords if w]

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

    units: list[tuple[int, int, list[list[int]]]] = []
    n_kept = 0
    n_seen = 0
    for word_idx, (cs, ce) in enumerate(text_words):
        if word_idx >= len(pwords):
            break
        n_seen += 1
        if rng_.random() >= keep_prob:
            continue
        syll_groups = encode_arpabet_per_syllable(pwords[word_idx])
        if not syll_groups:
            continue
        units.append(
            (cs + prefix_len, ce + prefix_len, [list(g) for g in syll_groups])
        )
        n_kept += 1

    new_text_ids, new_slots, new_mask = _substitute_units(
        text_ids, text_offsets, units, pad_id, K
    )
    return new_text_ids, new_slots, new_mask, n_kept, n_seen


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
        from soulxpodcast.inpaint.composer import ALPHABET_STR_TO_ID

        row = self.rows[idx]
        lang = row["lang"]
        text = normalise_text(row["text"], lang)
        phonemes: list[str] = row["phonemes"]
        speech_raw: list[int] = row["speech_tokens"]

        if not (self.cfg.min_speech_tokens <= len(speech_raw) <= self.cfg.max_speech_tokens):
            return None

        # Tokenize natural text+prefix once. The alignment functions splice
        # pad tokens into this base sequence at syllable-override points.
        prefix = DIALECT_PREFIX.get(lang, "")
        text_with_prefix = prefix + text
        enc = self.tokenizer(
            text_with_prefix, add_special_tokens=False, return_offsets_mapping=True
        )
        base_text_ids: list[int] = enc["input_ids"]
        base_offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]

        K = self.cfg.slots_per_token
        alphabet = LANG_TO_ALPHABET[lang]
        prefix_len = len(prefix)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id for padsub")

        # Per-call RNG so DataLoader workers see different draws each step.
        # deterministic_dropout=True uses the row index as the seed for
        # reproducible eval.
        if self.cfg.deterministic_dropout:
            rng = random.Random(idx)
        else:
            rng = random
        keep_prob = self.cfg.phoneme_keep_prob

        align_english = (
            _align_grapheme_subst_english if self.cfg.grapheme_subst
            else _align_padsub_english
        )
        align_chinese = (
            _align_grapheme_subst_chinese if self.cfg.grapheme_subst
            else _align_padsub_chinese
        )
        if alphabet == "cmu":
            text_ids, text_slot_blocks, text_phone_mask, kept, seen = align_english(
                text, base_offsets, base_text_ids, phonemes, self.phone_tok, K,
                pad_id=pad_id, prefix_len=prefix_len,
                keep_prob=keep_prob, rng=rng,
            )
        else:
            text_ids, text_slot_blocks, text_phone_mask, kept, seen = align_chinese(
                text, base_offsets, base_text_ids, phonemes, alphabet, self.phone_tok, K,
                pad_id=pad_id, prefix_len=prefix_len,
                keep_prob=keep_prob, rng=rng,
            )

        # Drop rows that have no phoneme info at all. Rows where dropout
        # produced kept==0 are kept (LLM also trains on text-only inputs
        # so it has to fall back gracefully when the composer doesn't fire).
        if seen == 0:
            return None

        # Speech tokens with offset.
        speech_ids = [t + self.cfg.speech_token_offset for t in speech_raw]

        # Assemble the full sequence: task prefix + (text BPEs + pads) + bridge + speech + EOS.
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
        phone_mask = torch.zeros(T, dtype=torch.bool)
        for local_ti, block in enumerate(text_slot_blocks):
            gti = text_token_global_start + local_ti
            phone_token[gti * K : gti * K + K] = torch.tensor(block, dtype=torch.long)
        for local_ti, m in enumerate(text_phone_mask):
            gti = text_token_global_start + local_ti
            phone_mask[gti] = m

        speech_mask = torch.zeros(T, dtype=torch.long)
        speech_mask[speech_global_start : speech_global_end + 1] = 1  # include EOS

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(T, dtype=torch.long),
            "speech_mask": speech_mask,
            "phone_token": phone_token,
            "phone_mask": phone_mask,
            "alphabet_id": torch.tensor(ALPHABET_STR_TO_ID[alphabet], dtype=torch.long),
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
    alphabet_id = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        T = b["input_ids"].numel()
        input_ids[i, :T] = b["input_ids"]
        attention_mask[i, :T] = b["attention_mask"]
        speech_mask[i, :T] = b["speech_mask"]
        phone_token[i, : K * T] = b["phone_token"]
        phone_mask[i, :T] = b["phone_mask"]
        alphabet_id[i] = b["alphabet_id"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "speech_mask": speech_mask,
        "phone_token": phone_token,
        "phone_mask": phone_mask,
        "alphabet_id": alphabet_id,
        "n_phonemes_kept": sum(b["n_phonemes_kept"] for b in batch),
        "n_phonemes_seen": sum(b["n_phonemes_seen"] for b in batch),
    }
