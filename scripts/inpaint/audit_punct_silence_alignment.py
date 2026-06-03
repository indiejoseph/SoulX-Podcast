"""Test the 'punct anchors the silence prior' hypothesis.

Mechanism under test: in yue (which has punctuation in training text),
the LLM learned to channel its silence-token emission to text positions
near punctuation. In zh (no punct), the silence prior has no anchor and
spreads diffusely across content positions — driving mode-collapse when
the composer injects there.

This audit:
  1. Samples N rows per language from tmp/dataset.jsonl.
  2. For each row:
       - Identifies punct character positions in the normalised text.
       - Identifies silence-token positions in speech_tokens (using
         the data-derived per-language ``SILENCE_TOKEN_IDS`` set).
       - Builds an approximate text-char ↔ speech-token alignment via
         proportional scaling (no explicit alignment is stored in the
         training data — assumes roughly uniform speech rate within a
         row, which is good enough at the row scale).
  3. For yue rows with ≥1 punct char: bins each silence-token position
     by its (normalised) distance to the nearest punct character.
  4. Reports silence-token concentration in each bin, plus a uniform-
     distribution null hypothesis to compare against.
  5. For zh / en: reports the silence position histogram (no punct
     anchor, so we expect roughly uniform).

If the hypothesis is correct we expect:
  * yue: silence-token density much higher in the bin closest to punct
  * zh: silence-token density flat across the sequence
  * en: TBD — memory says training text is no-punct, so should look
    like zh; if it does NOT (and en works), the hypothesis is
    incomplete.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.training.inpaint_dataset import (
    SILENCE_TOKEN_IDS, normalise_text,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("punct_audit")


# CJK + halfwidth punctuation that ought to signal pause/silence.
CJK_PUNCT = set("。，！？、；：…—「」『』《》（）")
ASCII_PUNCT = set(".,!?;:")
PAUSE_PUNCT = CJK_PUNCT | ASCII_PUNCT


def find_punct_char_positions(text: str) -> list[int]:
    """Return character indices (in the normalised text) of pause-punct."""
    return [i for i, ch in enumerate(text) if ch in PAUSE_PUNCT]


def map_text_pos_to_speech_pos(text_char_pos: int, text_len: int,
                               speech_len: int) -> int:
    """Proportional mapping. Row-level approximation."""
    if text_len <= 0:
        return 0
    frac = (text_char_pos + 0.5) / text_len
    return int(round(frac * speech_len))


def collect_rows(jsonl_path: str, lang: str, n: int,
                 min_text_len: int = 6,
                 min_speech_len: int = 30,
                 seed: int = 42) -> list[dict]:
    """Reservoir-style sampling so we don't load the whole file."""
    rng = random.Random(seed)
    out: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r["lang"] != lang:
                continue
            text = normalise_text(r["text"], lang)
            if len(text) < min_text_len:
                continue
            if len(r["speech_tokens"]) < min_speech_len:
                continue
            if len(out) < n:
                out.append({"text": text, "speech_tokens": r["speech_tokens"],
                           "id": r.get("id", "?"), "lang": lang})
            else:
                # Reservoir replacement
                j = rng.randint(0, len(out) - 1)
                if rng.random() < n / (n + 1):  # mild bias toward early
                    pass
                else:
                    out[j] = {"text": text, "speech_tokens": r["speech_tokens"],
                             "id": r.get("id", "?"), "lang": lang}
    return out


def _silence_alignment(rows: list[dict], lang: str,
                       anchor_predicate,
                       n_distance_bins: int = 10) -> dict:
    """Generic: anchor positions are wherever ``anchor_predicate(ch)`` is True.

    Same proportional alignment + binning as the yue-specific function;
    factored out so we can test space-as-anchor in en, char-as-anchor in
    any lang, etc.
    """
    silence_set = SILENCE_TOKEN_IDS.get(lang, frozenset())
    bin_silence = [0] * n_distance_bins
    bin_total = [0] * n_distance_bins
    rows_with_anchor = 0
    silence_at_anchor_pos = 0
    n_total_silence = 0

    for r in rows:
        text = r["text"]
        speech = r["speech_tokens"]
        T = len(text)
        S = len(speech)
        if T == 0 or S == 0:
            continue
        anchor_chars = [i for i, ch in enumerate(text) if anchor_predicate(ch)]
        if not anchor_chars:
            continue
        rows_with_anchor += 1
        anchor_speech_pos = [
            map_text_pos_to_speech_pos(p, T, S) for p in anchor_chars
        ]
        for s_idx, tok in enumerate(speech):
            nearest = min(abs(s_idx - p) for p in anchor_speech_pos)
            d_norm = nearest / S
            bin_idx = min(int(d_norm / 0.5 * n_distance_bins), n_distance_bins - 1)
            bin_total[bin_idx] += 1
            if tok in silence_set:
                bin_silence[bin_idx] += 1
                n_total_silence += 1
                if nearest <= 2:
                    silence_at_anchor_pos += 1
    bin_rates = [
        (bin_silence[i] / bin_total[i] if bin_total[i] > 0 else 0.0)
        for i in range(n_distance_bins)
    ]
    return {
        "lang": lang,
        "rows_with_anchor": rows_with_anchor,
        "bin_silence": bin_silence,
        "bin_total": bin_total,
        "bin_rates": bin_rates,
        "n_total_silence": n_total_silence,
        "silence_at_anchor_pos": silence_at_anchor_pos,
        "frac_silence_at_anchor": (
            silence_at_anchor_pos / n_total_silence if n_total_silence else 0.0
        ),
    }


def punct_silence_alignment_yue(rows: list[dict],
                                n_distance_bins: int = 10) -> dict:
    """For yue rows with punct: how does silence-token density vary with
    distance to nearest punct text-position (after proportional alignment)?

    Distance is normalised by speech_len so bins are language-/row-
    independent. Bin 0 = closest 10% of positions, bin 9 = farthest.
    """
    silence_set = SILENCE_TOKEN_IDS.get("yue", frozenset())
    bin_silence = [0] * n_distance_bins
    bin_total = [0] * n_distance_bins
    rows_with_punct = 0
    silence_at_punct_pos = 0
    n_total_silence = 0

    for r in rows:
        text = r["text"]
        speech = r["speech_tokens"]
        T = len(text)
        S = len(speech)
        if T == 0 or S == 0:
            continue
        punct_chars = find_punct_char_positions(text)
        if not punct_chars:
            continue
        rows_with_punct += 1

        # Project punct char positions into speech-token positions.
        punct_speech_pos = [
            map_text_pos_to_speech_pos(p, T, S) for p in punct_chars
        ]
        # For each speech position s, find min distance to any punct speech
        # position (in *normalised* speech units, 0..1).
        for s_idx, tok in enumerate(speech):
            nearest = min(abs(s_idx - p) for p in punct_speech_pos)
            d_norm = nearest / S
            # Bin: equal-width over [0, 0.5]. Distances >0.5 are rare and
            # lumped into the last bin.
            bin_idx = min(int(d_norm / 0.5 * n_distance_bins), n_distance_bins - 1)
            bin_total[bin_idx] += 1
            if tok in silence_set:
                bin_silence[bin_idx] += 1
                n_total_silence += 1
                # "At punct" = within ±2 speech positions of a punct pos
                if nearest <= 2:
                    silence_at_punct_pos += 1

    bin_rates = [
        (bin_silence[i] / bin_total[i] if bin_total[i] > 0 else 0.0)
        for i in range(n_distance_bins)
    ]
    return {
        "lang": "yue",
        "rows_with_punct": rows_with_punct,
        "bin_silence": bin_silence,
        "bin_total": bin_total,
        "bin_rates": bin_rates,
        "n_total_silence": n_total_silence,
        "silence_at_punct_pos": silence_at_punct_pos,
        "frac_silence_at_punct": (
            silence_at_punct_pos / n_total_silence if n_total_silence else 0.0
        ),
    }


def silence_position_histogram(rows: list[dict], lang: str,
                               n_bins: int = 10) -> dict:
    """For each row, bin silence-token positions by normalised position
    in the speech sequence. If silence is uniformly distributed across
    the sequence, every bin should have ~equal rate.
    """
    silence_set = SILENCE_TOKEN_IDS.get(lang, frozenset())
    bin_silence = [0] * n_bins
    bin_total = [0] * n_bins
    n_punct_rows = 0
    for r in rows:
        if find_punct_char_positions(r["text"]):
            n_punct_rows += 1
        S = len(r["speech_tokens"])
        if S == 0:
            continue
        for s_idx, tok in enumerate(r["speech_tokens"]):
            bin_idx = min(int(s_idx / S * n_bins), n_bins - 1)
            bin_total[bin_idx] += 1
            if tok in silence_set:
                bin_silence[bin_idx] += 1
    bin_rates = [
        (bin_silence[i] / bin_total[i] if bin_total[i] > 0 else 0.0)
        for i in range(n_bins)
    ]
    return {
        "lang": lang,
        "n_rows": len(rows),
        "n_punct_rows": n_punct_rows,
        "bin_silence": bin_silence,
        "bin_total": bin_total,
        "bin_rates": bin_rates,
    }


def text_punct_summary(rows: list[dict], lang: str) -> dict:
    """How prevalent IS punctuation in this language's training text?"""
    n_rows = len(rows)
    n_punct_rows = 0
    total_chars = 0
    total_punct = 0
    for r in rows:
        punct = find_punct_char_positions(r["text"])
        total_chars += len(r["text"])
        total_punct += len(punct)
        if punct:
            n_punct_rows += 1
    return {
        "lang": lang,
        "n_rows": n_rows,
        "n_punct_rows": n_punct_rows,
        "frac_rows_with_punct": n_punct_rows / max(1, n_rows),
        "total_chars": total_chars,
        "total_punct": total_punct,
        "punct_per_char": total_punct / max(1, total_chars),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="tmp/dataset.jsonl")
    p.add_argument("--n_per_lang", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows_yue = collect_rows(args.jsonl, "yue", args.n_per_lang, seed=args.seed)
    rows_zh = collect_rows(args.jsonl, "zh", args.n_per_lang, seed=args.seed)
    rows_en = collect_rows(args.jsonl, "en", args.n_per_lang, seed=args.seed)
    log.info(f"sampled yue={len(rows_yue)} zh={len(rows_zh)} en={len(rows_en)}")

    # === Punctuation prevalence in training text ===
    log.info("\n=== TRAINING TEXT PUNCTUATION PREVALENCE ===")
    log.info(f"  {'lang':<5s} {'rows':>6s} {'punct rows':>11s} {'%rows':>7s} "
             f"{'chars':>10s} {'punct chars':>12s} {'punct/char':>11s}")
    for rows, lang in [(rows_yue, "yue"), (rows_zh, "zh"), (rows_en, "en")]:
        s = text_punct_summary(rows, lang)
        log.info(f"  {lang:<5s} {s['n_rows']:>6d} "
                 f"{s['n_punct_rows']:>11d} {s['frac_rows_with_punct']:>6.1%}  "
                 f"{s['total_chars']:>10,d} {s['total_punct']:>12,d} "
                 f"{s['punct_per_char']:>10.2%}")

    # === yue: silence concentration vs distance to punct ===
    log.info("\n=== yue: SILENCE-TOKEN DENSITY vs DISTANCE-TO-PUNCT ===")
    log.info("(bin 0 = closest to punct, bin 9 = farthest)")
    log.info("If hypothesis correct: bin 0 silence rate >> uniform rate.")
    a = punct_silence_alignment_yue(rows_yue)
    log.info(f"  yue rows with punct: {a['rows_with_punct']}/{len(rows_yue)}")
    log.info(f"  total silence tokens: {a['n_total_silence']}")
    log.info(f"  silence tokens at ±2 of a punct text-pos: "
             f"{a['silence_at_punct_pos']} "
             f"({a['frac_silence_at_punct']:.1%} of all silence)")
    avg = (
        sum(a["bin_silence"]) / max(1, sum(a["bin_total"]))
        if sum(a["bin_total"]) > 0 else 0.0
    )
    log.info(f"\n  {'bin':>4s}  {'n_tokens':>9s}  {'n_silence':>10s}  "
             f"{'rate':>6s}  {'vs uniform':>11s}")
    for i in range(len(a["bin_rates"])):
        rate = a["bin_rates"][i]
        ratio = rate / avg if avg > 0 else 0.0
        log.info(f"   {i:>3d}  {a['bin_total'][i]:>9,d}  "
                 f"{a['bin_silence'][i]:>10,d}  {rate:>5.2%}  {ratio:>10.2f}x")

    # === English: space-as-anchor test ===
    log.info("\n=== en: SILENCE-TOKEN DENSITY vs DISTANCE-TO-SPACE ===")
    log.info("English training text has no punctuation BUT does retain word-boundary")
    log.info("spaces (normalise_text only strips spaces for zh/yue). Test: do silence")
    log.info("tokens cluster near space characters the way yue silence clusters near punct?")
    en_space = _silence_alignment(rows_en, "en", lambda ch: ch == " ")
    log.info(f"  en rows with ≥1 space: {en_space['rows_with_anchor']}/{len(rows_en)}")
    log.info(f"  total silence tokens: {en_space['n_total_silence']}")
    log.info(f"  silence tokens at ±2 of a space text-pos: "
             f"{en_space['silence_at_anchor_pos']} "
             f"({en_space['frac_silence_at_anchor']:.1%} of all silence)")
    avg_en = (
        sum(en_space["bin_silence"]) / max(1, sum(en_space["bin_total"]))
        if sum(en_space["bin_total"]) > 0 else 0.0
    )
    log.info(f"\n  {'bin':>4s}  {'n_tokens':>9s}  {'n_silence':>10s}  "
             f"{'rate':>6s}  {'vs uniform':>11s}")
    for i in range(len(en_space["bin_rates"])):
        rate = en_space["bin_rates"][i]
        ratio = rate / avg_en if avg_en > 0 else 0.0
        log.info(f"   {i:>3d}  {en_space['bin_total'][i]:>9,d}  "
                 f"{en_space['bin_silence'][i]:>10,d}  {rate:>5.2%}  {ratio:>10.2f}x")

    # === Silence position histograms (all langs) ===
    log.info("\n=== SILENCE POSITION HISTOGRAMS (10 bins, 0=start, 9=end) ===")
    log.info("If silence is anchored to specific positions (start/end pauses)")
    log.info("we expect U-shape. If diffuse, we expect flat.")
    for rows, lang in [(rows_yue, "yue"), (rows_zh, "zh"), (rows_en, "en")]:
        h = silence_position_histogram(rows, lang)
        log.info(f"\n  --- {lang} (n_rows={h['n_rows']}, "
                 f"with punct={h['n_punct_rows']}) ---")
        avg = sum(h["bin_silence"]) / max(1, sum(h["bin_total"]))
        log.info(f"  avg silence rate: {avg:.2%}")
        log.info(f"  {'bin':>4s}  {'tok':>9s}  {'sil':>8s}  {'rate':>6s}  {'vs avg':>7s}")
        for i in range(len(h["bin_rates"])):
            r = h["bin_rates"][i]
            ratio = r / avg if avg > 0 else 0.0
            log.info(f"   {i:>3d}  {h['bin_total'][i]:>9,d}  "
                     f"{h['bin_silence'][i]:>8,d}  {r:>5.2%}  {ratio:>6.2f}x")


if __name__ == "__main__":
    main()
