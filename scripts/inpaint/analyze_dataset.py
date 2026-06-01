"""Dataset distribution analysis for the position-tagged inpaint vocab.

Scans every row of ``tmp/dataset.jsonl`` and reports, for the proposed
vocab (position-tagged ARPAbet + split-form Chinese), the:

- vocab inventory + size for each alphabet
- realised usage frequency per id (which positions actually appear?)
- per-word / per-syllable / per-BPE-position slot demand
- truncation rate at various K (slots-per-text-token caps)
- pathological cases (zero-vowel "words", overflows, etc.)

Standalone — does not import the production tokenizer/composer because
the goal is to compare the *new* vocab proposal against the *existing*
dataset *before* committing the code change. Run as::

    .venv/bin/python scripts/inpaint/analyze_dataset.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

DATA_PATH = Path("tmp/dataset.jsonl")


# ===== Vocab definitions ===================================================

# CMUdict ARPAbet — vowels carry stress historically; in this dataset stress
# is stripped (the corpus uses bare symbols). So our vowel set is the 15
# canonical ARPAbet vowels.
ARPA_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY",
               "EH", "ER", "EY", "IH", "IY",
               "OW", "OY", "UH", "UW"}

# The 24 ARPAbet consonants present in CMUdict 0.7b.
ARPA_CONSONANTS = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L",
                   "M", "N", "NG", "P", "R", "S", "SH",
                   "T", "TH", "V", "W", "Y", "Z", "ZH"]
assert len(ARPA_CONSONANTS) == 24

# Word-boundary marker stays as one id.
ARPA_BOUNDARY = "|"

# Jyutping / Pinyin initials (incl. zero-initial "").
JP_INITIALS = ["gw", "kw", "ng",
               "b", "c", "d", "f", "g", "h", "j", "k", "l",
               "m", "n", "p", "s", "t", "w", "z", ""]
PY_INITIALS = ["zh", "ch", "sh",
               "b", "c", "d", "f", "g", "h", "j", "k", "l",
               "m", "n", "p", "q", "r", "s", "t", "w", "x", "y", "z", ""]


def build_new_vocab() -> tuple[dict[str, int], dict[str, int]]:
    """Return (token→id) and (per-alphabet → range-size) under the proposed scheme.

    Layout::

        pad                                            : 0
        ARPAbet vowels (nucleus only)                  : 15
        ARPAbet consonants, onset-tagged "_on"         : 24
        ARPAbet consonants, coda-tagged  "_co"         : 24
        ARPAbet boundary `|`                           : 1
        Jyutping initials (incl. zero-initial)         : 20
        Jyutping finals-with-tone                      : (loaded from data)
        Pinyin   initials (incl. zero-initial)         : 24
        Pinyin   finals-with-tone                      : (loaded from data)
    """
    vocab: dict[str, int] = {}
    sizes: dict[str, int] = {}
    nxt = 1

    sizes["cmu_vowel"] = 0
    for v in sorted(ARPA_VOWELS):
        vocab[f"cmu:{v}"] = nxt; nxt += 1; sizes["cmu_vowel"] += 1
    sizes["cmu_onset"] = 0
    for c in ARPA_CONSONANTS:
        vocab[f"cmu:{c}_on"] = nxt; nxt += 1; sizes["cmu_onset"] += 1
    sizes["cmu_coda"] = 0
    for c in ARPA_CONSONANTS:
        vocab[f"cmu:{c}_co"] = nxt; nxt += 1; sizes["cmu_coda"] += 1
    sizes["cmu_boundary"] = 1
    vocab["cmu:|"] = nxt; nxt += 1

    sizes["jp_initial"] = 0
    for ini in JP_INITIALS:
        vocab[f"jp:i:{ini}"] = nxt; nxt += 1; sizes["jp_initial"] += 1

    sizes["py_initial"] = 0
    for ini in PY_INITIALS:
        vocab[f"py:i:{ini}"] = nxt; nxt += 1; sizes["py_initial"] += 1

    # finals are populated as we scan the dataset (data-derived).
    return vocab, sizes


# ===== Syllabification (maximal-onset) =====================================

def syllabify_word(phs: list[str]) -> list[list[tuple[str, str]]]:
    """Return list of syllables; each syllable is [(phoneme, role)].

    Role is 'onset' | 'nucleus' | 'coda'.

    Rule:
    - Each vowel = its own syllable's nucleus.
    - Consonants before the first vowel = onset of first syllable.
    - Consonants between two vowels = ALL go to the NEXT syllable's onset
      (maximal-onset; conservative; gives 'L' onset role in 'hello').
    - Consonants after the last vowel = coda of the last syllable.

    Zero-vowel words (rare; abbreviations) → single "syllable" of all-coda
    consonants (caller can audit; not really linguistically meaningful).
    """
    vowel_idx = [i for i, p in enumerate(phs) if p in ARPA_VOWELS]
    if not vowel_idx:
        return [[(p, "coda") for p in phs]] if phs else []

    sylls: list[list[tuple[str, str]]] = []
    # First syll's onset = pre-first-vowel consonants
    first_v = vowel_idx[0]
    first = [(p, "onset") for p in phs[:first_v]]
    first.append((phs[first_v], "nucleus"))
    sylls.append(first)

    # Middle syllables: each subsequent vowel takes ALL prior intervening
    # consonants as its onset (maximal onset).
    for k in range(1, len(vowel_idx)):
        prev_v = vowel_idx[k - 1]
        cur_v = vowel_idx[k]
        between = phs[prev_v + 1 : cur_v]
        cur = [(p, "onset") for p in between]
        cur.append((phs[cur_v], "nucleus"))
        sylls.append(cur)

    # Trailing consonants after the last vowel = coda of the last syllable
    last_v = vowel_idx[-1]
    for p in phs[last_v + 1 :]:
        sylls[-1].append((p, "coda"))

    return sylls


def tag_arpabet_stream(phs: list[str]) -> list[str]:
    """Take a flat ARPAbet stream (incl. '|' markers) and emit position-tagged tokens.

    Vowels → bare symbol (always nucleus role).
    Consonants → '<SYM>_on' or '<SYM>_co' depending on syllable role.
    '|' is preserved as-is.
    """
    out: list[str] = []
    words: list[list[str]] = [[]]
    for p in phs:
        if p == ARPA_BOUNDARY:
            words.append([])
        else:
            words[-1].append(p)
    for wi, w in enumerate(words):
        if wi > 0:
            out.append(ARPA_BOUNDARY)
        if not w:
            continue
        sylls = syllabify_word(w)
        for syll in sylls:
            for ph, role in syll:
                if role == "nucleus":
                    out.append(ph)
                else:
                    suffix = "_on" if role == "onset" else "_co"
                    out.append(f"{ph}{suffix}")
    return out


# ===== Chinese split-form (already shipped) =================================

def split_initial(syl: str, initials: list[str]) -> tuple[str, str]:
    for ini in initials:
        if ini == "":
            continue
        if syl.startswith(ini):
            return ini, syl[len(ini) :]
    return "", syl


# ===== Main scan ===========================================================

def analyze() -> None:
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    vocab, sizes = build_new_vocab()
    lang_counts: Counter[str] = Counter()

    # English stats
    en_token_usage: Counter[str] = Counter()    # which position-tagged tokens fire
    en_word_phoneme_len: Counter[int] = Counter()
    en_syllables_per_word: Counter[int] = Counter()
    en_phonemes_per_syllable: Counter[int] = Counter()
    en_zero_vowel_words = 0
    en_consonant_role_split: dict[str, Counter[str]] = {c: Counter() for c in ARPA_CONSONANTS}
    en_consonant_only_onset: set[str] = set()
    en_consonant_only_coda: set[str] = set()
    en_rows = 0

    # Chinese stats
    yue_init_used: Counter[str] = Counter()
    yue_final_used: Counter[str] = Counter()
    zh_init_used: Counter[str] = Counter()
    zh_final_used: Counter[str] = Counter()
    yue_syllables_per_row = Counter()
    zh_syllables_per_row = Counter()

    # Generic
    n_rows = 0
    t0 = time.perf_counter()

    with DATA_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            n_rows += 1
            lang_counts[r["lang"]] += 1
            phs = r["phonemes"]

            if r["lang"] == "en":
                en_rows += 1
                # per-word counts
                cur: list[str] = []
                for p in phs:
                    if p == ARPA_BOUNDARY:
                        if cur:
                            en_word_phoneme_len[len(cur)] += 1
                            sylls = syllabify_word(cur)
                            n_v = sum(1 for ph in cur if ph in ARPA_VOWELS)
                            if n_v == 0:
                                en_zero_vowel_words += 1
                            en_syllables_per_word[len(sylls)] += 1
                            for s in sylls:
                                en_phonemes_per_syllable[len(s)] += 1
                                for ph, role in s:
                                    if ph in ARPA_VOWELS:
                                        en_token_usage[ph] += 1
                                    else:
                                        suf = "_on" if role == "onset" else "_co"
                                        en_token_usage[f"{ph}{suf}"] += 1
                                        en_consonant_role_split[ph][role] += 1
                        cur = []
                    else:
                        cur.append(p)
                if cur:
                    en_word_phoneme_len[len(cur)] += 1
                    sylls = syllabify_word(cur)
                    n_v = sum(1 for ph in cur if ph in ARPA_VOWELS)
                    if n_v == 0:
                        en_zero_vowel_words += 1
                    en_syllables_per_word[len(sylls)] += 1
                    for s in sylls:
                        en_phonemes_per_syllable[len(s)] += 1
                        for ph, role in s:
                            if ph in ARPA_VOWELS:
                                en_token_usage[ph] += 1
                            else:
                                suf = "_on" if role == "onset" else "_co"
                                en_token_usage[f"{ph}{suf}"] += 1
                                en_consonant_role_split[ph][role] += 1

            elif r["lang"] == "yue":
                used = 0
                for p in phs:
                    if not p or not p[-1].isdigit():
                        continue
                    used += 1
                    ini, fin = split_initial(p, JP_INITIALS)
                    yue_init_used[ini] += 1
                    yue_final_used[fin] += 1
                yue_syllables_per_row[used] += 1

            elif r["lang"] == "zh":
                used = 0
                for p in phs:
                    if not p or not p[-1].isdigit():
                        continue
                    used += 1
                    ini, fin = split_initial(p, PY_INITIALS)
                    zh_init_used[ini] += 1
                    zh_final_used[fin] += 1
                zh_syllables_per_row[used] += 1

    dt = time.perf_counter() - t0
    print(f"=== Dataset scan complete ===")
    print(f"  rows: {n_rows:,}")
    print(f"  langs: {dict(lang_counts)}")
    print(f"  scan time: {dt:.1f}s ({n_rows/dt:.0f} rows/s)")
    print()

    # ----- Proposed vocab summary -----
    py_finals_seen = set(zh_final_used) - {""}
    yp_finals_seen = set(yue_final_used) - {""}
    total = (
        1
        + sizes["cmu_vowel"]
        + sizes["cmu_onset"]
        + sizes["cmu_coda"]
        + sizes["cmu_boundary"]
        + sizes["jp_initial"]
        + len(yp_finals_seen)
        + sizes["py_initial"]
        + len(py_finals_seen)
    )
    print("=== Proposed vocab (position-tagged ARPAbet + split-form Chinese) ===")
    print(f"  pad                              : 1   ids [0..0]")
    cursor = 1
    for label, n in [
        ("cmu vowels (nucleus)",       sizes["cmu_vowel"]),
        ("cmu consonants onset (_on)", sizes["cmu_onset"]),
        ("cmu consonants coda  (_co)", sizes["cmu_coda"]),
        ("cmu boundary `|`",           sizes["cmu_boundary"]),
        ("jp initials (incl. zero)",   sizes["jp_initial"]),
        ("jp finals-with-tone",        len(yp_finals_seen)),
        ("py initials (incl. zero)",   sizes["py_initial"]),
        ("py finals-with-tone",        len(py_finals_seen)),
    ]:
        print(f"  {label:32s} : {n:3d}   ids [{cursor}..{cursor + n - 1}]")
        cursor += n
    print(f"  TOTAL_VOCAB_SIZE                 : {total}")
    print()

    # ----- English: tag distribution -----
    print("=== English (CMUdict, position-tagged) ===")
    print(f"  rows={en_rows:,}")
    print(f"  zero-vowel words: {en_zero_vowel_words:,}  ({100*en_zero_vowel_words/max(1,sum(en_word_phoneme_len.values())):.3f}%)")
    print(f"  total position-tagged tokens emitted: {sum(en_token_usage.values()):,}")
    print(f"  unique position-tagged tokens used:   {len(en_token_usage)}/{sizes['cmu_vowel']+sizes['cmu_onset']+sizes['cmu_coda']}")
    print()

    print("  per-consonant onset/coda split:")
    rows = []
    for c in ARPA_CONSONANTS:
        ons = en_consonant_role_split[c].get("onset", 0)
        cod = en_consonant_role_split[c].get("coda", 0)
        tot = ons + cod
        if tot == 0:
            continue
        on_frac = 100 * ons / tot
        rows.append((c, ons, cod, on_frac))
    rows.sort(key=lambda x: -x[1] - x[2])
    print(f"    {'sym':>4}   {'onset':>9}  {'coda':>9}   onset%")
    for c, ons, cod, on_frac in rows:
        print(f"    {c:>4}   {ons:>9,}  {cod:>9,}   {on_frac:5.1f}%")

    # Consonants that never appear in one role
    only_onset = [c for c in ARPA_CONSONANTS if en_consonant_role_split[c].get("coda", 0) == 0 and en_consonant_role_split[c].get("onset", 0) > 0]
    only_coda = [c for c in ARPA_CONSONANTS if en_consonant_role_split[c].get("onset", 0) == 0 and en_consonant_role_split[c].get("coda", 0) > 0]
    unused = [c for c in ARPA_CONSONANTS if en_consonant_role_split[c].get("onset", 0) == 0 and en_consonant_role_split[c].get("coda", 0) == 0]
    print(f"  consonants only-as-onset: {only_onset}")
    print(f"  consonants only-as-coda:  {only_coda}")
    print(f"  consonants unused:        {unused}")
    print()

    print("  English word-phoneme-count distribution (untagged count = tagged count):")
    total_w = sum(en_word_phoneme_len.values())
    cum = 0
    for k in sorted(en_word_phoneme_len):
        cum += en_word_phoneme_len[k]
        print(f"    {k:2d} phonemes/word: {en_word_phoneme_len[k]:>10,} ({100*en_word_phoneme_len[k]/total_w:5.2f}%)  cum={100*cum/total_w:5.2f}%")
    print()

    print("  English syllables-per-word:")
    total_w_s = sum(en_syllables_per_word.values())
    cum = 0
    for k in sorted(en_syllables_per_word):
        cum += en_syllables_per_word[k]
        print(f"    {k:2d} sylls/word:    {en_syllables_per_word[k]:>10,} ({100*en_syllables_per_word[k]/total_w_s:5.2f}%)  cum={100*cum/total_w_s:5.2f}%")
    print()

    print("  English phonemes-per-syllable (under maximal-onset rule):")
    total_s = sum(en_phonemes_per_syllable.values())
    cum = 0
    for k in sorted(en_phonemes_per_syllable):
        cum += en_phonemes_per_syllable[k]
        print(f"    {k:2d} phs/syll:      {en_phonemes_per_syllable[k]:>10,} ({100*en_phonemes_per_syllable[k]/total_s:5.2f}%)  cum={100*cum/total_s:5.2f}%")
    print()

    # K-coverage: with positional tags, phoneme count per word is unchanged
    # so K=8 fits 97% of words. Repeat the headline for clarity.
    print("  K-slot coverage if we keep K=8 slots per BPE token:")
    counts = sorted(en_word_phoneme_len.items())
    fits = sum(n for k, n in counts if k <= 8)
    print(f"    words fitting in K=8: {fits:,}/{total_w} ({100*fits/total_w:.2f}%)")
    fits12 = sum(n for k, n in counts if k <= 12)
    print(f"    words fitting in K=12: {fits12:,}/{total_w} ({100*fits12/total_w:.2f}%)")
    print()

    # ----- Cantonese -----
    print("=== Cantonese (Jyutping, split-form) ===")
    print(f"  rows={lang_counts['yue']:,}")
    print(f"  total syllables: {sum(yue_syllables_per_row.values()):,} → "
          f"avg {sum(k*v for k,v in yue_syllables_per_row.items())/max(1,sum(yue_syllables_per_row.values())):.1f}/row")
    print(f"  initials used: {len(yue_init_used)}/{sizes['jp_initial']}  finals used: {len(yp_finals_seen)} (dataset-derived)")
    print("  initial frequency (top 15):")
    for tok, n in yue_init_used.most_common(15):
        name = tok if tok else "(zero-initial)"
        print(f"    {name!r:>20s}  {n:>10,}")
    print("  final frequency (top 15):")
    for tok, n in yue_final_used.most_common(15):
        print(f"    {tok!r:>20s}  {n:>10,}")
    print()

    # ----- Mandarin -----
    print("=== Mandarin (Pinyin, split-form) ===")
    print(f"  rows={lang_counts['zh']:,}")
    print(f"  total syllables: {sum(zh_syllables_per_row.values()):,} → "
          f"avg {sum(k*v for k,v in zh_syllables_per_row.items())/max(1,sum(zh_syllables_per_row.values())):.1f}/row")
    print(f"  initials used: {len(zh_init_used)}/{sizes['py_initial']}  finals used: {len(py_finals_seen)} (dataset-derived)")
    print("  initial frequency (top 15):")
    for tok, n in zh_init_used.most_common(15):
        name = tok if tok else "(zero-initial)"
        print(f"    {name!r:>20s}  {n:>10,}")
    print("  final frequency (top 15):")
    for tok, n in zh_final_used.most_common(15):
        print(f"    {tok!r:>20s}  {n:>10,}")


if __name__ == "__main__":
    analyze()
