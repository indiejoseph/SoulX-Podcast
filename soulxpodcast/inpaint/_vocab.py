"""Frozen phoneme vocabularies for the three supported alphabets.

Layout (global id space, padding=0 reserved):

    id 0                                       = pad / no phoneme
    CMU vowels (nucleus only)                  = 15
    CMU consonants tagged "_on" (onset)        = 24
    CMU consonants tagged "_co" (coda)         = 24
    CMU word boundary "|"                      = 1
    Jyutping initials (incl. zero-initial "")  = 20
    Jyutping finals-with-tone                  = 257 (data-derived)
    Pinyin   initials (incl. zero-initial "")  = 24
    Pinyin   finals-with-tone                  = 210 (data-derived)

    TOTAL_VOCAB_SIZE = 576

The position-tagging on English consonants encodes the syllable role
(``L_on`` vs ``L_co`` are different ids → different embedding rows),
giving the composer the same kind of position information that
Chinese ``[initial, final+tone]`` carries by construction.

The zero-initial for Jyutping/Pinyin (syllables like ``aa3`` / ``a1``
that have no consonant onset) lives at an explicit ``""`` entry in
each initials list rather than overloading pad.

Initials and ARPAbet structure are hard-coded so they survive even
when a particular position-tag is unobserved in the training set
(empirically ``HH_co``, ``W_co``, ``Y_co`` never appear — those are
phonotactic gaps, kept for vocab symmetry). Chinese finals are
data-derived from ``vocab/finals.json``; regenerate that file from
the corpus if the underlying inventory changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

# ARPAbet vowels — always nucleus.
CMU_VOWELS: Final[tuple[str, ...]] = (
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
)
# ARPAbet consonants — get split into "_on" and "_co" variants.
CMU_CONSONANTS: Final[tuple[str, ...]] = (
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L",
    "M", "N", "NG", "P", "R", "S", "SH",
    "T", "TH", "V", "W", "Y", "Z", "ZH",
)
CMU_VOWEL_SET: Final[frozenset[str]] = frozenset(CMU_VOWELS)
CMU_CONSONANT_SET: Final[frozenset[str]] = frozenset(CMU_CONSONANTS)

# Word boundary marker (one id).
CMU_BOUNDARY: Final[str] = "|"

# Jyutping initials, greedy longest-first for syllable splitting.
JP_INITIALS: Final[tuple[str, ...]] = (
    "gw", "kw", "ng",
    "b", "c", "d", "f", "g", "h", "j", "k", "l",
    "m", "n", "p", "s", "t", "w", "z",
    "",  # zero-initial — must remain last so longest-first split still works
)

# Pinyin initials, greedy longest-first.
PY_INITIALS: Final[tuple[str, ...]] = (
    "zh", "ch", "sh",
    "b", "c", "d", "f", "g", "h", "j", "k", "l",
    "m", "n", "p", "q", "r", "s", "t", "w", "x", "y", "z",
    "",  # zero-initial
)


def _load_finals() -> tuple[tuple[str, ...], tuple[str, ...]]:
    path = Path(__file__).parent / "vocab" / "finals.json"
    with path.open() as f:
        blob = json.load(f)
    return tuple(blob["jyutping_finals"]), tuple(blob["pinyin_finals"])


JP_FINALS, PY_FINALS = _load_finals()


# ----- Global id-space layout (pad reserved at 0) ---------------------------

PAD_ID: Final[int] = 0

CMU_VOWEL_BASE: Final[int] = 1
N_CMU_VOWEL: Final[int] = len(CMU_VOWELS)

CMU_ONSET_BASE: Final[int] = CMU_VOWEL_BASE + N_CMU_VOWEL
N_CMU_ONSET: Final[int] = len(CMU_CONSONANTS)

CMU_CODA_BASE: Final[int] = CMU_ONSET_BASE + N_CMU_ONSET
N_CMU_CODA: Final[int] = len(CMU_CONSONANTS)

CMU_BOUNDARY_ID: Final[int] = CMU_CODA_BASE + N_CMU_CODA

JP_INITIAL_BASE: Final[int] = CMU_BOUNDARY_ID + 1
N_JP_INITIAL: Final[int] = len(JP_INITIALS)

JP_FINAL_BASE: Final[int] = JP_INITIAL_BASE + N_JP_INITIAL
N_JP_FINAL: Final[int] = len(JP_FINALS)

PY_INITIAL_BASE: Final[int] = JP_FINAL_BASE + N_JP_FINAL
N_PY_INITIAL: Final[int] = len(PY_INITIALS)

PY_FINAL_BASE: Final[int] = PY_INITIAL_BASE + N_PY_INITIAL
N_PY_FINAL: Final[int] = len(PY_FINALS)

TOTAL_VOCAB_SIZE: Final[int] = PY_FINAL_BASE + N_PY_FINAL


# ----- Forward maps token-string → global-id --------------------------------

CMU_VOWEL_TO_ID: Final[dict[str, int]] = {
    tok: CMU_VOWEL_BASE + i for i, tok in enumerate(CMU_VOWELS)
}
CMU_ONSET_TO_ID: Final[dict[str, int]] = {
    f"{tok}_on": CMU_ONSET_BASE + i for i, tok in enumerate(CMU_CONSONANTS)
}
CMU_CODA_TO_ID: Final[dict[str, int]] = {
    f"{tok}_co": CMU_CODA_BASE + i for i, tok in enumerate(CMU_CONSONANTS)
}
JP_INITIAL_TO_ID: Final[dict[str, int]] = {
    tok: JP_INITIAL_BASE + i for i, tok in enumerate(JP_INITIALS)
}
JP_FINAL_TO_ID: Final[dict[str, int]] = {
    tok: JP_FINAL_BASE + i for i, tok in enumerate(JP_FINALS)
}
PY_INITIAL_TO_ID: Final[dict[str, int]] = {
    tok: PY_INITIAL_BASE + i for i, tok in enumerate(PY_INITIALS)
}
PY_FINAL_TO_ID: Final[dict[str, int]] = {
    tok: PY_FINAL_BASE + i for i, tok in enumerate(PY_FINALS)
}

# Inverse map id → (alphabet, surface-token) for diagnostics.
ID_TO_TOKEN: Final[dict[int, tuple[str, str]]] = {PAD_ID: ("pad", "")}
for _tok, _id in CMU_VOWEL_TO_ID.items():
    ID_TO_TOKEN[_id] = ("cmu", _tok)
for _tok, _id in CMU_ONSET_TO_ID.items():
    ID_TO_TOKEN[_id] = ("cmu", _tok)
for _tok, _id in CMU_CODA_TO_ID.items():
    ID_TO_TOKEN[_id] = ("cmu", _tok)
ID_TO_TOKEN[CMU_BOUNDARY_ID] = ("cmu", CMU_BOUNDARY)
for _tok, _id in JP_INITIAL_TO_ID.items():
    ID_TO_TOKEN[_id] = ("jyutping", _tok)
for _tok, _id in JP_FINAL_TO_ID.items():
    ID_TO_TOKEN[_id] = ("jyutping", _tok)
for _tok, _id in PY_INITIAL_TO_ID.items():
    ID_TO_TOKEN[_id] = ("pinyin", _tok)
for _tok, _id in PY_FINAL_TO_ID.items():
    ID_TO_TOKEN[_id] = ("pinyin", _tok)


def alphabet_of(token_id: int) -> str:
    """Return one of {pad, cmu, jyutping, pinyin} for a global phoneme id."""
    return ID_TO_TOKEN[token_id][0]
