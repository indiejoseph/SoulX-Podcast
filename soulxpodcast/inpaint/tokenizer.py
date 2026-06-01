"""Multilingual phoneme tokenizer for pronunciation inpainting.

Accepts split-form phoneme strings as produced by the SSML frontend:

    CMUdict  (English):    "HH EH L OW W ER L D"
                           position roles auto-inferred via syllabification
                           (maximal-onset). "L" emerges as "L_on" in "Hello"
                           but as "L_co" in "world".

    Jyutping (Cantonese):  "j yut6 p ing4"          (initial, final+tone)
    Pinyin   (Mandarin):   "p in1 y in1"            (initial, final+tone)

Also exposes ``split_whole_syllable`` for offline data prep — the SoulX
training corpus stores Chinese phonemes in whole-syllable form
(``wo3``, ``nei1``) and needs to be re-tokenised into split form before
training.

All emitted ids live in a single global id space defined in
:mod:`soulxpodcast.inpaint._vocab`. Pad id 0 means "no phoneme".
"""

from __future__ import annotations

from typing import Literal

from soulxpodcast.inpaint._vocab import (
    CMU_BOUNDARY,
    CMU_BOUNDARY_ID,
    CMU_CODA_TO_ID,
    CMU_CONSONANT_SET,
    CMU_ONSET_TO_ID,
    CMU_VOWEL_SET,
    CMU_VOWEL_TO_ID,
    JP_FINAL_TO_ID,
    JP_INITIAL_TO_ID,
    JP_INITIALS,
    PY_FINAL_TO_ID,
    PY_INITIAL_TO_ID,
    PY_INITIALS,
    TOTAL_VOCAB_SIZE,
    alphabet_of,
)

Alphabet = Literal["cmu", "jyutping", "pinyin"]
_VALID_ALPHABETS: frozenset[str] = frozenset({"cmu", "jyutping", "pinyin"})


def syllabify_arpabet(phonemes: list[str]) -> list[list[tuple[str, str]]]:
    """Syllabify a flat ARPAbet phoneme list using maximal-onset.

    Input is a single word's phoneme stream (no ``|`` markers — split on
    those before calling). Returns a list of syllables; each syllable is
    a list of ``(phoneme_symbol, role)`` pairs where role is one of
    ``"onset" | "nucleus" | "coda"``.

    Rules:

    - Every vowel is its own syllable's nucleus.
    - Consonants before the first vowel = onset of first syllable.
    - Consonants between two vowels: maximal onset — ALL go to the NEXT
      syllable's onset. (This is conservative; phonotactically a coda is
      preferred in some clusters, but the model can learn the residue.)
    - Consonants after the last vowel = coda of last syllable.
    - Zero-vowel words (rare; abbreviations like "HMM") become a single
      synthetic syllable of all-coda consonants. Caller can audit.
    """
    vowel_idx = [i for i, p in enumerate(phonemes) if p in CMU_VOWEL_SET]
    if not vowel_idx:
        if not phonemes:
            return []
        return [[(p, "coda") for p in phonemes]]

    sylls: list[list[tuple[str, str]]] = []
    first_v = vowel_idx[0]
    first: list[tuple[str, str]] = [(p, "onset") for p in phonemes[:first_v]]
    first.append((phonemes[first_v], "nucleus"))
    sylls.append(first)

    for k in range(1, len(vowel_idx)):
        prev_v = vowel_idx[k - 1]
        cur_v = vowel_idx[k]
        between = phonemes[prev_v + 1 : cur_v]
        cur: list[tuple[str, str]] = [(p, "onset") for p in between]
        cur.append((phonemes[cur_v], "nucleus"))
        sylls.append(cur)

    for p in phonemes[vowel_idx[-1] + 1 :]:
        sylls[-1].append((p, "coda"))
    return sylls


def _cmu_id_for(phoneme: str, role: str) -> int:
    """Look up the position-tagged id for a single ARPAbet phoneme."""
    if phoneme in CMU_VOWEL_SET:
        return CMU_VOWEL_TO_ID[phoneme]
    if phoneme not in CMU_CONSONANT_SET:
        raise ValueError(f"unknown ARPAbet phoneme {phoneme!r}")
    if role == "onset":
        return CMU_ONSET_TO_ID[f"{phoneme}_on"]
    if role == "coda":
        return CMU_CODA_TO_ID[f"{phoneme}_co"]
    raise ValueError(f"unknown role {role!r} for consonant {phoneme!r}")


def encode_arpabet_word(phonemes: list[str]) -> list[int]:
    """Syllabify one English word's phoneme stream and return position-tagged ids.

    Equivalent to ``[id for syll in syllabify_arpabet(phs) for (p, r) in syll
    for id in [_cmu_id_for(p, r)]]`` but inlined and faster.
    """
    sylls = syllabify_arpabet(phonemes)
    out: list[int] = []
    for syll in sylls:
        for ph, role in syll:
            out.append(_cmu_id_for(ph, role))
    return out


class PhonemeTokenizer:
    """Token-string → global-id encoder for the three supported alphabets."""

    vocab_size: int = TOTAL_VOCAB_SIZE
    alphabet_of = staticmethod(alphabet_of)
    syllabify_arpabet = staticmethod(syllabify_arpabet)

    def encode_span(self, alphabet: str, ph_tokens: list[str]) -> list[int]:
        """Encode a phoneme span emitted by the SSML frontend.

        For ``cmu`` the tokens are a flat ARPAbet stream for one word (may
        contain ``|`` if the caller passes multi-word spans; this method
        will split and re-syllabify per word). Position roles are inferred
        from the syllabification.

        For ``jyutping`` / ``pinyin`` tokens alternate
        ``initial, final, initial, final, ...`` starting with an initial;
        the zero-initial ``""`` is a valid initial.

        Raises ``ValueError`` on an unknown alphabet or token.
        """
        if alphabet not in _VALID_ALPHABETS:
            raise ValueError(
                f"unknown alphabet {alphabet!r}; expected one of {_VALID_ALPHABETS}"
            )
        if alphabet == "cmu":
            return self._encode_cmu(ph_tokens)
        return self._encode_chinese(alphabet, ph_tokens)

    @staticmethod
    def _encode_cmu(ph_tokens: list[str]) -> list[int]:
        out: list[int] = []
        # Split on word-boundary markers if present, then per-word syllabify.
        word: list[str] = []
        for tok in ph_tokens:
            if tok == CMU_BOUNDARY:
                if word:
                    out.extend(encode_arpabet_word(word))
                    word = []
                out.append(CMU_BOUNDARY_ID)
            else:
                word.append(tok)
        if word:
            out.extend(encode_arpabet_word(word))
        return out

    @staticmethod
    def _encode_chinese(alphabet: str, ph_tokens: list[str]) -> list[int]:
        if len(ph_tokens) % 2 != 0:
            raise ValueError(
                f"{alphabet} phoneme span must contain pairs of [initial, final+tone]; "
                f"got odd length {len(ph_tokens)}: {ph_tokens!r}"
            )
        initial_map = JP_INITIAL_TO_ID if alphabet == "jyutping" else PY_INITIAL_TO_ID
        final_map = JP_FINAL_TO_ID if alphabet == "jyutping" else PY_FINAL_TO_ID
        out: list[int] = []
        for i, tok in enumerate(ph_tokens):
            if i % 2 == 0:
                if tok not in initial_map:
                    raise ValueError(f"unknown {alphabet} initial {tok!r}")
                out.append(initial_map[tok])
            else:
                if tok not in final_map:
                    raise ValueError(f"unknown {alphabet} final {tok!r}")
                out.append(final_map[tok])
        return out

    # -- offline data-prep helpers --------------------------------------

    @staticmethod
    def split_whole_syllable(alphabet: str, syllable: str) -> tuple[str, str]:
        """Greedy longest-initial split of a whole-syllable Chinese token.

        Examples::

            split_whole_syllable("jyutping", "nei1")  -> ("n",  "ei1")
            split_whole_syllable("jyutping", "aa3")   -> ("",   "aa3")
            split_whole_syllable("jyutping", "gwong2") -> ("gw", "ong2")
            split_whole_syllable("pinyin",   "zhuo1") -> ("zh", "uo1")
            split_whole_syllable("pinyin",   "a1")    -> ("",   "a1")

        Used to convert the SoulX training corpus's whole-syllable
        ``phonemes`` field into the model's expected split form.
        """
        if alphabet == "jyutping":
            initials = JP_INITIALS
        elif alphabet == "pinyin":
            initials = PY_INITIALS
        else:
            raise ValueError(
                f"split is only defined for jyutping/pinyin, not {alphabet!r}"
            )
        for ini in initials:
            if ini == "":
                continue
            if syllable.startswith(ini):
                return ini, syllable[len(ini) :]
        return "", syllable

    def encode_whole_syllable_sequence(
        self, alphabet: str, syllables: list[str]
    ) -> list[int]:
        """Convenience: split each whole-syllable token then encode as split-form.

        Convert a dataset row's ``phonemes`` list (whole-syllable) to a flat
        list of global ids in split form.
        """
        if alphabet not in {"jyutping", "pinyin"}:
            raise ValueError(
                f"encode_whole_syllable_sequence is jyutping/pinyin-only, got {alphabet!r}"
            )
        ph_tokens: list[str] = []
        for syl in syllables:
            if syl and not syl[-1].isdigit():
                # punctuation token; skip — it lives in the text stream, not phoneme
                continue
            ini, fin = self.split_whole_syllable(alphabet, syl)
            ph_tokens.append(ini)
            ph_tokens.append(fin)
        return self.encode_span(alphabet, ph_tokens)
