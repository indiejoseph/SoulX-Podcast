"""Adversarial input library for pre-v11 inpaint testing.

Each fixture is a single test case that exercises a specific known failure mode
in the inpaint pipeline. Fixtures are intentionally hand-curated:

  * BPE count for the target span is verified empirically and asserted at
    test-load time — if upstream Qwen3 tokenizer changes, tests fail loudly.
  * Each fixture carries BOTH inference-shape (SSML) and training-shape
    (whole-syllable phonemes per CJK char / per ARPAbet word) so the parity
    test can drive both pipelines with equivalent input.
  * Behavioural assertions live with the fixture (token-count bands,
    "correct should be close to baseline", etc).

Adding a fixture: pick a single failure-mode hypothesis to exercise, write
the source row to match it, and verify expected BPE behaviour by running
the assertions at the bottom of this file.

Failure modes covered so far:

  yue_keoi5_byte_fallback   — 1-char 2-BPE (byte fallback), broadcast stutter.
  yue_dei6_byte_fallback    — same shape, different syllable.
  zh_yinhang_merged         — 1-BPE 2-char merged word, mean-pool collapse on
                              multi-syllable override.
  zh_zhongguo_merged        — same shape, different vocab to rule out target-
                              specific accidents.
  zh_wo_singleton_baseline  — 1-char 1-BPE 1-syllable — should always work;
                              regression canary.
  en_banana_multibpe        — 6-letter 3-syllable word over 3 BPEs; exercises
                              the multi-BPE syllable distribution logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    """One adversarial test case.

    Fields:
      name          identifier, used in test output and file names.
      lang          'yue' | 'zh' | 'en'.
      text          source text without SSML (matches the training row's `text`
                    field shape, including punctuation conventions per lang).
      ssml_correct  SSML form whose phoneme override matches the natural reading
                    of the target chars/word. Inference output should be close
                    to the no-inpaint baseline for this prompt.
      ssml_wrong    SSML form whose phoneme override differs from the natural
                    reading. Inference output should diverge from baseline AND
                    from ssml_correct.
      train_phonemes_correct
                    Whole-syllable phonemes list aligned to text CJK chars
                    (Chinese) or alpha words (English). Non-target positions
                    use a non-syllable placeholder ('X') so the training-side
                    `_align_*` skips them — lets us drive training and inference
                    paths with the SAME logical override.
      train_phonemes_wrong
                    Same shape, with the wrong syllable.
      target_chars  Char range that the override applies to. Used by tests to
                    locate the BPE positions of interest.
      expected_bpe_count_for_target
                    Number of BPE tokens that the tokenizer assigns to the
                    target char range. Asserted at fixture load — see end of
                    file. Catches tokenizer-version drift.
      expected_inpaint_pads
                    Number of pad-substitution tokens the inference pipeline
                    should insert if pad-substitution alignment is used.
                    Equals syllable count.
      tok_ratio_band_correct
                    Expected (inpaint_correct_tokens / baseline_tokens) band.
                    Correct override should produce similar token count to
                    baseline since the override matches natural reading.
      tok_ratio_band_wrong
                    Expected (inpaint_wrong_tokens / baseline_tokens) band.
                    Wrong override may shift token count somewhat but should
                    not balloon (broadcast stutter would push it to ~1.5-2x).
      enable_behavioral
                    False for fixtures whose LLM baseline is known unstable
                    (e.g. English under current SoulX checkpoints). Parity
                    test still runs; behavioral test is skipped.
      enable_parity
                    False for fixtures where training-side and inference-side
                    intentionally produce different layouts (e.g. English,
                    where ``train_phonemes`` annotates every word but the SSML
                    only annotates one). Default True.
    """

    name: str
    lang: str
    text: str
    ssml_correct: str
    ssml_wrong: str
    train_phonemes_correct: list[str]
    train_phonemes_wrong: list[str]
    target_chars: tuple[int, int]
    expected_bpe_count_for_target: int
    expected_inpaint_pads: int
    tok_ratio_band_correct: tuple[float, float]
    tok_ratio_band_wrong: tuple[float, float]
    enable_behavioral: bool = True
    enable_parity: bool = True


# -- Cantonese fixtures -------------------------------------------------- #

YUE_KEOI5_BYTE_FALLBACK = Fixture(
    name="yue_keoi5_byte_fallback",
    lang="yue",
    # NB: the natural reading of 佢 is "keoi5" (he/she). Surface text uses
    # full-width punctuation matching the yue training distribution.
    text="我同佢一齊去飲茶。",
    ssml_correct='我同<phoneme alphabet="jyutping" ph="k eoi5">佢</phoneme>一齊去飲茶。',
    ssml_wrong='我同<phoneme alphabet="jyutping" ph="b aai6">佢</phoneme>一齊去飲茶。',
    # Whole-syllable phonemes per CJK char. 'X' is a non-syllable placeholder
    # that the training-side `_align_chinese` skips (line 324: requires the
    # final char to be a digit). Punctuation falls outside _is_cjk and is also
    # skipped — we don't need an entry for the final '。'.
    train_phonemes_correct=["X", "X", "keoi5", "X", "X", "X", "X", "X"],
    train_phonemes_wrong=["X", "X", "baai6", "X", "X", "X", "X", "X"],
    target_chars=(2, 3),
    expected_bpe_count_for_target=2,  # byte-fallback into 2 BPE tokens
    expected_inpaint_pads=1,           # 1 syllable
    tok_ratio_band_correct=(0.7, 1.3),
    tok_ratio_band_wrong=(0.7, 1.3),   # broadcast bug would push to 1.5+
)

YUE_DEI6_BYTE_FALLBACK = Fixture(
    name="yue_dei6_byte_fallback",
    lang="yue",
    text="我哋一齊去食飯。",
    ssml_correct='我<phoneme alphabet="jyutping" ph="d ei6">哋</phoneme>一齊去食飯。',
    ssml_wrong='我<phoneme alphabet="jyutping" ph="g aau3">哋</phoneme>一齊去食飯。',
    train_phonemes_correct=["X", "dei6", "X", "X", "X", "X", "X"],
    train_phonemes_wrong=["X", "gaau3", "X", "X", "X", "X", "X"],
    target_chars=(1, 2),
    expected_bpe_count_for_target=2,
    expected_inpaint_pads=1,
    tok_ratio_band_correct=(0.7, 1.3),
    tok_ratio_band_wrong=(0.7, 1.3),
)


# -- Mandarin fixtures --------------------------------------------------- #

ZH_YINHANG_MERGED = Fixture(
    name="zh_yinhang_merged",
    lang="zh",
    # zh training distribution = no punctuation.
    text="我去银行办事",
    ssml_correct='我去<phoneme alphabet="pinyin" ph="y in2 h ang2">银行</phoneme>办事',
    ssml_wrong='我去<phoneme alphabet="pinyin" ph="y in2 x ing2">银行</phoneme>办事',
    # 6 CJK chars; phonemes target positions 2-3 (银, 行).
    train_phonemes_correct=["X", "X", "yin2", "hang2", "X", "X"],
    train_phonemes_wrong=["X", "X", "yin2", "xing2", "X", "X"],
    target_chars=(2, 4),
    expected_bpe_count_for_target=1,   # 银行 is one merged BPE
    expected_inpaint_pads=2,            # 2 syllables — pad-sub should split
    tok_ratio_band_correct=(0.7, 1.3),
    tok_ratio_band_wrong=(0.7, 1.3),
)

ZH_ZHONGGUO_MERGED = Fixture(
    name="zh_zhongguo_merged",
    lang="zh",
    text="我们去中国旅游",
    ssml_correct='我们去<phoneme alphabet="pinyin" ph="zh ong1 g uo2">中国</phoneme>旅游',
    ssml_wrong='我们去<phoneme alphabet="pinyin" ph="zh ong1 sh i4">中国</phoneme>旅游',
    train_phonemes_correct=["X", "X", "X", "zhong1", "guo2", "X", "X"],
    train_phonemes_wrong=["X", "X", "X", "zhong1", "shi4", "X", "X"],
    target_chars=(3, 5),
    expected_bpe_count_for_target=1,
    expected_inpaint_pads=2,
    tok_ratio_band_correct=(0.7, 1.3),
    tok_ratio_band_wrong=(0.7, 1.3),
)

ZH_WO_SINGLETON_BASELINE = Fixture(
    name="zh_wo_singleton_baseline",
    lang="zh",
    text="我吃饭",
    ssml_correct='<phoneme alphabet="pinyin" ph="w o3">我</phoneme>吃饭',
    # wrong: replace "wo3" with "ta1" (he) — clearly different syllable
    ssml_wrong='<phoneme alphabet="pinyin" ph="t a1">我</phoneme>吃饭',
    train_phonemes_correct=["wo3", "X", "X"],
    train_phonemes_wrong=["ta1", "X", "X"],
    target_chars=(0, 1),
    expected_bpe_count_for_target=1,
    expected_inpaint_pads=1,
    tok_ratio_band_correct=(0.7, 1.3),
    tok_ratio_band_wrong=(0.7, 1.3),
)


# -- English fixture (behavioral disabled — known LLM-baseline loop) --- #

EN_BANANA_MULTIBPE = Fixture(
    name="en_banana_multibpe",
    lang="en",
    # All-caps, no punctuation — matches the SoulX en training corpus shape.
    text="I LIKE BANANA",
    ssml_correct='I LIKE <phoneme alphabet="cmu" ph="B AH N AE N AH">BANANA</phoneme>',
    ssml_wrong='I LIKE <phoneme alphabet="cmu" ph="P IH N AE P AH L">BANANA</phoneme>',
    # 3 words. Train side splits by '|'. Non-target words use the 'X'
    # placeholder (skipped by the aligner, mirroring the zh convention) so the
    # training side annotates ONLY BANANA — matching the SSML which wraps only
    # BANANA. This makes parity a like-for-like comparison.
    train_phonemes_correct=["X", "|", "X", "|", "B", "AH", "N", "AE", "N", "AH"],
    train_phonemes_wrong=["X", "|", "X", "|", "P", "IH", "N", "AE", "P", "AH", "L"],
    target_chars=(7, 13),
    expected_bpe_count_for_target=3,   # 'B', 'AN', 'ANA' or similar
    expected_inpaint_pads=3,            # 3 syllables: BA-NA-NA
    tok_ratio_band_correct=(0.5, 2.0),
    tok_ratio_band_wrong=(0.5, 2.0),
    enable_behavioral=False,  # LLM baseline currently loops on en prompts
    # v12 + whitespace-tolerant substitution: training (with 'X' placeholders
    # for non-target words) and inference (SSML annotates only BANANA) now
    # produce the SAME layout — only BANANA is substituted on both sides — so
    # parity is meaningful again. The leading-space BPE (" B") is cleanly
    # substituted via free_chars, so BANANA's graphemes are removed on both.
    enable_parity=True,
)


ALL_FIXTURES: tuple[Fixture, ...] = (
    YUE_KEOI5_BYTE_FALLBACK,
    YUE_DEI6_BYTE_FALLBACK,
    ZH_YINHANG_MERGED,
    ZH_ZHONGGUO_MERGED,
    ZH_WO_SINGLETON_BASELINE,
    EN_BANANA_MULTIBPE,
)


def verify_bpe_assertions(tokenizer) -> None:
    """Verify expected_bpe_count_for_target against the live tokenizer.

    Fails loudly if any fixture's expectation drifted — usually means a
    tokenizer version bump or a fixture text edit broke the failure-mode
    coverage we relied on.
    """
    for fx in ALL_FIXTURES:
        cs, ce = fx.target_chars
        enc = tokenizer(fx.text, add_special_tokens=False, return_offsets_mapping=True)
        offs = [tuple(o) for o in enc["offset_mapping"]]
        covering = [
            i for i, (s, e) in enumerate(offs) if not (e <= cs or s >= ce)
        ]
        actual = len(covering)
        if actual != fx.expected_bpe_count_for_target:
            raise AssertionError(
                f"fixture {fx.name!r}: target {fx.text[cs:ce]!r} now covers "
                f"{actual} BPE(s), fixture expected "
                f"{fx.expected_bpe_count_for_target}. Tokenizer drift or text "
                f"edit broke the test coverage."
            )


if __name__ == "__main__":
    # Sanity script — verify fixtures against the live model tokenizer.
    from transformers import AutoTokenizer

    MODEL_PATH = "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    verify_bpe_assertions(tok)
    print(f"OK — {len(ALL_FIXTURES)} fixtures verified against tokenizer")
    for fx in ALL_FIXTURES:
        print(f"  {fx.name:30s} lang={fx.lang}  BPEs={fx.expected_bpe_count_for_target}  pads={fx.expected_inpaint_pads}  behavioral={fx.enable_behavioral}")
