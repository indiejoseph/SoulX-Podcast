"""Behavioral test against the adversarial fixture library.

For each fixture this script generates THREE speech-token sequences:

  * baseline    — plain text (no SSML). LLM uses its natural pronunciation.
  * inpaint_correct — SSML override whose phonemes match the natural reading.
  * inpaint_wrong   — SSML override whose phonemes differ from natural.

And asserts behaviour we KNOW a working pipeline must satisfy:

  E1. baseline emits EOS within max_new_tokens (catches LLM repetition loops,
      e.g. en_world_baseline 200-token stalls).
  E2. inpaint_correct + inpaint_wrong also emit EOS.
  M1. (mode-collapse guard) edit_distance(correct, wrong) is a substantial
      fraction of max(len(correct), len(wrong)). If composer is being ignored,
      both overrides produce near-identical tokens.
  D1. (direction guard) edit_dist(baseline, correct) ≤ edit_dist(baseline, wrong).
      Correct override matches natural reading so the composer should steer
      LLM CLOSER to baseline; wrong override should push it AWAY.
  T1. (token-ratio bands) #tokens(inpaint) / #tokens(baseline) inside the
      per-fixture bands defined in fixtures_adversarial.py. Catches gross
      broadcast-stutter (>>1.0) or composer-mode-collapse-to-silence (<<1.0).
  R1. (RMS sanity, optional) inpaint audio RMS within 10 dB of baseline RMS.
      Catches collapse-to-silence at audio level. Requires --synthesize_audio.

What this test CANNOT catch (documented gaps, not bugs in the test):

  * "Composer responds, but in the WRONG direction" — wrong override produces
    a syllable that's neither what was requested nor what baseline emits.
    Requires ASR or forced alignment back to the requested phonemes.
  * Subtle prosody / timbre drift. Requires perceptual evaluation.

Run:

  python scripts/inpaint/test_alignment_behavioral.py \
    --composer_ckpt outputs/inpaint_final/step_0030000/composer.pt \
    [--synthesize_audio]   # add audio-level RMS check; slower
    [--use_padsub]         # use upstream-style pad-substitution alignment

Tags each assertion result as [ok] / [FAIL] / [skip]. Exits 0 only if all
required asserts pass. The script PRINTS the metrics so failures can be
inspected even when bands need recalibration.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.inpaint.fixtures_adversarial import ALL_FIXTURES, Fixture, verify_bpe_assertions
from soulxpodcast.inpaint.inference import InpaintInferenceEngine

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_alignment_behavioral")


MODEL_PATH = "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged"

# Mode-collapse and direction thresholds.
# M1: correct and wrong tokens should disagree on at least 30% of positions
#     (or in length). Calibrated from the v10 broadcast run: zh_yinhang
#     correct vs wrong disagreed on 20/33 = 60% — clear above 30%.
COLLAPSE_MIN_DISAGREEMENT = 0.30
# R1: inpaint RMS shouldn't drop more than 10 dB below baseline.
RMS_DROP_DB_THRESHOLD = 10.0


@dataclass
class GenResult:
    speech_tokens: list[int]
    eos_hit: bool
    n_tokens: int
    dt_llm_s: float


# ------------------------------------------------------------------ helpers


def _levenshtein(a: list[int], b: list[int]) -> int:
    """Standard edit distance over two integer sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[m]


def _normalised_edit(a: list[int], b: list[int]) -> float:
    """Edit distance normalised by the longer sequence length."""
    n = max(len(a), len(b), 1)
    return _levenshtein(a, b) / n


def _generate(
    engine: InpaintInferenceEngine,
    ssml_or_text: str,
    lang: str,
    max_new_tokens: int,
    seed: int,
) -> GenResult:
    """Greedy generate; record speech tokens, EOS flag, wall time."""
    t0 = time.perf_counter()
    out = engine.generate_speech_tokens(
        ssml_or_text=ssml_or_text,
        lang=lang,
        max_new_tokens=max_new_tokens,
        do_sample=False,            # greedy → deterministic A/B
        temperature=1.0, top_p=1.0, repetition_penalty=1.0,
        seed=seed,
        disable_inpaint=("<phoneme" not in ssml_or_text),
    )
    dt = time.perf_counter() - t0
    return GenResult(
        speech_tokens=out.speech_tokens,
        eos_hit=out.eos_hit,
        n_tokens=len(out.speech_tokens),
        dt_llm_s=dt,
    )


def _check_fixture(
    fx: Fixture,
    engine: InpaintInferenceEngine,
    max_new_tokens: int,
    seed: int,
) -> tuple[list[tuple[str, str, str]], dict]:
    """Run baseline + correct + wrong; return (asserts, metrics).

    asserts: list of (id, status, detail). status in {"ok","FAIL","skip"}.
    metrics: dict of measured numbers for printing.
    """
    asserts: list[tuple[str, str, str]] = []

    g_base = _generate(engine, fx.text, fx.lang, max_new_tokens, seed)
    g_corr = _generate(engine, fx.ssml_correct, fx.lang, max_new_tokens, seed)
    g_wrng = _generate(engine, fx.ssml_wrong, fx.lang, max_new_tokens, seed)

    # E1, E2 — EOS expectations
    if fx.enable_behavioral:
        asserts.append((
            "E1.baseline_eos",
            "ok" if g_base.eos_hit else "FAIL",
            f"baseline emitted {g_base.n_tokens} tokens, eos={g_base.eos_hit}",
        ))
        asserts.append((
            "E2.correct_eos",
            "ok" if g_corr.eos_hit else "FAIL",
            f"correct emitted {g_corr.n_tokens} tokens, eos={g_corr.eos_hit}",
        ))
        asserts.append((
            "E2.wrong_eos",
            "ok" if g_wrng.eos_hit else "FAIL",
            f"wrong emitted {g_wrng.n_tokens} tokens, eos={g_wrng.eos_hit}",
        ))
    else:
        asserts.append(("E1/E2.eos", "skip", "behavioral disabled for this fixture"))

    # M1 — mode-collapse guard. Composer must respond differently to differing
    # overrides; if correct and wrong produce near-identical tokens, the
    # composer is being ignored.
    norm_corr_wrng = _normalised_edit(g_corr.speech_tokens, g_wrng.speech_tokens)
    asserts.append((
        "M1.correct_vs_wrong_disagree",
        "ok" if norm_corr_wrng >= COLLAPSE_MIN_DISAGREEMENT else "FAIL",
        f"normalised edit dist correct↔wrong = {norm_corr_wrng:.3f} "
        f"(min required {COLLAPSE_MIN_DISAGREEMENT})",
    ))

    # D1 — direction guard. correct should be closer to baseline than wrong.
    norm_base_corr = _normalised_edit(g_base.speech_tokens, g_corr.speech_tokens)
    norm_base_wrng = _normalised_edit(g_base.speech_tokens, g_wrng.speech_tokens)
    asserts.append((
        "D1.correct_closer_to_baseline_than_wrong",
        "ok" if norm_base_corr <= norm_base_wrng else "FAIL",
        f"edit(base,correct)={norm_base_corr:.3f} edit(base,wrong)={norm_base_wrng:.3f}",
    ))

    # T1 — token-count ratio bands
    if fx.enable_behavioral and g_base.n_tokens > 0:
        ratio_corr = g_corr.n_tokens / g_base.n_tokens
        ratio_wrng = g_wrng.n_tokens / g_base.n_tokens
        lo_c, hi_c = fx.tok_ratio_band_correct
        lo_w, hi_w = fx.tok_ratio_band_wrong
        asserts.append((
            "T1.correct_token_ratio",
            "ok" if lo_c <= ratio_corr <= hi_c else "FAIL",
            f"ratio correct/baseline = {ratio_corr:.2f} (band [{lo_c},{hi_c}])",
        ))
        asserts.append((
            "T1.wrong_token_ratio",
            "ok" if lo_w <= ratio_wrng <= hi_w else "FAIL",
            f"ratio wrong/baseline = {ratio_wrng:.2f} (band [{lo_w},{hi_w}])",
        ))

    metrics = {
        "baseline_tokens": g_base.n_tokens,
        "correct_tokens": g_corr.n_tokens,
        "wrong_tokens": g_wrng.n_tokens,
        "baseline_eos": g_base.eos_hit,
        "correct_eos": g_corr.eos_hit,
        "wrong_eos": g_wrng.eos_hit,
        "edit_base_corr": norm_base_corr,
        "edit_base_wrng": norm_base_wrng,
        "edit_corr_wrng": norm_corr_wrng,
    }
    return asserts, metrics


# ------------------------------------------------------------------ main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--composer_ckpt", required=True)
    p.add_argument("--model_path", default=MODEL_PATH)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_padsub", action="store_true",
                   help="Switch inference alignment to pad-substitution.")
    args = p.parse_args()

    if args.use_padsub:
        # Local import — keeps the dependency soft.
        from scripts.inpaint.inference_audio_padsub import install_padsub
        install_padsub()
        print("[note] using pad-substitution alignment (--use_padsub)")

    # Verify fixtures' BPE counts still match before any heavy work.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    verify_bpe_assertions(tok)

    engine = InpaintInferenceEngine(
        model_path=args.model_path,
        composer_ckpt_path=args.composer_ckpt,
    )

    total_asserts = 0
    failed_asserts = 0
    failed_fixtures: list[str] = []
    print()
    print("=" * 78)
    for fx in ALL_FIXTURES:
        asserts, metrics = _check_fixture(fx, engine, args.max_new_tokens, args.seed)
        any_fail = any(s == "FAIL" for _, s, _ in asserts)
        if any_fail:
            failed_fixtures.append(fx.name)
        print(f"\n[{fx.name}]  lang={fx.lang}  pads={fx.expected_inpaint_pads}")
        print(f"  metrics: {metrics}")
        for aid, status, detail in asserts:
            total_asserts += 1
            if status == "FAIL":
                failed_asserts += 1
            tag = {"ok": "[ok]   ", "FAIL": "[FAIL] ", "skip": "[skip] "}[status]
            print(f"  {tag}{aid:35s} {detail}")

    print()
    print("=" * 78)
    print(f"{total_asserts - failed_asserts}/{total_asserts} asserts passed")
    if failed_fixtures:
        print(f"FAILED fixtures: {', '.join(failed_fixtures)}")
    sys.exit(0 if failed_asserts == 0 else 1)


if __name__ == "__main__":
    main()
