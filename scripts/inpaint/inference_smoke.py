"""Multi-language inference smoke test for pronunciation-inpaint.

For each of {en, yue, zh} we run **two** generations of the same
utterance:

  * baseline — plain text, composer effectively OFF
  * inpaint — SSML with one or two ``<phoneme>`` annotations,
              composer ON at those spans

Then we compare the generated speech-token sequences and report:

  * how many tokens were generated in each case
  * how many positions differ between baseline and inpaint
  * where the first divergence falls

A real inpaint should diverge from the baseline starting near the
annotated word's audio onset and re-converge after the word ends. We
can't fully verify acoustic correctness from token IDs alone, but a
**non-zero, localised** difference is the cheapest "the inject path
works end-to-end" signal.

Usage::

    .venv/bin/python scripts/inpaint/inference_smoke.py \\
      --model_path /home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/pretrained_models/SoulX-Podcast-1.7B-dialect \\
      --composer_ckpt outputs/inpaint_final/step_0030000/composer.pt

If the user trained on top of a merged-LoRA model (``runs/merged``),
prefer the same model for inference. Mismatch is OK for smoke but
audio quality will be off.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running as a plain script from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from soulxpodcast.inpaint.inference import InpaintInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inference_smoke")


# Three test cases — one per supported alphabet. The "inpaint_word"
# field records the user-facing word being phoneme-overridden; the SSML
# wraps it.
TEST_CASES: list[dict] = [
    {
        "name": "yue_keoi5",
        "lang": "yue",
        "inpaint_word": "佢",
        "ph": "k eoi5",
        "alphabet": "jyutping",
        "plain":  "我同佢去飲茶。",
        "ssml":   '我同<phoneme alphabet="jyutping" ph="k eoi5">佢</phoneme>去飲茶。',
    },
    {
        "name": "zh_zhuyi",
        "lang": "zh",
        "inpaint_word": "注意",
        "ph": "zh u4 y i4",
        "alphabet": "pinyin",
        "plain":  "请注意听。",
        "ssml":   '请<phoneme alphabet="pinyin" ph="zh u4 y i4">注意</phoneme>听。',
    },
    {
        "name": "en_world",
        "lang": "en",
        "inpaint_word": "world",
        "ph": "W ER L D",
        "alphabet": "cmu",
        "plain":  "Hello world today.",
        "ssml":   'Hello <phoneme alphabet="cmu" ph="W ER L D">world</phoneme> today.',
    },
]


def first_divergence(a: list[int], b: list[int]) -> int:
    """Index of first position where a and b differ; -1 if identical up to min length."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


def hamming(a: list[int], b: list[int]) -> tuple[int, int]:
    """Returns (n_differ, n_compared) over the shared prefix length."""
    n = min(len(a), len(b))
    diff = sum(1 for i in range(n) if a[i] != b[i])
    return diff, n


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        default="/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/pretrained_models/SoulX-Podcast-1.7B-dialect",
    )
    p.add_argument(
        "--composer_ckpt",
        default="outputs/inpaint_final/step_0030000/composer.pt",
    )
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_sample", action="store_true",
        help="Greedy decoding — makes the A/B deterministic so diff is purely composer-driven."
    )
    p.add_argument("--save_jsonl", type=str, default="",
                   help="If set, write per-case results to this file as JSON lines.")
    p.add_argument("--cases", type=str, default="",
                   help="Comma-separated case names to run (default: all).")
    return p.parse_args()


def main():
    args = parse_args()
    log.info(f"model_path     : {args.model_path}")
    log.info(f"composer_ckpt  : {args.composer_ckpt}")

    t_load = time.perf_counter()
    engine = InpaintInferenceEngine(
        model_path=args.model_path,
        composer_ckpt_path=args.composer_ckpt,
    )
    log.info(f"engine loaded in {time.perf_counter() - t_load:.1f}s")

    selected = {c.strip() for c in args.cases.split(",") if c.strip()}
    cases = [c for c in TEST_CASES if not selected or c["name"] in selected]

    out_records: list[dict] = []
    print()
    print("=" * 76)
    for case in cases:
        print(f"CASE {case['name']}  ({case['lang']})")
        print(f"  word     : {case['inpaint_word']!r}")
        print(f"  phonemes : {case['ph']!r}  (alphabet={case['alphabet']})")
        print(f"  plain    : {case['plain']!r}")
        print(f"  ssml     : {case['ssml']!r}")
        print("-" * 76)

        # --- baseline (no inpaint) ---
        t0 = time.perf_counter()
        base = engine.generate_speech_tokens(
            ssml_or_text=case["plain"],
            lang=case["lang"],
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.no_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            disable_inpaint=True,
        )
        dt_base = time.perf_counter() - t0

        # --- inpaint (composer ON) ---
        t0 = time.perf_counter()
        inp = engine.generate_speech_tokens(
            ssml_or_text=case["ssml"],
            lang=case["lang"],
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.no_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            disable_inpaint=False,
        )
        dt_inp = time.perf_counter() - t0

        # --- compare ---
        div_idx = first_divergence(base.speech_tokens, inp.speech_tokens)
        n_diff, n_cmp = hamming(base.speech_tokens, inp.speech_tokens)
        rng_ok_base = all(0 <= t < 6561 for t in base.speech_tokens)
        rng_ok_inp = all(0 <= t < 6561 for t in inp.speech_tokens)

        print(f"  baseline : n_tokens={len(base.speech_tokens):4d}  "
              f"eos={base.eos_hit}  wall={dt_base:.2f}s  range_ok={rng_ok_base}")
        print(f"  inpaint  : n_tokens={len(inp.speech_tokens):4d}  "
              f"eos={inp.eos_hit}  wall={dt_inp:.2f}s  range_ok={rng_ok_inp}  "
              f"phone_positions={sum(inp.phone_mask)}  aligned={inp.n_phonemes_aligned}")
        print(f"  diff     : first_divergence_at={div_idx}  "
              f"differ={n_diff}/{n_cmp}  ({100*n_diff/max(1,n_cmp):.1f}% of shared prefix)")

        # Verdict
        verdict_parts = []
        if not rng_ok_inp:
            verdict_parts.append("FAIL: inpaint output has tokens out of [0, 6561)")
        if inp.n_phonemes_aligned == 0:
            verdict_parts.append("FAIL: no phoneme spans aligned to BPE tokens")
        if sum(inp.phone_mask) == 0:
            verdict_parts.append("FAIL: phone_mask is all-False (composer didn't fire)")
        if div_idx == -1 and n_diff == 0:
            verdict_parts.append(
                "WARN: outputs identical (composer had no measurable effect — "
                "could be seed coincidence, or composer learned to no-op)"
            )
        if not verdict_parts:
            verdict_parts.append("OK")
        print(f"  verdict  : {'; '.join(verdict_parts)}")

        out_records.append({
            "case": case["name"],
            "lang": case["lang"],
            "alphabet": case["alphabet"],
            "plain": case["plain"],
            "ssml": case["ssml"],
            "baseline_tokens": base.speech_tokens,
            "inpaint_tokens": inp.speech_tokens,
            "phone_mask": inp.phone_mask,
            "phonemes_aligned": inp.n_phonemes_aligned,
            "first_divergence_at": div_idx,
            "n_diff": n_diff,
            "n_cmp": n_cmp,
            "baseline_eos": base.eos_hit,
            "inpaint_eos": inp.eos_hit,
            "verdict": "; ".join(verdict_parts),
        })
        print("=" * 76)

    if args.save_jsonl:
        Path(args.save_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_jsonl, "w") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"wrote {len(out_records)} records to {args.save_jsonl}")


if __name__ == "__main__":
    main()
