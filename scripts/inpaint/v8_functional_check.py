"""v8 functional check — does silence-masked CE + no-LN composer break
the v4-v7 zh mode-collapse AND respect SSML phoneme overrides?

For each of {zh, yue, en} we run two greedy decodes from the inpaint
engine against the v8 step_0030000 composer:

  (a) BASELINE  : plain text, composer effectively OFF (no <phoneme>).
                  Tests that the composer doesn't actively break the
                  default text→speech path.
  (b) OVERRIDE  : SSML with a single <phoneme> annotation on a chosen
                  word, composer ON at those slots only. Tests that
                  the composer respects the override (output diverges
                  from baseline at the right place).

Per case we report:

  * n_gen, n_unique, top_id @ top_frac, eos_hit
  * mode-collapse flag (top_frac > 0.6 AND n_gen >= 30)
  * first divergence index between baseline and override
  * audio file (synthesised through flow + HiFT, prompt voice = matched lang)

Text format follows the training distribution
([[inference-text-must-match-training-format]]):

  * en   : ALL CAPS, no terminal punct
  * zh   : no punct (Pinyin alphabet)
  * yue  : with punct (Jyutping alphabet)
"""

from __future__ import annotations

import logging
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.inpaint.inference import InpaintInferenceEngine
from soulxpodcast.utils.infer_utils import initiate_model
from scripts.inpaint.inference_audio import (
    PROMPT_FEMALE, PROMPT_MALE, synthesize_one, trim_trailing_silence,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("v8_check")

MODEL_PATH = "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged"
V8_CKPT = "outputs/inpaint_h100_v8_silencemasked_no_ln/step_0030000/composer.pt"
OUT_DIR = Path("outputs/inpaint_v8_functional")


# ---------------------------------------------------------------------
# Test cases. text format follows training distribution exactly.
# ---------------------------------------------------------------------

CASES = [
    # ---- zh (the critical case — v4-v7 collapsed on this) ----
    {
        "name": "zh_zhuyi", "lang": "zh", "prompt": "female",
        "plain": "请注意听",
        "ssml":  '请<phoneme alphabet="pinyin" ph="zh u4 y i4">注意</phoneme>听',
        "note":  "common zh phrase, 'pay attention'",
    },
    {
        "name": "zh_dianhua", "lang": "zh", "prompt": "female",
        "plain": "拨打这个电话",
        "ssml":  '拨打这个<phoneme alphabet="pinyin" ph="d ian4 h ua4">电话</phoneme>',
        "note":  "the classic v6 overfit row",
    },
    {
        "name": "zh_kexue", "lang": "zh", "prompt": "female",
        "plain": "尊重科学规律的要求",
        "ssml":  '尊重<phoneme alphabet="pinyin" ph="k e1 x ue2">科学</phoneme>规律的要求',
        "note":  "AISHELL-3 row, v6 overfit target",
    },

    # ---- yue (known-working since v4 — sanity baseline) ----
    {
        "name": "yue_keoi5", "lang": "yue", "prompt": "female",
        "plain": "我同佢去飲茶。",
        "ssml":  '我同<phoneme alphabet="jyutping" ph="k eoi5">佢</phoneme>去飲茶。',
        "note":  "common yue word",
    },
    {
        "name": "yue_jat1", "lang": "yue", "prompt": "female",
        "plain": "今日天氣好好。",
        "ssml":  '今日<phoneme alphabet="jyutping" ph="t in1 hei3">天氣</phoneme>好好。',
        "note":  "two-char override",
    },

    # ---- en (borderline — HK-LoRA backbone weakness) ----
    {
        "name": "en_world", "lang": "en", "prompt": "male",
        "plain": "HELLO WORLD TODAY",
        "ssml":  'HELLO <phoneme alphabet="cmu" ph="W ER L D">WORLD</phoneme> TODAY',
        "note":  "short common word",
    },
    {
        "name": "en_doctor", "lang": "en", "prompt": "male",
        "plain": "MY FATHER IS A DOCTOR",
        "ssml":  'MY FATHER IS A <phoneme alphabet="cmu" ph="D AA K T ER">DOCTOR</phoneme>',
        "note":  "2-syllable",
    },
]


PROMPTS = {"female": PROMPT_FEMALE, "male": PROMPT_MALE}


def diversity_stats(tokens: list[int]) -> dict:
    n = len(tokens)
    if n == 0:
        return {"n": 0, "n_unique": 0, "top_id": -1, "top_frac": 0.0,
                "collapsed": False}
    c = Counter(tokens)
    top_id, top_n = c.most_common(1)[0]
    top_frac = top_n / n
    return {
        "n": n, "n_unique": len(c),
        "top_id": int(top_id), "top_frac": top_frac,
        "collapsed": (top_frac > 0.6 and n >= 30),
    }


def first_div_idx(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"loading SoulXPodcast (flow + HiFT + audio_tokenizer)")
    soulx_model, dataset = initiate_model(
        seed=42, model_path=MODEL_PATH, llm_engine="hf", fp16_flow=True,
    )
    try:
        del soulx_model.llm
        torch.cuda.empty_cache()
    except AttributeError:
        pass

    log.info(f"loading v8 composer: {V8_CKPT}")
    engine = InpaintInferenceEngine(
        model_path=MODEL_PATH, composer_ckpt_path=V8_CKPT,
    )

    rows = []
    log.info(f"\n{'='*72}\nrunning {len(CASES)} cases\n{'='*72}")
    for case in CASES:
        name, lang = case["name"], case["lang"]
        prompt = PROMPTS[case["prompt"]]
        log.info(f"\n[{name}] lang={lang}  ({case['note']})")
        log.info(f"  plain: {case['plain']!r}")
        log.info(f"  ssml : {case['ssml']!r}")

        # Token-only generation through the inpaint engine
        out_plain = engine.generate_speech_tokens(
            ssml_or_text=case["plain"], lang=lang,
            max_new_tokens=200, do_sample=False, seed=42,
            disable_inpaint=True,
        )
        out_ssml = engine.generate_speech_tokens(
            ssml_or_text=case["ssml"], lang=lang,
            max_new_tokens=200, do_sample=False, seed=42,
        )
        dp = diversity_stats(out_plain.speech_tokens)
        di = diversity_stats(out_ssml.speech_tokens)
        first_div = first_div_idx(out_plain.speech_tokens, out_ssml.speech_tokens)
        log.info(f"  BASELINE: n={dp['n']:>3d}  unique={dp['n_unique']:>3d}  "
                 f"top={dp['top_id']:>5d}@{dp['top_frac']:5.1%}  "
                 f"eos={out_plain.eos_hit!s:>5s}  "
                 f"{'✗ COLLAPSE' if dp['collapsed'] else '✓ OK'}")
        log.info(f"  OVERRIDE: n={di['n']:>3d}  unique={di['n_unique']:>3d}  "
                 f"top={di['top_id']:>5d}@{di['top_frac']:5.1%}  "
                 f"eos={out_ssml.eos_hit!s:>5s}  "
                 f"{'✗ COLLAPSE' if di['collapsed'] else '✓ OK'}")
        log.info(f"  first divergence: idx={first_div}  "
                 f"({'no divergence' if first_div == -1 else 'good — composer changes output'})")

        # Synthesize audio for listening
        try:
            for tag, txt in [("baseline", case["plain"]), ("override", case["ssml"])]:
                wav, dbg = synthesize_one(
                    soulx_model, dataset, engine,
                    prompt_audio=prompt["audio"], prompt_text=prompt["text"],
                    ssml_or_text=txt, lang=lang,
                    max_new_tokens=200, do_sample=False,
                    temperature=0.8, top_p=0.95, repetition_penalty=1.1,
                    seed=42, fp16_flow=True,
                )
                w2 = wav.squeeze(0) if wav.dim() == 3 else wav
                w2 = trim_trailing_silence(w2, 24000, silence_db=-40.0)
                peak_db = 20 * math.log10(max(float(w2.abs().max()), 1e-12))
                out_path = OUT_DIR / f"{name}_{tag}.wav"
                torchaudio.save(str(out_path), w2.float(), 24000)
                log.info(f"  → {out_path}  ({w2.shape[-1]/24000:.2f}s peak={peak_db:+.1f}dB)")
        except Exception as e:
            log.error(f"  audio synth failed: {e}")

        rows.append({"case": name, "lang": lang,
                    "baseline_collapsed": dp["collapsed"],
                    "baseline_top_frac": dp["top_frac"],
                    "override_collapsed": di["collapsed"],
                    "override_top_frac": di["top_frac"],
                    "first_div": first_div, "n_baseline": dp["n"],
                    "n_override": di["n"]})

    # ---- Summary ----
    log.info(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    log.info(f"  {'case':<14s} {'lang':<4s} {'base_n':>6s} {'base_top%':>10s} "
             f"{'over_n':>6s} {'over_top%':>10s} {'div_idx':>8s} {'verdict':<12s}")
    n_pass = 0
    for r in rows:
        verdict = "✓ ok" if (not r["baseline_collapsed"]
                             and not r["override_collapsed"]
                             and r["first_div"] != -1) else "✗ FAIL"
        if verdict == "✓ ok":
            n_pass += 1
        log.info(f"  {r['case']:<14s} {r['lang']:<4s} {r['n_baseline']:>6d} "
                 f"{r['baseline_top_frac']:>9.1%} {r['n_override']:>6d} "
                 f"{r['override_top_frac']:>9.1%} {r['first_div']:>8d} "
                 f"{verdict:<12s}")
    log.info(f"\n  PASS: {n_pass}/{len(rows)}")
    log.info(f"  audio written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
