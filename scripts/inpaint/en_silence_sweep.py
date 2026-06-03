"""English inpaint sweep — short, medium, long words across various sentences.

Goal: confirm English doesn't have the same mode-collapse problem as zh.
Test format matches training distribution (ALL CAPS, no trailing punctuation).
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
    PROMPT_MALE, synthesize_one, trim_trailing_silence,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("en_sweep")


# Real English from CMUdict patterns — UPPERCASE, no terminal punct
CASES = [
    # (text, target_word, override_correct, override_wrong, note)
    ("HELLO WORLD TODAY", "WORLD", "W ER L D", "B AA T",
     "short word, common phonemes"),
    ("THE QUICK BROWN FOX JUMPS",  "FOX",   "F AA K S",   "Z UH P",
     "short, 4-phoneme word"),
    ("HE WAS A SCIENTIST",         "SCIENTIST", "S AY AH N T IH S T", "T AY P R AY T ER",
     "long word, 8 phonemes (potentially multi-BPE)"),
    ("SHE STUDIES MATHEMATICS",    "MATHEMATICS", "M AE TH AH M AE T IH K S", "K OW M P UW T ER",
     "very long word, 10 phonemes"),
    ("MY NAME IS JOHN",            "JOHN",  "JH AA N",    "M AA R K",
     "short common name"),
    ("WELCOME TO THE PROGRAM",     "PROGRAM", "P R OW G R AE M", "K AH N S ER T",
     "medium word, multi-syllable"),
    ("MY FATHER IS A DOCTOR",      "DOCTOR", "D AA K T ER", "L AA Y ER",
     "common 2-syllable word"),
    ("INTERNATIONAL ORGANIZATION", "INTERNATIONAL", "IH N T ER N AE SH AH N AH L",
     "K AO N T ER", "very long, 11 phonemes"),
]


def build_ssml(text: str, target_word: str, ph: str) -> str:
    return text.replace(target_word, f'<phoneme alphabet="cmu" ph="{ph}">{target_word}</phoneme>', 1)


def main():
    out_dir = Path("outputs/inpaint_en_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading models...")
    soulx_model, dataset = initiate_model(
        seed=42,
        model_path="/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged",
        llm_engine="hf", fp16_flow=True,
    )
    try: del soulx_model.llm; torch.cuda.empty_cache()
    except AttributeError: pass
    engine = InpaintInferenceEngine(
        model_path="/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged",
        composer_ckpt_path="outputs/inpaint_h100_v4_concat/step_0030000/composer.pt",
    )

    results = []
    for i, (text, target, ph_c, ph_w, note) in enumerate(CASES):
        log.info(f'\n[{i}] {text!r}  target={target!r}  ({note})')
        for variant, ph in [("baseline", None), ("correct", ph_c), ("wrong", ph_w)]:
            content = text if ph is None else build_ssml(text, target, ph)
            out = engine.generate_speech_tokens(
                ssml_or_text=content, lang="en",
                max_new_tokens=300, do_sample=False, seed=42,
            )
            tokens = out.speech_tokens
            n = len(tokens)
            if n > 0:
                c = Counter(tokens)
                top_id, top_n = c.most_common(1)[0]
                top_frac = top_n / n
                unique = len(c)
            else:
                top_id, top_frac, unique = -1, 0.0, 0
            mode_collapsed = top_frac >= 0.6 and n >= 30

            # Synthesize audio (small batch path through flow + HiFT)
            try:
                wav, dbg = synthesize_one(
                    soulx_model, dataset, engine,
                    prompt_audio=PROMPT_MALE["audio"],
                    prompt_text=PROMPT_MALE["text"],
                    ssml_or_text=content, lang="en",
                    max_new_tokens=300, do_sample=False,
                    temperature=0.8, top_p=0.95, repetition_penalty=1.1,
                    seed=42, fp16_flow=True,
                )
                w2 = wav.squeeze(0) if wav.dim() == 3 else wav
                w2 = trim_trailing_silence(w2, 24000, silence_db=-40.0)
                torchaudio.save(str(out_dir / f"en_{i}_{variant}.wav"), w2.float(), 24000)
                peak_db = 20 * math.log10(float(w2.abs().max()) + 1e-12)
            except Exception as e:
                peak_db = float('nan')

            mark = "✗ COLLAPSE" if mode_collapsed else ("✗ NO_EOS" if not out.eos_hit else "✓ OK")
            log.info(f'  {variant:8s}  n={n:>4d}  eos={out.eos_hit!s:>5s}  unique={unique:>3d}  '
                     f'top_id={top_id:>5d}@{top_frac:>5.1%}  peak={peak_db:+5.1f}dB  {mark}')
            results.append({
                "i": i, "variant": variant, "text": text, "n": n,
                "eos": out.eos_hit, "unique": unique,
                "top_id": top_id, "top_frac": top_frac, "peak_db": peak_db,
                "collapsed": mode_collapsed,
            })

    # Summary
    log.info("\n" + "=" * 80)
    log.info("SUMMARY:")
    log.info(f"  case               baseline       correct        wrong")
    for i, (text, target, *_) in enumerate(CASES):
        per = {r['variant']: r for r in results if r['i'] == i}
        def fmt(r):
            if r is None: return "-"
            tag = "COLLAPSE" if r['collapsed'] else ("no-EOS" if not r['eos'] else "OK")
            return f"{r['n']:3d}t {tag:8s}"
        log.info(f"  s{i}: {text[:18]:18s} {fmt(per.get('baseline')):18s} "
                 f"{fmt(per.get('correct')):18s} {fmt(per.get('wrong')):18s}")

    # Mode-collapse counts
    n_collapsed = sum(1 for r in results if r['collapsed'])
    log.info(f"\nmode-collapse cases: {n_collapsed}/{len(results)}")


if __name__ == "__main__":
    main()
