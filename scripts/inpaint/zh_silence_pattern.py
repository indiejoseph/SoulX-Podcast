"""Hunt the pattern that triggers zh inpaint silence.

Hypothesis space:
  (a) Override SYLLABLE matters — some phoneme combinations push LLM
      to silence regardless of sentence context.
  (b) SENTENCE matters — some sentences are robust to composer injection,
      others aren't, regardless of which syllable is overridden.
  (c) POSITION in sentence matters — last vs middle vs first.
  (d) Interaction between override content + sentence context.

Approach: build a grid of (sentence × override_syllable × position) and
record per-cell silence/audible. Look for axis-aligned patterns.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.inpaint.inference import InpaintInferenceEngine
from soulxpodcast.utils.infer_utils import initiate_model
from scripts.inpaint.inference_audio import (
    PROMPT_FEMALE, synthesize_one, trim_trailing_silence,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("zh_pattern")


# Six representative zh sentences from the training set, varying in:
# length, topic, phoneme diversity
SENTENCES = [
    ("七路无人售票",            ["qi1", "lu4", "wu2", "ren2", "shou4", "piao4"]),
    ("尊重科学规律的要求",      ["zun1", "zhong4", "ke1", "xue2", "gui1", "lv4", "de5", "yao1", "qiu2"]),
    ("北京天气很好",            ["bei3", "jing1", "tian1", "qi4", "hen3", "hao3"]),
    ("我喜欢吃苹果",            ["wo3", "xi3", "huan1", "chi1", "ping2", "guo3"]),
    ("中国人民解放军",          ["zhong1", "guo2", "ren2", "min2", "jie3", "fang4", "jun1"]),
    ("时间过得真快",            ["shi2", "jian1", "guo4", "de5", "zhen1", "kuai4"]),
]


def split_pinyin(syl: str) -> tuple[str, str]:
    INITIALS = ["zh", "ch", "sh",
                "b", "c", "d", "f", "g", "h", "j", "k", "l",
                "m", "n", "p", "q", "r", "s", "t", "w", "x", "y", "z"]
    for ini in INITIALS:
        if syl.startswith(ini):
            return ini, syl[len(ini):]
    return "", syl


def build_ssml(text: str, char_idx: int, ph: str) -> str:
    return (text[:char_idx]
            + f'<phoneme alphabet="pinyin" ph="{ph}">{text[char_idx]}</phoneme>'
            + text[char_idx + 1:])


@torch.no_grad()
def run_one(soulx_model, dataset, engine, text, ssml, name, out_dir):
    wav, dbg = synthesize_one(
        soulx_model, dataset, engine,
        prompt_audio=PROMPT_FEMALE["audio"], prompt_text=PROMPT_FEMALE["text"],
        ssml_or_text=ssml, lang="zh",
        max_new_tokens=200, do_sample=False,
        temperature=0.8, top_p=0.95, repetition_penalty=1.1,
        seed=42, fp16_flow=True,
    )
    w2 = wav.squeeze(0) if wav.dim() == 3 else wav
    w2 = trim_trailing_silence(w2, 24000, silence_db=-40.0)
    torchaudio.save(str(out_dir / f"{name}.wav"), w2.float(), 24000)
    peak = float(w2.abs().max())
    peak_db = 20 * math.log10(peak + 1e-12)
    return {
        "name": name, "n_gen": dbg["n_generated_tokens"],
        "eos": dbg["eos_hit"], "peak_db": peak_db,
        "silent": peak_db < -50.0,
    }


def main():
    out_dir = Path("outputs/inpaint_zh_pattern")
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

    # Grid: for each (sentence, char_position), inject the NATURAL syllable.
    # This isolates whether silence depends on which char/position is overridden
    # for each sentence — without confounding with "wrong phoneme content".
    results = []
    for sent_idx, (text, phs) in enumerate(SENTENCES):
        cjk_chars = [(i, c) for i, c in enumerate(text)
                     if 0x4e00 <= ord(c) <= 0x9fff]
        if len(cjk_chars) != len(phs):
            log.warning(f"skipping {text!r}: char/phon mismatch")
            continue

        # Also run baseline (no inpaint) as control
        name_b = f"s{sent_idx}_baseline"
        log.info(f"\n[{sent_idx}] {text!r}  (baseline)")
        r = run_one(soulx_model, dataset, engine, text, text, name_b, out_dir)
        r["sent"] = text; r["pos"] = -1; r["syl"] = None
        results.append(r)

        # For each CJK position, do a NATURAL-syllable inpaint
        for (ci, ch), syl in zip(cjk_chars, phs):
            ini, fin = split_pinyin(syl)
            ph_string = f"{ini} {fin}".strip()
            ssml = build_ssml(text, ci, ph_string)
            name = f"s{sent_idx}_pos{ci}_{ch}_{syl}"
            log.info(f"[{sent_idx}] pos {ci} ({ch}={syl}) → '{ph_string}'")
            r = run_one(soulx_model, dataset, engine, text, ssml, name, out_dir)
            r["sent"] = text; r["pos"] = ci; r["syl"] = syl
            results.append(r)

    # Per-sentence silence rate
    log.info("\n" + "=" * 80)
    log.info("PER-SENTENCE SILENCE RATE (excluding baseline):")
    for sent_idx, (text, _) in enumerate(SENTENCES):
        per_sent = [r for r in results if r["sent"] == text and r["pos"] >= 0]
        if not per_sent: continue
        n_silent = sum(1 for r in per_sent if r["silent"])
        log.info(f"  s{sent_idx} {text!r:30s}  {n_silent}/{len(per_sent)} silent")

    # Per-position: does override position matter within a sentence?
    log.info("\nPER-POSITION VIEW (silent? rows = sentences, cols = positions):")
    for sent_idx, (text, phs) in enumerate(SENTENCES):
        per_sent = [r for r in results if r["sent"] == text and r["pos"] >= 0]
        marks = []
        for r in per_sent:
            marks.append("✗" if r["silent"] else "✓")
        log.info(f"  s{sent_idx} {text!r:30s}  {' '.join(marks)}")

    # Detail table
    log.info("\nDETAIL:")
    log.info(f"  {'name':40s} {'n_gen':>6s} {'eos':>6s} {'peak_dB':>9s} {'silent':>8s}")
    for r in results:
        log.info(f"  {r['name']:40s} {r['n_gen']:>6} {str(r['eos']):>6s} "
                 f"{r['peak_db']:+8.1f}  {'✗' if r['silent'] else '✓'}")


if __name__ == "__main__":
    main()
