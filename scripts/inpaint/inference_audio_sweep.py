"""Per-syllable inpaint silence sweep.

Tests whether silent inpaint output correlates with:
  (a) zh-specific annotation issues (z/zh confusion)
  (b) rare (initial, final) PAIR frequency (any language)
  (c) sentence properties (length, voice prompt mismatch)

For each test case we override ONE syllable, record token count, EOS,
audio amplitude. A silent output (peak < -50 dB) flags the case.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.inpaint.inference import InpaintInferenceEngine
from soulxpodcast.inpaint.tokenizer import PhonemeTokenizer
from soulxpodcast.utils.infer_utils import initiate_model
from scripts.inpaint.inference_audio import (
    PROMPT_FEMALE, synthesize_one, trim_trailing_silence,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("sweep")


def compute_pair_frequencies():
    """Per-language (initial, final) PAIR frequencies from training corpus."""
    tok = PhonemeTokenizer()
    zh_pairs, yue_pairs = Counter(), Counter()
    with open("tmp/dataset.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["lang"] == "zh":
                for p in r["phonemes"]:
                    if not p or not p[-1:].isdigit(): continue
                    i, fin = tok.split_whole_syllable("pinyin", p)
                    zh_pairs[(i, fin)] += 1
            elif r["lang"] == "yue":
                for p in r["phonemes"]:
                    if not p or not p[-1:].isdigit(): continue
                    i, fin = tok.split_whole_syllable("jyutping", p)
                    yue_pairs[(i, fin)] += 1
    return zh_pairs, yue_pairs


# ===== Test cases =====
# zh: pick chars where we know the pair frequency
ZH_CASES = [
    # (text, char_idx, override_correct, override_wrong, note)
    ("七路无人售票",      5, "p iao4",  "ch ang2", "all common pairs"),
    ("尊重科学规律的要求", 8, "q iu2",   "x ing2",  "obey — fails in v3"),
    ("北京天气很好",       1, "j ing1",  "x ue2",   "common Beijing word"),
    ("我喜欢吃苹果",       4, "p ing2",  "h ua1",   "common everyday"),
    ("中国人民解放军",     6, "j un1",   "l ai2",   "compound common"),
]

# yue: pick chars from the dataset with KNOWN rare pairs
# We'll compute and select 3-5 rows below where the override pair is rare
YUE_RARE_TEST_CASES = []  # populated dynamically


def find_yue_rare_test_cases(yue_pairs, n_rare=3, n_common=2):
    """Find yue rows whose syllables include rare pairs (≤5 occurrences)."""
    rare_pairs = {p for p, c in yue_pairs.items() if c <= 5}
    common_test_added = 0
    rare_test_added = 0
    cases = []
    tok = PhonemeTokenizer()
    with open("tmp/dataset.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["lang"] != "yue":
                continue
            if "phonemes" not in r:
                continue
            phs = [p for p in r["phonemes"] if p[-1:].isdigit()]
            if not (4 <= len(phs) <= 10):
                continue
            text = r["text"].replace(" ", "")
            # Walk syllables, find first CJK char with rare-pair syllable
            cjk_chars = [(i, c) for i, c in enumerate(text)
                         if 0x4e00 <= ord(c) <= 0x9fff]
            if len(cjk_chars) != len(phs):
                continue  # skip alignment mismatches (erhua etc.)
            for syl_idx, ((ci, ch), syl) in enumerate(zip(cjk_chars, phs)):
                pair = tok.split_whole_syllable("jyutping", syl)
                pair_freq = yue_pairs.get(pair, 0)
                if pair in rare_pairs and rare_test_added < n_rare:
                    cases.append({
                        "text": text, "char_idx": ci, "syl": syl,
                        "pair": pair, "pair_freq": pair_freq,
                        "tag": "yue_rare",
                    })
                    rare_test_added += 1
                    break
                if pair_freq > 5000 and common_test_added < n_common:
                    cases.append({
                        "text": text, "char_idx": ci, "syl": syl,
                        "pair": pair, "pair_freq": pair_freq,
                        "tag": "yue_common",
                    })
                    common_test_added += 1
                    break
            if rare_test_added >= n_rare and common_test_added >= n_common:
                break
    return cases


def build_ssml(text: str, char_idx: int, alphabet: str, ph: str) -> str:
    before = text[:char_idx]
    target = text[char_idx]
    after = text[char_idx + 1 :]
    return f'{before}<phoneme alphabet="{alphabet}" ph="{ph}">{target}</phoneme>{after}'


@torch.no_grad()
def run_case(soulx_model, dataset, engine, lang, text, ssml, name, prompt):
    wav_b, dbg_b = synthesize_one(
        soulx_model, dataset, engine,
        prompt_audio=prompt["audio"], prompt_text=prompt["text"],
        ssml_or_text=text, lang=lang,
        max_new_tokens=200, do_sample=False,
        temperature=0.8, top_p=0.95, repetition_penalty=1.1,
        seed=42, fp16_flow=True,
    )
    wav_i, dbg_i = synthesize_one(
        soulx_model, dataset, engine,
        prompt_audio=prompt["audio"], prompt_text=prompt["text"],
        ssml_or_text=ssml, lang=lang,
        max_new_tokens=200, do_sample=False,
        temperature=0.8, top_p=0.95, repetition_penalty=1.1,
        seed=42, fp16_flow=True,
    )
    out_dir = Path("outputs/inpaint_sweep_v4")
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, wav, dbg in [("baseline", wav_b, dbg_b), ("inpaint", wav_i, dbg_i)]:
        wav_2d = wav.squeeze(0) if wav.dim() == 3 else wav
        wav_2d = trim_trailing_silence(wav_2d, 24000, silence_db=-40.0)
        torchaudio.save(str(out_dir / f"{name}_{tag}.wav"), wav_2d.float(), 24000)
    # Audit
    def peak_db(wav):
        w = wav.squeeze(0) if wav.dim() == 3 else wav
        peak = float(w.abs().max())
        return 20 * math.log10(peak + 1e-12)
    return {
        "name": name,
        "baseline_eos": dbg_b["eos_hit"], "baseline_peak": peak_db(wav_b),
        "inpaint_eos":  dbg_i["eos_hit"], "inpaint_peak":  peak_db(wav_i),
        "silent": peak_db(wav_i) < -50.0,
    }


def main():
    log.info("computing pair frequencies from training corpus...")
    zh_pairs, yue_pairs = compute_pair_frequencies()
    log.info(f"zh pairs total: {len(zh_pairs)}  yue pairs: {len(yue_pairs)}")

    # Annotate zh test cases with pair frequency for the override syllable
    tok = PhonemeTokenizer()
    for i, case in enumerate(ZH_CASES):
        text, ci, ph_c, ph_w, note = case
        target = text[ci]
        pair_c = tuple(ph_c.split())
        pair_w = tuple(ph_w.split())
        # Map back to (initial, final) for frequency lookup
        # ph string is split-form like "q iu2" → ("q", "iu2")
        freq_c = zh_pairs.get((pair_c[0], pair_c[1]), 0)
        freq_w = zh_pairs.get((pair_w[0], pair_w[1]), 0)
        log.info(f"  zh[{i}] '{text}' char {target} (idx {ci})  "
                 f"correct=({pair_c[0]},{pair_c[1]}):{freq_c}  "
                 f"wrong=({pair_w[0]},{pair_w[1]}):{freq_w}  -- {note}")

    yue_cases = find_yue_rare_test_cases(yue_pairs, n_rare=3, n_common=2)
    log.info(f"\nyue test cases:")
    for case in yue_cases:
        log.info(f"  yue[{case['tag']}] '{case['text']}' char {case['text'][case['char_idx']]} "
                 f"(idx {case['char_idx']})  syl={case['syl']}  pair={case['pair']}  "
                 f"pair_freq={case['pair_freq']}")

    log.info("\nloading SoulXPodcast + inpaint engine")
    soulx_model, dataset = initiate_model(
        seed=42,
        model_path="/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged",
        llm_engine="hf", fp16_flow=True,
    )
    try:
        del soulx_model.llm; torch.cuda.empty_cache()
    except AttributeError: pass
    engine = InpaintInferenceEngine(
        model_path="/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged",
        composer_ckpt_path="outputs/inpaint_h100_v4_concat/step_0030000/composer.pt",
    )

    results = []
    log.info("\n=== zh sweep ===")
    for i, case in enumerate(ZH_CASES):
        text, ci, ph_c, ph_w, note = case
        # Correct override
        ssml = build_ssml(text, ci, "pinyin", ph_c)
        r = run_case(soulx_model, dataset, engine, "zh", text, ssml,
                     f"zh_{i}_correct", PROMPT_FEMALE)
        r["override_pair"] = ph_c
        r["pair_freq"] = zh_pairs.get(tuple(ph_c.split()), 0)
        results.append(r)
        # Wrong override
        ssml = build_ssml(text, ci, "pinyin", ph_w)
        r = run_case(soulx_model, dataset, engine, "zh", text, ssml,
                     f"zh_{i}_wrong", PROMPT_FEMALE)
        r["override_pair"] = ph_w
        r["pair_freq"] = zh_pairs.get(tuple(ph_w.split()), 0)
        results.append(r)

    log.info("\n=== yue sweep ===")
    for i, case in enumerate(yue_cases):
        # For yue, just use the natural syllable (no wrong test needed —
        # we want to see if the natural pronunciation triggers silence).
        ini, fin = case["pair"]
        ph = f"{ini} {fin}".strip()
        ssml = build_ssml(case["text"], case["char_idx"], "jyutping", ph)
        r = run_case(soulx_model, dataset, engine, "yue", case["text"], ssml,
                     f"yue_{case['tag']}_{i}", PROMPT_FEMALE)
        r["pair_freq"] = case["pair_freq"]
        r["tag"] = case["tag"]
        results.append(r)

    log.info("\n" + "=" * 80)
    log.info("RESULTS:")
    log.info(f"  {'name':30s} {'pair_freq':>10s} {'inp_peak':>10s} {'inp_eos':>8s} silent?")
    for r in results:
        log.info(f"  {r['name']:30s} {r.get('pair_freq', '-'):>10}  {r['inpaint_peak']:+8.1f} dB  "
                 f"{str(r['inpaint_eos']):>8s}  {'✗ SILENT' if r['silent'] else 'OK'}")


if __name__ == "__main__":
    main()
