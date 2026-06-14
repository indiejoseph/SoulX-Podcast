"""Mandarin inpaint test on REAL training-distribution samples.

Hypothesis to disprove: "the composer's silence-on-Pinyin is caused by
my synthesised test text being OOD". To test, pick zh rows directly
from the training set (guaranteed in-distribution by construction),
override one syllable each, and observe baseline + inpaint audio.

If baseline produces clean Mandarin AND inpaint produces silence on
these in-distribution samples, then the silence is purely a composer
bug — not a test-input artifact.

Each case overrides the LAST syllable of the sentence so we can hear:
  baseline:  full sentence as the model naturally pronounces it
  inpaint:   same sentence up to the override, then whatever the
             composer produces for the overridden final syllable
"""

from __future__ import annotations

import json
import logging
import sys
import time
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inference_audio_zh_real")


# Real zh rows from tmp/dataset.jsonl — guaranteed in-distribution.
# For each case we override the LAST syllable with either:
#   - the natural reading (correct), as a no-op control
#   - a different valid Pinyin syllable (wrong), to test composer override
REAL_ZH_CASES = [
    {
        "name":       "zh_real_seven_road",
        "text":       "七路无人售票",
        "phonemes":   ["qi1", "lu4", "wu2", "ren2", "shou4", "piao4"],
        # Override last char 票 — natural reading is piao4. Try a wrong
        # syllable for a clear A/B.
        "override_char_idx": 5,         # 0-indexed char position
        "override_correct":   "p iao4", # split form of natural piao4
        "override_wrong":     "ch ang2",# clearly different Pinyin
    },
    {
        "name":       "zh_real_obey",
        "text":       "尊重科学规律的要求",
        "phonemes":   ["zhun1", "zhong4", "ke1", "xue2", "gui1", "lv4", "de5", "yao1", "qiu2"],
        "override_char_idx": 8,
        "override_correct":   "q iu2",
        "override_wrong":     "x ing2",
    },
    {
        "name":       "zh_real_son_phone",
        "text":       "儿子趁机玩儿爸爸的手机",
        "phonemes":   ["er2", "zi5", "chen4", "ji1", "wanr2", "ba4", "ba5", "de5", "shou3", "ji1"],
        # Last char 机. Note: 玩儿 is one phoneme 'wanr2' but two chars
        # in text — so phoneme index 4 covers chars 4-5 in the text.
        "override_char_idx": 10,        # 机 (last char) in the 11-char text
        "override_correct":   "j i1",
        "override_wrong":     "b an4",
    },
]


def build_ssml(text: str, char_idx: int, ph: str) -> str:
    """Wrap a single character with a <phoneme> tag."""
    before = text[:char_idx]
    target = text[char_idx]
    after = text[char_idx + 1 :]
    return f'{before}<phoneme alphabet="pinyin" ph="{ph}">{target}</phoneme>{after}'


def main():
    out_dir = Path("outputs/inpaint_audio_zh_real")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading SoulXPodcast (flow + HiFT + dataset)")
    soulx_model, dataset = initiate_model(
        seed=42,
        model_path="/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged",
        llm_engine="hf",
        fp16_flow=True,
    )
    try:
        del soulx_model.llm
        torch.cuda.empty_cache()
        log.info("freed SoulXPodcast.llm")
    except AttributeError:
        pass

    log.info("loading InpaintInferenceEngine")
    engine = InpaintInferenceEngine(
        model_path="/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged",
        composer_ckpt_path="outputs/inpaint_final/step_0030000/composer.pt",
    )

    summary: list[dict] = []
    for case in REAL_ZH_CASES:
        text = case["text"]
        # Build three variants: baseline (no SSML), inpaint-correct, inpaint-wrong.
        variants = [
            ("baseline",         text),
            ("inpaint_correct",  build_ssml(text, case["override_char_idx"], case["override_correct"])),
            ("inpaint_wrong",    build_ssml(text, case["override_char_idx"], case["override_wrong"])),
        ]
        for variant_name, content in variants:
            name = f"{case['name']}_{variant_name}"
            log.info(f"--- {name} ---  text={content!r}")
            wav, debug = synthesize_one(
                soulx_model, dataset, engine,
                prompt_audio=PROMPT_FEMALE["audio"],
                prompt_text=PROMPT_FEMALE["text"],
                ssml_or_text=content,
                lang="zh",
                max_new_tokens=200,
                do_sample=False,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                seed=42,
                fp16_flow=True,
            )
            wav_2d = wav.squeeze(0) if wav.dim() == 3 else wav
            wav_2d = trim_trailing_silence(wav_2d, 24000, silence_db=-40.0)
            dur_sec = wav_2d.shape[-1] / 24000.0
            out_path = out_dir / f"{name}.wav"
            torchaudio.save(str(out_path), wav_2d.float(), 24000)
            log.info(
                f"  ✓ {out_path}  dur={dur_sec:.2f}s  "
                f"gen_toks={debug['n_generated_tokens']}  eos={debug['eos_hit']}  "
                f"phone_pos={debug['phone_positions']}"
            )
            summary.append({
                "name": name, "case": case["name"], "variant": variant_name,
                "text": content, "dur_sec": dur_sec, **debug,
            })

    log.info("=" * 64)
    log.info("Summary:")
    for s in summary:
        log.info(f"  {s['name']:42s} gen={s['n_generated_tokens']:4d} eos={s['eos_hit']} dur={s['dur_sec']:.2f}s")


if __name__ == "__main__":
    main()
