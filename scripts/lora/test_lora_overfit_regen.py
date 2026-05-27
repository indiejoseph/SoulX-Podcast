"""Strong overfit verification: regenerate audio from the actual 16 training
samples' text via the LoRA-merged pipeline.

For each test sample:
  - Use the SAME text the LoRA was trained on
  - Use a standard voice prompt (female_mandarin)
  - Generate audio
  - Compare wall duration to expected training duration

If the LoRA correctly memorized the samples (top1=1.0 was reached in training),
inference on that same text should produce intelligible audio close to the
expected duration. If even that's garbled, the pipeline has a remaining bug.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import time
from pathlib import Path

import torch
import torchaudio
from datasets import load_from_disk
from peft import PeftModel

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser

BASE = "pretrained_models/SoulX-Podcast-1.7B-dialect"
ADAPTER = "runs/overfit_smoke_lora/adapter"
DATASET = "tmp/dataset_small_with_tokens"
OUT = Path("outputs/overfit_regen")

# Standard voice prompt — same one used in mtp_audio_ab.py / test_lora_merged.py
S1_PROMPT_WAV = "example/audios/female_mandarin.wav"
S1_PROMPT_TEXT = "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。"
S1_DIALECT_PROMPT = "<|Yue|>真係冇讲错啊！攀山滑雪嘅语言专家几巴闭，都唔及我听日拖成副身家去景德镇玩泥巴，呢铺真系发哂白日梦咯！"

# Same dialect-prefix mapping as MtpDataset._apply_dialect_prefix
DIALECT_PREFIX = {"yue": "<|Yue|>"}


def maybe_with_prefix(text: str, lang: str) -> str:
    """Match MtpDataset's training-time text transform."""
    prefix = DIALECT_PREFIX.get(lang, "")
    return f"{prefix}{text}" if prefix else text


def make_data(text: str, lang: str, dh):
    """Build inference input for one prompt+target pair."""
    pd = {"prompt_audio": Path(S1_PROMPT_WAV), "prompt_text": S1_PROMPT_TEXT}
    # For Cantonese targets, include the dialect prompt the inference path expects
    if lang == "yue":
        pd["dialect_prompt"] = S1_DIALECT_PROMPT
    speakers = {"S1": pd}
    parsed = podcast_format_parser({"speakers": speakers, "text": [["S1", maybe_with_prefix(text, lang)]]})
    prepared = process_single_input(
        dh, parsed["text"], parsed["prompt_wav"], parsed["prompt_text"],
        parsed["use_dialect_prompt"], parsed["dialect_prompt_text"],
    )
    return prepared


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[load] base SoulX pipeline")
    sp_model, dh = initiate_model(seed=42, model_path=BASE, llm_engine="hf", fp16_flow=True)

    print(f"[merge] applying LoRA from {ADAPTER}")
    peft_model = PeftModel.from_pretrained(sp_model.llm.model, ADAPTER)
    sp_model.llm.model = peft_model.merge_and_unload()
    print(f"[merge] done")

    print(f"[ds] loading {DATASET}")
    ds = load_from_disk(DATASET).remove_columns(["audio"])

    # Run on first 4 training samples (a subset of the 16 the LoRA was trained on)
    for idx in [0, 1, 2, 3]:
        s = ds[idx]
        text = s["text"]
        lang = s["lang"]
        n_speech_train = len(s["speech_tokens"].split())
        expected = n_speech_train / 25.0

        print(f"\n=== sample {idx} (lang={lang}) expected ~{expected:.2f}s of audio ===")
        print(f"    text: {text!r}")

        prepared = make_data(text, lang, dh)
        t0 = time.perf_counter()
        result = sp_model.forward_longform(**prepared)
        dt = time.perf_counter() - t0
        wav = result["generated_wavs"][0].cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        asec = wav.shape[-1] / 24000
        path = OUT / f"sample{idx}_{lang}_overfit_regen.wav"
        torchaudio.save(str(path), wav, 24000)
        ratio = asec / max(expected, 0.01)
        marker = "✓" if 0.5 <= ratio <= 2.5 else "⚠"
        print(f"    {marker} generated {asec:.2f}s in {dt:.2f}s wall  "
              f"(expected ~{expected:.2f}s, ratio {ratio:.2f}x)  → {path}")


if __name__ == "__main__":
    main()
