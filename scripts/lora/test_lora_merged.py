"""LoRA merged audio test — base vs final-merged vs SWA-average-merged.

Generates the same 3 test cases as mtp_audio_ab.py, but compares:
  - base SoulX trunk
  - LoRA final adapter merged into trunk
  - SWA average (4 late adapters) merged into trunk

No MTP — this isolates the LoRA style transfer.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
import torchaudio
from peft import PeftModel

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser


BASE_MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect"
ADAPTERS_DIR = Path("runs/hpc/adaptors")

S1_PROMPT_WAV = "example/audios/female_mandarin.wav"
S1_PROMPT_TEXT = "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。"
S1_DIALECT_PROMPT = "<|Yue|>真係冇讲错啊！攀山滑雪嘅语言专家几巴闭，都唔及我听日拖成副身家去景德镇玩泥巴，呢铺真系发哂白日梦咯！"

TEST_CASES = [
    (
        "english_short",
        "Hello everyone, welcome to our show.",
        {"prompt_audio": S1_PROMPT_WAV, "prompt_text": S1_PROMPT_TEXT},
    ),
    (
        "cantonese_short",
        "<|Yue|>哈囉大家好啊，歡迎收聽我哋嘅節目。",
        {
            "prompt_audio": S1_PROMPT_WAV,
            "prompt_text": S1_PROMPT_TEXT,
            "dialect_prompt": S1_DIALECT_PROMPT,
        },
    ),
    (
        "mandarin_medium",
        "今天天气真好，我们一起出去走走吧。听说附近新开了一家咖啡店，环境很不错。",
        {"prompt_audio": S1_PROMPT_WAV, "prompt_text": S1_PROMPT_TEXT},
    ),
]


def make_data(name, target_text, prompt_dict, dataset_handler):
    speakers = {"S1": {**prompt_dict, "prompt_audio": Path(prompt_dict["prompt_audio"])}}
    data = {"speakers": speakers, "text": [["S1", target_text]]}
    parsed = podcast_format_parser(data)
    prepared = process_single_input(
        dataset_handler,
        parsed["text"],
        parsed["prompt_wav"],
        parsed["prompt_text"],
        parsed["use_dialect_prompt"],
        parsed["dialect_prompt_text"],
    )
    return prepared


@torch.inference_mode()
def gen_wav(model, prepared):
    t0 = time.perf_counter()
    result = model.forward_longform(**prepared)
    dt = time.perf_counter() - t0
    wav = result["generated_wavs"][0].cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav, dt


def save_wav(wav, path, sr=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav, sr)


def build_pipeline():
    sp_model, dataset_handler = initiate_model(
        seed=42, model_path=BASE_MODEL_PATH, llm_engine="hf", fp16_flow=True,
    )
    return sp_model, dataset_handler


def merge_final_adapter(sp_model):
    """Apply final LoRA adapter to trunk in-place. Returns merged HF model."""
    trunk = sp_model.llm.model
    peft_model = PeftModel.from_pretrained(trunk, str(ADAPTERS_DIR / "adapter"))
    merged = peft_model.merge_and_unload()
    sp_model.llm.model = merged
    return merged


def merge_swa(sp_model, weights=None):
    """Average 4 late adapters via peft add_weighted_adapter, then merge."""
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    adapter_names = ["final", "s27k", "s265k", "s26k"]
    adapter_paths = [
        ADAPTERS_DIR / "adapter",
        ADAPTERS_DIR / "adapter_step27000",
        ADAPTERS_DIR / "adapter_step26500",
        ADAPTERS_DIR / "adapter_step26000",
    ]
    trunk = sp_model.llm.model
    peft_model = PeftModel.from_pretrained(trunk, str(adapter_paths[0]), adapter_name=adapter_names[0])
    for nm, pth in zip(adapter_names[1:], adapter_paths[1:]):
        peft_model.load_adapter(str(pth), adapter_name=nm)
    peft_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name="avg",
        combination_type="linear",
    )
    peft_model.set_adapter("avg")
    merged = peft_model.merge_and_unload()
    sp_model.llm.model = merged
    return merged


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_variant(label, out_dir, build_fn, sp_model_existing=None):
    """build_fn: ()-> (sp_model, dataset_handler). label written into wav names."""
    print(f"\n{'='*60}\n=== VARIANT: {label}\n{'='*60}")
    if sp_model_existing is not None:
        sp_model, dataset_handler = sp_model_existing
    else:
        sp_model, dataset_handler = build_fn()

    results = []
    for name, text, pd in TEST_CASES:
        prepared = make_data(name, text, pd, dataset_handler)
        wav, dt = gen_wav(sp_model, prepared)
        audio_sec = wav.shape[-1] / 24000
        rtf = dt / audio_sec
        path = out_dir / f"{name}_{label}.wav"
        save_wav(wav, path)
        print(f"  {name:24s} {dt:6.2f}s wall  {audio_sec:5.2f}s audio  RTF={rtf:.2f}  → {path}")
        results.append((name, dt, audio_sec, rtf))
    return sp_model, dataset_handler, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/lora_merged_ab")
    parser.add_argument("--skip_base", action="store_true")
    parser.add_argument("--skip_final", action="store_true")
    parser.add_argument("--skip_swa", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir.resolve()}")

    all_results = {}

    # --- Variant 1: base ---
    if not args.skip_base:
        sp_model, dataset_handler, res = run_variant("base", out_dir, build_pipeline)
        all_results["base"] = res
        del sp_model
        free_gpu()

    # --- Variant 2: lora final merged ---
    if not args.skip_final:
        print("\n[build] loading fresh pipeline for lora_final...")
        sp_model, dataset_handler = build_pipeline()
        print("[merge] applying final LoRA adapter...")
        merge_final_adapter(sp_model)
        _, _, res = run_variant("lora_final", out_dir, None,
                                sp_model_existing=(sp_model, dataset_handler))
        all_results["lora_final"] = res
        del sp_model
        free_gpu()

    # --- Variant 3: lora swa merged ---
    if not args.skip_swa:
        print("\n[build] loading fresh pipeline for lora_swa...")
        sp_model, dataset_handler = build_pipeline()
        print("[merge] averaging 4 adapters then merging...")
        merge_swa(sp_model)
        _, _, res = run_variant("lora_swa", out_dir, None,
                                sp_model_existing=(sp_model, dataset_handler))
        all_results["lora_swa"] = res
        del sp_model
        free_gpu()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY (wall sec / RTF)")
    print("=" * 60)
    case_names = [c[0] for c in TEST_CASES]
    header = f"{'variant':14s}" + "".join(f"  {n:24s}" for n in case_names)
    print(header)
    for variant, res in all_results.items():
        row = f"{variant:14s}"
        for name, dt, audio_sec, rtf in res:
            row += f"  {dt:6.2f}s/RTF{rtf:5.2f}        "
        print(row)
    print()
    print(f"wavs at {out_dir.resolve()}")
    print("Listen to each prompt across variants — focus on Cantonese for style transfer.")


if __name__ == "__main__":
    main()
