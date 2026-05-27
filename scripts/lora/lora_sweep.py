"""Generate audio for each LoRA adapter checkpoint in runs/hpc/adaptors3/.

Loads the base model once, attaches all adapters into a single PeftModel,
and switches between them with set_adapter() — no base reload between runs.
All checkpoints synthesize the same prompt with the same seed so results
are directly comparable.

Outputs land in outputs/lora_sweep/step_<N>.wav (and step_final.wav for the
unsuffixed `adapter/` checkpoint).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import re
import time
from pathlib import Path

import torch
import torchaudio
from peft import PeftModel

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser

ADAPTER_ROOT = Path("runs/hpc/adaptors3")
OUT_DIR = Path("outputs/lora_sweep")
MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect"
SEED = 198964

DATA = {
    "speakers": {
        "S1": {
            "prompt_audio": "/notebooks/bert-vits2/frankenstein-matcha/ref123.wav",
            "prompt_text": "到咗落車嘅時候，連媽媽都走失埋。",
        }
    },
    "text": [
        ["S1", "<|Yue|>哈囉大家好啊，歡迎收聽我哋嘅節目。<|laughter|>今日我哋會傾下香港人最關心嘅事，住屋同埋通脹點樣影響緊我哋嘅生活。"]
    ],
}


def adapter_sort_key(p: Path):
    """Order: step4000, step4500, ..., step22500, then the unsuffixed 'adapter'."""
    m = re.match(r"adapter_step(\d+)", p.name)
    if m:
        return (0, int(m.group(1)))
    return (1, 0)  # 'adapter' (final) goes last


def adapter_name(p: Path) -> str:
    """PeftModel adapter_name — must be a valid identifier."""
    if p.name == "adapter":
        return "final"
    return p.name.replace("adapter_step", "step_")


def out_filename(p: Path) -> str:
    if p.name == "adapter":
        return "step_final.wav"
    return p.name.replace("adapter_step", "step_") + ".wav"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adapters = sorted(
        [p for p in ADAPTER_ROOT.iterdir() if p.is_dir() and (p / "adapter_config.json").exists()],
        key=adapter_sort_key,
    )
    print(f"[INFO] Found {len(adapters)} adapter checkpoints:")
    for p in adapters:
        print(f"  - {p.name}")

    print(f"\n[INFO] Loading base model from {MODEL_PATH}")
    model, dataset = initiate_model(SEED, MODEL_PATH, "hf", fp16_flow=True)

    offset = model.config.hf_config.speech_token_offset
    print(f"[INFO] speech_token_offset = {offset}")
    assert offset == 153595, f"wrong offset: {offset} (LoRA was trained against 153595)"

    print("\n[INFO] Pre-processing prompt input (shared across all adapters)")
    inputs = podcast_format_parser(DATA)
    prepared = process_single_input(
        dataset,
        inputs["text"],
        inputs["prompt_wav"],
        inputs["prompt_text"],
        inputs["use_dialect_prompt"],
        inputs["dialect_prompt_text"],
    )

    # Wrap base trunk in PeftModel and load every adapter under a unique name.
    print(f"\n[INFO] Attaching first adapter to base trunk: {adapters[0].name}")
    base_trunk = model.llm.model
    first_name = adapter_name(adapters[0])
    peft_wrapped = PeftModel.from_pretrained(base_trunk, str(adapters[0]), adapter_name=first_name)

    for p in adapters[1:]:
        name = adapter_name(p)
        print(f"[INFO] Loading adapter {p.name} -> name '{name}'")
        peft_wrapped.load_adapter(str(p), adapter_name=name)

    model.llm.model = peft_wrapped
    print(f"[INFO] All {len(adapters)} adapters loaded; switching via set_adapter()")

    for p in adapters:
        name = adapter_name(p)
        out_path = OUT_DIR / out_filename(p)
        if out_path.exists():
            print(f"\n[SKIP] {out_path} already exists")
            continue

        print(f"\n[INFO] === {p.name} (adapter '{name}') ===")
        peft_wrapped.set_adapter(name)

        # Re-seed each run so stochastic differences between adapters
        # aren't conflated with sampler RNG drift.
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        t0 = time.perf_counter()
        results = model.forward_longform(**prepared)
        elapsed = time.perf_counter() - t0

        wav = results["generated_wavs"][0].cpu()
        duration_s = wav.shape[-1] / 24000
        rtf = elapsed / duration_s if duration_s > 0 else float("nan")
        print(f"[INFO] generated {duration_s:.2f}s in {elapsed:.2f}s (RTF={rtf:.2f})")

        wav_to_save = wav.unsqueeze(0) if wav.dim() == 1 else wav
        torchaudio.save(str(out_path), wav_to_save, 24000)
        print(f"[INFO] saved -> {out_path}")

    print(f"\n[DONE] All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
