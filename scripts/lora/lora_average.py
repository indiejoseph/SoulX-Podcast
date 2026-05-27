"""Average selected LoRA adapter checkpoints into a single merged adapter.

Element-wise averages every tensor in adapter_model.safetensors — both the
LoRA A/B matrices and the modules_to_save `lm_head.weight`. For checkpoints
from the same training run (smoothly-varying weights), the cross-term error
introduced by naive A/B averaging is small. This is the standard "model
soup" / SWA pattern adapted to LoRA.

Outputs:
  - runs/hpc/adaptors3_avg/<AVG_NAME>/ — new PEFT-loadable adapter
  - outputs/lora_sweep/<AVG_NAME>.wav — inference with the averaged adapter
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import shutil
import time
from pathlib import Path

import torch
import torchaudio
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser

ADAPTER_ROOT = Path("runs/hpc/adaptors3")
OUT_ADAPTER_ROOT = Path("runs/hpc/adaptors3_avg")
OUT_WAV_DIR = Path("outputs/lora_sweep")
MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect"
SEED = 198964

# 3-point trajectory average: epoch-1 endpoint, mid-epoch-2, epoch-2 endpoint.
SOURCES = [
    "adapter_step11500",
    "adapter_step17000",
    "adapter_step22500",
]
# Equal weights → arithmetic mean. Must sum to 1.0 for a true average.
WEIGHTS = [1.0 / len(SOURCES)] * len(SOURCES)
AVG_NAME = "avg_11500_17000_22500"

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


def average_safetensors(source_paths: list[Path], weights: list[float], out_path: Path):
    """Element-wise weighted average of every tensor across sources.

    Upcasts to float32 for the accumulation to avoid bf16 precision loss,
    then casts each result back to its original dtype.
    """
    assert len(source_paths) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6, f"weights sum to {sum(weights)}, expected 1.0"

    print(f"[INFO] Reading first source to get key list + dtypes: {source_paths[0].name}")
    with safe_open(str(source_paths[0]), framework="pt") as f:
        keys = list(f.keys())
        dtypes = {k: f.get_tensor(k).dtype for k in keys}

    accum: dict[str, torch.Tensor] = {}
    for src_path, w in zip(source_paths, weights):
        print(f"[INFO]   adding {src_path.name} * {w:.4f}")
        with safe_open(str(src_path), framework="pt") as f:
            src_keys = set(f.keys())
            assert src_keys == set(keys), f"key mismatch in {src_path.name}"
            for k in keys:
                t = f.get_tensor(k).to(torch.float32) * w
                if k in accum:
                    accum[k] += t
                else:
                    accum[k] = t

    # Cast each averaged tensor back to its original dtype.
    averaged = {k: v.to(dtypes[k]) for k, v in accum.items()}
    print(f"[INFO] Saving averaged adapter -> {out_path}")
    save_file(averaged, str(out_path))


def main():
    OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
    out_adapter_dir = OUT_ADAPTER_ROOT / AVG_NAME
    out_adapter_dir.mkdir(parents=True, exist_ok=True)

    src_dirs = [ADAPTER_ROOT / s for s in SOURCES]
    src_safetensors = [d / "adapter_model.safetensors" for d in src_dirs]
    for p in src_safetensors:
        assert p.exists(), f"missing source: {p}"

    print(f"[INFO] Averaging {len(SOURCES)} adapters into '{AVG_NAME}':")
    for s, w in zip(SOURCES, WEIGHTS):
        print(f"  - {s} (weight={w:.4f})")

    # 1. Copy adapter_config.json (and README) from the first source — config
    #    is identical across same-run checkpoints (same rank, alpha, targets).
    shutil.copy(src_dirs[0] / "adapter_config.json", out_adapter_dir / "adapter_config.json")
    if (src_dirs[0] / "README.md").exists():
        shutil.copy(src_dirs[0] / "README.md", out_adapter_dir / "README.md")

    # 2. Average the safetensors weights.
    average_safetensors(
        src_safetensors,
        WEIGHTS,
        out_adapter_dir / "adapter_model.safetensors",
    )
    print(f"[DONE] Averaged adapter written to {out_adapter_dir}")

    # 3. Verify by running inference with the merged adapter.
    print(f"\n[INFO] Loading base model from {MODEL_PATH}")
    model, dataset = initiate_model(SEED, MODEL_PATH, "hf", fp16_flow=True)

    offset = model.config.hf_config.speech_token_offset
    print(f"[INFO] speech_token_offset = {offset}")
    assert offset == 153595, f"wrong offset: {offset}"

    print("[INFO] Preparing input")
    inputs = podcast_format_parser(DATA)
    prepared = process_single_input(
        dataset,
        inputs["text"],
        inputs["prompt_wav"],
        inputs["prompt_text"],
        inputs["use_dialect_prompt"],
        inputs["dialect_prompt_text"],
    )

    print(f"[INFO] Attaching averaged adapter {AVG_NAME}")
    peft_wrapped = PeftModel.from_pretrained(model.llm.model, str(out_adapter_dir), adapter_name=AVG_NAME)
    model.llm.model = peft_wrapped

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("[INFO] Running inference with averaged adapter")
    t0 = time.perf_counter()
    results = model.forward_longform(**prepared)
    elapsed = time.perf_counter() - t0

    wav = results["generated_wavs"][0].cpu()
    duration_s = wav.shape[-1] / 24000
    rtf = elapsed / duration_s if duration_s > 0 else float("nan")
    print(f"[INFO] generated {duration_s:.2f}s in {elapsed:.2f}s (RTF={rtf:.2f})")

    wav_to_save = wav.unsqueeze(0) if wav.dim() == 1 else wav
    out_wav = OUT_WAV_DIR / f"{AVG_NAME}.wav"
    torchaudio.save(str(out_wav), wav_to_save, 24000)
    print(f"[INFO] saved -> {out_wav}")


if __name__ == "__main__":
    main()
