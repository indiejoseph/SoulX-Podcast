"""Quantize the merged SoulX-Podcast trunk to AWQ-INT4 using SoulX-formatted
calibration prompts (text + speech tokens) so AWQ's activation-aware scales
are tuned for the actual token distribution we care about (not generic text).

Outputs:
    pretrained_models/SoulX-Podcast-1.7B-dialect-avg-awq/
    — model.safetensors (AWQ-INT4 packed weights, ~1 GB)
    — config.json + quantization_config (AWQ marker)
    — tokenizer files (copied)
    — flow/vocoder/etc symlinks (copied from the merged dir, same as merge step)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import shutil
from pathlib import Path
from typing import List

import torch
from awq import AutoAWQForCausalLM
from datasets import load_from_disk
from transformers import AutoTokenizer

from soulxpodcast.training.mtp_dataset import DIALECT_PREFIX, SPECIAL_TOKENS


BASE = Path("pretrained_models/SoulX-Podcast-1.7B-dialect-avg")
OUT = Path("pretrained_models/SoulX-Podcast-1.7B-dialect-avg-awq")
DATASET = "/notebooks/projects/SoulX-Podcast/tmp/dataset_small_with_tokens"
N_CALIB = 256
SPEECH_TOKEN_OFFSET = 153595

# Same heavy-file symlink + small-file copy lists as merge_adapter_to_base.py.
SYMLINK_FILES = {
    "flow.pt", "flow.cache.pt", "flow.decoder.estimator.fp32.onnx",
    "hift.pt", "campplus.onnx",
}
COPY_FILES = {
    "soulxpodcast_config.json", "added_tokens.json", "merges.txt",
    "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json",
    "vocab.json", "README.md", ".gitattributes",
}


def build_calib_strings(tokenizer, n: int) -> List[str]:
    """Assemble SoulX-formatted text prompts from the local dataset.

    AWQ wants plain strings; the calibration loader inside autoawq will
    tokenize them. We build the same `<|task_podcast|><|SPEAKER_0|>...` prefix
    that training and inference use, then concatenate text + speech tokens
    as their decoded string forms.

    Speech tokens decode to `<|N|>` for N=0..6560 (the added_tokens vocab).
    Including them in calibration is critical — AWQ scales need to reflect
    activation patterns at speech-token positions, not just text positions.
    """
    ds = load_from_disk(DATASET).remove_columns(
        [c for c in load_from_disk(DATASET).column_names
         if c not in ["text", "speech_tokens", "lang"]]
    )
    strings = []
    for i in range(min(n, len(ds))):
        s = ds[i]
        text = s["text"]
        lang = s["lang"]
        prefix = DIALECT_PREFIX.get(lang, "")
        text_full = prefix + text if prefix else text

        speech_tokens_raw = [int(x) for x in s["speech_tokens"].split()]
        if not speech_tokens_raw:
            continue
        # Decode speech tokens back to the `<|N|>` string form so the
        # calibration tokenizer maps them to the right vocab ids.
        speech_str = "".join(f"<|{t}|>" for t in speech_tokens_raw)

        full = (
            f"<|task_podcast|><|SPEAKER_0|><|text_start|>{text_full}"
            f"<|text_end|><|semantic_token_start|>{speech_str}<|semantic_token_end|>"
        )
        strings.append(full)
    print(f"[INFO] Built {len(strings)} calibration strings "
          f"(avg len {sum(len(s) for s in strings) // max(len(strings), 1)} chars)")
    return strings


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists — refuse to overwrite")
    OUT.mkdir(parents=True)

    print(f"[INFO] Loading base for quantization: {BASE}")
    model = AutoAWQForCausalLM.from_pretrained(
        str(BASE), low_cpu_mem_usage=True, use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True)

    print(f"[INFO] Building calibration set ({N_CALIB} samples)")
    calib = build_calib_strings(tokenizer, N_CALIB)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,   # standard for AWQ-INT4
        "w_bit": 4,
        "version": "GEMM",     # GEMM kernel; vLLM supports this
    }
    print(f"[INFO] Quantizing → AWQ-INT4 (this takes a while)")
    print(f"       config: {quant_config}")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib)

    print(f"[INFO] Saving quantized model -> {OUT}")
    model.save_quantized(str(OUT))
    tokenizer.save_pretrained(str(OUT))

    print(f"[INFO] Symlinking heavy artifacts from {BASE}")
    for fname in sorted(SYMLINK_FILES):
        src = (BASE / fname).resolve()
        if src.exists():
            (OUT / fname).symlink_to(src)
    for fname in sorted(COPY_FILES):
        src = BASE / fname
        if src.exists() and not (OUT / fname).exists():
            shutil.copy(src, OUT / fname)
    if (BASE / "assets").exists():
        shutil.copytree(BASE / "assets", OUT / "assets", dirs_exist_ok=True)

    print(f"\n[DONE] AWQ model at {OUT}")
    print("Use with vLLM: pass quantization='awq' (or 'awq_marlin') to LLM().")


if __name__ == "__main__":
    main()
