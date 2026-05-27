"""Definitive overfit verification: does the LoRA-merged model reproduce
its memorized training tokens under GREEDY decoding from the EXACT training
format (no voice prompt, no sampling)?

This eliminates:
  - sampling drift (do_sample=False)
  - training/inference format mismatch (we feed the exact format MtpDataset built)

Expected for a model with top1=1.0 at training: 100% greedy match on
training-set positions, EOS emitted at the end.

If even THIS produces wrong tokens, there's a deeper bug — possibly in the
LoRA merge path, weight tying, or how we're constructing the prefix.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from soulxpodcast.training.mtp_dataset import (
    MtpDataset, MtpDatasetConfig, SPECIAL_TOKENS,
)

BASE = "pretrained_models/SoulX-Podcast-1.7B-dialect"
ADAPTER = "runs/overfit_smoke_lora/adapter"
DATASET = "tmp/dataset_small_with_tokens"


@torch.inference_mode()
def main():
    print("[load] tokenizer + base model")
    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cuda",
    )
    print(f"[merge] applying LoRA from {ADAPTER}")
    peft_model = PeftModel.from_pretrained(base, ADAPTER)
    model = peft_model.merge_and_unload()
    model.eval()
    print("[merge] done")

    print(f"[ds] loading {DATASET}")
    ds = load_from_disk(DATASET).remove_columns(["audio"])
    mtp_ds = MtpDataset(ds, tokenizer, MtpDatasetConfig(
        include_eos_in_speech_mask=True,
    ))

    eos_id = tokenizer.encode(SPECIAL_TOKENS["semantic_token_end"], add_special_tokens=False)[0]
    text_end_id = tokenizer.encode(SPECIAL_TOKENS["text_end"], add_special_tokens=False)[0]
    sem_start_id = tokenizer.encode(SPECIAL_TOKENS["semantic_token_start"], add_special_tokens=False)[0]

    for idx in [0, 1, 2, 3]:
        s = mtp_ds[idx]
        if s is None:
            print(f"sample {idx} filtered, skipping")
            continue
        input_ids = s["input_ids"]
        speech_mask = s["speech_mask"]

        # Find where speech starts (the position right after semantic_token_start)
        sem_positions = (input_ids == sem_start_id).nonzero(as_tuple=True)[0]
        if len(sem_positions) == 0:
            print(f"sample {idx}: no semantic_token_start, skipping")
            continue
        speech_start = int(sem_positions[0].item()) + 1

        prefix = input_ids[:speech_start].cuda().unsqueeze(0)
        gt_speech = input_ids[speech_start:].tolist()   # includes trailing semantic_token_end
        gt_len = len(gt_speech)

        # Generate greedily for up to 2x gt_len tokens or until EOS
        max_new = max(int(gt_len * 1.5), gt_len + 20)
        out = model.generate(
            prefix,
            do_sample=False,
            max_new_tokens=max_new,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
        gen_speech = out[0, speech_start:].tolist()

        # Trim trailing pad if any (HF generate pads to max if EOS not hit; eos becomes EOS)
        if eos_id in gen_speech:
            gen_speech = gen_speech[:gen_speech.index(eos_id) + 1]
        gen_len = len(gen_speech)

        # Compare token by token (up to common length)
        common = min(gt_len, gen_len)
        n_match = sum(1 for a, b in zip(gt_speech, gen_speech) if a == b)
        prefix_match = 0
        for a, b in zip(gt_speech, gen_speech):
            if a == b:
                prefix_match += 1
            else:
                break

        marker = "✓" if n_match / max(common, 1) > 0.9 else "✗"
        print(f"\n=== sample {idx} (len={int(s['length'])}) ===")
        print(f"  gt speech len = {gt_len}  (incl trailing EOS={gt_speech[-1] == eos_id})")
        print(f"  gen speech len = {gen_len}  (ends with EOS={gen_speech[-1] == eos_id if gen_speech else '?'})")
        print(f"  per-pos match: {n_match}/{common} ({n_match/max(common,1):.1%})  {marker}")
        print(f"  prefix match: {prefix_match}/{common}")
        if prefix_match < common:
            i = prefix_match
            print(f"  first divergence at speech-pos {i}: gt={gt_speech[i]}  gen={gen_speech[i]}")
            print(f"    gt [{max(0,i-2)}:{i+3}]  = {gt_speech[max(0,i-2):i+3]}")
            print(f"    gen[{max(0,i-2)}:{i+3}]  = {gen_speech[max(0,i-2):i+3]}")


if __name__ == "__main__":
    main()
