"""Self-consistency check: trunk CE on its own generated tokens.

Logic:
  - If training pipeline (frontend tokenization, special tokens, speech offset,
    sequence assembly) MATCHES what inference uses, then the trunk should
    assign **very low CE** to its own generated speech tokens.
  - If training CE on dataset tokens is high (~9 nats) BUT self-gen CE is
    also high → wiring bug in the training pipeline.
  - If self-gen CE is low (~1-3 nats) but dataset CE is high → no bug,
    just expected dataset/model preference mismatch.

This tells us whether the high CE we see during training is "expected" data
divergence or a smoking-gun for a bug in the pipeline.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from soulxpodcast.training.mtp_dataset import (
    DIALECT_PREFIX, MtpDatasetConfig, SPECIAL_TOKENS,
)


MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect"
DATASET_PATH = "/notebooks/projects/SoulX-Podcast/tmp/dataset_small_with_tokens"


def build_training_input(text: str, lang: str, speech_tokens_0based, special, offset):
    """Construct the EXACT same input that MtpDataset.build_sample would create
    for this sample. Returns a flat list of token ids."""
    prefix = DIALECT_PREFIX.get(lang, "")
    text_with_prefix = prefix + text if prefix else text
    text_ids = tokenizer.encode(text_with_prefix, add_special_tokens=False)
    speech_llm = [t + offset for t in speech_tokens_0based]
    return (
        [special["task_podcast"], special["speaker_0"], special["text_start"]]
        + text_ids
        + [special["text_end"], special["semantic_token_start"]]
        + speech_llm
        + [special["semantic_token_end"]]
    ), len(text_ids), len(speech_llm)


@torch.inference_mode()
def trunk_ce_on_speech_positions(
    base, input_ids, speech_start: int, speech_len: int, label: str
) -> float:
    """Run trunk forward and compute mean CE on the speech-token positions.

    input_ids[t] predicts token at position t+1.
    Speech-token region in `input_ids` is [speech_start, speech_start + speech_len).
    The logit predicting position t is at logits[t-1].
    """
    out = base.model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = base.lm_head(out.last_hidden_state)        # [1, T, V]
    speech_logits = logits[0, speech_start - 1 : speech_start - 1 + speech_len]  # [L, V]
    speech_targets = input_ids[0, speech_start : speech_start + speech_len]      # [L]
    ce_per_pos = F.cross_entropy(speech_logits, speech_targets, reduction="none")
    mean_ce = float(ce_per_pos.mean().item())
    # Also compute top-1 acc — how often trunk's argmax matches the actual token
    top1 = speech_logits.argmax(dim=-1)
    top1_acc = float((top1 == speech_targets).float().mean().item())
    print(f"  [{label}]  speech positions: {speech_len}  "
          f"mean CE = {mean_ce:.3f}  top1 acc = {top1_acc:.1%}")
    return mean_ce


def generate_speech_tokens(model, prompt_ids, eos_id, sampling_params, label: str):
    """Use the existing HFLLMEngine to generate speech tokens from the prompt.
    Returns list of generated token IDs (LLM-vocab space, including EOS at end)."""
    print(f"  [{label}]  generating from prompt of length {len(prompt_ids)}...")
    sp = sampling_params  # SamplingParams object
    sp.stop_token_ids = [eos_id]
    out = model.llm.generate(prompt_ids, sp, past_key_values=None)
    return out["token_ids"]


def main():
    global tokenizer

    print("=" * 72)
    print("TRAINING-PIPELINE SELF-CONSISTENCY CHECK")
    print("=" * 72)

    # ---- Setup ---------------------------------------------------------
    print(f"\n[setup] loading model from {MODEL_PATH}")
    from soulxpodcast.utils.infer_utils import initiate_model
    model, _ = initiate_model(seed=42, model_path=MODEL_PATH,
                              llm_engine="hf", fp16_flow=True)
    base = model.llm.model            # the AutoModelForCausalLM
    base.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    special = {k: tokenizer.encode(v, add_special_tokens=False)[0]
               for k, v in SPECIAL_TOKENS.items()}
    cfg = MtpDatasetConfig()
    print(f"[setup] special tokens: {special}")
    print(f"[setup] speech_token_offset = {cfg.speech_token_offset}")

    # ---- Load dataset --------------------------------------------------
    print(f"\n[setup] loading dataset from {DATASET_PATH}")
    hf_ds = load_from_disk(DATASET_PATH).remove_columns(["audio", "id", "phone"])

    # Pick one sample per language for the check
    samples_to_test = []
    for target_lang in ["en", "zh", "yue"]:
        for i in range(len(hf_ds)):
            if hf_ds[i]["lang"] == target_lang:
                samples_to_test.append((target_lang, hf_ds[i]))
                break

    # Sampling params for self-generation. Use temperature=0.6 (production
    # setting); we ALSO test temperature=0 (greedy) which gives the trunk's
    # argmax path → CE on that should be as low as possible.
    from soulxpodcast.config import SamplingParams

    # ---- Per-sample diagnostic ----------------------------------------
    for lang, sample in samples_to_test:
        print()
        print("─" * 72)
        print(f"SAMPLE  lang={lang}  text={sample['text'][:70]!r}")
        print("─" * 72)

        dataset_speech = [int(x) for x in sample["speech_tokens"].split()]
        print(f"  Dataset speech tokens: {len(dataset_speech)} tokens"
              f"  (first 5: {dataset_speech[:5]})")

        # === Test 1: training format with DATASET speech tokens ===
        # This is exactly what MtpDataset.build_sample produces.
        ds_input, n_text, n_speech = build_training_input(
            sample["text"], sample["lang"], dataset_speech, special,
            cfg.speech_token_offset
        )
        ds_input_t = torch.tensor([ds_input], dtype=torch.long, device="cuda")
        # Speech tokens start after: 3 prefix tokens + n_text text tokens + 2 (text_end, semantic_start)
        speech_start = 3 + n_text + 2
        ce_dataset = trunk_ce_on_speech_positions(
            base, ds_input_t, speech_start, n_speech, "dataset_tokens"
        )

        # === Test 2: training format with GREEDY self-generated tokens ===
        prompt_for_gen = ds_input[:speech_start]  # everything before speech tokens
        sp_greedy = SamplingParams(
            temperature=0.001, top_k=1, top_p=1.0,
            repetition_penalty=1.0, use_ras=False,
            max_tokens=400, min_tokens=8,
            stop_token_ids=[special["semantic_token_end"]],
        )
        gen_greedy = generate_speech_tokens(
            model, prompt_for_gen, special["semantic_token_end"], sp_greedy,
            "self_gen_greedy"
        )
        # gen_greedy includes the EOS at end. Strip it for the "speech tokens" portion.
        if gen_greedy and gen_greedy[-1] == special["semantic_token_end"]:
            gen_speech_with_eos = gen_greedy
            gen_speech_len = len(gen_greedy) - 1
        else:
            gen_speech_with_eos = gen_greedy + [special["semantic_token_end"]]
            gen_speech_len = len(gen_greedy)

        selfgen_input = prompt_for_gen + gen_speech_with_eos
        selfgen_t = torch.tensor([selfgen_input], dtype=torch.long, device="cuda")
        ce_selfgen_greedy = trunk_ce_on_speech_positions(
            base, selfgen_t, speech_start, gen_speech_len, "self_gen_greedy"
        )

        # === Test 3: training format with SAMPLED self-generated tokens ===
        # Match production sampling settings.
        sp_sample = SamplingParams(
            temperature=0.6, top_k=100, top_p=0.9,
            repetition_penalty=1.25, use_ras=False,  # disable RAS for clean test
            max_tokens=400, min_tokens=8,
            stop_token_ids=[special["semantic_token_end"]],
        )
        gen_sample = generate_speech_tokens(
            model, prompt_for_gen, special["semantic_token_end"], sp_sample,
            "self_gen_sampled"
        )
        if gen_sample and gen_sample[-1] == special["semantic_token_end"]:
            gen_speech_with_eos = gen_sample
            gen_speech_len = len(gen_sample) - 1
        else:
            gen_speech_with_eos = gen_sample + [special["semantic_token_end"]]
            gen_speech_len = len(gen_sample)

        selfgen_sample_input = prompt_for_gen + gen_speech_with_eos
        selfgen_sample_t = torch.tensor([selfgen_sample_input], dtype=torch.long, device="cuda")
        ce_selfgen_sample = trunk_ce_on_speech_positions(
            base, selfgen_sample_t, speech_start, gen_speech_len, "self_gen_sampled"
        )

        # === Diagnostic interpretation ===
        print(f"\n  ──── verdict for lang={lang} ────")
        print(f"  CE on DATASET tokens:        {ce_dataset:.3f}   "
              "(what training sees on this sample)")
        print(f"  CE on SELF-GEN (greedy):     {ce_selfgen_greedy:.3f}   "
              "(should be near 0 if pipeline correct)")
        print(f"  CE on SELF-GEN (sampled):    {ce_selfgen_sample:.3f}   "
              "(should be small; some variance from sampling)")
        gap_dataset = ce_dataset - ce_selfgen_greedy
        if ce_selfgen_greedy < 1.0:
            print(f"  ✓ Pipeline OK: greedy self-gen CE < 1.0 — trunk recognizes "
                  "its own output")
        elif ce_selfgen_greedy < 3.0:
            print(f"  ⚠ Marginal: greedy self-gen CE = {ce_selfgen_greedy:.2f}, "
                  "expected near 0. Possibly noise from RAS or sampling defaults.")
        else:
            print(f"  ✗ LIKELY BUG: greedy self-gen CE = {ce_selfgen_greedy:.2f} "
                  ">> 0. Training pipeline may have wiring mismatch with inference.")
        print(f"  Dataset/SelfGen gap:         {gap_dataset:.3f}   "
              "(measures how 'far' dataset tokens are from trunk's preference)")


if __name__ == "__main__":
    main()
