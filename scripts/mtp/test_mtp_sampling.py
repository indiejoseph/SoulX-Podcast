"""Sampling-based MTP spec-decode acceptance test (Leviathan-Kalman rule).

The greedy test (`test_mtp_inference.py`) showed mean accept length ≈ 1.0
because KL-distilled heads converge to match the trunk's *distribution* but
not its argmax — and greedy spec-decode requires argmax equality.

The Leviathan-Kalman rule accepts draft d_k with probability min(1, p(d_k)/q(d_k))
where p is trunk's true distribution and q is the student's. This is much
more lenient: drafts get accepted whenever the student doesn't massively
under-rate the trunk's choice. The output distribution provably matches
sampling from the trunk directly under the given sampling parameters.

Uses production sampling params (from `soulxpodcast/config.py`):
  temperature=0.6, top_k=100, top_p=0.9, repetition_penalty=1.25,
  RAS (Repetition-Aware Sampling): win_size=25, tau_r=0.2.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from soulxpodcast.training.mtp_dataset import (
    DIALECT_PREFIX, MtpDatasetConfig, SPECIAL_TOKENS,
)
from soulxpodcast.training.mtp_inference import (
    baseline_greedy_decode_cached, mtp_speculative_sample_cached,
)
from soulxpodcast.training.mtp_module import MtpConfig, SequentialMTP


DEFAULT_BASE = "pretrained_models/SoulX-Podcast-1.7B-dialect-avg"
DEFAULT_DATASET = "/notebooks/projects/SoulX-Podcast/tmp/dataset_small_with_tokens"
DEFAULT_CKPT = "/notebooks/projects/SoulX-Podcast/runs/hpc/mtp_v2/mtp_final.pt"

# Production sampling params from soulxpodcast/config.py:SamplingParams.
SAMPLING = dict(
    temperature=0.6,
    top_k=100,
    top_p=0.9,
    repetition_penalty=1.25,
    use_ras=True,
    ras_win_size=25,
    ras_tau_r=0.2,
)
SEED = 198964


def load_mtp(ckpt_path: str, base):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mtp_config = MtpConfig(**ckpt["mtp_config"])
    decoder_layer_cls = base.model.layers[0].__class__
    mtp = SequentialMTP(mtp_config, decoder_layer_cls, base.config)
    mtp.load_state_dict(ckpt["mtp_state"])
    mtp = mtp.to(device="cuda", dtype=torch.bfloat16).eval()
    return mtp, ckpt.get("step", -1)


def build_prompt(sample, tokenizer):
    cfg = MtpDatasetConfig()
    special = {k: tokenizer.encode(v, add_special_tokens=False)[0]
               for k, v in SPECIAL_TOKENS.items()}
    text = sample["text"]
    lang = sample["lang"]
    speech_tokens = [int(x) for x in sample["speech_tokens"].split()]
    prefix = DIALECT_PREFIX.get(lang, "")
    text_with_prefix = prefix + text if prefix else text
    text_ids = tokenizer.encode(text_with_prefix, add_special_tokens=False)
    prompt_ids = (
        [special["task_podcast"], special["speaker_0"], special["text_start"]]
        + text_ids
        + [special["text_end"], special["semantic_token_start"]]
    )
    return prompt_ids, special["semantic_token_end"], len(speech_tokens)


def run_sample_test(label, sample, base, mtp, tokenizer, n_drafts: int,
                    max_new_tokens: int = 200):
    prompt_ids, eos_id, target_len = build_prompt(sample, tokenizer)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    cap = min(max_new_tokens, target_len + 16)

    print(f"\n=== {label} ===")
    print(f"  lang={sample['lang']}  text={sample['text'][:60]!r}"
          f"{'...' if len(sample['text']) > 60 else ''}")
    print(f"  prompt_len={len(prompt_ids)}  target_speech_len={target_len}")

    # ---- Spec-decode with sampling (Leviathan-Kalman) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    spec = mtp_speculative_sample_cached(
        base, mtp, prompt_tensor,
        max_new_tokens=cap, eos_token_id=eos_id, seed=SEED, **SAMPLING,
    )
    torch.cuda.synchronize()
    t_spec = time.perf_counter() - t0
    print(f"  [spec ] {spec.n_committed} tokens in {spec.n_steps} steps "
          f"({t_spec:.2f}s)")
    print(f"           mean accept length: {spec.mean_accept_length:.2f}  "
          f"tokens/step: {spec.tokens_per_step:.2f}  eos_hit: {spec.eos_hit}")
    other = t_spec - spec.t_trunk - spec.t_mtp
    print(f"           breakdown: trunk={spec.t_trunk:.2f}s  "
          f"mtp={spec.t_mtp:.2f}s  other={other:.2f}s")
    if spec.accept_lengths:
        hist = Counter(spec.accept_lengths)
        bucket = " ".join(f"{i}:{hist.get(i, 0)}" for i in range(1, n_drafts + 2))
        print(f"           accept-len histogram (1..{n_drafts + 1}): {bucket}")

    # ---- Trunk-only greedy baseline for wall-time comparison ----
    # NOTE: This is greedy not sampled. We can't directly compare wall-time
    # of sampled spec-decode against sampled greedy (because both involve RNG
    # paths), but the greedy baseline is the same trunk forward cost and the
    # most stable wall-time reference. Sampled-vs-sampled would differ only
    # by negligible RNG overhead.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    base_ids, base_steps = baseline_greedy_decode_cached(
        base, prompt_tensor, max_new_tokens=cap, eos_token_id=eos_id,
    )
    torch.cuda.synchronize()
    t_base = time.perf_counter() - t0
    base_new = base_ids.shape[1] - prompt_tensor.shape[1]
    speedup = t_base / t_spec if t_spec > 0 else 0
    print(f"  [base ] {base_new} tokens in {base_steps} steps ({t_base:.2f}s)")
    flag = "FASTER" if speedup > 1 else "slower"
    print(f"           >>> speedup vs trunk-only greedy: {speedup:.2f}x ({flag})")
    return spec.mean_accept_length, speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    args = ap.parse_args()

    print(f"[init] base: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print(f"[init] mtp ckpt: {args.ckpt}")
    mtp, step = load_mtp(args.ckpt, base)
    n_drafts = len(mtp.layers)
    print(f"[init] loaded MTP step={step}  n_layers={n_drafts}  K_max={n_drafts + 1}")
    print(f"[init] sampling params: {SAMPLING}")

    hf_ds = load_from_disk(args.dataset).remove_columns(["audio", "id", "phone"])

    samples = [(0, "sample_0"), (1, "sample_1"), (2, "sample_2"),
               (100, "sample_100"), (1000, "sample_1000"), (2500, "sample_2500")]
    results = []
    for idx, name in samples:
        if idx < len(hf_ds):
            mean_acc, sp = run_sample_test(
                name, hf_ds[idx], base, mtp, tokenizer, n_drafts
            )
            results.append((name, mean_acc, sp))

    print("\n" + "=" * 70)
    print("SUMMARY (Leviathan-Kalman sampling spec-decode)")
    print("=" * 70)
    print(f"{'sample':<20} {'mean_accept':>14} {'speedup':>10}")
    for name, ma, sp in results:
        print(f"{name:<20} {ma:>14.2f} {sp:>10.2f}x")
    avg_acc = sum(r[1] for r in results) / max(len(results), 1)
    avg_sp = sum(r[2] for r in results) / max(len(results), 1)
    print(f"{'AVERAGE':<20} {avg_acc:>14.2f} {avg_sp:>10.2f}x")
    print(f"(theoretical max accept length = {n_drafts + 1}; "
          f"break-even speedup ~= 1.0×)")


if __name__ == "__main__":
    main()
