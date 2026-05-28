"""Greedy-ceiling and checkpoint sweep for MTP spec decoding.

For each checkpoint in a run directory, measures `mean_accept` under two
decoding regimes:

  - greedy : `baseline_greedy_decode_cached` vs `mtp_speculative_decode_cached`
             (argmax everywhere; the MTP heads' ceiling — no sampling mismatch
             with the trunk).
  - sample : `baseline_sample_decode_cached` vs `mtp_speculative_sample_cached`
             (production sampler: RAS + top_k=100 + top_p=0.9 + temp=0.6 +
             rep_pen=1.25; what real inference looks like).

If greedy mean_accept is high but sample mean_accept is low, the heads
learned to match argmax but disagree with the production sampler — fixable
by retraining with `kl_top_k` distillation. If greedy is also low, the heads
haven't learned to predict future tokens at all.

Skips flow + HiFT — we only care about LLM-side acceptance, so the run is
fast (~30s/ckpt instead of minutes).

Usage:
    python scripts/mtp/mtp_ceiling_sweep.py <run_dir> [--prompt mandarin_medium]
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import time
from pathlib import Path

import torch

from soulxpodcast.config import SamplingParams
from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.training.mtp_inference import (
    baseline_greedy_decode_cached,
    baseline_sample_decode_cached,
    mtp_speculative_decode_cached,
    mtp_speculative_sample_cached,
)
from soulxpodcast.training.mtp_module import MtpConfig, SequentialMTP


MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect"

S1_PROMPT_WAV = "example/audios/female_mandarin.wav"
S1_PROMPT_TEXT = "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。"

PROMPTS = {
    "english_short": "Hello everyone, welcome to our show.",
    "mandarin_medium": "今天天气真好，我们一起出去走走吧。听说附近新开了一家咖啡店，环境很不错。",
}


def load_mtp(ckpt_path, base):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mtp_config = MtpConfig(**ckpt["mtp_config"])
    decoder_layer_cls = base.model.layers[0].__class__
    mtp = SequentialMTP(mtp_config, decoder_layer_cls, base.config)
    mtp.load_state_dict(ckpt["mtp_state"])
    mtp = mtp.to(device="cuda", dtype=torch.bfloat16).eval()
    return mtp, ckpt


def build_input_ids(model, prepared):
    """Assemble first-turn LLM input_ids — mirrors generate_mtp() in mtp_audio_ab.py."""
    cfg_off = model.config.hf_config.speech_token_offset
    eos_id = model.config.hf_config.eos_token_id

    pm = prepared["prompt_mels_for_llm"].cuda()
    pml = prepared["prompt_mels_lens_for_llm"].cuda()
    pst, pst_lens = model.audio_tokenizer.quantize(pm, pml)
    pst0 = pst[0, : pst_lens[0].item()].tolist()
    speech_tokens_0 = [t + cfg_off for t in pst0] + [eos_id]
    prompt_input = prepared["prompt_text_tokens_for_llm"][0] + speech_tokens_0
    inputs = list(prompt_input) + list(prepared["text_tokens_for_llm"][0])
    return torch.tensor([inputs], dtype=torch.long, device="cuda"), eos_id


def fmt_hist(accept_lengths, n_drafts):
    from collections import Counter
    hist = Counter(accept_lengths)
    return " ".join(f"{i}:{hist.get(i, 0)}" for i in range(1, n_drafts + 1))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--prompt", default="mandarin_medium", choices=list(PROMPTS))
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--model-path", default=MODEL_PATH,
                    help="Trunk model directory. Use the same one MTP was trained against "
                         "(see train_config.json:model_path).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_paths = sorted(run_dir.glob("mtp_step*.pt"))
    final = run_dir / "mtp_final.pt"
    if final.exists():
        ckpt_paths = list(ckpt_paths) + [final]
    if not ckpt_paths:
        raise SystemExit(f"no checkpoints under {run_dir}")
    print(f"[init] {len(ckpt_paths)} checkpoints under {run_dir}")
    for p in ckpt_paths:
        print(f"  - {p.name}")

    print(f"[init] loading SoulXPodcast model from {args.model_path}")
    model, dataset_handler = initiate_model(
        seed=42, model_path=args.model_path, llm_engine="hf", fp16_flow=True,
    )
    base = model.llm.model

    # Build the prompt once.
    speakers = {"S1": {"prompt_audio": Path(S1_PROMPT_WAV), "prompt_text": S1_PROMPT_TEXT}}
    parsed = podcast_format_parser({"speakers": speakers, "text": [["S1", PROMPTS[args.prompt]]]})
    prepared = process_single_input(
        dataset_handler, parsed["text"], parsed["prompt_wav"], parsed["prompt_text"],
        parsed["use_dialect_prompt"], parsed["dialect_prompt_text"],
    )
    input_ids, eos_id = build_input_ids(model, prepared)
    print(f"[init] prompt={args.prompt!r}  input_ids={input_ids.shape[1]} tokens")

    sp = SamplingParams()
    print(f"[init] sampling: T={sp.temperature} top_k={sp.top_k} top_p={sp.top_p} "
          f"rep={sp.repetition_penalty} ras={sp.use_ras}")

    # --- Warmup + baselines (computed once; same for every ckpt) ---
    print("\n[warmup] one greedy + one sample baseline pass...")
    _ = baseline_greedy_decode_cached(base, input_ids, max_new_tokens=16, eos_token_id=eos_id)
    _ = baseline_sample_decode_cached(
        base, input_ids, max_new_tokens=16, eos_token_id=eos_id,
        temperature=sp.temperature, top_k=sp.top_k, top_p=sp.top_p,
        repetition_penalty=sp.repetition_penalty, seed=42,
    )

    print("[baseline] greedy autoregressive (cached)...")
    t0 = time.perf_counter()
    g_ids, g_steps = baseline_greedy_decode_cached(
        base, input_ids, max_new_tokens=args.max_new, eos_token_id=eos_id,
    )
    t_g = time.perf_counter() - t0
    g_new = g_ids.shape[1] - input_ids.shape[1]
    print(f"[baseline] greedy: {g_new} tokens in {g_steps} steps, {t_g:.2f}s "
          f"({g_steps/t_g:.1f} tok/s)")

    print("[baseline] sample (RAS off — fair comparison to spec sampler)...")
    t0 = time.perf_counter()
    s_ids, s_steps = baseline_sample_decode_cached(
        base, input_ids, max_new_tokens=args.max_new, eos_token_id=eos_id,
        temperature=sp.temperature, top_k=sp.top_k, top_p=sp.top_p,
        repetition_penalty=sp.repetition_penalty, seed=42,
    )
    t_s = time.perf_counter() - t0
    s_new = s_ids.shape[1] - input_ids.shape[1]
    print(f"[baseline] sample: {s_new} tokens in {s_steps} steps, {t_s:.2f}s "
          f"({s_steps/t_s:.1f} tok/s)")

    # --- Per-checkpoint: greedy spec + sample spec ---
    rows = []
    for ckpt_path in ckpt_paths:
        print(f"\n{'=' * 70}\nCKPT: {ckpt_path.name}\n{'=' * 70}")
        mtp, ckpt = load_mtp(str(ckpt_path), base)
        n_mtp = len(mtp.layers)
        step = ckpt.get("step", "?")

        # Greedy spec
        t0 = time.perf_counter()
        r_g = mtp_speculative_decode_cached(
            base, mtp, input_ids, max_new_tokens=args.max_new, eos_token_id=eos_id,
        )
        t_gs = time.perf_counter() - t0
        n_new_g = sum(r_g.accept_lengths)
        print(f"  [greedy] {n_new_g} new toks, {r_g.n_steps} steps, "
              f"mean_accept={r_g.mean_accept_length:.3f} (max={n_mtp + 1}), "
              f"{t_gs:.2f}s  speedup={t_g/t_gs:.2f}x")
        print(f"           hist {fmt_hist(r_g.accept_lengths, n_mtp + 1)}")

        # Sample spec (RAS off so it's directly comparable to baseline_sample)
        t0 = time.perf_counter()
        r_s = mtp_speculative_sample_cached(
            base, mtp, input_ids, max_new_tokens=args.max_new, eos_token_id=eos_id,
            temperature=sp.temperature, top_k=sp.top_k, top_p=sp.top_p,
            repetition_penalty=sp.repetition_penalty,
            use_ras=False, seed=42,
        )
        t_ss = time.perf_counter() - t0
        n_new_s = sum(r_s.accept_lengths)
        print(f"  [sample] {n_new_s} new toks, {r_s.n_steps} steps, "
              f"mean_accept={r_s.mean_accept_length:.3f} (max={n_mtp + 1}), "
              f"{t_ss:.2f}s  speedup={t_s/t_ss:.2f}x")
        print(f"           hist {fmt_hist(r_s.accept_lengths, n_mtp + 1)}")

        rows.append((ckpt_path.name, step, r_g.mean_accept_length, t_g/t_gs,
                     r_s.mean_accept_length, t_s/t_ss))

        del mtp
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}\nSUMMARY (prompt={args.prompt}, baseline greedy={t_g:.2f}s, "
          f"sample={t_s:.2f}s)\n{'=' * 70}")
    print(f"{'ckpt':<22s} {'step':>8s} {'greedy_acc':>11s} {'g_spdup':>9s} "
          f"{'sample_acc':>11s} {'s_spdup':>9s}")
    for name, step, ga, gx, sa, sx in rows:
        print(f"{name:<22s} {str(step):>8s} {ga:>11.3f} {gx:>8.2f}x {sa:>11.3f} {sx:>8.2f}x")
    print("\nInterpretation:")
    print("  - greedy_acc ~ 1.0 → MTP heads not learning; further training won't help spec decode")
    print("  - greedy_acc > 1.8 but sample_acc near 1.0 → train/inference sampler mismatch (fixable)")
    print("  - both growing across steps → keep training")
    print("  - both flat across steps → loss is converged but to a non-useful objective")


if __name__ == "__main__":
    main()
