"""Per-head CE diagnostic for the MTP run: trunk-only vs student.

Loads the merged trunk + an MTP checkpoint, runs one batch from the local
dataset, and reports per-k:
  - trunk-only CE (project trunk_hidden[k:k+valid_len] through lm_head)
  - student CE (project mtp_h_slice through lm_head)
  - student↔trunk KL (the actual training objective)

If trunk-only CE has the same U-shape as the logs, the k=2 anomaly is
inherent to speech-token statistics, and the student is correctly mimicking
it. If trunk CE is monotonic but student CE has the U-shape, head 2 has
a real problem.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from soulxpodcast.training.mtp_dataset import MtpCollator, MtpDataset, MtpDatasetConfig
from soulxpodcast.training.mtp_module import MtpConfig, SequentialMTP, build_causal_mask_4d


DEFAULT_CKPT = "runs/hpc/mtp_v2/mtp_step8000.pt"
DEFAULT_BASE = "pretrained_models/SoulX-Podcast-1.7B-dialect-avg"
DEFAULT_DS = "tmp/dataset_small_with_tokens"
DEFAULT_N = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--dataset", default=DEFAULT_DS)
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="Number of samples to use for the diagnostic.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"[INFO] Loading base trunk: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map=device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    embed_tokens = base.model.embed_tokens
    rotary_emb = base.model.rotary_emb
    lm_head = base.lm_head

    print(f"[INFO] Loading MTP checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mtp_cfg = MtpConfig(**ckpt["mtp_config"])
    print(f"[INFO] num_mtp_layers={mtp_cfg.num_mtp_layers}  step={ckpt['step']}")

    decoder_layer_cls = base.model.layers[0].__class__
    mtp = SequentialMTP(mtp_cfg, decoder_layer_cls, base.config)
    mtp.load_state_dict(ckpt["mtp_state"])
    mtp = mtp.to(device=device, dtype=dtype)
    mtp.eval()
    for p in mtp.parameters():
        p.requires_grad_(False)

    print(f"[INFO] Loading dataset: {args.dataset}")
    hf_ds = load_from_disk(args.dataset).remove_columns(
        [c for c in load_from_disk(args.dataset).column_names
         if c not in ["text", "speech_tokens", "lang"]]
    )
    hf_ds = hf_ds.select(range(args.n))
    ds_cfg = MtpDatasetConfig(
        speech_token_offset=base.config.speech_token_offset
        if hasattr(base.config, "speech_token_offset") else 153595,
    )
    dataset = MtpDataset(hf_ds, tokenizer, ds_cfg)
    collator = MtpCollator(pad_token_id=tokenizer.pad_token_id or 0)

    # Build a batch from valid (non-None) samples.
    samples = []
    for i in range(len(dataset)):
        s = dataset[i]
        if s is not None:
            samples.append(s)
        if len(samples) >= args.n:
            break
    if not samples:
        raise RuntimeError("no usable samples")
    batch = collator(samples)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    speech_mask = batch["speech_mask"].to(device)
    B, T = input_ids.shape
    print(f"[INFO] Batch: B={B}  T={T}  speech_positions/sample={speech_mask.sum(dim=1).tolist()}")

    # ----- Trunk forward (frozen) -----
    print("\n[INFO] Trunk forward")
    with torch.no_grad():
        out = base.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        )
        trunk_hidden = out.last_hidden_state  # [B, T, H]

    # ----- MTP forward -----
    print("[INFO] MTP forward")
    with torch.no_grad():
        causal_mask_4d = build_causal_mask_4d(attention_mask, dtype=trunk_hidden.dtype)
        mtp_hiddens = mtp(
            trunk_hidden=trunk_hidden,
            input_ids=input_ids,
            embed_tokens=embed_tokens,
            rotary_emb=rotary_emb,
            position_ids=None,
            causal_mask_4d=causal_mask_4d,
        )

    # ----- Per-head CE: trunk-only vs student -----
    print("\n[INFO] Per-head CE / KL (averaged over masked speech positions)")
    print(f"{'k':>3} {'n':>8} {'trunk_ce':>10} {'student_ce':>12} {'kl(s||t)':>10} "
          f"{'trunk_acc':>10} {'student_acc':>12}")
    print("-" * 75)

    for k_idx, mtp_h in enumerate(mtp_hiddens, start=1):
        valid_len = T - k_idx - 1
        if valid_len <= 0:
            continue

        # Slices matching train_mtp.mtp_loss exactly.
        student_h = mtp_h[:, :valid_len]                                # [B, valid, H]
        trunk_h = trunk_hidden[:, k_idx: k_idx + valid_len]              # [B, valid, H]
        target_ids = input_ids[:, k_idx + 1: k_idx + 1 + valid_len]      # [B, valid]
        mask = speech_mask[:, k_idx + 1: k_idx + 1 + valid_len].float()  # [B, valid]
        n_total = mask.sum().clamp_min(1)

        # Compute both logits (chunk to control memory).
        chunk = 64
        s_ce_sum = student_h.new_zeros((), dtype=torch.float32)
        t_ce_sum = student_h.new_zeros((), dtype=torch.float32)
        kl_sum = student_h.new_zeros((), dtype=torch.float32)
        s_correct = student_h.new_zeros((), dtype=torch.float32)
        t_correct = student_h.new_zeros((), dtype=torch.float32)

        for start in range(0, valid_len, chunk):
            end = min(start + chunk, valid_len)
            m = mask[:, start:end]
            if m.sum() == 0:
                continue
            tgt = target_ids[:, start:end]
            s_log = lm_head(student_h[:, start:end])     # [B, c, V]
            t_log = lm_head(trunk_h[:, start:end])       # [B, c, V]

            # Per-position CE (full vocab — exact, no top-K).
            V = s_log.size(-1)
            s_ce_pos = F.cross_entropy(
                s_log.reshape(-1, V), tgt.reshape(-1), reduction="none"
            ).view(B, end - start)
            t_ce_pos = F.cross_entropy(
                t_log.reshape(-1, V), tgt.reshape(-1), reduction="none"
            ).view(B, end - start)
            s_ce_sum = s_ce_sum + (s_ce_pos * m).sum()
            t_ce_sum = t_ce_sum + (t_ce_pos * m).sum()

            # KL(student || teacher) full-vocab at T=1 (no temperature here so
            # it's comparable to a raw posterior gap — different scale than
            # the training-time KL which uses T=2.0).
            s_lp = F.log_softmax(s_log, dim=-1)
            t_lp = F.log_softmax(t_log, dim=-1)
            t_p = t_lp.exp()
            kl_pos = (t_p * (t_lp - s_lp)).sum(dim=-1)
            kl_sum = kl_sum + (kl_pos * m).sum()

            s_correct = s_correct + ((s_log.argmax(-1) == tgt).float() * m).sum()
            t_correct = t_correct + ((t_log.argmax(-1) == tgt).float() * m).sum()

            del s_log, t_log, s_lp, t_lp, t_p

        trunk_ce = (t_ce_sum / n_total).item()
        student_ce = (s_ce_sum / n_total).item()
        kl = (kl_sum / n_total).item()
        trunk_acc = (t_correct / n_total).item()
        student_acc = (s_correct / n_total).item()
        print(f"{k_idx:>3d} {int(n_total.item()):>8d} {trunk_ce:>10.4f} "
              f"{student_ce:>12.4f} {kl:>10.4f} {trunk_acc:>10.4f} {student_acc:>12.4f}")

    print()
    print("Interpretation:")
    print("  If trunk_ce has the same shape as student_ce (e.g. k=2 spike),")
    print("  then the U-shape is inherent to speech statistics and the heads")
    print("  are correctly distilling the trunk. The training CE log is just")
    print("  reflecting that distribution.")
    print("  If trunk_ce is monotonic but student_ce spikes at k=2, then head 2")
    print("  has a real problem (initialization, optimization, or wiring).")


if __name__ == "__main__":
    main()
