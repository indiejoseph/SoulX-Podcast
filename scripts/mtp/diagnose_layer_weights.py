"""Per-layer weight + output statistics on the MTP checkpoint.

For each MTP layer, prints:
  - weight stats per submodule (mean, std, norm, abs-max) — looks for
    anomalies in layer 1 specifically.
  - output activation stats (mean, std, abs-max) from one batch — looks
    for saturated / collapsed hidden states feeding the next head.

If layer 1's stats are markedly different from layers 0/2/3, that's a
strong signal head 2 has a structural / optimization issue independent
of the loss math.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from typing import Dict, List

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from soulxpodcast.training.mtp_dataset import MtpCollator, MtpDataset, MtpDatasetConfig
from soulxpodcast.training.mtp_module import MtpConfig, SequentialMTP, build_causal_mask_4d


DEFAULT_CKPT = "runs/hpc/mtp_v2/mtp_step8000.pt"
DEFAULT_BASE = "pretrained_models/SoulX-Podcast-1.7B-dialect-avg"
DEFAULT_DS = "tmp/dataset_small_with_tokens"


def tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().to(torch.float32)
    return {
        "mean": float(t.mean()),
        "std": float(t.std()),
        "norm": float(t.norm()),
        "absmax": float(t.abs().max()),
        "shape": tuple(t.shape),
    }


def print_layer_weight_stats(state_dict: Dict[str, torch.Tensor], n_layers: int):
    """Per-layer breakdown of trained MTP weights."""
    submods = [
        "norm_h.weight",
        "norm_e.weight",
        "proj.weight",
        "transformer.self_attn.q_proj.weight",
        "transformer.self_attn.k_proj.weight",
        "transformer.self_attn.v_proj.weight",
        "transformer.self_attn.o_proj.weight",
        "transformer.self_attn.q_norm.weight",
        "transformer.self_attn.k_norm.weight",
        "transformer.mlp.gate_proj.weight",
        "transformer.mlp.up_proj.weight",
        "transformer.mlp.down_proj.weight",
        "transformer.input_layernorm.weight",
        "transformer.post_attention_layernorm.weight",
    ]
    print("\n" + "=" * 80)
    print("PER-LAYER WEIGHT STATISTICS")
    print("=" * 80)
    print(f"{'submodule':<48} | " + " | ".join(f"L{i:<3}{'norm':<10}" for i in range(n_layers)))
    print("-" * (48 + 18 * n_layers))
    for sm in submods:
        row = [f"{sm:<48}"]
        for li in range(n_layers):
            key = f"layers.{li}.{sm}"
            if key in state_dict:
                row.append(f"{tensor_stats(state_dict[key])['norm']:>13.4f}")
            else:
                row.append(f"{'(missing)':>13}")
        print(" | ".join(row))

    print("\n" + "=" * 80)
    print("PER-LAYER WEIGHT ABS-MAX (saturation check)")
    print("=" * 80)
    print(f"{'submodule':<48} | " + " | ".join(f"L{i:<3}{'absmax':<8}" for i in range(n_layers)))
    print("-" * (48 + 18 * n_layers))
    for sm in submods:
        row = [f"{sm:<48}"]
        for li in range(n_layers):
            key = f"layers.{li}.{sm}"
            if key in state_dict:
                row.append(f"{tensor_stats(state_dict[key])['absmax']:>13.4f}")
            else:
                row.append(f"{'(missing)':>13}")
        print(" | ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--dataset", default=DEFAULT_DS)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    print(f"[INFO] Loading MTP checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mtp_cfg = MtpConfig(**ckpt["mtp_config"])
    n_layers = mtp_cfg.num_mtp_layers
    state = ckpt["mtp_state"]
    print(f"[INFO] step={ckpt['step']}, num_mtp_layers={n_layers}")

    # ---- (1) WEIGHT STATS ----
    print_layer_weight_stats(state, n_layers)

    # ---- (2) OUTPUT ACTIVATION STATS (forward pass) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"\n[INFO] Loading base for forward pass: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map=device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)

    decoder_layer_cls = base.model.layers[0].__class__
    mtp = SequentialMTP(mtp_cfg, decoder_layer_cls, base.config)
    mtp.load_state_dict(state)
    mtp = mtp.to(device=device, dtype=dtype)
    mtp.eval()

    hf_ds = load_from_disk(args.dataset)
    hf_ds = hf_ds.remove_columns([c for c in hf_ds.column_names
                                   if c not in ["text", "speech_tokens", "lang"]])
    hf_ds = hf_ds.select(range(args.n))
    ds_cfg = MtpDatasetConfig(
        speech_token_offset=getattr(base.config, "speech_token_offset", 153595)
    )
    dataset = MtpDataset(hf_ds, tokenizer, ds_cfg)
    collator = MtpCollator(pad_token_id=tokenizer.pad_token_id or 0)
    samples = [s for s in (dataset[i] for i in range(len(dataset))) if s is not None][:args.n]
    batch = collator(samples)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    B, T = input_ids.shape

    with torch.no_grad():
        trunk_out = base.model(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=False, return_dict=True)
        trunk_hidden = trunk_out.last_hidden_state
        causal_mask_4d = build_causal_mask_4d(attention_mask, dtype=trunk_hidden.dtype)
        mtp_hiddens = mtp(
            trunk_hidden=trunk_hidden, input_ids=input_ids,
            embed_tokens=base.model.embed_tokens,
            rotary_emb=base.model.rotary_emb,
            causal_mask_4d=causal_mask_4d,
        )

    print("\n" + "=" * 80)
    print("HEAD OUTPUT ACTIVATION STATS (one batch, all positions)")
    print("=" * 80)
    print(f"{'head':<8} {'shape':<20} {'mean':>10} {'std':>10} {'absmax':>10} {'%saturated_>30':>14}")
    print("-" * 80)
    # Compare against trunk hidden as the reference distribution.
    s = tensor_stats(trunk_hidden)
    sat = float((trunk_hidden.abs() > 30).float().mean()) * 100
    print(f"{'trunk':<8} {str(s['shape']):<20} {s['mean']:>10.4f} {s['std']:>10.4f} "
          f"{s['absmax']:>10.4f} {sat:>14.4f}")
    for k_idx, h_k in enumerate(mtp_hiddens, start=1):
        s = tensor_stats(h_k)
        sat = float((h_k.abs() > 30).float().mean()) * 100
        print(f"{'k='+str(k_idx):<8} {str(s['shape']):<20} {s['mean']:>10.4f} {s['std']:>10.4f} "
              f"{s['absmax']:>10.4f} {sat:>14.4f}")

    # ---- (3) DIRECT CHECK: would head 2 work if fed trunk_hidden directly? ----
    # Replace prev_h for head 2 with trunk's own h at position [k, k+T-k_idx] and
    # see if CE recovers. Tests the hypothesis that the chained h_1 is what's
    # poisoning head 2 specifically.
    print("\n" + "=" * 80)
    print("ISOLATION TEST: head 2 with prev_h = trunk_hidden[1:] instead of h_1")
    print("=" * 80)
    lm_head = base.lm_head
    import torch.nn.functional as F
    speech_mask = batch["speech_mask"].to(device)

    # Run head 2 in isolation with prev_h = trunk_hidden[:, 1:] (skipping head 1 entirely).
    k_idx = 2
    valid_len_mod = T - k_idx  # what mtp_module would use
    layer_2 = mtp.layers[1]
    prev_h_isolated = trunk_hidden[:, 1:1+valid_len_mod]   # [B, T-2, H]  trunk view at p+1
    target_embed_2 = base.model.embed_tokens(input_ids[:, k_idx:])  # [B, T-2, H]
    pos_ids = torch.arange(valid_len_mod, device=device).unsqueeze(0).expand(B, -1)
    causal_mask_2 = causal_mask_4d[:, :, :valid_len_mod, :valid_len_mod]
    cos_sin = base.model.rotary_emb(prev_h_isolated, pos_ids)
    with torch.no_grad():
        h_iso = layer_2(prev_h_isolated, target_embed_2, pos_ids, cos_sin, causal_mask_2)

    # Compute CE for the isolated head vs the chained head, at the same valid positions.
    valid_len_loss = T - k_idx - 1
    target_ids = input_ids[:, k_idx + 1: k_idx + 1 + valid_len_loss]
    mask = speech_mask[:, k_idx + 1: k_idx + 1 + valid_len_loss].float()
    n_total = mask.sum().clamp_min(1)

    def head_ce(h_slice):
        s_log = lm_head(h_slice[:, :valid_len_loss])
        V = s_log.size(-1)
        ce_pos = F.cross_entropy(
            s_log.reshape(-1, V), target_ids.reshape(-1), reduction="none"
        ).view(B, valid_len_loss)
        return float((ce_pos * mask).sum() / n_total), float(
            ((s_log.argmax(-1) == target_ids).float() * mask).sum() / n_total
        )

    with torch.no_grad():
        chained_ce, chained_acc = head_ce(mtp_hiddens[1])  # h_2 chained
        iso_ce, iso_acc = head_ce(h_iso)                   # h_2 isolated (prev_h=trunk)

    print(f"  chained  (prev_h = h_1):           CE={chained_ce:.4f}  acc={chained_acc:.4f}")
    print(f"  isolated (prev_h = trunk_hidden):  CE={iso_ce:.4f}  acc={iso_acc:.4f}")
    if iso_ce + 1.0 < chained_ce:
        print("  >> Isolated CE is significantly LOWER → h_1 (head 1's output) is")
        print("     poisoning head 2's input. The chained signal is the problem.")
    elif chained_ce + 1.0 < iso_ce:
        print("  >> Chained CE is significantly LOWER → head 2 NEEDS h_1's signal;")
        print("     the layer-2 weights are doing fine.")
    else:
        print("  >> Both CEs are similar → head 2's badness is intrinsic to its")
        print("     own weights, not the prev_h source.")


if __name__ == "__main__":
    main()
