"""Diagnose why MTP drafts never match trunk.

Tests TWO things:
  1. Does MTP layer 1 at an INTERIOR position match the trunk's prediction?
     (Tests whether training actually worked.)
  2. What tokens does MTP layer 1 produce at the inference position vs trunk?
     (Tests whether the inference forward is consistent with training.)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from __future__ import annotations

import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from soulxpodcast.training.mtp_dataset import (
    DIALECT_PREFIX, MtpDatasetConfig, SPECIAL_TOKENS,
)
from soulxpodcast.training.mtp_inference import (
    _build_causal_mask, _mtp_layer_forward_seq,
)
from soulxpodcast.training.mtp_module import MtpConfig, SequentialMTP


MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect"
DATASET_PATH = "/notebooks/projects/SoulX-Podcast/tmp/dataset_small_with_tokens"


def load_mtp(ckpt_path, base):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mtp = SequentialMTP(MtpConfig(**ckpt["mtp_config"]),
                        base.model.layers[0].__class__, base.config)
    mtp.load_state_dict(ckpt["mtp_state"])
    return mtp.to(device="cuda", dtype=torch.bfloat16).eval()


@torch.inference_mode()
def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/mtp_overfit_klonly/mtp_final.pt"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    mtp = load_mtp(ckpt_path, base)

    # Build a full training-like sequence from sample 0 of the overfit set.
    hf_ds = load_from_disk(DATASET_PATH).remove_columns(["audio", "id", "phone"])
    sample = hf_ds[0]
    special = {k: tokenizer.encode(v, add_special_tokens=False)[0]
               for k, v in SPECIAL_TOKENS.items()}
    cfg = MtpDatasetConfig()

    text_ids = tokenizer.encode(sample["text"], add_special_tokens=False)
    speech_ids = [int(x) + cfg.speech_token_offset for x in sample["speech_tokens"].split()]
    full = ([special["task_podcast"], special["speaker_0"], special["text_start"]]
            + text_ids
            + [special["text_end"], special["semantic_token_start"]]
            + speech_ids
            + [special["semantic_token_end"]])
    full_ids = torch.tensor([full], dtype=torch.long, device="cuda")
    T = full_ids.shape[1]
    print(f"[setup] full training sequence length: T={T}")

    embed_tokens = base.model.embed_tokens
    lm_head = base.lm_head
    rotary_emb = base.model.rotary_emb

    # ---- Trunk forward on the full true sequence ----
    out = base.model(input_ids=full_ids, use_cache=False, return_dict=True)
    trunk_hidden = out.last_hidden_state  # [1, T, H]
    trunk_logits = lm_head(trunk_hidden)
    trunk_argmax = trunk_logits.argmax(dim=-1)  # [1, T]
    # trunk_argmax[t] is trunk's prediction for the token AT POSITION t+1
    # (given context up to t). We can compare to full_ids[t+1] for next-token acc.

    # ---- MTP layer 1 forward, training-style ----
    # In training, MTP layer 1 takes prev_h[:, :T-1] with target_embed of
    # length T-1. Let's match exactly.
    valid_len_1 = T - 1
    pos_ids_1 = torch.arange(valid_len_1, device="cuda").unsqueeze(0)
    mask_1 = _build_causal_mask(valid_len_1, trunk_hidden.dtype, "cuda")

    target_embed_1_train = embed_tokens(full_ids[:, 1:T])  # [1, T-1, H]
    h_slice_1 = trunk_hidden[:, :T-1]                       # [1, T-1, H]

    mtp1_out_train = _mtp_layer_forward_seq(
        mtp.layers[0], h_slice_1, target_embed_1_train, pos_ids_1, rotary_emb, mask_1,
    )  # [1, T-1, H]
    mtp1_argmax_train = lm_head(mtp1_out_train).argmax(dim=-1)  # [1, T-1]
    # mtp1_argmax_train[p] is MTP layer 1's prediction at training position p
    # → it should predict full_ids[p+2] (the training CE target — but we
    # trained with KL only, so we expect it to match trunk's argmax instead).

    # ---- Comparison: MTP layer 1 (training-style at interior positions) ----
    # vs trunk's prediction at the same offset (+2).
    # Trunk's prediction for position p+2 given context up to p+1 is
    # trunk_argmax[p+1].
    # So at training MTP position p (predicting position p+2), we compare to
    # trunk_argmax[p+1].
    valid_compare = T - 2  # positions p in [0..T-3], avoiding edge
    n_match_trunk = (mtp1_argmax_train[0, :valid_compare] ==
                     trunk_argmax[0, 1:1+valid_compare]).sum().item()
    n_match_data  = (mtp1_argmax_train[0, :valid_compare] ==
                     full_ids[0, 2:2+valid_compare]).sum().item()
    print(f"\n[interior] MTP layer 1 at TRAINING positions (p in [0, T-2)):")
    print(f"  matches trunk's argmax at the same target: "
          f"{n_match_trunk}/{valid_compare} ({n_match_trunk/valid_compare:.1%})")
    print(f"  matches dataset token at the same target:  "
          f"{n_match_data}/{valid_compare} ({n_match_data/valid_compare:.1%})")
    print(f"  (KL-only training should give HIGH trunk-match, LOWER data-match)")

    # ---- Now the INFERENCE-style forward: extend by 1 ----
    # Treat first T-1 tokens as the "prompt"; predict token at T-1 (which is
    # full_ids[T-1]) using MTP layer 1.
    # Inference setup:
    #   prompt = full_ids[:, :T-1]    (length T-1)
    #   we want d_0 to predict full_ids[T-1]
    prompt = full_ids[:, :T-1]
    out_prompt = base.model(input_ids=prompt, use_cache=False, return_dict=True)
    trunk_hidden_p = out_prompt.last_hidden_state  # [1, T-1, H]
    h_t = trunk_hidden_p[:, -1:]
    d0 = lm_head(h_t).argmax(dim=-1)  # primary head pick (= trunk's pick for position T-1)

    # Build MTP layer 1 input: prev_h = trunk_hidden_p (length T-1),
    # target_embed = [embed(full_ids[1]), embed(full_ids[2]), ..., embed(full_ids[T-2]), embed(d_0)]
    L = T - 1
    shift = full_ids[:, 1:L]  # length L-1
    tgt_ids = torch.cat([shift, d0], dim=1)  # length L
    target_embed_infer = embed_tokens(tgt_ids)
    pos_ids_infer = torch.arange(L, device="cuda").unsqueeze(0)
    mask_infer = _build_causal_mask(L, trunk_hidden_p.dtype, "cuda")
    mtp1_out_infer = _mtp_layer_forward_seq(
        mtp.layers[0], trunk_hidden_p, target_embed_infer, pos_ids_infer, rotary_emb, mask_infer,
    )
    d1 = lm_head(mtp1_out_infer[:, -1:]).argmax(dim=-1)

    # What would trunk predict at position T (after seeing full_ids[:T-1] + d_0)?
    extended = torch.cat([prompt, d0], dim=1)
    out_ext = base.model(input_ids=extended, use_cache=False, return_dict=True)
    trunk_pred_for_T = lm_head(out_ext.last_hidden_state[:, -1:]).argmax(dim=-1)

    print(f"\n[inference-style] predicting at position T-1 of an extended sequence:")
    print(f"  d_0 (primary head) = {d0.item()}  (trunk would also pick this)")
    print(f"  d_1 (MTP layer 1)  = {d1.item()}")
    print(f"  trunk pred at T    = {trunk_pred_for_T.item()}")
    print(f"  d_1 matches trunk?  {bool(d1.item() == trunk_pred_for_T.item())}")
    print(f"  full_ids at T-1 (the true next token from dataset) = {full_ids[0, T-1].item()}")
    print(f"  d_0 matches full_ids[T-1]? {bool(d0.item() == full_ids[0, T-1].item())}")

    # ---- Inference-style interior positions ----
    # Check whether MTP layer 1's INTERIOR positions of the inference forward
    # match trunk's argmax at the matching target. If yes, the issue is
    # specifically the LAST position (where target_embed differs from training).
    mtp1_logits_infer = lm_head(mtp1_out_infer)            # [1, T-1, V]
    mtp1_argmax_infer = mtp1_logits_infer.argmax(dim=-1)   # [1, T-1]
    # Position p (in inference forward) predicts position p+2.
    # Compare to trunk_argmax[p+1] (trunk's prediction for position p+2).
    valid_inf = T - 3
    n_match_inf = (mtp1_argmax_infer[0, :valid_inf] ==
                   trunk_argmax[0, 1:1+valid_inf]).sum().item()
    print(f"\n[inference-style interior] MTP layer 1 at p in [0..T-3) of inference forward:")
    print(f"  matches trunk's argmax: {n_match_inf}/{valid_inf} ({n_match_inf/valid_inf:.1%})")
    print(f"  (should match the training-style interior result if forward is consistent)")

    # ---- Top-K analysis at the LAST inference position ----
    last_mtp_logits = mtp1_logits_infer[:, -1]  # [1, V]
    # Trunk's prediction for position T (after seeing d_0). We computed this
    # above as trunk_pred_for_T.
    extended_again = torch.cat([prompt, d0], dim=1)
    out_ext2 = base.model(input_ids=extended_again, use_cache=False, return_dict=True)
    trunk_logits_T = lm_head(out_ext2.last_hidden_state[:, -1])  # [1, V]

    for topk in [1, 5, 20, 100]:
        trunk_topk = trunk_logits_T.topk(topk, dim=-1).indices[0].tolist()
        mtp_top1 = last_mtp_logits.argmax(dim=-1).item()
        in_topk = mtp_top1 in trunk_topk
        print(f"  MTP's top-1 in trunk's top-{topk}? {in_topk}")

    # Rank of MTP's top-1 in trunk's full ranking.
    trunk_sorted = trunk_logits_T.argsort(dim=-1, descending=True)[0].tolist()
    mtp_top1 = last_mtp_logits.argmax(dim=-1).item()
    rank = trunk_sorted.index(mtp_top1)
    print(f"  rank of MTP's top-1 in trunk's distribution: {rank} (0 = best)")


if __name__ == "__main__":
    main()
