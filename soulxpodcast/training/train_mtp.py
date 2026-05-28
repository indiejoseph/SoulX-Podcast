"""Train Sequential MTP heads on top of frozen SoulX-Podcast LLM.

PLAN.md Phase 2 — Medusa-2 style:
  - Base 1.7B Qwen3 trunk FROZEN (no fine-tune)
  - K-1 MTP layers + shared lm_head — TRAINABLE
  - Loss = depth-weighted ( α · CE(hard) + (1-α) · KL(trunk_soft) )
  - Loss applied only at speech-token positions

Target compute: 1×H100 80GB (your HPC node). Bf16 throughout.
For dev / smoke-test on 1×3090, drop `--batch_size` to ~4 and use the
debug-100 dataset slice.

Usage:
    # Smoke test on debug dataset (1 epoch, tiny batch)
    python -m soulxpodcast.training.train_mtp \\
        --dataset_path data/debug_100 \\
        --model_path  pretrained_models/SoulX-Podcast-1.7B-dialect \\
        --output_dir  runs/mtp_smoke \\
        --batch_size  2 --max_steps 20 --log_every 1

    # Full training
    python -m soulxpodcast.training.train_mtp \\
        --dataset_path data/your_full_dataset \\
        --model_path  pretrained_models/SoulX-Podcast-1.7B-dialect \\
        --output_dir  runs/mtp_full \\
        --batch_size  16 --num_epochs 2 \\
        --lr 1e-4 --warmup_steps 500
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from soulxpodcast.training.mtp_dataset import (
    MtpCollator, MtpDataset, MtpDatasetConfig,
)
from soulxpodcast.training.mtp_module import (
    MtpConfig, SequentialMTP, build_causal_mask_4d,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_mtp")


def resolve_speech_token_offset(tokenizer) -> int:
    """Resolve the LLM-vocab id of raw s3tokenizer token 0."""
    ids = tokenizer.encode("<|0|>", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            "Tokenizer did not encode '<|0|>' as one token; cannot derive "
            f"speech_token_offset safely. Got ids={ids}"
        )
    return int(ids[0])


# ------------------------------------------------------------------------- #
# Loss
# ------------------------------------------------------------------------- #

def _masked_mean(loss_per_position: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `loss_per_position` over positions where `mask==1`. Returns 0
    when the mask is all zeros (caller should still log to detect dead batches)."""
    denom = mask.sum().clamp_min(1).to(loss_per_position.dtype)
    return (loss_per_position * mask).sum() / denom


def mtp_loss(
    mtp_hiddens_list: list,                # length K-1, each [B, T-k, H]   (req grad)
    trunk_hidden: torch.Tensor,             # [B, T, H]  (no grad — teacher hidden)
    lm_head: nn.Module,                     # shared lm_head (frozen)
    input_ids: torch.LongTensor,            # [B, T]
    speech_mask: torch.LongTensor,          # [B, T]
    ce_weight: float = 0.3,
    kl_weight: float = 0.7,
    kl_temperature: float = 1.0,
    kl_top_k: int = 0,                      # 0 = full-vocab KL; >0 = top-K KL distillation
    depth_decay: float = 0.5,
    ce_target: str = "data",                # "data" = next-token from dataset;
                                            # "trunk_argmax" = teacher argmax (richer
                                            # hard-label distillation — directly
                                            # optimizes spec-decode acceptance)
    loss_chunk_size: int = 256,             # chunk along seq to bound peak memory
):
    """Memory-efficient MTP loss.

    The naive version materializes [B, T, V] tensors for student logits,
    teacher logits, log_softmax outputs, and teacher probs — for each of K-1
    MTP heads. With V≈159K this is many GB and OOMs at production batch
    sizes. This version chunks the sequence dimension: for each chunk we
      1) project hiddens through lm_head to get [B, chunk, V]
      2) compute CE + KL on that chunk
      3) accumulate the sums; free intermediates before next chunk

    Peak memory per chunk: O(B * chunk_size * V). With chunk=256 and B=16
    that's ~1.3 GiB per logits tensor (bf16), comfortable on H100.

    NOTE: trunk_hidden[t] is the trunk's hidden at position t. Projecting
    through lm_head gives the trunk's prediction for position t+1. The CE
    target for MTP head k_idx at position p (in the sliced frame) is the
    token at position k_idx+1+p in `input_ids`; the matching teacher logit
    comes from `lm_head(trunk_hidden[k_idx + p])` — i.e. teacher hidden
    slice starts at offset k_idx, not k_idx+1 (off-by-one was the original
    bug fixed earlier in development).
    """
    B, T = input_ids.shape
    total_loss = input_ids.new_zeros((), dtype=torch.float32)
    per_head = []
    do_ce = ce_weight != 0.0
    do_kl = kl_weight != 0.0

    for k_idx, mtp_h in enumerate(mtp_hiddens_list, start=1):
        # mtp_h: [B, T-k_idx, H]. The first valid position predicts token at
        # offset +k_idx+1. To have a real target we need T - k_idx - 1 positions.
        valid_len = T - k_idx - 1
        if valid_len <= 0:
            per_head.append({"k": k_idx, "ce": 0.0, "kl": 0.0, "n": 0,
                             "acc_top1": 0.0})
            continue

        mtp_h_slice = mtp_h[:, :valid_len]                                  # [B, valid, H]
        trunk_h_slice = trunk_hidden[:, k_idx: k_idx + valid_len]            # [B, valid, H]
        target_ids = input_ids[:, k_idx + 1: k_idx + 1 + valid_len]          # [B, valid]
        mask = speech_mask[:, k_idx + 1: k_idx + 1 + valid_len].to(torch.float32)  # [B, valid]

        ce_sum = mtp_h.new_zeros((), dtype=torch.float32)
        kl_sum = mtp_h.new_zeros((), dtype=torch.float32)
        correct_sum = mtp_h.new_zeros((), dtype=torch.float32)
        n_total = mask.sum().clamp_min(1).to(torch.float32)

        # Iterate chunks along the sequence dim.
        for start in range(0, valid_len, loss_chunk_size):
            end = min(start + loss_chunk_size, valid_len)
            mask_chunk = mask[:, start:end]
            # Skip chunks with no mask-active positions — saves a full lm_head.
            if mask_chunk.sum() == 0:
                continue

            s_logits = lm_head(mtp_h_slice[:, start:end])  # [B, chunk, V]
            need_teacher_logits = do_kl or ce_target == "trunk_argmax"
            t_logits = None
            if need_teacher_logits:
                with torch.no_grad():
                    t_logits = lm_head(trunk_h_slice[:, start:end])  # [B, chunk, V]

            if ce_target == "trunk_argmax":
                # Teacher's argmax as the CE pseudo-label. Richer than the raw
                # data label (trunk has integrated context and committed to its
                # top choice) and directly optimizes the spec-decode metric
                # (student.argmax == trunk.argmax).
                with torch.no_grad():
                    assert t_logits is not None
                    target_chunk = t_logits.argmax(dim=-1)
            else:
                target_chunk = target_ids[:, start:end]

            if do_ce:
                # CE (fused log_softmax + nll, memory-efficient).
                ce_pos = F.cross_entropy(
                    s_logits.reshape(-1, s_logits.size(-1)),
                    target_chunk.reshape(-1),
                    reduction="none",
                ).view(B, end - start)
                ce_sum = ce_sum + (ce_pos * mask_chunk).sum()

            # KL(student || teacher).
            #
            # Two variants:
            #   - kl_top_k == 0: full-vocab KL. Materializes log/probs over all
            #     V≈159K positions per chunk position. Strong supervision
            #     including tail suppression, but expensive.
            #   - kl_top_k >  0: top-K KL. Distills only against teacher's
            #     top-K positions, renormalizing both teacher and student
            #     over that set. Tail mass is left unconstrained.
            #
            # Top-K KL is the natural fit for SoulX-Podcast because:
            #   1. At inference we already apply top_k=100 sampling, so tail
            #      mass never gets sampled — distilling it wastes capacity.
            #   2. Vocab has 152K text + 6.5K speech tokens; at a speech
            #      position the teacher assigns ~0 mass to text region.
            #      Top-K skips this uninformative bulk.
            #   3. ~1600× memory saving on the KL intermediates (K=100 vs V=159K).
            if do_kl:
                assert t_logits is not None
                if kl_top_k > 0:
                    with torch.no_grad():
                        t_topk_vals, t_topk_idx = t_logits.topk(kl_top_k, dim=-1)
                        t_log_topk = F.log_softmax(t_topk_vals / kl_temperature, dim=-1)
                        t_prob_topk = t_log_topk.exp()
                    # Gather student logits at teacher's top-K positions, then
                    # renormalize within that K-element support. This matches
                    # MiniLLM / DistiLLM truncated-KL convention.
                    s_topk = s_logits.gather(-1, t_topk_idx)
                    s_log_topk = F.log_softmax(s_topk / kl_temperature, dim=-1)
                    kl_pos = (t_prob_topk * (t_log_topk - s_log_topk)).sum(dim=-1)
                else:
                    s_log = F.log_softmax(s_logits / kl_temperature, dim=-1)
                    with torch.no_grad():
                        t_log = F.log_softmax(t_logits / kl_temperature, dim=-1)
                        t_prob = t_log.exp()
                    kl_pos = (t_prob * (t_log - s_log)).sum(dim=-1)
                kl_sum = kl_sum + (kl_pos * mask_chunk).sum() * (kl_temperature ** 2)

            # Top-1 acc against dataset tokens (metric only; under KL-only
            # training this stays low because the model mimics trunk, not data).
            with torch.no_grad():
                top1 = s_logits.argmax(dim=-1)
                correct_sum = correct_sum + (
                    (top1 == target_chunk).to(torch.float32) * mask_chunk
                ).sum()

            # Explicitly drop chunk tensors before the next iteration so the
            # autograd graph doesn't retain them all simultaneously.
            del s_logits
            if t_logits is not None:
                del t_logits
            if do_kl:
                if kl_top_k > 0:
                    del t_topk_vals, t_topk_idx, t_log_topk, t_prob_topk, s_topk, s_log_topk
                else:
                    del s_log, t_log, t_prob

        ce = ce_sum / n_total
        kl = kl_sum / n_total
        acc = correct_sum / n_total

        layer_loss = ce_weight * ce + kl_weight * kl
        weight = depth_decay ** (k_idx - 1)
        total_loss = total_loss + weight * layer_loss

        per_head.append({
            "k": k_idx,
            "ce": float(ce.detach().item()),
            "kl": float(kl.detach().item()),
            "n": int(n_total.item()),
            "acc_top1": float(acc.item()),
        })

    return total_loss, per_head


# ------------------------------------------------------------------------- #
# Training step / loop
# ------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    dataset_path: str
    model_path: str
    output_dir: str
    batch_size: int = 8
    num_epochs: int = 1
    max_steps: int = -1                    # -1 = run full epochs
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    ce_weight: float = 0.3
    kl_weight: float = 0.7
    kl_temperature: float = 1.0
    kl_top_k: int = 0                      # 0 = full-vocab KL; >0 = top-K KL distillation
    depth_decay: float = 0.5
    ce_target: str = "data"                # "data" or "trunk_argmax" (see mtp_loss)
    num_mtp_layers: int = 3
    max_total_tokens: int = 2048
    max_speech_tokens: int = 750           # drop clips with >this many speech tokens (~30s @ 25Hz)
    num_workers: int = 2
    log_every: int = 10
    save_every: int = 500
    seed: int = 42
    dtype: str = "bf16"                    # "bf16" | "fp16" | "fp32"
    eval_split_size: int = 0               # 0 = no eval split
    max_samples: int = 0                   # 0 = use full dataset, >0 = first N (overfit tests)
    shuffle: bool = True                   # disable for repeatable overfit cycles
    # Weights & Biases logging
    wandb: bool = False                    # enable wandb logging
    wandb_project: str = "soulxpodcast-mtp"
    wandb_run_name: str = ""               # blank = let wandb autogenerate
    wandb_entity: str = ""                 # blank = default user/team


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--ce_weight", type=float, default=0.3)
    p.add_argument("--kl_weight", type=float, default=0.7)
    p.add_argument("--kl_temperature", type=float, default=1.0)
    p.add_argument("--kl_top_k", type=int, default=0,
                   help="If >0, distill against teacher's top-K positions only "
                        "(MiniLLM-style). 0 = full-vocab KL. Recommended: 100-200 "
                        "to match inference top_k.")
    p.add_argument("--depth_decay", type=float, default=0.5)
    p.add_argument("--ce_target", choices=["data", "trunk_argmax"], default="data",
                   help="CE label source: 'data' (next-token from dataset) or "
                        "'trunk_argmax' (teacher argmax pseudo-label — better for "
                        "spec-decode acceptance).")
    p.add_argument("--num_mtp_layers", type=int, default=3)
    p.add_argument("--max_total_tokens", type=int, default=2048)
    p.add_argument("--max_speech_tokens", type=int, default=750)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--eval_split_size", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0,
                   help="If >0, use only the first N samples (overfit tests).")
    p.add_argument("--no_shuffle", action="store_true",
                   help="Disable shuffling (useful for repeatable overfit).")
    # wandb
    p.add_argument("--wandb", action="store_true",
                   help="Log metrics + config to Weights & Biases.")
    p.add_argument("--wandb_project", default="soulxpodcast-mtp")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--wandb_entity", default="")
    args = p.parse_args()
    args_dict = vars(args)
    args_dict["shuffle"] = not args_dict.pop("no_shuffle")
    return TrainConfig(**args_dict)


def get_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def build_model_and_mtp(cfg: TrainConfig, dtype: torch.dtype):
    """Load frozen Qwen3 trunk + fresh MTP module sharing trunk embeddings/lm_head."""
    log.info(f"loading frozen base model: {cfg.model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=dtype, device_map="cuda",
    )
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    # Pull architectural info from the base config so the MTP matches.
    bc = base.config
    mtp_config = MtpConfig(
        hidden_size=bc.hidden_size,
        num_attention_heads=bc.num_attention_heads,
        num_key_value_heads=getattr(bc, "num_key_value_heads", bc.num_attention_heads),
        intermediate_size=bc.intermediate_size,
        num_mtp_layers=cfg.num_mtp_layers,
        rms_norm_eps=getattr(bc, "rms_norm_eps", 1e-6),
        max_position_embeddings=bc.max_position_embeddings,
        rope_theta=getattr(bc, "rope_theta", 1000000.0),
    )

    # Architectural template: trunk's decoder-layer class + the trunk's HF
    # config (so attention impl / RoPE / GQA / SwiGLU all match exactly).
    decoder_layer_cls = base.model.layers[0].__class__
    mtp = SequentialMTP(mtp_config, decoder_layer_cls, bc)

    # Warm-start every MTP layer from the trunk's LAST decoder layer + its
    # final RMSNorm. This gives each head a competent starting point (instead
    # of random init) and makes its `final_norm` match the trunk's post-norm
    # distribution that `lm_head` was trained on.
    mtp.init_from_trunk(base.model.layers[-1], base.model.norm)
    log.info("warm-started MTP layers from base.model.layers[-1] + base.model.norm")

    # Keep MTP params in fp32 (optimizer master copy). Compute runs in `dtype`
    # via `torch.autocast` in the train loop. Storing params directly in bf16
    # made small AdamW updates (~1e-5) round to zero, leaving norm weights
    # (norm_h, norm_e, q_norm, k_norm, layernorms, final_norm) stuck at init —
    # confirmed by inspecting runs/mtp_h100_v5_trunkargmax/mtp_step8000.pt where
    # all norm params were bit-identical across the 3 MTP layers.
    mtp = mtp.to(device="cuda")

    # Sanity log on param counts.
    trainable = sum(p.numel() for p in mtp.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in base.parameters())
    log.info(f"params: trainable_MTP={trainable/1e6:.1f}M  frozen_base={frozen/1e6:.1f}M")

    return base, mtp, mtp_config


def trunk_forward(base, input_ids, attention_mask):
    """One frozen forward pass — returns (hidden_states, logits)."""
    with torch.no_grad():
        out = base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state  # [B, T, H]
        logits = base.lm_head(hidden)   # [B, T, V]
    return hidden, logits


def _maybe_init_wandb(cfg: TrainConfig):
    """Initialize wandb if enabled. Returns the wandb module or None."""
    if not cfg.wandb:
        return None
    try:
        import wandb
    except ImportError:
        log.warning("--wandb passed but `wandb` package not installed; "
                    "skipping (run `pip install wandb` to enable).")
        return None
    init_kwargs = dict(
        project=cfg.wandb_project,
        config=asdict(cfg),
        dir=cfg.output_dir,
    )
    if cfg.wandb_run_name:
        init_kwargs["name"] = cfg.wandb_run_name
    if cfg.wandb_entity:
        init_kwargs["entity"] = cfg.wandb_entity
    wandb.init(**init_kwargs)
    log.info(f"wandb logging enabled: project={cfg.wandb_project} "
             f"run={wandb.run.name}")
    return wandb


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    dtype = get_dtype(cfg.dtype)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    wandb_run = _maybe_init_wandb(cfg)

    # ---- Tokenizer + dataset --------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
    speech_token_offset = resolve_speech_token_offset(tokenizer)
    log.info(f"speech_token_offset resolved from tokenizer: {speech_token_offset}")
    log.info(f"loading dataset: {cfg.dataset_path}")
    hf_ds = load_from_disk(cfg.dataset_path)
    # Drop unused columns. Keep `audio` only if `speech_tokens` is absent —
    # the dataloader fallback needs raw audio bytes to tokenize on the fly.
    keep_cols = ["text", "speech_tokens", "lang"]
    if "speech_tokens" not in hf_ds.column_names:
        keep_cols.append("audio")
        log.warning(
            "speech_tokens column missing — fallback will tokenize audio "
            "on the fly (~100x slower). Precompute the column for full runs."
        )
    drop_cols = [c for c in hf_ds.column_names if c not in keep_cols]
    if drop_cols:
        hf_ds = hf_ds.remove_columns(drop_cols)
        log.info(f"dropped columns: {drop_cols}")

    # Truncate to first N samples for overfit / debug runs.
    if cfg.max_samples > 0 and cfg.max_samples < len(hf_ds):
        hf_ds = hf_ds.select(range(cfg.max_samples))
        log.info(f"truncated dataset to first {cfg.max_samples} samples (overfit mode)")

    ds_cfg = MtpDatasetConfig(
        speech_token_offset=speech_token_offset,
        max_total_tokens=cfg.max_total_tokens,
        max_speech_tokens=cfg.max_speech_tokens,
    )
    dataset = MtpDataset(hf_ds, tokenizer, ds_cfg)

    collator = MtpCollator(pad_token_id=tokenizer.pad_token_id or 0)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
        collate_fn=collator, num_workers=cfg.num_workers, pin_memory=True,
        drop_last=True,
    )

    # ---- Model + MTP module ---------------------------------------------
    base, mtp, mtp_config = build_model_and_mtp(cfg, dtype)
    embed_tokens = base.model.embed_tokens
    lm_head = base.lm_head

    # ---- Optimizer / scheduler ------------------------------------------
    optim = torch.optim.AdamW(
        [p for p in mtp.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    # Estimate total steps for scheduler.
    steps_per_epoch = max(1, len(loader) // cfg.grad_accum_steps)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,
    )

    log.info(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}")

    # ---- Train loop ------------------------------------------------------
    global_step = 0
    accum_loss = 0.0
    accum_loss_count = 0
    accum_count = 0
    t_last_log = time.perf_counter()
    tokens_since_log = 0

    # When max_steps > 0, keep cycling the loader until we hit max_steps —
    # don't let num_epochs cut things short (matters for overfit tests where
    # the dataset is tiny and we want many passes over it).
    mtp.train()
    epoch = 0
    while True:
        log.info(f"=== epoch {epoch} ===")
        for batch_idx, batch in enumerate(loader):
            if not batch:
                continue  # all samples filtered

            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            speech_mask = batch["speech_mask"].cuda(non_blocking=True)

            # Forward in autocast(bf16): fp32 master params (mtp + accumulators)
            # with bf16 compute. lm_head is called inside mtp_loss so it must
            # also be under autocast — keep the whole forward block inside.
            with torch.autocast("cuda", dtype=dtype):
                # 1) Trunk forward (frozen). Don't materialize trunk_logits here —
                # the chunked loss applies lm_head() on the fly per chunk to avoid
                # the [B, T, 159K] tensor that would OOM at production batch sizes.
                with torch.no_grad():
                    out = base.model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True,
                    )
                    trunk_hidden = out.last_hidden_state

                # 2) MTP forward (trainable).
                causal_4d = build_causal_mask_4d(attention_mask, dtype=dtype)
                mtp_hiddens = mtp(
                    trunk_hidden=trunk_hidden,
                    input_ids=input_ids,
                    embed_tokens=embed_tokens,
                    rotary_emb=base.model.rotary_emb,
                    causal_mask_4d=causal_4d,
                )

                # 3) Loss (chunked: lm_head applied per sequence chunk).
                raw_loss, per_head = mtp_loss(
                    mtp_hiddens_list=mtp_hiddens,
                    trunk_hidden=trunk_hidden,
                    lm_head=lm_head,
                    input_ids=input_ids,
                    speech_mask=speech_mask,
                    ce_weight=cfg.ce_weight, kl_weight=cfg.kl_weight,
                    kl_temperature=cfg.kl_temperature, kl_top_k=cfg.kl_top_k,
                    depth_decay=cfg.depth_decay,
                    ce_target=cfg.ce_target,
                )
            loss = raw_loss / cfg.grad_accum_steps

            # 4) Backprop.
            loss.backward()
            accum_loss += float(raw_loss.detach().item())
            accum_loss_count += 1
            accum_count += 1
            tokens_since_log += int(speech_mask.sum().item())

            if accum_count >= cfg.grad_accum_steps:
                # Always compute the grad norm — when grad_clip <= 0 we still
                # log it (just don't clip). clip_grad_norm_ returns the
                # pre-clip total L2 norm across all trainable params.
                clip_max = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    mtp.parameters(), max_norm=clip_max
                )
                grad_norm = float(grad_norm_t.item()) if grad_norm_t is not None else 0.0
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                accum_count = 0

                if global_step % cfg.log_every == 0:
                    dt = time.perf_counter() - t_last_log
                    tps = tokens_since_log / max(dt, 1e-6)
                    lr_now = scheduler.get_last_lr()[0]
                    avg_loss = accum_loss / max(accum_loss_count, 1)
                    head_summary = " | ".join(
                        f"k={h['k']} acc={h['acc_top1']:.3f} ce={h['ce']:.3f} kl={h['kl']:.3f}"
                        for h in per_head
                    )
                    log.info(
                        f"step={global_step}/{total_steps}  "
                        f"loss={avg_loss:.4f}  lr={lr_now:.2e}  "
                        f"|grad|={grad_norm:.3f}  "
                        f"tok/s={tps:.0f}  | {head_summary}"
                    )
                    if wandb_run is not None:
                        metrics = {
                            "train/loss": avg_loss,
                            "train/lr": lr_now,
                            "train/grad_norm": grad_norm,
                            "train/tok_per_sec": tps,
                            "train/step": global_step,
                        }
                        for h in per_head:
                            k = h["k"]
                            metrics[f"head_{k}/acc_top1"] = h["acc_top1"]
                            metrics[f"head_{k}/ce"] = h["ce"]
                            metrics[f"head_{k}/kl"] = h["kl"]
                            metrics[f"head_{k}/n_positions"] = h["n"]
                        wandb_run.log(metrics, step=global_step)
                    accum_loss = 0.0
                    accum_loss_count = 0
                    tokens_since_log = 0
                    t_last_log = time.perf_counter()

                if cfg.save_every > 0 and global_step % cfg.save_every == 0:
                    save_path = out_dir / f"mtp_step{global_step}.pt"
                    torch.save({
                        "mtp_state": mtp.state_dict(),
                        "mtp_config": asdict(mtp_config),
                        "train_config": asdict(cfg),
                        "step": global_step,
                    }, save_path)
                    log.info(f"saved {save_path}")
                    if wandb_run is not None:
                        wandb_run.log({"checkpoint/step": global_step},
                                      step=global_step)

                if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    break
        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            break
        epoch += 1
        # Stop when we've done the requested number of epochs (only enforced
        # when max_steps is not set; max_steps takes priority).
        if cfg.max_steps <= 0 and epoch >= cfg.num_epochs:
            break

    # Final save.
    save_path = out_dir / "mtp_final.pt"
    torch.save({
        "mtp_state": mtp.state_dict(),
        "mtp_config": asdict(mtp_config),
        "train_config": asdict(cfg),
        "step": global_step,
    }, save_path)
    log.info(f"done. saved {save_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train(parse_args())
