"""LoRA fine-tune the Qwen3 trunk for style bias (PLAN.md Phase B).

Targets the SoulX-Podcast use case where you want to shift the trunk's
output preference — e.g. toward Hong Kong–style Cantonese — without:
  (a) catastrophic-forgetting English / Mandarin (LoRA is conservative)
  (b) destroying MTP head compatibility (small distribution shift can be
      recovered by a quick MTP refresh; see PLAN.md Phase C / Phase A')

What this trains:
  - LoRA adapters on Q/K/V/O attention projections + gate/up/down MLP
    projections of every Qwen3 decoder layer
  - Embeddings, lm_head, and norms stay FROZEN (LoRA's standard recipe)

Loss:
  - Next-token cross-entropy on SPEECH-TOKEN positions only by default
    (the audio style lives in speech-token preferences; we don't want
    to shift text-side behavior). Pass --include_text_loss to also train
    on text-token CE.

After training:
  - Output is a PEFT adapter (~30-100 MB) at `output_dir/adapter`.
  - Load at inference with:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, "<output_dir>/adapter")
        base = base.merge_and_unload()  # or keep separate
  - Run MTP refresh (Phase C) on the merged model to restore spec acceptance.

Usage (debug):
    python -m soulxpodcast.training.train_lora_trunk \\
        --dataset_path tmp/dataset_small_with_tokens \\
        --model_path  pretrained_models/SoulX-Podcast-1.7B-dialect \\
        --output_dir  runs/lora_hk_smoke \\
        --max_samples 50 --batch_size 2 --max_steps 30 \\
        --log_every 5 --num_workers 0

Usage (H100, full HK bias run):
    python -m soulxpodcast.training.train_lora_trunk \\
        --dataset_path /path/to/full_dataset \\
        --model_path  pretrained_models/SoulX-Podcast-1.7B-dialect \\
        --output_dir  runs/lora_hk_full \\
        --lang_filter yue \\
        --batch_size 8 --grad_accum_steps 2 \\
        --num_epochs 2 \\
        --lr 5e-5 --warmup_steps 500 \\
        --lora_rank 32 --lora_alpha 64 \\
        --save_every 1000 \\
        --wandb --wandb_run_name lora-hk-r32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from soulxpodcast.training.mtp_dataset import (
    MtpCollator, MtpDataset, MtpDatasetConfig,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_lora_trunk")


# ------------------------------------------------------------------------- #
# Config
# ------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    dataset_path: str
    model_path: str
    output_dir: str
    # Training
    batch_size: int = 4
    grad_accum_steps: int = 1
    num_epochs: int = 1
    max_steps: int = -1
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    dtype: str = "bf16"
    seed: int = 42
    # LoRA — matches CosyVoice2's working recipe (r=8, alpha=16, scale=2.0).
    # r=8 is sufficient to steer language/style bias and dramatically reduces
    # hidden-state drift (vs r=32) → safer co-adaptation with lm_head.
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    # Fully-trained modules (NOT LoRA-factorized). Default: just lm_head, which
    # matches CosyVoice2's working recipe and is required to prevent the
    # frozen-lm_head-can't-decode-drifted-hidden-states garbling. Qwen3 uses
    # `tie_word_embeddings=True`, so including embed_tokens here would unmake
    # the tie and double memory; lm_head alone is sufficient and safe.
    lora_modules_to_save: str = "lm_head"
    # Loss
    include_text_loss: bool = False         # if False, CE only on speech-token positions
    # Label smoothing for CE — borrowed from CosyVoice2's LabelSmoothingLoss.
    # Prevents the model from becoming over-confident on training tokens, which
    # is especially important when the dataset's "correct" speech token is just
    # one of many acoustically-equivalent draws. 0.0 = no smoothing (vanilla
    # CE); 0.1 is the common default in seq2seq / speech-LM training.
    label_smoothing: float = 0.1
    # Activation memory saver — necessary when lm_head is in modules_to_save
    # (adds 326M trainable params worth of optimizer/grad state).
    gradient_checkpointing: bool = True
    # Dataset filtering
    lang_filter: str = ""                   # if non-empty, only train on this lang (en/zh/yue)
    max_total_tokens: int = 2048
    max_speech_tokens: int = 750
    max_samples: int = 0                    # 0 = full dataset; >0 = first N (debug)
    shuffle: bool = True
    # Logging / saving
    num_workers: int = 2
    log_every: int = 20
    save_every: int = 500
    # Wandb
    wandb: bool = False
    wandb_project: str = "soulxpodcast-lora"
    wandb_run_name: str = ""
    wandb_entity: str = ""


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Comma-separated list of module suffix names to apply LoRA to.")
    p.add_argument("--lora_modules_to_save", type=str,
                   default="lm_head",
                   help="Modules to fully unfreeze (NOT LoRA-factorized). Set "
                        "empty string to disable. Default unfreezes lm_head — "
                        "required to prevent frozen-lm_head garbling when the "
                        "LoRA shifts hidden states.")
    p.add_argument("--include_text_loss", action="store_true",
                   help="Include text-token positions in CE loss (default: speech only).")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing for CE (CosyVoice2 recipe). 0.0 disables. "
                        "Default 0.1 prevents over-confidence on training tokens.")
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                   help="Disable gradient checkpointing (default ON; needed for lm_head trainable).")
    p.add_argument("--lang_filter", type=str, default="",
                   help="If set (en/zh/yue), train only on samples of this language.")
    p.add_argument("--max_total_tokens", type=int, default=2048)
    p.add_argument("--max_speech_tokens", type=int, default=750)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--no_shuffle", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="soulxpodcast-lora")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--wandb_entity", default="")
    args = p.parse_args()
    args_dict = vars(args)
    args_dict["shuffle"] = not args_dict.pop("no_shuffle")
    args_dict["gradient_checkpointing"] = not args_dict.pop("no_gradient_checkpointing")
    return TrainConfig(**args_dict)


def get_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


# ------------------------------------------------------------------------- #
# Model setup with LoRA
# ------------------------------------------------------------------------- #

def build_model_with_lora(cfg: TrainConfig, dtype: torch.dtype):
    """Load Qwen3 base + attach LoRA adapters. Base stays frozen; only LoRA trains."""
    log.info(f"loading base model: {cfg.model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=dtype, device_map="cuda",
    )
    if cfg.gradient_checkpointing:
        # Necessary headroom for lm_head being trainable (326M extra params +
        # AdamW state). Without checkpointing, peak activation memory ≈ 30 GB
        # at batch=8/T=512; checkpointing trades ~30% compute for ~3x memory.
        base.gradient_checkpointing_enable()
        log.info("gradient checkpointing: ON")

    target_modules = [m.strip() for m in cfg.lora_target_modules.split(",") if m.strip()]
    modules_to_save = [m.strip() for m in cfg.lora_modules_to_save.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    log.info(f"applying LoRA: rank={cfg.lora_rank} alpha={cfg.lora_alpha} "
             f"dropout={cfg.lora_dropout} targets={target_modules} "
             f"modules_to_save={modules_to_save}")
    model = get_peft_model(base, lora_config)

    # Sanity log: param counts.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"params: trainable={trainable/1e6:.1f}M  total={total/1e6:.1f}M  "
             f"({100*trainable/total:.2f}% trainable)")

    return model


# ------------------------------------------------------------------------- #
# Loss
# ------------------------------------------------------------------------- #

def compute_lm_loss(
    logits: torch.Tensor,           # [B, T, V]
    input_ids: torch.LongTensor,    # [B, T]
    loss_mask: torch.LongTensor,    # [B, T]  — 1 at positions where loss applies
    top_ks: tuple = (1, 5, 20),
    label_smoothing: float = 0.0,
):
    """Standard causal LM loss, masked by `loss_mask`.

    Shift convention: position t in input_ids predicts input_ids[t+1].
    The loss at position t therefore uses logits[t] vs target input_ids[t+1].
    We apply the mask AT THE TARGET POSITION (t+1) — if the target is a speech
    token (mask=1), include the loss; otherwise skip.

    Label smoothing borrowed from CosyVoice2's LabelSmoothingLoss recipe:
    PyTorch's built-in `label_smoothing` parameter on F.cross_entropy gives
    the same effect (slight numerical difference vs CosyVoice's `eps/(V-1)`
    formulation that's irrelevant at V=159K) and uses the optimized
    log-softmax kernel — no [B*T, V] target distribution materialization.

    Returns (loss, accs_dict, n_masked) where accs_dict maps each k in top_ks
    to top-k accuracy at masked positions. With a 159K vocab and CE training,
    top-1 acc near 0 is normal in early steps — top-5/top-20 are better
    indicators of whether the model is "close" (i.e. correct token within
    the top-K most likely).
    """
    # Shift: predictions = logits[:, :-1], targets = input_ids[:, 1:]
    shift_logits = logits[:, :-1].contiguous()       # [B, T-1, V]
    shift_targets = input_ids[:, 1:].contiguous()    # [B, T-1]
    shift_mask = loss_mask[:, 1:].contiguous().to(torch.float32)  # [B, T-1]

    B, Tm1, V = shift_logits.shape
    ce_per_pos = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view(B, Tm1)
    denom = shift_mask.sum().clamp_min(1)
    loss = (ce_per_pos * shift_mask).sum() / denom

    # Metrics: top-K accuracy at masked positions, for several K.
    # Compute once for max(top_ks), then slice.
    with torch.no_grad():
        max_k = max(top_ks)
        top_preds = shift_logits.topk(max_k, dim=-1).indices  # [B, T-1, max_k]
        target_expanded = shift_targets.unsqueeze(-1)         # [B, T-1, 1]
        accs = {}
        for k in sorted(top_ks):
            in_top_k = (top_preds[..., :k] == target_expanded).any(dim=-1)  # [B, T-1]
            acc = (in_top_k.to(torch.float32) * shift_mask).sum() / denom
            accs[k] = float(acc.item())

    return loss, accs, int(denom.item())


# ------------------------------------------------------------------------- #
# Wandb (optional)
# ------------------------------------------------------------------------- #

def _maybe_init_wandb(cfg: TrainConfig):
    if not cfg.wandb:
        return None
    try:
        import wandb
    except ImportError:
        log.warning("--wandb passed but `wandb` package not installed; skipping.")
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


# ------------------------------------------------------------------------- #
# Training loop
# ------------------------------------------------------------------------- #

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
    log.info(f"loading dataset: {cfg.dataset_path}")
    hf_ds = load_from_disk(cfg.dataset_path)

    # Filter columns (drop audio if speech_tokens exists, etc. — same as MTP).
    keep_cols = ["text", "speech_tokens", "lang"]
    if "speech_tokens" not in hf_ds.column_names:
        keep_cols.append("audio")
        log.warning("speech_tokens column missing — fallback will tokenize audio "
                    "on the fly (slow). Precompute the column upstream.")
    drop_cols = [c for c in hf_ds.column_names if c not in keep_cols]
    if drop_cols:
        hf_ds = hf_ds.remove_columns(drop_cols)
        log.info(f"dropped columns: {drop_cols}")

    # Optional language filter.
    if cfg.lang_filter:
        n_before = len(hf_ds)
        hf_ds = hf_ds.filter(lambda r: r["lang"] == cfg.lang_filter)
        log.info(f"language filter '{cfg.lang_filter}': "
                 f"{n_before} → {len(hf_ds)} samples")

    if cfg.max_samples > 0 and cfg.max_samples < len(hf_ds):
        hf_ds = hf_ds.select(range(cfg.max_samples))
        log.info(f"truncated dataset to first {cfg.max_samples} samples")

    ds_cfg = MtpDatasetConfig(
        max_total_tokens=cfg.max_total_tokens,
        max_speech_tokens=cfg.max_speech_tokens,
        # Trunk LM training needs the EOS (semantic_token_end) position in the
        # loss mask so it learns when to stop. The default (False) is correct
        # for MTP training but wrong for trunk LM — fixes the runaway-generation
        # bug seen in the first lora_hk_full run.
        include_eos_in_speech_mask=True,
    )
    dataset = MtpDataset(hf_ds, tokenizer, ds_cfg)

    collator = MtpCollator(pad_token_id=tokenizer.pad_token_id or 0)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
        collate_fn=collator, num_workers=cfg.num_workers, pin_memory=True,
        drop_last=True,
    )

    # ---- Model + LoRA ----------------------------------------------------
    model = build_model_with_lora(cfg, dtype)
    model.train()

    # ---- Optimizer / scheduler ------------------------------------------
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    steps_per_epoch = max(1, len(loader) // cfg.grad_accum_steps)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,
    )

    log.info(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}")
    log.info(f"loss target: {'speech + text positions' if cfg.include_text_loss else 'speech positions only'}")

    # ---- Train loop ------------------------------------------------------
    global_step = 0
    accum_loss = 0.0
    accum_count = 0
    t_last_log = time.perf_counter()
    tokens_since_log = 0
    epoch = 0

    while True:
        log.info(f"=== epoch {epoch} ===")
        for batch_idx, batch in enumerate(loader):
            if not batch:
                continue

            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            speech_mask = batch["speech_mask"].cuda(non_blocking=True)

            # Pick loss mask: speech-only or all-content (attention_mask).
            if cfg.include_text_loss:
                loss_mask = attention_mask
            else:
                loss_mask = speech_mask

            # Forward.
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits  # [B, T, V]

            loss, accs, n_loss_tokens = compute_lm_loss(
                logits, input_ids, loss_mask,
                label_smoothing=cfg.label_smoothing,
            )
            loss = loss / cfg.grad_accum_steps

            loss.backward()
            accum_loss += loss.item()
            accum_count += 1
            tokens_since_log += n_loss_tokens

            if accum_count >= cfg.grad_accum_steps:
                clip_max = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=clip_max,
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
                    acc_str = "  ".join(
                        f"top{k}={accs[k]:.3f}" for k in sorted(accs.keys())
                    )
                    log.info(
                        f"step={global_step}/{total_steps}  "
                        f"loss={accum_loss:.4f}  {acc_str}  "
                        f"lr={lr_now:.2e}  |grad|={grad_norm:.3f}  "
                        f"loss_tok/s={tps:.0f}"
                    )
                    if wandb_run is not None:
                        metrics = {
                            "train/loss": accum_loss,
                            "train/lr": lr_now,
                            "train/grad_norm": grad_norm,
                            "train/loss_tok_per_sec": tps,
                            "train/step": global_step,
                        }
                        for k, v in accs.items():
                            metrics[f"train/acc_top{k}"] = v
                        wandb_run.log(metrics, step=global_step)
                    accum_loss = 0.0
                    tokens_since_log = 0
                    t_last_log = time.perf_counter()

                if cfg.save_every > 0 and global_step % cfg.save_every == 0:
                    save_path = out_dir / f"adapter_step{global_step}"
                    model.save_pretrained(str(save_path))
                    log.info(f"saved LoRA adapter → {save_path}")
                    if wandb_run is not None:
                        wandb_run.log({"checkpoint/step": global_step},
                                      step=global_step)

                if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    break
        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            break
        epoch += 1
        if cfg.max_steps <= 0 and epoch >= cfg.num_epochs:
            break

    # Final save — this is what you load at inference.
    final_path = out_dir / "adapter"
    model.save_pretrained(str(final_path))
    log.info(f"done. final LoRA adapter saved → {final_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train(parse_args())
