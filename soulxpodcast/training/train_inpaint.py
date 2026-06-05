"""Train the pronunciation-inpaint composer on top of a frozen Qwen3 trunk.

Mirrors :mod:`soulxpodcast.training.train_lora_trunk` in shape so a future
reader recognises the pattern. Differences from the LoRA trainer:

- The backbone is frozen ENTIRELY (no LoRA, no lm_head trainable). The
  only trainable params live in
  :class:`soulxpodcast.inpaint.composer.PhonemeComposer` (~10 M for
  ``d_model=2048, K=8``).
- The forward goes via ``inputs_embeds=`` instead of ``input_ids=`` so
  the composer's output can replace the LLM's embedding at phoneme
  positions (``apply_phoneme_inpaint``).
- Speech-token CE on the LLM logits. v7 masks silence-target positions
  out of the supervised loss via a precomputed LUT so the composer is
  never rewarded for predicting silence (root cause of v4-v6 zh mode
  collapse). The aux 3-way alphabet head used in v1-v6 was removed —
  its loss was always ~0 (alphabet is trivially decodable from the
  disjoint per-alphabet vocab ranges).
- Phoneme dropout is **unit-level** (one CJK character or one English
  word) with default ``phoneme_keep_prob=0.25`` — mirrors the upstream
  CosyVoice-Inpaint recipe and matches the sparse inference distribution.

Usage (small sanity run)::

    .venv/bin/python -m soulxpodcast.training.train_inpaint \\
        --dataset_path tmp/dataset.jsonl \\
        --model_path /path/to/SoulX-Podcast-1.7B-dialect \\
        --output_dir runs/inpaint_v1 \\
        --batch_size 4 --max_steps 300 --eval_every 100 --save_every 200

Usage (overnight run on a single 3090, ~1 epoch on the full corpus)::

    .venv/bin/python -m soulxpodcast.training.train_inpaint \\
        --dataset_path tmp/dataset.jsonl \\
        --model_path /path/to/SoulX-Podcast-1.7B-dialect \\
        --output_dir runs/inpaint_full \\
        --batch_size 4 --grad_accum_steps 2 \\
        --max_steps 50000 --eval_every 1000 --save_every 2000 \\
        --lr 5e-4 --warmup_steps 500 \\
        --phoneme_keep_prob 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from soulxpodcast.inpaint import PhonemeComposer, apply_phoneme_inpaint
from soulxpodcast.training.inpaint_dataset import (
    InpaintDataset,
    InpaintDatasetConfig,
    collate,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_inpaint")


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    dataset_path: str
    model_path: str
    output_dir: str
    # Training
    batch_size: int = 4
    grad_accum_steps: int = 1
    max_steps: int = 50_000
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    dtype: str = "bf16"
    seed: int = 42
    # Composer
    slots_per_token: int = 8
    # On the 160K-vocab LLM, label_smoothing=0.1 adds ~1.5 nats of irreducible
    # CE floor that makes train loss un-interpretable. Composer already has
    # phoneme-dropout + weight decay + grad clip; extra smoothing is redundant.
    # Set >0 only when needed.
    label_smoothing: float = 0.0
    init_from_text_embed: bool = True
    # Resume a prior composer (loads composer weights only — NOT optimizer
    # / scheduler / step count). Use for "continuation fine-tuning" on a
    # rebalanced subset to specialise an existing checkpoint.
    resume_composer: str = ""
    # Dataset
    phoneme_keep_prob: float = 0.25
    lang_filter: str = ""             # "" = all langs (yue,zh,en); else comma-separated
    # Language-rebalanced sampling. One of:
    #   "none"     — uniform sampling (matches the corpus distribution)
    #   "balanced" — each language gets equal expected representation
    #                (per-row weight = 1 / count[lang]).
    #   "sqrt"     — sqrt-tempered rebalance: minority langs are boosted
    #                but not all the way to uniform. Per-row weight =
    #                1 / sqrt(count[lang]). Safer when the minority lang
    #                is tiny — avoids overfitting on the few rows.
    lang_balance: str = "none"
    max_total_tokens: int = 2048
    max_speech_tokens: int = 750
    eval_fraction: float = 0.01       # held-out fraction (capped at 2000 rows)
    eval_max_rows: int = 2000
    # Memory / runtime
    gradient_checkpointing: bool = True
    attn_implementation: str = "flash_attention_2"  # "flash_attention_2" | "sdpa" | "eager"
    use_8bit_adam: bool = True
    num_workers: int = 2
    # Logging / saving
    log_every: int = 20
    eval_every: int = 1000
    save_every: int = 2000
    # Wandb
    wandb: bool = False
    wandb_project: str = "soulxpodcast-inpaint"
    wandb_run_name: str = ""
    wandb_entity: str = ""


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slots_per_token", type=int, default=8)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--no_init_from_text_embed", action="store_true")
    p.add_argument("--resume_composer", type=str, default="",
                   help="Path to a prior composer.pt to warm-start from "
                        "(loads composer weights only — optimizer / scheduler "
                        "start fresh).")
    p.add_argument("--lang_balance", choices=["none", "balanced", "sqrt"], default="none",
                   help="Per-row sampling weight scheme based on language frequency.")
    p.add_argument("--phoneme_keep_prob", type=float, default=0.25)
    p.add_argument("--lang_filter", type=str, default="")
    p.add_argument("--max_total_tokens", type=int, default=2048)
    p.add_argument("--max_speech_tokens", type=int, default=750)
    p.add_argument("--eval_fraction", type=float, default=0.01)
    p.add_argument("--eval_max_rows", type=int, default=2000)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--attn_implementation", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--no_8bit_adam", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="soulxpodcast-inpaint")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--wandb_entity", default="")
    args = p.parse_args()
    d = vars(args)
    d["init_from_text_embed"] = not d.pop("no_init_from_text_embed")
    d["gradient_checkpointing"] = not d.pop("no_gradient_checkpointing")
    d["use_8bit_adam"] = not d.pop("no_8bit_adam")
    return TrainConfig(**d)


def get_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def _maybe_init_wandb(cfg: TrainConfig):
    if not cfg.wandb:
        return None
    try:
        import wandb
    except ImportError:
        log.warning("--wandb passed but the `wandb` package is not importable; skipping.")
        return None
    init_kwargs: dict = dict(
        project=cfg.wandb_project,
        config=asdict(cfg),
        dir=cfg.output_dir,
    )
    if cfg.wandb_run_name:
        init_kwargs["name"] = cfg.wandb_run_name
    if cfg.wandb_entity:
        init_kwargs["entity"] = cfg.wandb_entity
    wandb.init(**init_kwargs)
    log.info(f"wandb logging enabled: project={cfg.wandb_project} run={wandb.run.name}")
    return wandb


# --------------------------------------------------------------------- #
# Forward + loss
# --------------------------------------------------------------------- #

def forward_step(
    model,
    composer: PhonemeComposer,
    batch: dict,
    dtype: torch.dtype,
    label_smoothing: float,
):
    """One forward through frozen backbone + composer; returns scalar loss + parts."""
    device = next(composer.parameters()).device
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    speech_mask = batch["speech_mask"].to(device, non_blocking=True)
    phone_token = batch["phone_token"].to(device, non_blocking=True)
    phone_mask = batch["phone_mask"].to(device, non_blocking=True)
    alphabet_id = batch["alphabet_id"].to(device, non_blocking=True)

    text_emb = model.get_input_embeddings()(input_ids).to(dtype)
    composed, _ = composer(phone_token, alphabet_id)
    inputs_embeds = apply_phoneme_inpaint(text_emb, composed, phone_mask)

    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits

    shift_logits = logits[:, :-1].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    shift_mask = speech_mask[:, 1:].to(torch.float32)
    V = shift_logits.size(-1)
    ce = F.cross_entropy(
        shift_logits.reshape(-1, V).float(),
        shift_targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view(*shift_logits.shape[:2])
    denom = shift_mask.sum().clamp_min(1)
    lm_loss = (ce * shift_mask).sum() / denom

    return lm_loss, lm_loss.detach(), int(denom.item())


# --------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    model,
    composer: PhonemeComposer,
    eval_loader: DataLoader,
    eval_loader_full: DataLoader,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Compute eval lm_loss at three phoneme regimes for trend tracking.

    Returns dict with:
      - ``eval/lm_loss_dropout``  : composer ON with same p_keep as train
      - ``eval/lm_loss_full``     : composer ON, all units kept (p_keep=1.0)
      - ``eval/lm_loss_text_only``: composer fully OFF (pure text baseline)
    """
    composer.eval()
    sums = {"dropout": 0.0, "full": 0.0, "text_only": 0.0}
    n = {"dropout": 0, "full": 0, "text_only": 0}

    def _step(batch: dict, want_text_only: bool):
        device = next(composer.parameters()).device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        speech_mask = batch["speech_mask"].to(device, non_blocking=True)
        phone_token = batch["phone_token"].to(device, non_blocking=True)
        phone_mask = batch["phone_mask"].to(device, non_blocking=True)
        alphabet_id = batch["alphabet_id"].to(device, non_blocking=True)

        text_emb = model.get_input_embeddings()(input_ids).to(dtype)
        composed, _ = composer(phone_token, alphabet_id)
        emb_inject = apply_phoneme_inpaint(text_emb, composed, phone_mask)

        def _lm(inputs_embeds):
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            shift_logits = out.logits[:, :-1].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            shift_mask = speech_mask[:, 1:].to(torch.float32)
            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_targets.reshape(-1),
                reduction="none",
            ).view(*shift_logits.shape[:2])
            d = shift_mask.sum().clamp_min(1)
            return float((ce * shift_mask).sum() / d), int(d.item())

        results: dict[str, tuple[float, int]] = {}
        results["inject"] = _lm(emb_inject)
        if want_text_only:
            results["text_only"] = _lm(text_emb)
        return results

    # ---- dropout-regime + text-only baseline (same dataset, dropout=0.25) ----
    for batch in eval_loader:
        if not batch:  # collate returns {} when every row in the micro-batch filtered out
            continue
        out = _step(batch, want_text_only=True)
        ld, nd = out["inject"]
        sums["dropout"] += ld * nd; n["dropout"] += nd
        if "text_only" in out:
            lt, nt = out["text_only"]
            sums["text_only"] += lt * nt; n["text_only"] += nt

    # ---- full-inject regime (same rows, but keep_prob=1.0) ----
    for batch in eval_loader_full:
        if not batch:
            continue
        out = _step(batch, want_text_only=False)
        lf, nf = out["inject"]
        sums["full"] += lf * nf; n["full"] += nf

    composer.train()
    out: dict[str, float] = {}
    for k, key in [
        ("dropout", "eval/lm_loss_dropout"),
        ("full", "eval/lm_loss_full"),
        ("text_only", "eval/lm_loss_text_only"),
    ]:
        out[key] = sums[k] / max(1, n[k])
    out["eval/gap_vs_text"] = out["eval/lm_loss_dropout"] - out["eval/lm_loss_text_only"]
    out["eval/full_vs_text"] = out["eval/lm_loss_full"] - out["eval/lm_loss_text_only"]
    return out


# --------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------- #

def save_checkpoint(
    out_dir: Path,
    step: int,
    composer: PhonemeComposer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    extra: Optional[dict] = None,
) -> Path:
    ckpt_dir = out_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "step": step,
        "config": {
            "d_model": composer.d_model,
            "slots_per_token": composer.K,
            "vocab_size": composer.vocab_size,
        },
        "composer": composer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "extra": extra or {},
    }
    path = ckpt_dir / "composer.pt"
    torch.save(bundle, path)
    log.info(f"saved checkpoint → {path}")
    return path


# --------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------- #

def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    wandb_run = _maybe_init_wandb(cfg)

    dtype = get_dtype(cfg.dtype)

    # ---- Tokenizer + dataset ----------------------------------------- #
    log.info(f"loading tokenizer from {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)

    lang_filter = {x.strip() for x in cfg.lang_filter.split(",") if x.strip()} or None
    ds_cfg = InpaintDatasetConfig(
        slots_per_token=cfg.slots_per_token,
        phoneme_keep_prob=cfg.phoneme_keep_prob,
        max_total_tokens=cfg.max_total_tokens,
        max_speech_tokens=cfg.max_speech_tokens,
    )
    log.info(f"loading dataset from {cfg.dataset_path}")
    full_ds = InpaintDataset(
        cfg.dataset_path, tokenizer, config=ds_cfg,
        lang_filter=lang_filter,
    )

    # ---- Train / eval split (deterministic) -------------------------- #
    n_total = len(full_ds)
    eval_n = min(cfg.eval_max_rows, int(n_total * cfg.eval_fraction))
    eval_n = max(eval_n, 32)  # at least 32 eval rows
    rng = random.Random(cfg.seed)
    eval_idx = sorted(rng.sample(range(n_total), eval_n))
    train_idx = sorted(set(range(n_total)) - set(eval_idx))
    log.info(f"split: train={len(train_idx):,}  eval={len(eval_idx):,}")

    train_ds = Subset(full_ds, train_idx)

    # Eval datasets use deterministic per-row dropout so the eval distribution
    # is stable across evaluations (essential for comparing curves).
    # Two views of the same eval indices: one at the training dropout rate,
    # one with all phonemes kept (p_keep=1.0).
    eval_ds_obj = InpaintDataset(
        cfg.dataset_path, tokenizer,
        config=InpaintDatasetConfig(
            slots_per_token=cfg.slots_per_token,
            phoneme_keep_prob=cfg.phoneme_keep_prob,
            max_total_tokens=cfg.max_total_tokens,
            max_speech_tokens=cfg.max_speech_tokens,
            deterministic_dropout=True,
        ),
        lang_filter=lang_filter,
    )
    eval_ds_full_obj = InpaintDataset(
        cfg.dataset_path, tokenizer,
        config=InpaintDatasetConfig(
            slots_per_token=cfg.slots_per_token,
            phoneme_keep_prob=1.0,
            max_total_tokens=cfg.max_total_tokens,
            max_speech_tokens=cfg.max_speech_tokens,
            deterministic_dropout=True,
        ),
        lang_filter=lang_filter,
    )
    eval_ds = Subset(eval_ds_obj, eval_idx)
    eval_ds_full = Subset(eval_ds_full_obj, eval_idx)

    # ---- Optional language-rebalanced sampler ------------------------ #
    sampler = None
    do_shuffle = True
    if cfg.lang_balance != "none":
        import math
        from collections import Counter

        lang_per_train_idx = [full_ds.rows[i]["lang"] for i in train_idx]
        lang_counts = Counter(lang_per_train_idx)
        log.info(f"lang counts in train split: {dict(lang_counts)}")

        if cfg.lang_balance == "balanced":
            weight_for = {lang: 1.0 / cnt for lang, cnt in lang_counts.items()}
        elif cfg.lang_balance == "sqrt":
            weight_for = {lang: 1.0 / math.sqrt(cnt) for lang, cnt in lang_counts.items()}
        else:
            raise ValueError(f"unknown lang_balance: {cfg.lang_balance}")

        sample_weights = torch.tensor(
            [weight_for[lang] for lang in lang_per_train_idx], dtype=torch.float64
        )
        # Effective dataset size = same as a full epoch; sampler runs with
        # replacement so minority langs get oversampled in expectation.
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_ds), replacement=True
        )
        do_shuffle = False  # sampler controls ordering
        # Log the expected per-lang draws under this scheme.
        total_w = sample_weights.sum().item()
        per_lang_draws = {
            lang: weight_for[lang] * cnt / total_w * len(train_ds)
            for lang, cnt in lang_counts.items()
        }
        log.info(
            f"lang_balance={cfg.lang_balance}  expected per-epoch draws: "
            + ", ".join(f"{lang}={int(n):,}" for lang, n in per_lang_draws.items())
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=do_shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate(b, pad_token_id=tokenizer.pad_token_id or 0),
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate(b, pad_token_id=tokenizer.pad_token_id or 0),
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    eval_loader_full = DataLoader(
        eval_ds_full,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate(b, pad_token_id=tokenizer.pad_token_id or 0),
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )

    # ---- Model + composer -------------------------------------------- #
    log.info(f"loading model from {cfg.model_path} dtype={dtype} attn={cfg.attn_implementation}")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        dtype=dtype,
        device_map="cuda",
        attn_implementation=cfg.attn_implementation,
    )
    log.info(f"model loaded in {time.perf_counter() - t0:.1f}s")
    for p in model.parameters():
        p.requires_grad = False
    # Don't call model.eval(): HF silently disables gradient checkpointing
    # in eval mode, which blows activation memory up at long T. Qwen3's
    # dropout is tiny so leaving model in train() mode is a non-issue.
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("gradient checkpointing: ON")

    composer = PhonemeComposer(
        d_model=model.config.hidden_size,
        slots_per_token=cfg.slots_per_token,
    ).to("cuda", dtype=dtype)
    if cfg.init_from_text_embed:
        composer.init_from_text_embed(model.get_input_embeddings())
    if cfg.resume_composer:
        log.info(f"resuming composer weights from {cfg.resume_composer}")
        prior = torch.load(cfg.resume_composer, map_location="cuda", weights_only=False)
        prior_cfg = prior.get("config", {})
        if prior_cfg.get("d_model") not in (None, composer.d_model):
            raise ValueError(
                f"resume_composer d_model={prior_cfg.get('d_model')} != "
                f"current {composer.d_model}"
            )
        if prior_cfg.get("slots_per_token") not in (None, composer.K):
            raise ValueError(
                f"resume_composer slots_per_token={prior_cfg.get('slots_per_token')} != "
                f"current {composer.K}"
            )
        # Strict load — composer arch must match the checkpoint's. Any
        # arch change requires a fresh run, not a resume.
        composer.load_state_dict(prior["composer"], strict=True)
        log.info(
            f"  loaded {sum(p.numel() for p in composer.parameters())/1e6:.2f}M composer params "
            f"(prior step={prior.get('step', '?')}). Optimizer & scheduler start fresh."
        )
    composer.train()

    n_train = sum(p.numel() for p in composer.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters())
    log.info(
        f"trainable composer params: {n_train/1e6:.3f}M  "
        f"frozen backbone params: {n_frozen/1e6:.1f}M"
    )

    # ---- Optimizer + scheduler --------------------------------------- #
    if cfg.use_8bit_adam:
        try:
            from bitsandbytes.optim import AdamW8bit  # type: ignore
        except ImportError:
            log.warning("bitsandbytes not available; falling back to torch AdamW")
            AdamW8bit = None
    else:
        AdamW8bit = None
    if AdamW8bit is not None:
        log.info("using bitsandbytes AdamW8bit")
        optim = AdamW8bit(
            composer.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
    else:
        log.info("using torch AdamW (fp32 state)")
        optim = torch.optim.AdamW(
            composer.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps,
    )

    # ---- Train loop -------------------------------------------------- #
    train_log = (out_dir / "train_log.jsonl").open("a", buffering=1)
    eval_log = (out_dir / "eval_log.jsonl").open("a", buffering=1)
    global_step = 0
    accum_loss = 0.0
    accum_lm = 0.0
    accum_count = 0           # micro-batches since last optimizer step
    window_steps = 0          # gradient steps since last log
    tokens_since_log = 0
    units_kept_since_log = 0
    units_seen_since_log = 0
    t_last_log = time.perf_counter()
    epoch = 0

    log.info(f"starting training: max_steps={cfg.max_steps}  "
             f"batch={cfg.batch_size}  accum={cfg.grad_accum_steps}  "
             f"effective_batch={cfg.batch_size * cfg.grad_accum_steps}")

    while global_step < cfg.max_steps:
        for batch in train_loader:
            if not batch:
                continue
            loss, lm_loss, n_loss_tokens = forward_step(
                model, composer, batch, dtype, cfg.label_smoothing,
            )
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += float(loss)
            accum_lm += float(lm_loss) / cfg.grad_accum_steps
            accum_count += 1
            tokens_since_log += n_loss_tokens
            units_kept_since_log += batch.get("n_phonemes_kept", 0)
            units_seen_since_log += batch.get("n_phonemes_seen", 0)

            if accum_count >= cfg.grad_accum_steps:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        composer.parameters(),
                        max_norm=cfg.grad_clip if cfg.grad_clip > 0 else float("inf"),
                    )
                )
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                accum_count = 0
                window_steps += 1

                if global_step % cfg.log_every == 0:
                    dt = time.perf_counter() - t_last_log
                    tps = tokens_since_log / max(dt, 1e-6)
                    lr_now = sched.get_last_lr()[0]
                    keep_frac = (
                        units_kept_since_log / max(1, units_seen_since_log)
                    )
                    # Report MEAN per gradient step, not sum over window.
                    n = max(1, window_steps)
                    mean_loss = accum_loss / n
                    mean_lm = accum_lm / n
                    log.info(
                        f"step {global_step:6d}/{cfg.max_steps}  "
                        f"loss={mean_loss:.4f}  lm={mean_lm:.4f}  "
                        f"|grad|={grad_norm:.3f}  "
                        f"lr={lr_now:.2e}  keep={keep_frac:.2%}  "
                        f"loss_tok/s={tps:.0f}"
                    )
                    row = {
                        "step": global_step,
                        "loss": mean_loss,
                        "lm_loss": mean_lm,
                        "grad_norm": grad_norm,
                        "lr": lr_now,
                        "keep_frac": keep_frac,
                        "loss_tok_per_s": tps,
                    }
                    train_log.write(json.dumps(row) + "\n")
                    if wandb_run is not None:
                        wandb_run.log({
                            "train/loss": row["loss"],
                            "train/lm_loss": row["lm_loss"],
                            "train/grad_norm": row["grad_norm"],
                            "train/lr": row["lr"],
                            "train/keep_frac": row["keep_frac"],
                            "train/loss_tok_per_s": row["loss_tok_per_s"],
                        }, step=global_step)
                    accum_loss = 0.0; accum_lm = 0.0
                    window_steps = 0
                    tokens_since_log = 0
                    units_kept_since_log = 0; units_seen_since_log = 0
                    t_last_log = time.perf_counter()

                if cfg.eval_every > 0 and global_step % cfg.eval_every == 0:
                    log.info(f"running eval at step {global_step} ...")
                    t_eval = time.perf_counter()
                    metrics = evaluate(
                        model, composer, eval_loader, eval_loader_full,
                        dtype,
                    )
                    dt_eval = time.perf_counter() - t_eval
                    log.info(
                        f"eval @ {global_step}: "
                        f"text_only={metrics['eval/lm_loss_text_only']:.4f}  "
                        f"dropout={metrics['eval/lm_loss_dropout']:.4f}  "
                        f"full={metrics['eval/lm_loss_full']:.4f}  "
                        f"gap_vs_text={metrics['eval/gap_vs_text']:+.4f}  "
                        f"full_vs_text={metrics['eval/full_vs_text']:+.4f}  "
                        f"({dt_eval:.1f}s)"
                    )
                    eval_log.write(json.dumps({"step": global_step, **metrics}) + "\n")
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=global_step)

                if cfg.save_every > 0 and global_step % cfg.save_every == 0:
                    save_checkpoint(out_dir, global_step, composer, optim, sched, cfg)

                if global_step >= cfg.max_steps:
                    break
        epoch += 1
        log.info(f"=== epoch {epoch} done; step {global_step}/{cfg.max_steps} ===")

    # ---- Final checkpoint -------------------------------------------- #
    save_checkpoint(out_dir, global_step, composer, optim, sched, cfg)
    train_log.close()
    eval_log.close()
    if wandb_run is not None:
        wandb_run.finish()
    log.info(f"training done at step {global_step}")


if __name__ == "__main__":
    train(parse_args())
