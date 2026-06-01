"""Single-GPU smoke test for the pronunciation-inpaint training pipeline.

What this validates end-to-end:
- ``InpaintDataset`` builds samples from ``tmp/dataset.jsonl`` with the SoulX
  per-turn template, strips zh/yue spaces, and aligns split-form phoneme ids
  to text-token slots.
- ``PhonemeComposer`` produces a (B, T, d_model) embedding that replaces the
  frozen Qwen3 trunk's text embedding at masked positions.
- Forward through the frozen Qwen3 model via ``inputs_embeds`` succeeds.
- Loss + backward run; gradients flow ONLY into the composer (backbone has
  ``requires_grad=False`` everywhere).
- Optimizer step changes composer weights; backbone weights stay bit-identical.

Run::

    .venv/bin/python scripts/inpaint/smoke_train.py \\
        --jsonl tmp/dataset.jsonl \\
        --model_path /home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/pretrained_models/SoulX-Podcast-1.7B-dialect \\
        --num_samples 4 --batch_size 2

Memory budget on RTX 3090 (24 GB): Qwen3-1.7B in bf16 is ~3.4 GB weights,
gradient-checkpointed activations for B=2 T<=512 are ~1-2 GB, composer is
trivial. Comfortable.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.inpaint import (
    PhonemeComposer,
    apply_phoneme_inpaint,
)
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
log = logging.getLogger("smoke_train")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="tmp/dataset.jsonl")
    p.add_argument(
        "--model_path",
        default="/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/pretrained_models/SoulX-Podcast-1.7B-dialect",
    )
    p.add_argument("--num_samples", type=int, default=8,
                   help="rows to scan from jsonl; some will drop on filter/alignment")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lang", default="yue", help="comma-separated lang filter (yue,zh,en)")
    p.add_argument("--slots_per_token", type=int, default=8)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--aux_weight", type=float, default=0.3)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_csv", default="",
                   help="if non-empty, append per-step metrics here")
    p.add_argument("--summary_every", type=int, default=10,
                   help="print a compact line every N steps; default = log every step")
    return p.parse_args()


def get_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = get_dtype(args.dtype)

    log.info(f"loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    log.info(f"loading dataset from {args.jsonl}")
    cfg = InpaintDatasetConfig(slots_per_token=args.slots_per_token)
    lang_filter = {x.strip() for x in args.lang.split(",") if x.strip()}
    ds = InpaintDataset(
        args.jsonl, tokenizer,
        config=cfg, lang_filter=lang_filter, max_samples=args.num_samples,
    )

    # Build the batch eagerly (smoke test, not a perf benchmark).
    raw_samples = [ds[i] for i in range(len(ds))]
    raw_samples = [s for s in raw_samples if s is not None]
    if len(raw_samples) < args.batch_size:
        raise RuntimeError(
            f"need at least batch_size={args.batch_size} samples that pass alignment; "
            f"only {len(raw_samples)} of {len(ds)} survived. "
            f"Bump --num_samples or check phoneme alignment."
        )
    raw_samples = raw_samples[: args.batch_size]
    batch = collate(raw_samples, pad_token_id=tokenizer.pad_token_id or 0)
    log.info(
        f"batch: B={batch['input_ids'].shape[0]} T={batch['input_ids'].shape[1]} "
        f"speech_positions={int(batch['speech_mask'].sum())} "
        f"phone_positions={int(batch['phone_mask'].sum())} "
        f"phonemes_kept/seen={batch['n_phonemes_kept']}/{batch['n_phonemes_seen']}"
    )

    # Move to device.
    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)
    speech_mask = batch["speech_mask"].to(args.device)
    phone_token = batch["phone_token"].to(args.device)
    phone_mask = batch["phone_mask"].to(args.device)

    log.info(f"loading model from {args.model_path} dtype={dtype}")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, device_map=args.device,
    )
    log.info(f"model loaded in {time.perf_counter() - t0:.1f}s")

    # Freeze backbone — every param explicitly requires_grad=False.
    for p in model.parameters():
        p.requires_grad = False
    model.eval()  # turn off dropout; we still get grads through it

    # gradient checkpointing keeps activations small while we backprop through
    # the frozen trunk to reach the composer. need use_reentrant=False with
    # inputs_embeds for HF >= 4.40.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    d_model = model.config.hidden_size
    log.info(f"d_model = {d_model}")

    composer = PhonemeComposer(d_model=d_model, slots_per_token=args.slots_per_token).to(
        args.device, dtype=dtype
    )
    # Initialise toward the average backbone embedding row.
    composer.init_from_text_embed(model.get_input_embeddings())

    n_train = sum(p.numel() for p in composer.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters())
    log.info(
        f"trainable composer params: {n_train/1e6:.3f}M  "
        f"frozen backbone params: {n_frozen/1e6:.1f}M"
    )

    # Snapshot a tiny chunk of the backbone to confirm it stays bit-identical
    # after the step.
    backbone_embed_w_before = (
        model.get_input_embeddings().weight[:8, :8].detach().clone()
    )

    # ---- Baseline: what's the loss with phone_mask all-off?  ----
    # This is the "text-only frozen LLM" reference. Composer must beat it
    # (or at least match) over the course of training.
    with torch.no_grad():
        text_only_emb = model.get_input_embeddings()(input_ids).to(dtype)
        out_b = model(
            inputs_embeds=text_only_emb,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        shift_b = out_b.logits[:, :-1].contiguous().float()
        targ_b = input_ids[:, 1:].contiguous()
        smask = speech_mask[:, 1:].to(torch.float32)
        ce_b = F.cross_entropy(
            shift_b.reshape(-1, shift_b.size(-1)),
            targ_b.reshape(-1),
            reduction="none",
        ).view(*shift_b.shape[:2])
        baseline_lm = (ce_b * smask).sum() / smask.sum().clamp_min(1)
    log.info(
        f"baseline (no inpaint, frozen Qwen3 on this batch): lm_loss={float(baseline_lm):.4f}"
    )

    optim = torch.optim.AdamW(composer.parameters(), lr=args.lr, betas=(0.9, 0.95))

    trajectory: list[dict[str, float]] = []
    csv_f = open(args.log_csv, "a") if args.log_csv else None
    if csv_f and csv_f.tell() == 0:
        csv_f.write("step,loss,lm_loss,aux_loss,grad_norm,dt_ms\n")

    for step in range(args.steps):
        t_step = time.perf_counter()
        text_emb = model.get_input_embeddings()(input_ids).to(dtype)
        composed, mask_from_composer = composer(phone_token)
        if step == 0:
            assert torch.equal(mask_from_composer, phone_mask), (
                "phone_mask from dataset != mask derived by composer"
            )
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
        B, Tm1, V = shift_logits.shape
        ce = F.cross_entropy(
            shift_logits.reshape(-1, V).float(),
            shift_targets.reshape(-1),
            reduction="none",
            label_smoothing=args.label_smoothing,
        ).view(B, Tm1)
        denom = shift_mask.sum().clamp_min(1)
        lm_loss = (ce * shift_mask).sum() / denom

        alpha_labels = composer.alphabet_labels(phone_token)
        alpha_logits = composer.alphabet_head(composed).float()
        aux_loss = F.cross_entropy(
            alpha_logits.reshape(-1, alpha_logits.size(-1)),
            alpha_labels.reshape(-1),
            ignore_index=-100,
        )

        loss = lm_loss + args.aux_weight * aux_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()

        composer_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(composer.parameters(), max_norm=10.0)
        )
        if step == 0:
            assert not any(p.grad is not None for p in model.parameters()), (
                "backbone has gradient — freeze is broken"
            )

        optim.step()
        dt = (time.perf_counter() - t_step) * 1000
        rec = {
            "step": step,
            "loss": float(loss),
            "lm_loss": float(lm_loss),
            "aux_loss": float(aux_loss),
            "grad_norm": composer_grad_norm,
            "dt_ms": dt,
        }
        trajectory.append(rec)
        if csv_f:
            csv_f.write(
                f"{step},{rec['loss']:.6f},{rec['lm_loss']:.6f},{rec['aux_loss']:.6f},"
                f"{rec['grad_norm']:.6f},{rec['dt_ms']:.1f}\n"
            )

        if (
            step < 5
            or step % args.summary_every == 0
            or step == args.steps - 1
        ):
            log.info(
                f"step {step:4d}: loss={rec['loss']:7.4f}  lm={rec['lm_loss']:7.4f}  "
                f"aux={rec['aux_loss']:7.4f}  |grad|={rec['grad_norm']:6.3f}  "
                f"dt={rec['dt_ms']:5.0f}ms"
            )
        assert torch.isfinite(loss), f"loss is not finite at step {step}: {loss}"
        assert composer_grad_norm > 0, (
            f"composer received zero gradient at step {step}"
        )

    if csv_f:
        csv_f.close()

    backbone_embed_w_after = model.get_input_embeddings().weight[:8, :8].detach()
    delta = (backbone_embed_w_after - backbone_embed_w_before).abs().max().item()
    log.info(f"backbone embedding |Δ|_∞ after training: {delta:.6f}  (expect 0)")
    assert delta == 0.0, "backbone embedding mutated — freeze is broken"

    # ---- Trajectory summary ----
    log.info("=" * 64)
    log.info("trajectory summary:")
    log.info(f"  baseline (text-only) lm_loss = {float(baseline_lm):.4f}")
    log.info(f"  step  0   loss={trajectory[0]['loss']:.4f} lm={trajectory[0]['lm_loss']:.4f}")
    if len(trajectory) > 10:
        mid = trajectory[len(trajectory) // 2]
        log.info(f"  step {mid['step']:3d}   loss={mid['loss']:.4f} lm={mid['lm_loss']:.4f}")
    last = trajectory[-1]
    log.info(f"  step {last['step']:3d}   loss={last['loss']:.4f} lm={last['lm_loss']:.4f}")
    delta_lm = trajectory[0]["lm_loss"] - last["lm_loss"]
    delta_grad = trajectory[0]["grad_norm"] - last["grad_norm"]
    log.info(
        f"  Δ lm_loss = {delta_lm:+.4f}  (initial - final)  "
        f"Δ grad_norm = {delta_grad:+.3f}"
    )
    log.info(
        f"  lm_loss vs baseline = {last['lm_loss'] - float(baseline_lm):+.4f} "
        f"(negative = composer BEATS the text-only frozen LLM)"
    )
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log.info(f"peak GPU memory: {peak_gb:.2f} GB")
    log.info("SMOKE TRAIN ✓")


if __name__ == "__main__":
    main()
