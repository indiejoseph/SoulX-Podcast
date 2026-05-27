"""Overfit smoke test: verify EOS-in-mask fix lets LoRA learn termination.

Two phases:
  1. STATIC: build samples with include_eos_in_speech_mask=True vs False.
     Mask sum should differ by exactly 1, and the extra position should be
     the semantic_token_end (speech EOS) target.

  2. OVERFIT: train LoRA on 4 samples for 150 steps. With the fix in place,
     the LoRA should drive *both* speech-token loss AND EOS-specific loss
     toward zero. Probes P(EOS) at the pre-EOS position before vs. after.

Expected after fix:
  - STATIC PASS
  - Overall loss → near-zero on training samples
  - Loss measured ONLY at EOS-target positions → near-zero (would NOT happen
    without the fix because there's no gradient there)
  - P(semantic_token_end | last_speech_token) climbs from ~0 to >0.5
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import sys
import logging

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from soulxpodcast.training.mtp_dataset import (
    MtpDataset, MtpDatasetConfig, MtpCollator, SPECIAL_TOKENS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("smoke")

MODEL = "pretrained_models/SoulX-Podcast-1.7B-dialect"
DATASET = "tmp/dataset_small_with_tokens"
N_OVERFIT_SAMPLES = 4
N_STEPS = 150


def get_special_ids(tokenizer):
    return {k: tokenizer.encode(v, add_special_tokens=False)[0]
            for k, v in SPECIAL_TOKENS.items()}


# ---------------------------------------------------------------------------
# PHASE 1 — static mask check
# ---------------------------------------------------------------------------
def static_check(tokenizer, hf_ds, eos_id):
    log.info("=" * 60)
    log.info("STATIC — mask shape with/without EOS-in-mask")
    log.info("=" * 60)
    ds_off = MtpDataset(hf_ds, tokenizer, MtpDatasetConfig(include_eos_in_speech_mask=False))
    ds_on = MtpDataset(hf_ds, tokenizer, MtpDatasetConfig(include_eos_in_speech_mask=True))

    ok = True
    for i in range(5):
        s_off = ds_off[i]
        s_on = ds_on[i]
        if s_off is None or s_on is None:
            continue
        n_off = int(s_off["speech_mask"].sum().item())
        n_on = int(s_on["speech_mask"].sum().item())

        eos_positions = (s_on["input_ids"] == eos_id).nonzero(as_tuple=True)[0].tolist()
        eos_pos = eos_positions[0] if eos_positions else -1
        m_off_at_eos = int(s_off["speech_mask"][eos_pos].item()) if eos_pos >= 0 else -1
        m_on_at_eos = int(s_on["speech_mask"][eos_pos].item()) if eos_pos >= 0 else -1

        delta = n_on - n_off
        passed = delta == 1 and m_off_at_eos == 0 and m_on_at_eos == 1
        marker = "✓" if passed else "✗"
        log.info(f"  sample {i}: len={len(s_off['input_ids'])} "
                 f"mask_sum off={n_off} on={n_on} (Δ={delta})  "
                 f"EOS@pos{eos_pos} mask off={m_off_at_eos} on={m_on_at_eos}  {marker}")
        if not passed:
            ok = False
    log.info(f"STATIC: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# PHASE 2 — overfit training
# ---------------------------------------------------------------------------
def probe_eos(model, batch, eos_id, label):
    """For each row in batch, report P(EOS) at the predictor position for EOS.

    The predictor position is the LAST speech-token position (i.e. mask==1 and
    next token == EOS). logits there should put high prob on eos_id.
    """
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].cuda()
        speech_mask = batch["speech_mask"].cuda()
        attn_mask = batch["attention_mask"].cuda()
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits
        results = []
        for b in range(input_ids.shape[0]):
            eos_pos_list = (input_ids[b] == eos_id).nonzero(as_tuple=True)[0].tolist()
            if not eos_pos_list:
                continue
            eos_pos = eos_pos_list[0]
            pred_pos = eos_pos - 1
            if pred_pos < 0:
                continue
            row_logits = logits[b, pred_pos]
            probs = F.softmax(row_logits.float(), dim=-1)
            eos_prob = probs[eos_id].item()
            top1 = int(row_logits.argmax().item())
            top1_prob = probs[top1].item()
            rank = int((row_logits > row_logits[eos_id]).sum().item())
            results.append((eos_prob, top1, top1_prob, rank))
            log.info(f"  [{label}] row{b}: P(EOS)={eos_prob:.4f}  "
                     f"top1={top1} (P={top1_prob:.4f})  EOS rank={rank}")
    model.train()
    return results


def overfit_train(tokenizer, hf_ds, eos_id):
    log.info("=" * 60)
    log.info(f"OVERFIT — {N_OVERFIT_SAMPLES} samples × {N_STEPS} steps")
    log.info("=" * 60)

    # Pick N samples that survive the dataset filter (not None).
    sub_ds = MtpDataset(
        hf_ds.select(range(64)), tokenizer,
        MtpDatasetConfig(
            max_total_tokens=2048, max_speech_tokens=750,
            include_eos_in_speech_mask=True,
        ),
    )
    picked = []
    for i in range(len(sub_ds)):
        s = sub_ds[i]
        if s is not None:
            picked.append(s)
        if len(picked) >= N_OVERFIT_SAMPLES:
            break
    log.info(f"picked {len(picked)} samples; lengths={[int(s['length']) for s in picked]}")

    collator = MtpCollator(pad_token_id=tokenizer.pad_token_id or 0)
    fixed_batch = collator(picked)  # one fixed mega-batch for overfit
    log.info(f"batched shape: input_ids={tuple(fixed_batch['input_ids'].shape)}  "
             f"mask_sum={int(fixed_batch['speech_mask'].sum().item())}")

    log.info("loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
    )
    base.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["lm_head"],   # critical — see CosyVoice2 recipe
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"trainable params: {n_trainable/1e6:.1f}M")

    # Verify lm_head is unfrozen.
    lm_head_trainable = any(
        p.requires_grad for n, p in model.named_parameters() if "lm_head" in n
    )
    log.info(f"lm_head trainable={lm_head_trainable}")
    assert lm_head_trainable, "lm_head should be in modules_to_save"

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=0.0, betas=(0.9, 0.95),
    )

    # --- BEFORE training ---
    log.info("--- probe BEFORE training ---")
    pre = probe_eos(model, fixed_batch, eos_id, "pre")

    # --- training ---
    log.info("--- training ---")
    losses_all = []
    losses_eos = []
    input_ids = fixed_batch["input_ids"].cuda()
    attn_mask = fixed_batch["attention_mask"].cuda()
    speech_mask = fixed_batch["speech_mask"].cuda()

    for step in range(1, N_STEPS + 1):
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits
        shift_logits = logits[:, :-1]
        shift_targets = input_ids[:, 1:]
        shift_mask = speech_mask[:, 1:].float()

        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_targets.reshape(-1),
            reduction="none",
            label_smoothing=0.1,    # borrowed from CosyVoice2 recipe
        ).reshape(shift_targets.shape)
        loss = (ce * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)

        eos_target_mask = (shift_targets == eos_id).float() * shift_mask
        eos_n = eos_target_mask.sum()
        eos_loss = (ce * eos_target_mask).sum() / eos_n.clamp_min(1.0)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        )
        optim.step()
        losses_all.append(loss.item())
        losses_eos.append(eos_loss.item())
        if step == 1 or step % 25 == 0:
            log.info(f"  step={step:3d}  loss={loss.item():6.3f}  "
                     f"loss@EOS={eos_loss.item():6.3f}  "
                     f"(EOS-positions in batch={int(eos_n.item())})")

    # --- AFTER training ---
    log.info("--- probe AFTER training ---")
    post = probe_eos(model, fixed_batch, eos_id, "post")

    log.info("=" * 60)
    log.info(f"loss      first 3: {[f'{x:.3f}' for x in losses_all[:3]]}")
    log.info(f"loss      last  3: {[f'{x:.3f}' for x in losses_all[-3:]]}")
    log.info(f"loss@EOS  first 3: {[f'{x:.3f}' for x in losses_eos[:3]]}")
    log.info(f"loss@EOS  last  3: {[f'{x:.3f}' for x in losses_eos[-3:]]}")

    pre_eos_probs = [r[0] for r in pre]
    post_eos_probs = [r[0] for r in post]
    log.info(f"P(EOS) pre  → mean={sum(pre_eos_probs)/max(len(pre_eos_probs),1):.4f}")
    log.info(f"P(EOS) post → mean={sum(post_eos_probs)/max(len(post_eos_probs),1):.4f}")

    eos_loss_start = sum(losses_eos[:3]) / 3
    eos_loss_end = sum(losses_eos[-3:]) / 3
    eos_drop = eos_loss_start - eos_loss_end
    log.info(f"EOS-loss drop: {eos_loss_start:.3f} → {eos_loss_end:.3f} (Δ={eos_drop:.3f})")

    avg_post_prob = sum(post_eos_probs) / max(len(post_eos_probs), 1)
    ok = eos_drop > 2.0 and avg_post_prob > 0.5
    if ok:
        log.info("OVERFIT: PASS — LoRA learned EOS termination on training samples.")
    else:
        log.warning("OVERFIT: WEAK — EOS not strongly learned. Investigate before HPC.")
    return ok


# ---------------------------------------------------------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    special = get_special_ids(tokenizer)
    eos_id = special["semantic_token_end"]
    log.info(f"semantic_token_end id = {eos_id}")

    hf_ds = load_from_disk(DATASET).remove_columns(["audio"])
    log.info(f"dataset size: {len(hf_ds)}  cols: {hf_ds.column_names}")

    ok1 = static_check(tokenizer, hf_ds, eos_id)
    ok2 = overfit_train(tokenizer, hf_ds, eos_id)
    print()
    print("=" * 60)
    print(f"STATIC:  {'PASS' if ok1 else 'FAIL'}")
    print(f"OVERFIT: {'PASS' if ok2 else 'WEAK / FAIL'}")
    print("=" * 60)
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
