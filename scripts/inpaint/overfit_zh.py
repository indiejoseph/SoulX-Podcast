"""Overfit a small zh batch under three silence policies — validates v5 fix.

Three training runs on the same 8 zh rows:
  (0) strip=False, inline=False — baseline v4 behaviour
  (1) strip=True,  inline=False — v5 boundary-only (insufficient: still collapses)
  (2) strip=True,  inline=True  — v5 + inline removal (proposed v6 fix)

Train each for ~500 steps with high phoneme_keep_prob (0.5) and small LR
(1e-4) to force the composer to fit the small set. Then check:

  * Final train loss
  * Final composer output on those exact rows (via inference): does
    inpaint generation produce diverse tokens or mode-collapse?

If the inline-filter fix is real, run (2) should produce non-collapsing
generation on these overfit-fit samples. Run (0) and (1) should still
collapse.

Time budget: ~10 min on 3090 (per run × 3 = 30 min total).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.inpaint import PhonemeComposer, apply_phoneme_inpaint
from soulxpodcast.training.inpaint_dataset import (
    InpaintDataset, InpaintDatasetConfig, collate,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("overfit")

MODEL_PATH = "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged"


def pick_zh_rows(n: int = 8) -> list[int]:
    """Return indices of n zh rows in tmp/dataset.jsonl with medium length."""
    idx = []
    with open("tmp/dataset.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r["lang"] == "zh" and 40 <= len(r["speech_tokens"]) <= 120:
                idx.append(i)
                if len(idx) >= n:
                    break
    return idx


@torch.no_grad()
def evaluate_inpaint_diversity(
    model, composer, tokenizer, dataset_obj, indices: list[int]
) -> dict:
    """For each row in indices, generate inpaint output and count token diversity.
    Returns mode-collapse rate (top-token frac across all gens)."""
    from soulxpodcast.inpaint.inference import InpaintInferenceEngine

    composer.eval()
    diversity = []
    mode_collapse = 0
    for i in indices:
        sample = dataset_obj[i]
        if sample is None: continue
        input_ids = sample["input_ids"].unsqueeze(0).cuda()
        attention_mask = sample["attention_mask"].unsqueeze(0).cuda()
        phone_token = sample["phone_token"].unsqueeze(0).cuda()
        phone_mask = sample["phone_mask"].unsqueeze(0).cuda()

        text_emb = model.get_input_embeddings()(input_ids).to(torch.bfloat16)
        composed, _ = composer(phone_token)
        inputs_embeds = apply_phoneme_inpaint(text_emb, composed, phone_mask)
        # Generate just the speech-token portion. Find <|semantic_token_start|>
        # position and truncate the prompt to up through that token, then generate.
        sem_start = tokenizer.encode("<|semantic_token_start|>", add_special_tokens=False)[0]
        sem_end_id = tokenizer.encode("<|semantic_token_end|>", add_special_tokens=False)[0]
        starts = (input_ids[0] == sem_start).nonzero(as_tuple=True)[0]
        if len(starts) == 0: continue
        prompt_end = int(starts[0].item()) + 1
        out = model.generate(
            inputs_embeds=inputs_embeds[:, :prompt_end],
            attention_mask=attention_mask[:, :prompt_end],
            max_new_tokens=150, do_sample=False,
            eos_token_id=sem_end_id,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
        gen = out[0].tolist()
        if gen and gen[-1] == sem_end_id:
            gen = gen[:-1]
        if not gen: continue
        c = Counter(gen)
        top_id, top_n = c.most_common(1)[0]
        top_frac = top_n / len(gen)
        diversity.append({"top_id": top_id, "top_frac": top_frac, "n_unique": len(c), "n_gen": len(gen)})
        if top_frac > 0.6 and len(gen) >= 30:
            mode_collapse += 1
    composer.train()
    return {"collapse_rate": mode_collapse / max(1, len(diversity)),
            "n_rows": len(diversity),
            "examples": diversity}


def run_one(strip_silence: bool, remove_inline: bool, indices: list[int],
            n_steps: int = 500) -> dict:
    tag = f"strip={strip_silence}, inline={remove_inline}"
    log.info(f"\n{'='*60}\nrun: {tag}, n_steps={n_steps}\n{'='*60}")
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    ds_cfg = InpaintDatasetConfig(
        phoneme_keep_prob=0.5,
        strip_silence_tokens=strip_silence,
        remove_silence_tokens_inline=remove_inline,
    )
    full_ds = InpaintDataset("tmp/dataset.jsonl", tokenizer, config=ds_cfg)
    train_ds = Subset(full_ds, indices)

    log.info(f"loaded {len(train_ds)} zh rows")
    for i in indices[:3]:
        r = full_ds.rows[i]
        log.info(f"  {r['id']!r}: text={r['text']!r}  n_tokens={len(r['speech_tokens'])}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    composer = PhonemeComposer(
        d_model=model.config.hidden_size, slots_per_token=8
    ).to("cuda", dtype=torch.bfloat16)
    composer.init_from_text_embed(model.get_input_embeddings())
    composer.train()

    optim = torch.optim.AdamW(composer.parameters(), lr=1e-4, weight_decay=0.0)

    loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate(b, pad_token_id=tokenizer.pad_token_id or 0),
        drop_last=True,
    )

    losses = []
    step = 0
    while step < n_steps:
        for batch in loader:
            if not batch: continue
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            speech_mask = batch["speech_mask"].cuda()
            phone_token = batch["phone_token"].cuda()
            phone_mask = batch["phone_mask"].cuda()

            text_emb = model.get_input_embeddings()(input_ids).to(torch.bfloat16)
            composed, _ = composer(phone_token)
            inputs_embeds = apply_phoneme_inpaint(text_emb, composed, phone_mask)
            out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                       use_cache=False, return_dict=True)
            logits = out.logits
            sh = logits[:, :-1].contiguous()
            tg = input_ids[:, 1:].contiguous()
            sm = speech_mask[:, 1:].to(torch.float32)
            V = sh.size(-1)
            ce = F.cross_entropy(sh.reshape(-1, V).float(), tg.reshape(-1),
                                reduction="none", label_smoothing=0.1).view(*sh.shape[:2])
            d = sm.sum().clamp_min(1)
            loss = (ce * sm).sum() / d

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(composer.parameters(), 1.0)
            optim.step()

            step += 1
            losses.append(float(loss))
            if step % 50 == 0:
                log.info(f"  step {step:4d}/{n_steps}  loss={loss:.4f}")
            if step >= n_steps:
                break

    log.info(f"final loss: {losses[-1]:.4f}  (step 0: {losses[0]:.4f})")

    # Now evaluate inpaint diversity on the overfit batch
    diversity = evaluate_inpaint_diversity(
        model, composer, tokenizer, full_ds, indices
    )
    log.info(f"mode-collapse rate on training set: {diversity['collapse_rate']:.0%}")
    log.info("per-row top-token domination:")
    for ex in diversity["examples"]:
        marker = "✗" if ex["top_frac"] > 0.6 else "✓"
        log.info(f"  {marker} top_id={ex['top_id']:>5d}@{ex['top_frac']:5.1%}  "
                 f"n_unique={ex['n_unique']:>3d}/{ex['n_gen']:>3d}")

    return {"final_loss": losses[-1], "first_loss": losses[0],
            "collapse_rate": diversity["collapse_rate"],
            "loss_curve": losses,
            "diversity": diversity}


def main():
    indices = pick_zh_rows(n=8)
    log.info(f"selected {len(indices)} zh rows from tmp/dataset.jsonl: {indices}")

    # Run 0: baseline v4 — no silence handling at all
    r0 = run_one(strip_silence=False, remove_inline=False,
                 indices=indices, n_steps=500)
    torch.cuda.empty_cache()

    # Run 1: v5 — boundary-only strip (known-insufficient)
    r1 = run_one(strip_silence=True, remove_inline=False,
                 indices=indices, n_steps=500)
    torch.cuda.empty_cache()

    # Run 2: v6 — boundary strip + inline removal
    r2 = run_one(strip_silence=True, remove_inline=True,
                 indices=indices, n_steps=500)

    log.info(f"\n{'='*60}\nFINAL COMPARISON\n{'='*60}")
    log.info(f"  (0) baseline (off/off)    final_loss={r0['final_loss']:.4f}  "
             f"collapse_rate={r0['collapse_rate']:.0%}")
    log.info(f"  (1) v5 boundary-only      final_loss={r1['final_loss']:.4f}  "
             f"collapse_rate={r1['collapse_rate']:.0%}")
    log.info(f"  (2) v6 boundary+inline    final_loss={r2['final_loss']:.4f}  "
             f"collapse_rate={r2['collapse_rate']:.0%}")

    if r2['collapse_rate'] < min(r0['collapse_rate'], r1['collapse_rate']):
        log.info(f"\n  ✓ Inline filter REDUCES mode collapse "
                 f"(baseline {r0['collapse_rate']:.0%} / "
                 f"strip-only {r1['collapse_rate']:.0%} → "
                 f"strip+inline {r2['collapse_rate']:.0%}). v6 fix validated.")
    else:
        log.info(f"\n  ✗ Inline filter does NOT reduce mode collapse. "
                 f"Need a different intervention before H100 spend.")


if __name__ == "__main__":
    main()
