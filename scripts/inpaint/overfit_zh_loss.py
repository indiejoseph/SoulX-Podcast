"""Overfit zh under three LOSS formulations — validates the v7 objective fix.

The v6 dataset transforms (boundary-strip + inline-removal) and the LN
composer were workarounds for a wrongly-shaped loss. This script tests
whether *fixing the loss* removes the need for those workarounds.

All three runs use the SAME composer (LN-aware) and SAME dataset
transform (boundary-strip only, NO inline-removal) so the only axis
that varies is the training objective. Inline-removal is OFF in all
three because the loss is now responsible for excluding silence
positions, not the dataset transform.

Loss formulations:
  (A) standard CE          — current v4-v6 objective. CE on every
                             supervised speech-token position
                             (incl. silence). Should mode-collapse —
                             this is the bug we're naming.
  (B) silence-masked CE    — CE only on non-silence-target positions.
                             Composer no longer rewarded for emitting
                             silence. Cheapest fix; no second forward.
  (C) masked CE + KL       — (B) plus a KL-to-baseline regulariser at
                             silence-target positions. Explicitly
                             penalises the composer for shifting the
                             LLM's distribution where it shouldn't.
                             Requires an extra no-grad forward pass.

Each runs 500 steps on the same 8 zh rows.
Per-run time: ~5 min (A/B) / ~8 min (C) → ~20 min total on 3090.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.inpaint import PhonemeComposer, apply_phoneme_inpaint
from soulxpodcast.training.inpaint_dataset import (
    InpaintDataset, InpaintDatasetConfig, SILENCE_TOKEN_IDS, collate,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("overfit-loss")

MODEL_PATH = "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged"
SPEECH_TOKEN_OFFSET = 153595  # s3 id 0 → LLM id 153595


def pick_zh_rows(n: int = 8) -> list[int]:
    idx = []
    with open("tmp/dataset.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r["lang"] == "zh" and 40 <= len(r["speech_tokens"]) <= 120:
                idx.append(i)
                if len(idx) >= n:
                    break
    return idx


def build_silence_id_lut(vocab_size: int, lang: str = "zh") -> torch.Tensor:
    """Bool LUT over the full LLM vocab. True at LLM ids that are silence."""
    lut = torch.zeros(vocab_size, dtype=torch.bool)
    for s3 in SILENCE_TOKEN_IDS.get(lang, frozenset()):
        llm_id = s3 + SPEECH_TOKEN_OFFSET
        if 0 <= llm_id < vocab_size:
            lut[llm_id] = True
    return lut


@torch.no_grad()
def evaluate_inpaint_diversity(model, composer, tokenizer, dataset_obj,
                               indices: list[int]) -> dict:
    composer.eval()
    diversity = []
    mode_collapse = 0
    sem_start = tokenizer.encode("<|semantic_token_start|>", add_special_tokens=False)[0]
    sem_end_id = tokenizer.encode("<|semantic_token_end|>", add_special_tokens=False)[0]
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
        diversity.append({"top_id": top_id, "top_frac": top_frac,
                          "n_unique": len(c), "n_gen": len(gen)})
        if top_frac > 0.6 and len(gen) >= 30:
            mode_collapse += 1
    composer.train()
    return {"collapse_rate": mode_collapse / max(1, len(diversity)),
            "n_rows": len(diversity),
            "examples": diversity}


def run_one(loss_mode: str, indices: list[int], n_steps: int = 500,
            kl_weight: float = 0.5) -> dict:
    """loss_mode ∈ {'ce_full', 'ce_masked', 'ce_masked_kl'}.

    All three keep boundary-strip ON (it's about discarding the leading/
    trailing silence padding noise) and inline-removal OFF (the loss
    decides which positions count, not the dataset).
    """
    assert loss_mode in {"ce_full", "ce_masked", "ce_masked_kl"}
    log.info(f"\n{'='*60}\nrun: loss={loss_mode}, n_steps={n_steps}\n{'='*60}")
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    ds_cfg = InpaintDatasetConfig(
        phoneme_keep_prob=0.5,
        strip_silence_tokens=True,
        remove_silence_tokens_inline=False,  # loss handles silence — not dataset
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

    silence_lut = build_silence_id_lut(model.config.vocab_size, "zh").cuda()

    loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate(b, pad_token_id=tokenizer.pad_token_id or 0),
        drop_last=True,
    )

    losses_lm: list[float] = []
    losses_kl: list[float] = []
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

            # Per-position silence-target mask (lookup against the LLM-id LUT)
            is_silence_tg = silence_lut[tg]                 # bool (B, T-1)
            silence_pos = sm * is_silence_tg.float()        # silence positions only
            content_pos = sm * (~is_silence_tg).float()     # non-silence positions

            ce_all = F.cross_entropy(sh.reshape(-1, V).float(), tg.reshape(-1),
                                     reduction="none", label_smoothing=0.1
                                    ).view(*sh.shape[:2])

            if loss_mode == "ce_full":
                # Current v4-v6 behaviour: CE on every supervised position.
                d = sm.sum().clamp_min(1)
                lm_loss = (ce_all * sm).sum() / d
                kl_loss = torch.zeros((), device=lm_loss.device)
            elif loss_mode in {"ce_masked", "ce_masked_kl"}:
                # CE only at non-silence-target positions.
                d_ce = content_pos.sum().clamp_min(1)
                lm_loss = (ce_all * content_pos).sum() / d_ce
                if loss_mode == "ce_masked_kl":
                    # Baseline forward — composer OFF, no grad.
                    with torch.no_grad():
                        baseline_logits = model(
                            inputs_embeds=text_emb,
                            attention_mask=attention_mask,
                            use_cache=False, return_dict=True,
                        ).logits
                    bsh = baseline_logits[:, :-1].contiguous()
                    # KL(composer || baseline) at silence-target positions.
                    log_p_c = F.log_softmax(sh.float(), dim=-1)
                    log_p_b = F.log_softmax(bsh.float(), dim=-1)
                    kl_per_pos = F.kl_div(
                        log_p_c, log_p_b, log_target=True, reduction="none"
                    ).sum(-1)                                # (B, T-1)
                    d_kl = silence_pos.sum().clamp_min(1)
                    kl_loss = (kl_per_pos * silence_pos).sum() / d_kl
                else:
                    kl_loss = torch.zeros((), device=lm_loss.device)
            loss = lm_loss + (kl_weight * kl_loss if loss_mode == "ce_masked_kl" else 0.0)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(composer.parameters(), 1.0)
            optim.step()

            step += 1
            losses_lm.append(float(lm_loss))
            losses_kl.append(float(kl_loss))
            if step % 50 == 0:
                log.info(f"  step {step:4d}/{n_steps}  lm={lm_loss:.4f}  "
                         f"kl={kl_loss:.4f}  total={float(loss):.4f}")
            if step >= n_steps:
                break

    log.info(f"final lm_loss: {losses_lm[-1]:.4f}  (step 0: {losses_lm[0]:.4f})")
    if loss_mode == "ce_masked_kl":
        log.info(f"final kl_loss: {losses_kl[-1]:.4f}  (step 0: {losses_kl[0]:.4f})")

    diversity = evaluate_inpaint_diversity(
        model, composer, tokenizer, full_ds, indices
    )
    log.info(f"mode-collapse rate on training set: {diversity['collapse_rate']:.0%}")
    log.info("per-row top-token domination:")
    for ex in diversity["examples"]:
        marker = "✗" if ex["top_frac"] > 0.6 else "✓"
        log.info(f"  {marker} top_id={ex['top_id']:>5d}@{ex['top_frac']:5.1%}  "
                 f"n_unique={ex['n_unique']:>3d}/{ex['n_gen']:>3d}")

    return {"loss_mode": loss_mode,
            "final_lm": losses_lm[-1], "first_lm": losses_lm[0],
            "final_kl": losses_kl[-1],
            "collapse_rate": diversity["collapse_rate"],
            "diversity": diversity}


def main():
    indices = pick_zh_rows(n=8)
    log.info(f"selected {len(indices)} zh rows from tmp/dataset.jsonl: {indices}")

    # (A) standard CE — the wrong-shape baseline (mirrors current v4-v6 loss).
    rA = run_one(loss_mode="ce_full", indices=indices, n_steps=500)
    torch.cuda.empty_cache()

    # (B) silence-masked CE — cheapest fix; composer never sees silence in loss.
    rB = run_one(loss_mode="ce_masked", indices=indices, n_steps=500)
    torch.cuda.empty_cache()

    # (C) masked CE + KL — principled; baseline regularises non-target positions.
    rC = run_one(loss_mode="ce_masked_kl", indices=indices, n_steps=500)

    log.info(f"\n{'='*60}\nFINAL COMPARISON (zh overfit, 8 rows, 500 steps)\n{'='*60}")
    log.info(f"  (A) standard CE          final_lm={rA['final_lm']:.4f}  "
             f"collapse_rate={rA['collapse_rate']:.0%}")
    log.info(f"  (B) silence-masked CE    final_lm={rB['final_lm']:.4f}  "
             f"collapse_rate={rB['collapse_rate']:.0%}")
    log.info(f"  (C) masked CE + KL       final_lm={rC['final_lm']:.4f}  "
             f"final_kl={rC['final_kl']:.4f}  "
             f"collapse_rate={rC['collapse_rate']:.0%}")

    best = min([(rA, "A"), (rB, "B"), (rC, "C")],
               key=lambda x: x[0]["collapse_rate"])[1]
    log.info(f"\n  best collapse rate: {best}")


if __name__ == "__main__":
    main()
