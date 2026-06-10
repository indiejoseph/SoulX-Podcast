# SoulX pronunciation-inpaint v12 spec — grapheme substitution

Single-source spec for the v12 composer iteration. v12 changes exactly ONE
thing relative to v11 and holds everything else constant, so the behavioral
gate cleanly attributes the result to that change.

## 0. The one change, and why it is the whole ballgame

**v11 and every prior version made the phoneme annotation non-load-bearing.**
The alignment ([`_splice_pads`](../soulxpodcast/training/inpaint_dataset.py))
INSERTED a phoneme pad *after* the covering grapheme BPE and **kept the
grapheme**. So at an annotated position the frozen trunk saw BOTH the natural
reading (the surviving grapheme) AND the composer's phoneme. Since the
training corpus's phonemes are always the G2P of the text (the natural reading
— there is never a `phoneme ≠ default` counterfactual), the cheapest way to
minimise CE was for the composer to ignore phoneme identity entirely and emit
a per-alphabet "there is an annotation here, language = X" marker, letting the
trunk read the answer off the grapheme.

This was proven, not guessed. Diagnostic
([`scripts/inpaint/diagnose_composer_noop.py`](../scripts/inpaint/diagnose_composer_noop.py))
on the v11 30K checkpoint:

- composed-embedding norm ≈ **300× the text-embedding norm** (a giant constant
  component),
- `cos(composed_correct, composed_wrong) ≈ 0.99` for EVERY fixture, including
  ones the behavioral gate PASSES — i.e. embedding geometry does not predict
  behaviour,
- cross-fixture cosine forms a per-alphabet block: all jyutping inputs share
  one direction (cos 0.994), all pinyin another (0.96–0.97); phoneme identity
  is only a ~1–4% perturbation on a per-alphabet constant.

**v12 = true grapheme substitution.** At a kept/annotated unit, REMOVE the
covering grapheme BPE token(s) and replace them with the phoneme-carrying pad
token(s). The composed embedding becomes the ONLY signal for that position;
the trunk cannot fall back on a grapheme because there is none. This is the
counterfactual pressure that was missing for 11 versions — even though the
phoneme still equals the natural reading, the composer now MUST encode
pronunciation to recover the masked grapheme, so a wrong phoneme at inference
genuinely produces a different pronunciation.

**This is not a new idea — it is the upstream recipe v11 silently broke.**
The v11 spec (§1.1) documents that CosyVoice-Inpaint upstream does
`re.sub(r"\[.*?\]", pad_token, text)` — it *replaces* the bracketed grapheme
with a pad before BPE tokenization. v11's `_splice_pads` mis-implemented
"pad substitution" as "pad insertion" (kept the grapheme). v12 restores the
true substitution semantics.

## 1. Alignment mechanics

Shared splicer
[`_substitute_units(text_ids, offsets, units, pad_id, K)`](../soulxpodcast/training/inpaint_dataset.py)
is called by BOTH training and inference, so the train/inference contract
(guarded by [`test_alignment_parity.py`](../scripts/inpaint/test_alignment_parity.py))
is a single code path. A `unit` is `(char_start, char_end, blocks)`; each
block is one pad's phoneme ids. The splicer removes the unit's covering BPE(s)
and emits its pads in their place.

- **Substitution vs insertion fallback.** A unit's covering BPEs are removed
  only when every char they cover lies inside the substituted set — so a
  removal can never delete a neighbouring un-annotated grapheme. If a covering
  BPE bleeds onto an unsubstituted char (the English `" B"` leading-space
  token), that unit falls back to INSERTION (grapheme kept). For zh/yue every
  BPE covers whole CJK chars, so substitution is always clean there.
- **Merged BPEs** (`银行`, `中国` = 1 BPE, 2 chars): removed and replaced by 2
  pads (one syllable each). Verified.
- **Byte-fallback chars** (`哋`, `佢` = 2 BPEs, same char): both removed,
  replaced by 1 pad. Verified.

**BPE-group-coherent dropout (training).** Under v11's per-char dropout,
a merged BPE's two chars were kept independently (p_keep²≈6% both kept), so
merged BPEs almost never got masked — exactly the hard cases. v12 decides
keep/drop once per [`_bpe_coherence_groups`](../soulxpodcast/training/inpaint_dataset.py)
group (chars sharing a BPE are one group), so `银行` is kept/dropped whole and
gets masked at the full `keep_prob` rate. The per-CJK-char phoneme-queue
advance (incl. skipping `X` placeholders) is preserved, so fixtures align.

## 2. Held constant from v11 (so the gate attributes cleanly)

- Composer architecture UNCHANGED: per-alphabet heads (`head_cn` 2d→d→d,
  `head_cmu` 6d→d→d), shared `phone_emb`, K_STORAGE=6, ~43.1M params. The v11
  diagnostic suggested the heads were the wrong lever, but they were starved of
  pressure; v12 gives them correct supervision before judging the architecture.
  If v12 STILL collapses, the head/aux-loss/output-norm levers are v13.
- Loss UNCHANGED: silence-masked speech-token CE, label_smoothing=0.0.
- `phoneme_keep_prob=0.25`, `lang_balance=sqrt`, `init_from_text_embed=True`.
- 30K steps, lr 5e-4 cosine, warmup 500, bf16, AdamW8bit.

## 3. Eval metric — read it precisely, do NOT use it as the gate

Under substitution the `text_only` eval baseline (composer OFF) sees pad
tokens at the masked positions. So be precise about what each number means:

- `gap_vs_text = lm_loss_dropout − lm_loss_text_only` measures the composer's
  contribution **versus having pads (no info) at those positions** — a real,
  non-trivially-satisfiable signal (a marker can't fake it the way it could
  under v11's grapheme-insertion). It is **NOT** comparable to v11's numbers
  (different baseline) and it is **NOT** the composer-vs-grapheme gap.
- Review flagged that a cleaner curve would baseline against the *natural*
  (un-substituted) text. That is a valid optional add (return natural
  `input_ids` from the dataset and run the text_only branch on them), but it
  is eval interpretability only — it does not change training or the model.

**The verdict is the behavioral gate + diagnostic (§4), never the loss curve.**
Under v11 the loss looked great while the model was broken; do not repeat that.

## 4. Acceptance criteria (the gate, unchanged from v11's discipline)

Run after 30K steps:

1. **Parity**: `test_alignment_parity.py` 10/10. (Already green on the v12
   code.)
2. **Behavioral**: `test_alignment_behavioral.py` ≥ 30/38 AND the
   discriminative asserts (`M1.correct_vs_wrong_disagree`,
   `D1.correct_closer_to_baseline_than_wrong`) PASS on `yue_dei6` and
   `zh_yinhang` — the two combos where v11 produced bit-identical tokens for
   correct vs wrong phonemes (corr↔wrong edit = 0.0). v12 is GO only if those
   flip to non-zero edit distance.
3. **Diagnostic**: `diagnose_composer_noop.py` cross-fixture cosine drops well
   below v11's per-alphabet-block pattern, and `cos(correct, wrong)` per
   fixture drops below ~0.9 on dei6/yinhang. This is the mechanistic
   confirmation that phoneme identity now lives in the composed direction.
4. No new FAIL vs v11's set beyond what items 2–3 fix.

If item 2 still shows corr↔wrong edit = 0.0 on dei6/yinhang while loss looks
good, v12's data fix was insufficient and the residual shortcut (context
disambiguation / frozen-trunk readout) is active. The fallback levers are
ranked and the first is **already implemented, default-off**:

1. **`--aux_phoneme_loss_weight 0.2` (v12.1)** — a direct phoneme-recovery BCE:
   a `Linear(d→vocab)` head (trainer-only, not in the composer state, so
   inference/parity are unaffected) must recover each masked position's
   phoneme ids from the composed embedding. This forces phoneme-discriminative
   composed vectors at the source — it attacks the cos≈0.99 collapse directly
   rather than relying on the indirect speech-token CE. Verified to run + the
   aux loss decreases on a 30-step smoke. Flip the flag and rerun; no code
   change. (Held off the primary v12 run to keep it a clean one-variable test
   and to avoid an untuned loss term regressing the expensive run.)
2. **`--phoneme_keep_prob 0.5`** — substitution makes only the kept positions
   exercise the composer; at 0.25, 75% of steps give zero composer gradient.
   0.5 doubles the composer gradient per step (training annotation density is
   decoupled from the sparse inference distribution since the composer is
   position-local). Free, no code change.
3. **Small LoRA on the trunk attention** — last resort if the FROZEN trunk
   genuinely cannot read composed embeddings that lie off the text-embedding
   manifold. Larger change (unfreezes a sliver of the backbone).

Adversarial review (4 independent lenses) judged substitution NECESSARY but
possibly INSUFFICIENT alone — chiefly because Chinese is highly predictable
from context (the trunk may guess the masked syllable without the phoneme) and
because the only phoneme pressure is indirect. Levers 1–2 close both gaps and
are cheap; that is why lever 1 is pre-built.

## 5. Rollback

`--no_grapheme_subst` reproduces v11 insertion behaviour (ablation only;
inference always substitutes, so such a model is train/inference-mismatched).
Legacy `_align_padsub_*` / `_splice_pads` kept in-tree for reference.
