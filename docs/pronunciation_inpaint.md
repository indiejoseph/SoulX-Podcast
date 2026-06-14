# Pronunciation inpaint — design & training

How SoulX-Podcast lets a user override the pronunciation of specific
words/characters via inline phoneme annotation (SSML `<phoneme>` tags), without
retraining or unfreezing the LLM.

## Mechanism

A tiny **PhonemeComposer** (~43M params,
[soulxpodcast/inpaint/composer.py](../soulxpodcast/inpaint/composer.py)) sits on
top of the **frozen** Qwen3 trunk. At an annotated position it produces a
`d_model` embedding that is injected into the trunk's input via
`apply_phoneme_inpaint`, in place of the text embedding.

- **Phoneme alphabets**: jyutping (yue), pinyin (zh), CMUdict/ARPAbet (en),
  routed by `alphabet_id`. Slot storage `K=6`. Two heads share one `phone_emb`:
  `head_cn` (2 slots: initial, final-with-tone) for jyutping ∪ pinyin, and
  `head_cmu` (6 slots) for English.
- **Alignment = grapheme substitution.** This is the load-bearing design choice.
  At a kept/annotated unit, the covering grapheme BPE token(s) are **removed**
  from `input_ids` and **replaced** by the phoneme-carrying pad token(s). The
  composer fires only at those pads. Because the grapheme is gone, the composed
  embedding is the *only* signal for that position — the LM cannot read the
  natural pronunciation off a surviving grapheme.

### Why substitution (not insertion)

The training corpus's phonemes are always the G2P of the text (the natural
reading) — there is no `phoneme ≠ default` counterfactual. If the grapheme is
*kept* and a pad merely *inserted* next to it, the cheapest way to fit the
speech-token CE is for the composer to emit a constant per-alphabet "marker"
and let the trunk read the answer off the grapheme. The composer then learns
nothing phoneme-specific (composed embeddings collapse to ~one direction per
alphabet). Removing the grapheme is what forces the composer to encode
pronunciation, so a *wrong* phoneme at inference actually changes the output.

This matches the upstream CosyVoice-Inpaint recipe
(`re.sub(r"\[.*?\]", pad, text)` — substitute the bracketed grapheme with a
pad before BPE tokenization).

### Alignment internals
([soulxpodcast/training/inpaint_dataset.py](../soulxpodcast/training/inpaint_dataset.py))

- `_substitute_units(text_ids, offsets, units, pad_id, K, free_chars=…)` —
  shared by **both** training and inference (single code path → the
  train/inference contract is guarded by `test_alignment_parity.py`). A unit's
  covering BPEs are removed when every char they cover is either part of the
  substituted set or whitespace (`free_chars`); a bleed onto a non-whitespace
  neighbour falls back to insertion (and warns at inference).
- `_bpe_coherence_groups` — chars that share a BPE are kept/dropped together, so
  merged BPEs (`银行`, `中国` = 1 BPE / 2 chars) are masked as a whole at the
  full `keep_prob`, and byte-fallback chars (`哋`, `佢` = 2 BPEs / 1 char) are
  removed together.
- `free_chars` (whitespace) is what makes **English** work: the Qwen tokenizer
  bundles a word's leading space into its first BPE (`" B"` in `I LIKE BANANA`);
  marking whitespace free lets that BPE be cleanly substituted instead of
  falling back to insertion.

## Training
([soulxpodcast/training/train_inpaint.py](../soulxpodcast/training/train_inpaint.py))

- Backbone frozen entirely; only the composer trains. Forward via
  `inputs_embeds=`. Speech-token CE with **silence-target positions masked out**
  (so the composer is never rewarded for predicting silence-padding tokens).
- **Unit-level dropout** (`phoneme_keep_prob=0.25`, BPE-group-coherent) matches
  the sparse inference distribution (a user annotates a few words per sentence).
- `lang_balance=sqrt` upsamples minority languages (corpus is ~261k yue / 85k
  en / 5.6k zh).
- Optional `--aux_phoneme_loss_weight` (default 0): a `Linear(d→vocab)` head must
  recover each masked position's phoneme ids from the composed embedding —
  directly enforces phoneme-discriminative embeddings. Off for the shipped
  model; use to tighten a data-starved alphabet (pinyin).
- Recipe: [scripts/inpaint/train_h100_inpaint.pjm](../scripts/inpaint/train_h100_inpaint.pjm)
  (30K steps, bf16, AdamW8bit, lr 5e-4 cosine).

## Test gate (do NOT trust loss alone)

Loss curves looked healthy through many broken iterations; the behavioral gate
is the verdict.

1. **Parity** — `scripts/inpaint/test_alignment_parity.py`: training and
   inference build byte-identical `input_ids`/slots for the same annotation
   (12/12 incl. English).
2. **Behavioral** — `scripts/inpaint/test_alignment_behavioral.py`: on
   adversarial fixtures (`scripts/inpaint/fixtures_adversarial.py`), `correct`
   vs `wrong` phonemes must produce different speech tokens (M1), and `correct`
   must be closer to the no-inpaint baseline than `wrong` (D1).
3. **Diagnostic** — `scripts/inpaint/diagnose_composer_noop.py`: composed
   embeddings for different phonemes must be distinct (no per-alphabet-marker
   collapse).

### English validation caveat

The behavioral gate decodes greedily with `repetition_penalty=1.0` for
deterministic A/B. The base SoulX LLM loops under that setting on short English
prompts, so the en baseline doesn't EOS and the en gate asserts are unreliable —
this is a **harness artifact, not a model failure** (the base LLM terminates on
English fine under deployment decoding: sampling + `repetition_penalty=1.1`).
Validate English steering with sampling + `repetition_penalty=1.1`, or compare
pre-EOS token prefixes.

## Status

Steers pronunciation across **yue, zh, en**. zh (pinyin) is the weakest alphabet
purely from data scarcity (5.6k rows); the aux-loss lever or more zh data is the
way to tighten it. Audio A/B synthesis (composer → flow → HiFT):
[scripts/inpaint/inference_audio.py](../scripts/inpaint/inference_audio.py).
Runtime injection of composed embeddings: the cosyvoice-inpaint skill +
[soulxpodcast/inpaint/inference.py](../soulxpodcast/inpaint/inference.py).
