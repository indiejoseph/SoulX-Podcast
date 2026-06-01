# Upstream CosyVoice-Inpaint — verbatim module references

Location on disk: `/home/joseph/projects/notebooks/notebooks/projects/CosyVoice-Inpaint`

This page lists the exact upstream signatures and constants we depend on
when porting to SoulX-Podcast. When the upstream changes, this page must
be re-derived from the source files listed.

## `pron_inpaint/tokenizer.py`

```python
ONSETS    = "b d g gw z p t k kw c m n ng f h s l w j".split()   # 19
NUCLEUSES = "aa a i yu u oe e eo o m n ng".split()                # 12 (m, n, ng are syllabic nasals)
CODAS     = "p t k m n ng i u".split()                             # 8
TONES     = "1 2 3 4 5 6".split()                                  # 6

# global offset layout — every component lives in its own id range,
# pad=0 is shared
ONSET_OFFSET   = 0                                  # ids 1..19
NUCLEUS_OFFSET = len(ONSETS)                        # ids 20..31
CODA_OFFSET    = NUCLEUS_OFFSET + len(NUCLEUSES)    # ids 32..39
TONE_OFFSET    = CODA_OFFSET + len(CODAS)           # ids 40..45
TOTAL_PHONEME_VOCAB = TONE_OFFSET + len(TONES) + 1  # = 46
```

`convert_phone_str_to_flat_ids(phone_str, L)` returns a flat list of
length `4*L` with the interleaved per-token layout
`[onset_0, nucleus_0, coda_0, tone_0, onset_1, ...]`. Pad ids (0) fill
positions that have no Jyutping.

## `pron_inpaint/modeling.py:Qwen2LMInpaint`

Constructor:
```python
Qwen2LMInpaint(
    qwen2lm,                        # frozen Qwen2LM instance
    vocab_size,                     # = TOTAL_PHONEME_VOCAB
    tone_offset,                    # = TONE_OFFSET, used by aux tone loss
    composition='concat_linear',    # or 'sum'
    num_tones=7,                    # 6 + pad
    tone_weight=0.3,
)
```

Trainable parameters:
- `phone_emb: nn.Embedding(vocab_size, d_model, padding_idx=0)`
- `composer: Sequential(Linear(3·d → d), GELU, Linear(d, d))`
  (or `nn.Identity` for `composition='sum'`)
- `tone_classifier: nn.Linear(d_model, num_tones)`

`compose_phoneme(phoneme_flat, phone_token_len=None)` returns
`(composed (B, L, d_model), pf_mask (B, L) bool)`. If
`phone_token_len` is provided it counts *phoneme tokens* (P), NOT
`4·P` and NOT `4·L`.

`forward(batch, device)` produces:
```python
{
  "loss":      lm_loss + tone_weight * tone_loss,
  "acc":       float,
  "lm_loss":   float (detached),
  "tone_loss": float (detached),
}
```

`inference(...)` is a generator over speech token ids; it composes
`phone_token` for the **text part only** (skipping the prompt prefix),
zeros the text embedding at masked positions, adds the composed
embedding, then feeds the assembled `lm_input` to the wrapped
`Qwen2LM.inference_wrapper`.

`init_component_from_text_embed()` initialises `phone_emb` as
`N(0, 0.02)` plus the average row of the LLM's `embed_tokens.weight`.
Pad row (index 0) is zeroed.

## `pron_inpaint/frontend_wrapper.py:InpaintFrontendWrapper`

Bracket regex: `r"\[([a-z]+[1-6]{1})\]"` — exact Jyutping syllable
form, single tone digit required.

`text_normalize(text, split, text_frontend)`:
- Splits on bracketed segments, runs wetext/ttsfrd only on the
  outside text, then re-joins with brackets preserved.
- For SSML input we will replace this with a `parse_ssml(text)` →
  `(plain_text, spans)` step.

`_extract_text_token(text)`:
- Replaces every `[…]` with `self.frontend_tokenizer.tokenizer.pad_token`
  (the literal pad string, not its id).
- Runs the tokenizer with `allowed_special='all'`.
- Returns `text_token: (1, text_len), text_token_len: (1,)`.

`_extract_phone_token(text, text_token)`:
- Asserts `phoneme_count == pad_count`.
- Encodes each bracketed Jyutping via
  `JyutpingTokenizer.encode([" ".join(phonemes)])` → flat list of
  length `phoneme_count * 4`.
- Walks `text_token`, writes the next 4-block into each pad position.
- Returns `phone_token: (1, text_len * 4), phone_token_len: (1,)`
  (which is `phone_token.shape[1]` = `4 * text_len` — confusing,
  because the model expects this slot to count `P`; the wrapper
  passes a degenerate value because at inference it isn't read).

## `pron_inpaint/utils.py:tokenize_add_label`

Stable PRNG seeded with `sha256(text + "|" + phone) ^ user_seed` so
two different processes (e.g. dataset workers with `num_proc>1`)
emit the same masked output. For each syllable `s`, with probability
`insert_prob`, append `[s]` after the corresponding character.
Handles three alignment cases:

1. `len(words) == len(sylls)` — direct 1:1 (the common case for
   CJK characters).
2. `len(non-space-chars) == len(sylls)` — character-level fallback.
3. otherwise — best-effort first `min(len, len)` words.

Returns `{text, text_with_jyutping, speech_token, phone, valid_phon}`.

## `train.py` — pipeline contract

CSV columns expected: `text`, `speech_tokens` (or `speech_token`),
optional `phone`.

Two dataset.map passes:
1. `tokenize_add_label` (multi-proc safe).
2. `_apply_frontend` (single-process, caches the wrapper on the fn
   object).

Hyperparameters that hit quality:
- `--phoneme_keep_prob` (default 0.25)
- `--dataset_factor` (default 1, with 4 you hit ~100% annotated)
- `--bf16`
- `--warmup` (default 1000)

`CustomDataCollatorWithPadding.pad_value` table:
| field | pad value |
|---|---|
| `text_token` | 151643 (Qwen2 pad id) |
| `speech_token` | 6561 (CosyVoice2 vocab size) |
| `phone_token` | 0 (phoneme pad) |

`freeze_backbone_except_inpaint` zeros `requires_grad` on the wrapped
LM and re-enables it on `phone_emb`, `composer`, and the legacy
per-component embeddings if present (forward-compat shim).

`save_safetensors=False` is required — Qwen2 ties input/output
embeddings and safetensors refuses shared-storage tensors.

## `pron_inpaint/patch.py`

`patch_cosyvoice2(cosyvoice)`:
1. Wraps `cosyvoice.frontend` with `InpaintFrontendWrapper`.
2. Rebinds `cosyvoice.model.llm_job` and `cosyvoice.model.tts` to
   versions that thread `phone_token` through.
3. Rebuilds `cosyvoice.model.llm = Qwen2LMInpaint(orig, ...)`.

This patch-by-monkey-bind pattern is replaced in SoulX-Podcast by an
explicit `InpaintEngine` class so the change is reviewable. The
upstream pattern is fine for an experiment but bad for code review.
