---
name: cosyvoice-inpaint
description: Reference for the CosyVoice-Inpaint pronunciation-inpainting technique and its planned adaptation to SoulX-Podcast. Covers the training objective (frozen LLM + tiny phoneme composition module + auxiliary tone loss), the compositional phoneme embedding mechanism (onset/nucleus/coda/tone with global offsets and interleaved per-text-token slots), the inline-annotation frontend pipeline, and the SoulX-specific design choices for SSML input and multi-lingual phoneme alphabets (CMUdict / Jyutping / Pinyin). Use whenever implementing, debugging, or extending the SoulX-Podcast pronunciation-inpaint training pipeline, the SSML frontend, or the runtime that injects composed phoneme embeddings into the Qwen3 trunk.
---

# CosyVoice-Inpaint — Pronunciation Inpainting Skill

This skill documents two things together:

1. The **upstream CosyVoice-Inpaint** technique (lives at
   `/home/joseph/projects/notebooks/notebooks/projects/CosyVoice-Inpaint`,
   working on Qwen2-based CosyVoice2-0.5B with Jyutping only).
2. The **SoulX-Podcast adaptation** planned on this branch
   (`experiment/pronunciation-inpaint`): SSML input, three phoneme alphabets,
   Qwen3-1.7B trunk, and SoulX's bi-stream / chunked streaming runtime.

Read [`references/upstream.md`](references/upstream.md) for verbatim
upstream module references and [`references/soulx-adaptation.md`](references/soulx-adaptation.md)
for the SoulX wiring map.

---

## 1. What pronunciation inpainting is

A way to **locally override pronunciation** of a single word or syllable
without fine-tuning the LLM and without growing the LLM tokenizer's vocab.

```
Input:  '<speak>Hello <phoneme alphabet="ipa" ph="wɜːld">world</phoneme>!</speak>'
Output: 24 kHz audio where "world" is pronounced as the supplied phonemes,
        everything else uses the LLM's learned pronunciation.
```

The trick: replace the **input embedding** at the annotated text positions
with a *composed phoneme embedding* that lives in the same `d_model` space.
The transformer never sees a new token id; it sees a different vector at
that position. Because the embedding is composed from a small trainable
table, you only train a few hundred K to a couple M parameters.

---

## 2. Training objective (upstream — and what we will keep)

**Backbone is frozen.** All Qwen2LM parameters have `requires_grad=False`.

**Trainable modules** (≤ ~5 M params total):
- `phone_emb: nn.Embedding(vocab_size, d_model, padding_idx=0)` — unified
  phoneme component table. Index 0 is reserved as "no phoneme / pad".
- `composer` — either `nn.Identity` (for `composition='sum'`) or a small
  MLP `Linear(3·d_model → d_model) → GELU → Linear(d_model → d_model)`
  for `composition='concat_linear'`.
- `tone_classifier: nn.Linear(d_model, num_tones)` — auxiliary head used
  *only* by the auxiliary loss; never used at inference.

**Primary loss** is the unchanged CosyVoice acoustic CE on next speech
token. Gradients only flow into the phoneme modules because everything
else is frozen.

**Auxiliary loss** is a tone-classification CE applied to the *composed*
phoneme embedding (not to text embeddings — they are stop-gradient'd via
`+ 0.0 * text_token_emb.detach()` so the classifier is forced to read
tone from the phoneme path). Weighted by `tone_weight` (upstream default
`0.3`):

```
loss = lm_loss + tone_weight * tone_loss
```

This auxiliary loss is the linchpin that prevents the composer from
shortcutting to a copy of the text embedding — it forces tone signal to
*live inside* the phoneme embedding rather than being recovered from
neighbouring text context.

**Initialization heuristic** (`init_component_from_text_embed`): the
non-pad rows of `phone_emb` are initialised to `N(0, 0.02)` plus the
average of the LLM's text embedding matrix. This keeps the composed
embedding on the manifold the frozen transformer was trained on, so
gradient signal is sane from step 1.

**Optional anchoring regulariser** (mentioned in upstream README, not
implemented):
```
|| LLM(E_phoneme) − LLM(E_text_equivalent) ||
```
i.e. for character positions where you know the canonical Jyutping
already, the composed phoneme embedding's downstream hidden state should
not drift far from the text-token-only forward. Worth trying when
embedding-distribution-mismatch shows up as audio artefacts.

---

## 3. The phoneme embedding mechanism

### 3.1 Components (upstream, Jyutping)

Each syllable is decomposed into four phonological components:

| Component | Examples | Pad id |
|---|---|---|
| **Onset** (initial consonant cluster) | `b`, `gw`, `c`, `m`, `""` | 0 |
| **Nucleus** (vowel) | `aa`, `i`, `oe`, `yu`, `eo` | 0 |
| **Coda** (final consonant) | `p`, `t`, `ng`, `i`, `u`, `""` | 0 |
| **Tone** | `1`..`6` (Cantonese has 6) | 0 |

All four IDs share one `nn.Embedding` table; component disambiguation is
done by **global offsets** so an onset id is in `[1, n_onset]`, a nucleus
in `[n_onset+1, n_onset+n_nucleus]`, etc. (see
`pron_inpaint/tokenizer.py:ONSET_OFFSET / NUCLEUS_OFFSET / CODA_OFFSET /
TONE_OFFSET`). Index 0 is **always** pad.

```
TOTAL_PHONEME_VOCAB = 1 + n_onset + n_nucleus + n_coda + n_tone
                    = 1 + 19 + 11 + 8  + 6   = 45 (upstream)
```

### 3.2 Composition function

For each text-token position the model gathers four embeddings
`(on, nu, co, to)` of shape `(B, L, d_model)` and composes:

```
sum:           E_phoneme = on + nu + co + to
concat_linear: E_phoneme = composer(cat([on, nu, co], -1)) + to    # tone as residual
```

The output sits in the LLM's `d_model` space and **replaces** the text
embedding at that position:

```python
text_token_emb = text_token_emb * (~pf_mask).unsqueeze(-1).float()
text_token_emb = text_token_emb + composed * pf_mask.unsqueeze(-1).float()
```

`pf_mask: (B, L) bool` is True where any of the four components is
non-zero. Where it is False the text embedding passes through untouched.

### 3.3 Interleaved per-token slot layout

The buffer that carries phoneme ids to the model is **flat, per-token,
interleaved**:

```
phone_token:  (B, 4 * L)
              [onset_0, nucleus_0, coda_0, tone_0,
               onset_1, nucleus_1, coda_1, tone_1,
               ...
               onset_{L-1}, nucleus_{L-1}, coda_{L-1}, tone_{L-1}]
```

`L` = padded text-token length. A text token that has no phoneme
annotation gets `[0, 0, 0, 0]` in its slot block and is unaffected.

The variant with `phone_token_len: (B,)` describing the *number of
phoneme syllables `P`* (not `4·P`!) supports collators that pad on the
flattened axis instead of the per-token axis. See
`Qwen2LMInpaint.compose_phoneme` for the reconstruction logic.

**Important** for any future change: when the user passes
`phone_token_len`, it counts *phoneme tokens (syllables) `P`*, NOT
`4·P` and NOT `4·L`. This has bitten the upstream once already; the
docstring spells it out.

---

## 4. Frontend / insertion mechanism (upstream)

Inline annotation form: `你好呀[aa3]！` — a square-bracketed
Jyutping syllable directly after the character it annotates.

`InpaintFrontendWrapper` does four things:

1. **Text normalisation** (`text_normalize`) — runs wetext or ttsfrd
   only on the *outside* of bracketed segments, then re-joins the
   bracketed parts so the annotations survive normalisation intact.
2. **Bracket → pad token** (`_extract_text_token`) — runs the
   normalised text through Qwen tokenizer after replacing every `[…]`
   with the tokenizer's `pad_token` (the literal pad string). One pad
   token id per annotation is what carries the position into the
   text-token tensor.
3. **Build phoneme buffer** (`_extract_phone_token`) — re-extracts the
   bracketed Jyutping strings, asserts the count matches the pad count
   in `text_token`, encodes each syllable into 4 global ids via
   `JyutpingTokenizer.encode`, and writes them into the 4-slot block at
   each pad position. Non-pad positions stay zero.
4. **Assert alignment** — `phoneme_count == pad_count` is hard-asserted
   so a parse mismatch fails loudly rather than silently misaligning.

At inference, the same wrapper is invoked from `frontend_zero_shot` /
`frontend_sft`; the resulting `phone_token` is threaded through
`cosyvoice2_llm_job` → `Qwen2LMInpaint.inference` which composes and
injects the embedding *only on the text part* (the speech-prompt and
sos/task-id positions are left untouched).

---

## 5. Training pipeline (upstream)

`train.py` ingests a CSV with columns `text`, `speech_tokens`, and
optionally `phone` (space-separated Jyutping syllables aligned 1:1
with text characters).

Two map passes:

1. **`tokenize_add_label`** (`pron_inpaint/utils.py`) — for each sample
   with a parsable `phone` column, deterministically (seeded by
   sha256(text+phone) ^ user seed) decide for each syllable whether to
   reveal it as an inline bracket. The fraction revealed is
   `--phoneme_keep_prob` (default `0.25`). Emits `text_with_jyutping`.

2. **`_apply_frontend`** — single-process pass that instantiates
   `InpaintFrontendWrapper` once and reuses it via a function attribute
   (avoids pickling the tokenizer across `num_proc` workers). Produces
   the per-sample `text_token` and `phone_token` lists.

Then `dataset.train_test_split` and a HuggingFace `Trainer` with:

- `CustomDataCollatorWithPadding` pads `text_token` to `pad_token_id =
  151643`, `speech_token` to `num_speech_tokens = 6561`, `phone_token`
  to `0` (the phoneme pad).
- `CustomTrainer.compute_loss` delegates to the bound
  `Qwen2LMInpaint.forward(batch, device)` which returns the dict with
  `loss / lm_loss / tone_loss / acc`.
- `freeze_backbone_except_inpaint` zeros `requires_grad` on the wrapped
  Qwen2LM and re-enables it on `phone_emb`, `composer`, and (legacy)
  per-component embeddings if present.
- `--dataset_factor N` duplicates the dataset N times before shuffling,
  giving the deterministic per-sample mask N independent draws — used
  with `--phoneme_keep_prob 0.25 --dataset_factor 4` to approach 100%
  annotated training samples without changing the per-sample
  distribution.
- `save_safetensors=False` is mandatory because Qwen2's tied
  embedding/lm_head trips safetensors' shared-storage check.

After training the only weights that changed are inside the
`phone_emb`, `composer`, and `tone_classifier` modules. The notebook's
`infer.py` patches them onto a clean CosyVoice via
`patch_cosyvoice2(cosyvoice)` then `load_state_dict(...,
strict=False)`.

---

## 6. SoulX-Podcast adaptation (this branch)

We are porting this technique to SoulX-Podcast. Four things are
different from upstream:

### 6.1 Input format — SSML, not square brackets

Upstream takes `你好[aa3]！` raw. We accept the W3C-style SSML subset
that matches Google Cloud TTS:

```xml
<speak>
  Hello <phoneme alphabet="ipa" ph="wɜːld">world</phoneme>!
  It is indeed a beautiful <phoneme alphabet="ipa" ph="wɜːld">world</phoneme>!
</speak>
```

Per the Google TTS spec we'll support three `alphabet` values:

| Alphabet | Language | Example `ph=` value | Per-syllable token shape |
|---|---|---|---|
| `cmu` / `x-arpabet` | English | `H EH L OW W ER L D` | flat list of ARPAbet phonemes |
| `jyutping` | Cantonese | `j yut4 p ing6` | `[initial, final+tone]` × N |
| `pinyin` | Mandarin | `p in1 y in1` | `[initial, final+tone]` × N |
| `ipa` | any (best effort) | — | resolved by a per-language IPA→native mapper |

User's exact spec, mirrored from the task message:

```
CMUdict: H EH L OW W ER L D       # per-phoneme tokens
Jyutping: j yut4 p ing6           # [initial, final+tone] per syllable
Pinyin:   p in1 y in1             # [initial, final+tone] per syllable

Chinese: [initial, final + tone]
```

**Implication for slot layout**: upstream's fixed 4-slot
`[onset, nucleus, coda, tone]` is replaced by a more general design.
See section 7.

### 6.2 Phoneme vocabulary — unified across three alphabets

Three disjoint id ranges, all sharing one `phone_emb`. Sizes are
**frozen and shipped** in `soulxpodcast/inpaint/_vocab.py` and
`soulxpodcast/inpaint/vocab/finals.json`:

```
id 0                = pad / no-phoneme
ids 1..40           = CMUdict ARPAbet + word boundary `|`  (40 tokens)
ids 41..60          = Jyutping initials (incl. zero-initial)  (20)
ids 61..317         = Jyutping finals-with-tone  (257, dataset-derived)
ids 318..341        = Pinyin initials (incl. zero-initial)  (24)
ids 342..551        = Pinyin finals-with-tone  (210, dataset-derived)
TOTAL_VOCAB_SIZE    = 552
```

The **alphabet itself is not** a separate component — the global offset
disambiguates which alphabet a token belongs to (exactly as upstream
disambiguates onset/nucleus/coda/tone).

The data-derived final inventories are produced from the SoulX corpus
(`tmp/dataset.jsonl`) via greedy longest-initial split — see
`references/soulx-adaptation.md` §2. Regenerate the JSON if the corpus
changes.

We do **not** sub-decompose Chinese finals into nucleus+coda+tone.
Per the user spec the final-with-tone (e.g. `ing6`, `in1`) is one
token. This is a deliberate simplification — it means the composer
can no longer do an `on+nu+co+to` decomposition. See 6.4.

**Dataset note** — the training corpus stores Chinese phonemes in
**whole-syllable** form (`wo3`, `nei1`, `aa3`), but the model trains on
**split form** (`p in1`, `n ei1`, `'' aa3`). Convert at dataset-load
time via `PhonemeTokenizer.encode_whole_syllable_sequence(alphabet,
sylls)`.

### 6.3 Slot layout — `K` slots per text token, K ≥ longest sequence

Variable phoneme count per annotated span is the structural change
from upstream. Three cases:

- **Pinyin / Jyutping**: each Chinese character = 1 text token = 2
  phoneme tokens (`initial`, `final+tone`). So K=2 fills a CJK
  character cleanly.
- **English CMUdict**: a word like "world" might tokenize as 1-3 Qwen
  subword tokens but supplies 4 ARPAbet phonemes (`W ER L D`).
  Phonemes need to be **distributed across the subword tokens of the
  word**, not crammed into the first one.
- **Mixed**: SSML may interleave English `<phoneme alphabet="cmu">`
  spans inside a Chinese sentence; the buffer must handle both in the
  same example.

We pick **K = 8** as the fixed per-token slot width — enough for
ARPAbet words of up to 8 phonemes mapping to a single subword, which
covers the long tail; longer words simply spread across more subwords.
The buffer is `phone_token: (B, K * L)` with the same interleaved
layout philosophy as upstream.

Distribution rule for English: align ARPAbet phonemes to the **first
subword token of the word** the `<phoneme>` element wraps. If the word
needs more than K phonemes, overflow spills into the next subword
token's slot block. This keeps the contract simple — one composed
embedding per text-token position, no positional resampling.

### 6.4 Composer — multi-set, not multi-component

Upstream's `concat_linear` assumes a fixed semantic role per slot
(`[onset, nucleus, coda]` concatenated, tone as residual). With our
variable layout (K slots, each holding a *phoneme of unknown role*),
that no longer fits.

Default for SoulX:

```python
composer(slots) = MLP(mean_pool(non-pad-slots))
```

i.e. mean-pool the non-pad slot embeddings then run a small MLP to land
back in `d_model`. The pad mask is the same `slot_id != 0` mask used
upstream, just at K granularity. We keep the option to swap in
`sum` for parity with upstream when running ablations.

The **tone auxiliary loss is replaced** by a **language auxiliary
loss**: a tiny classifier on the composed embedding predicting
`{cmu, jyutping, pinyin}` (3-way CE). The motivation is the same —
force a language-specific signal to live inside the composed embedding
rather than being recovered from neighbouring context. For Cantonese
spans we additionally predict tone-from-final (since each
final-with-tone uniquely determines tone), giving the same prosodic
benefit as upstream's tone-CE. See [`references/soulx-adaptation.md`](references/soulx-adaptation.md)
for the loss formulation.

### 6.5 LLM trunk — Qwen3-1.7B, not Qwen2-0.5B

`d_model = 2048` instead of 896. Embedding row count is ~160K (the
SoulX tokenizer includes speech tokens), but only the text-token
subset is in play for inpainting since we never overwrite speech-token
positions.

**Tie point** with the rest of the codebase: the embedding lookup
SoulX uses is `model.get_input_embeddings()` /
`model.model.embed_tokens` on the HF Qwen3 model the engine wraps. The
inpaint module must hook there. See section 8.

### 6.6 vLLM compatibility

The runtime currently runs vLLM V1 with ngram speculative decoding
([CLAUDE.md](../../../CLAUDE.md): "V1 + ngram speculative decoding —
production default"). vLLM does **not** accept `inputs_embeds` through
its standard API — it always tokenizes from ids. Two options:

- **Training path**: use HF `Qwen3ForCausalLM` directly (same as
  `train_lora_trunk.py`). No vLLM involvement.
- **Inference path**: either (a) fall back to the HF engine for
  inpaint requests (lose the perf), or (b) use the vLLM
  `prompt_embeds` extension if/when it lands. Option (a) is fine for a
  v1 ship — the inpaint API is a per-request opt-in.

See [`references/soulx-adaptation.md`](references/soulx-adaptation.md)
for the concrete wire-up plan.

---

## 7. Slot layout summary (the contract a future implementer must follow)

```
K = 8                          # max phonemes per text token slot block
phone_token:      LongTensor (B, K * L)
phone_token_mask: BoolTensor  (B, L)   — True where any of the K slots != 0
                                          (computed from phone_token by the model)

Per-text-token block layout (K consecutive ids):
  [slot_0, slot_1, ..., slot_{K-1}]

Per-slot id:
  0                          = pad / no phoneme
  [1 .. N_cmu]               = CMUdict ARPAbet
  [N_cmu+1 .. +N_jp]         = Jyutping {initial, final+tone}
  [+N_jp+1 .. +N_py]         = Pinyin   {initial, final+tone}

Composition: mean-pool slot embeddings → MLP → d_model
Replacement: text_emb at masked positions ← composed embedding
             text_emb elsewhere passes through
```

`K` is fixed at module init and must match the trained checkpoint.
Bumping it later requires re-training (or zero-padding old checkpoints
on the slot axis, since pad id is 0).

---

## 8. SoulX wire-up points (what code will need to change)

Reference: `.claude/skills/soulx-model/SKILL.md` for the canonical
SoulX layout. The inpaint adds:

```
soulxpodcast/
  inpaint/
    __init__.py
    tokenizer.py       # multilingual phoneme tokenizer (cmu/jyutping/pinyin)
    composer.py        # PhonemeComposer (phone_emb + composer MLP + aux classifier)
    ssml.py            # parse <speak>/<phoneme> into (text, spans)
    frontend.py        # InpaintFrontend — builds text_token + phone_token
    inject.py          # embedding-replacement hook (wraps Qwen3 embed_tokens)
  training/
    train_inpaint.py   # frozen-trunk training loop (mirrors train_lora_trunk.py)
    inpaint_dataset.py # CSV/HF dataset producing text + speech_tokens + phoneme labels
```

**Inference inject point**: `engine/llm_engine.py:HFLLMEngine` already
runs HF generate; the inpaint engine adds a forward pre-hook on
`model.get_input_embeddings()` that, for the current request's
text-token positions, replaces the row output of `embed_tokens` with
the composed phoneme embedding. The composer module is stored on the
engine and consults the per-request `phone_token` buffer.

**Training inject point**: explicit — call `embed_tokens(input_ids)`,
call `composer(phone_token, phone_token_len)`, blend, then call
`Qwen3Model(inputs_embeds=...)`. Identical to upstream's
`Qwen2LMInpaint.forward`.

---

## 9. Open questions / TBD before training

1. **CMUdict vocab finalisation** — include stress variants
   (`AH0 / AH1 / AH2`) as separate ids, or strip stress? The user's
   example `H EH L OW W ER L D` shows no stress digits. We will strip
   stress for v1 (simpler vocab, lower risk of train/eval token
   mismatch) and add stress later if quality suffers.
2. **IPA support** — the user mentioned `alphabet="ipa"` as the SSML
   surface form. We'll add a per-language IPA→native mapper
   (ipa→arpabet for English, ipa→jyutping for Cantonese,
   ipa→pinyin for Mandarin) on the frontend so the model only ever
   sees the three native alphabets. Build the mappers from existing
   tables (`pycantonese`, `pypinyin`, `g2p_en` already cover most of
   this). Out of scope for the first training run.
3. **Word-level vs char-level alignment for English** — keep "first
   subword token gets the full ARPAbet sequence, overflow spills into
   next" rule, or do equal-width spread? Equal-width is more
   linguistically defensible but harder to implement. Start with
   first-subword overflow and revisit if quality regresses on long
   English words.
4. **Speaker conditioning** — SoulX is multi-speaker dialogue. Inpaint
   span audio inherits the active speaker's prosody from context. Verify
   this empirically on a held-out dialogue with the same word inpainted
   across two different speakers.
5. **Bi-stream / chunked streaming interaction** — phoneme injection is
   a pre-LLM op, so streaming chunk size and the dual-CUDA-stream B3
   path should be untouched. Smoke-test once wired.

---

## 10. References

- Upstream README (verbatim):
  `/home/joseph/projects/notebooks/notebooks/projects/CosyVoice-Inpaint/README.md`
- Upstream model:
  `/home/joseph/projects/notebooks/notebooks/projects/CosyVoice-Inpaint/pron_inpaint/modeling.py`
  → `Qwen2LMInpaint`
- Upstream tokenizer:
  `/home/joseph/projects/notebooks/notebooks/projects/CosyVoice-Inpaint/pron_inpaint/tokenizer.py`
  → `JyutpingTokenizer`, `convert_phone_str_to_flat_ids`
- Upstream frontend:
  `/home/joseph/projects/notebooks/notebooks/projects/CosyVoice-Inpaint/pron_inpaint/frontend_wrapper.py`
  → `InpaintFrontendWrapper`
- Upstream training:
  `/home/joseph/projects/notebooks/notebooks/projects/CosyVoice-Inpaint/train.py`
- Upstream runtime patch:
  `/home/joseph/projects/notebooks/notebooks/projects/CosyVoice-Inpaint/pron_inpaint/patch.py`
- SoulX layout (must coexist with this skill):
  `.claude/skills/soulx-model/SKILL.md`
- SSML spec we mirror:
  https://docs.cloud.google.com/text-to-speech/docs/ssml#phoneme
