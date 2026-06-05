# SoulX pronunciation-inpaint v11 spec

Single-source spec for the next composer iteration. Captures the upstream
audit, per-alphabet design decisions, training data shape, training loss,
and the acceptance criteria the test gate will enforce.

Intent: no PJM is written until this doc and the matching implementation
agree across all eight axes below.


## 1. Upstream audit (what we keep, what we don't)

CosyVoice-Inpaint upstream is Cantonese-only and uses Jyutping. It is
load-bearing as the reference for principles, not as a drop-in copy. Three
findings that transfer regardless of alphabet, and one that doesn't.

**Transfer-yes principles (alphabet-agnostic):**

1. **Pad-token substitution at training AND inference.**
   - Source rows carry `text` and `phone` fields. `tokenize_add_label`
     produces a `text_with_jyutping` string where some syllables are wrapped
     in `[ph]` brackets. The frontend then does
     `re.sub(r"\[.*?\]", pad_token, text)` before BPE tokenization.
   - Same recipe runs at inference: user writes `[ph]` brackets, frontend
     substitutes pads, composer fires at pad positions.
   - There is NO multi-BPE broadcast and NO train/inference alignment drift.
     The position layout the composer sees is identical in both regimes.

2. **Concat + linear preserves slot order via dedicated weight bands.**
   - Component embeddings (onset / nucleus / coda) are concatenated along
     the feature dim, then `Linear(3d → d) → GELU → Linear(d → d)`. Each
     component occupies a fixed band of input channels, so the first Linear
     learns role-specific transforms by construction.
   - Mean-pool is NOT used. Upstream considered both modes (`concat_linear`
     and `sum`); `concat_linear` is the default and ships in their code.

3. **Tone is a residual ADD, not concatenated.**
   - `composed = mlp(concat(onset, nucleus, coda)) + tone_emb`.
   - Forces the LLM to read tone from a known channel direction
     (the tone-emb row) rather than from interactions among other slots.
   - Backed by a small auxiliary `tone_classifier(composed + 0 * text_emb.detach())`
     loss with weight 0.3 — pushes tone information to live inside the
     composed embedding rather than leak from the surrounding text context.

**Transfer-no principle (alphabet-specific):**

4. **K=4 fixed-role layout (onset, nucleus, coda, tone) is Cantonese-shaped.**
   - Pinyin doesn't separate nucleus from coda (`in`, `ang`, `ong` are
     single finals).
   - English has variable-length syllables with no canonical
     onset/nucleus/coda mapping; clusters can have up to 3 onset phones
     (e.g. `STR` in `STRENGTHS`) and up to 4 coda phones.
   - We need per-alphabet slot layouts, not one K=4 fits all.


## 2. Per-alphabet decisions

The SoulX phone vocab ([`_vocab.py`](../soulxpodcast/inpaint/_vocab.py))
already bakes tone into Chinese finals (e.g. `in2`, `ang3` are single
ids — separate phone_emb rows per (final, tone) combination), and
already position-tags CMU consonants as `_on` / `_co`. We use the vocab
as-is for v11 — re-deriving it to match upstream's (onset, nucleus,
coda, tone) decomposition is a v12 lever, not v11.

| alphabet | K | slot layout | composer head | tone? |
|---|---|---|---|---|
| jyutping | 2 | `[initial, final-with-tone]` | `Linear(2d→d) → GELU → Linear(d→d)` | no — baked into final ids |
| pinyin | 2 | `[initial, final-with-tone]` | `Linear(2d→d) → GELU → Linear(d→d)` | no — baked into final ids |
| cmu | 6 | `[phone₀, …, phone₅]` (syllable-internal order, pad with 0) | `Linear(6d→d) → GELU → Linear(d→d)` | n/a — consonants already position-tagged in vocab |

**Why K=2 for Chinese:** the vocab stores finals like `in2` and `ang3`
as single ids (see `_load_finals()` in `_vocab.py`). The composer's
input for `银 (yin2)` is `[y_initial_emb, in2_final_emb]` →
`Linear(2d→d)`. The frozen LLM was trained on full-precision text
embeddings of 银; the composer's job is to land in the same neighborhood
of embedding space.

**Why no tone residual:** upstream needed it because their final vocab
was nucleus+coda only (12+9=21 components), so tone had to be added in.
Our final-with-tone vocab is 257 rows for jyutping and 210 for pinyin —
each (final-shape, tone) is its own row. The phone_emb table already
learns tone-specific embeddings.

**Why CMU K=6 not K=8:** empirically 99%+ of English syllables fit in 6
phones. K=8 (current v10) wastes capacity. Internal slot order =
syllable-internal phoneme order from the SoulX phone vocab; the vocab
already tags consonants as `_on` / `_co` so slot role is encoded in the
ID before the composer sees it.

**Shared Chinese head:** jyutping and pinyin both have K=2 with the same
shape and a shared phone_emb table (their initials and finals live in
disjoint id ranges in the global vocab, so the same Linear(2d→d) can
learn alphabet-specific patterns through which embedding rows the
indices point at). v11 ships TWO heads: one for K=2 Chinese, one for
K=6 CMU.

**Cross-alphabet contract:** both heads emit a single `d_model` vector
per pad position. The output dimensionality is uniform so the inject
path (`apply_phoneme_inpaint`) is alphabet-agnostic.


## 3. Source data format

Training rows: `{"text", "phonemes", "speech_tokens", "lang", "spk"}`.
Same shape as current v10 dataset rows.

- `text`: source string in the SoulX corpus normalization for that lang.
  - yue: with full-width punctuation, e.g. `"我同佢一齊去飲茶。"`
  - zh: no punctuation, e.g. `"我去银行办事"`
  - en: all-caps no punctuation, e.g. `"HELLO WORLD TODAY"`
- `phonemes`: list[str]. One whole-syllable token per CJK char (zh/yue) or
  per ARPAbet phoneme with `"|"` word boundary (en). Examples:
  - yue: `["ngo5","tung4","keoi5","jat1","cai4","heoi3","jam2","caa4","。"]`
  - zh: `["wo3","qu4","yin2","hang2","ban4","shi4"]`
  - en: `["HH","AH","L","OW","|","W","ER","L","D","|","T","AH","D","EY"]`
- `speech_tokens`: list[int]. s3tokenizer 25 Hz speech tokens for the
  source audio.
- `lang`: one of `"yue"`, `"zh"`, `"en"`.

**Filtered dataset:** v10's `scripts/inpaint/filter_dataset_silence.py`
strips silence-heavy rows up-front. v11 keeps this filter; the dataset
fed to training is the filtered output, not raw.


## 4. Training-time alignment

For each row, build `text_with_marks` by walking source text and inserting
markers around the syllables to be exposed. Then BPE-tokenize and replace
markers with pad tokens.

**Marker insertion rule (per alphabet):**

- **yue / zh** (per-CJK-char): for each CJK character in `text`, with
  probability `phoneme_keep_prob` (default 0.25), wrap the character with
  `[ph]` brackets where `ph` is the syllable for that char (from the
  aligned `phonemes` list). Non-CJK chars (punctuation, ASCII) are left
  untouched and consume no phoneme.

- **en** (per-syllable inside per-word): split `phonemes` on `"|"` into
  per-word lists. For each word in `text`, syllabify the per-word ARPAbet
  via `encode_arpabet_per_syllable`. With probability `phoneme_keep_prob`
  per word, insert ONE `[ph]` bracket per syllable adjacent to the word
  (e.g. `BANANA` → `BANANA[B AH][N AH][N AH]`). The position of the
  brackets relative to the word is semantically a marker for "this
  syllable replaces the next BPE pad slot"; the actual BPE re-tokenization
  collapses bracket-removed text back to its natural form.

**BPE tokenization rule:**

After building `text_with_marks`, run the LLM tokenizer with
`re.sub(r"\[.*?\]", pad_token, text_with_marks)`. The pad token id used is
`tokenizer.pad_token_id` (Qwen3 = 151643). Each `[ph]` block becomes
exactly one pad position.

This produces:
- `input_ids: LongTensor (L,)` with pad ids at syllable positions
- For each pad position, a known associated `[ph]` whose phone-ids we
  write into the slot block at that position

**Slot writing rule:**

For each pad position `i` and its associated phoneme `ph`:
- yue: parse `ph` (jyutping) into (onset, nucleus, coda, tone) via our
  PhonemeTokenizer; write 4 ids into `phone_token[i*K : (i+1)*K]` per the
  jyutping slot layout.
- zh: parse `ph` (pinyin) into (initial, final-w-tone-stripped, tone);
  write 3 ids into the pinyin slot layout, slot 3 stays 0.
- en: each `[ph]` corresponds to one syllable's ARPAbet sequence; write up
  to 6 phone ids into the cmu slot layout, remaining slots stay 0.

`phone_mask` is True at every pad position.

**Determinism:** seed the keep_prob RNG with row index (matches v10
`deterministic_dropout=True` for eval rows; train rows still use
per-process randomness).


## 5. Inference-time alignment

Input: SSML string OR plain text. Plain text disables composer.

SSML form (unchanged from v10):
```
<phoneme alphabet="jyutping" ph="k eoi5">佢</phoneme>
<phoneme alphabet="pinyin" ph="y in2">银</phoneme><phoneme alphabet="pinyin" ph="h ang2">行</phoneme>
HELLO <phoneme alphabet="cmu" ph="W ER L D">WORLD</phoneme> TODAY
```

**SSML contract changes from v10:**

1. **One `<phoneme>` per syllable.** Multi-syllable Chinese words must be
   wrapped per character: `<phoneme ph="y in2">银</phoneme><phoneme ph="h ang2">行</phoneme>`,
   not `<phoneme ph="y in2 h ang2">银行</phoneme>`. Multi-syllable English
   words may have multi-syllable phonemes in a single span; the engine
   splits them internally.
2. **Reject ambiguous spans.** A `<phoneme ph="y in2 h ang2">银行</phoneme>`
   wrap raises `ValueError` at inference parse time. (v10 would have
   silently packed both syllables into one BPE — that was the source of
   the mean-pool collapse.)

Build the LLM input by walking the SSML:
- Text segments outside `<phoneme>` spans: BPE-tokenize normally.
- Each `<phoneme>` span: emit ONE pad token per syllable. Slot block
  follows the per-alphabet rule from §4.

This mirrors training §4 byte-for-byte. The parity test asserts this.


## 6. Composer architecture

```python
class PhonemeComposer(nn.Module):
    # K=2 for Chinese (jyutping / pinyin), K=6 for CMU.
    # Storage K = max = 6; Chinese rows leave slots 2..5 as pad id 0.
    K_PER_ALPHABET = {"jyutping": 2, "pinyin": 2, "cmu": 6}
    SLOTS_STORAGE = 6

    def __init__(self, d_model):
        self.phone_emb = nn.Embedding(TOTAL_VOCAB_SIZE, d_model, padding_idx=0)
        self.head_cn  = nn.Sequential(nn.Linear(2*d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.head_cmu = nn.Sequential(nn.Linear(6*d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
```

Forward, given `phone_token: (B, SLOTS_STORAGE*L)`, `alphabet: (B,)`
row-level alphabet id (`0`=jyutping, `1`=pinyin, `2`=cmu):

1. Look up `phone_emb` on each slot → `(B, L, SLOTS_STORAGE, d)`.
2. Route per-row to its head:
   - Chinese rows (alphabet 0 or 1): concat slots `[0,1]` → `head_cn`.
     Slots `[2..5]` are not read.
   - CMU rows (alphabet 2): concat slots `[0..5]` → `head_cmu`.
3. Output `composed: (B, L, d_model)` zeroed at positions with no phoneme,
   plus `position_mask: (B, L)`.

**Why per-alphabet heads:** the two head shapes encode different role
layouts. A single `Linear(6d, d)` with zero-padding for Chinese would
waste 4×d_model channels per Chinese position and still need to learn
two role conventions through the LLM context — slower and noisier.

**Parameter cost vs v10:** v10 = 9.6M. v11 estimate at d_model=2048:
phone_emb (576 × 2048) ≈ 1.2M, head_cn ((2d → d → d): 8M + 4M = 12M),
head_cmu ((6d → d → d): 24M + 4M = 28M). Total ≈ 41M — ~4× larger than
v10. Justified if it fixes the alignment bugs that v10 couldn't; if not,
the 6d→d projection on head_cmu is the obvious place to cut (down-project
to 2d first, then to d).

**Init:** `init_from_text_embed` heuristic still applies — bias
phone_emb rows toward the average text-embedding row so the frozen
backbone sees a sane vector at step 0.


## 7. Loss

```
loss = lm_loss
```

Plain speech-token CE on positions where `speech_mask=True`. No tone
auxiliary loss in v11 — tone is encoded in the final-with-tone phone_emb
rows so there's no separate tone embedding to push gradient through.
No silence masking — handled by dataset filtering up-front. No label
smoothing.

If v11's tone discrimination is empirically weak (e.g. zh fixtures
ship audio with the wrong tone), v12 can:
1. Re-derive the vocab to split finals into nucleus / coda / tone.
2. Add the tone-residual + aux-classifier from upstream (spec §6 had
   this; deferred).


## 8. What's trained vs frozen

**Trained:**
- `phone_emb` (shared across alphabets)
- Two per-alphabet composer heads (`head_cn`, `head_cmu`)

**Frozen:** Qwen3 LLM backbone (every parameter under `model.*`).

Apply via `for p in model.parameters(): p.requires_grad = False`; keep
`model.train()` (per [[training-eval-mode-disables-grad-ckpt]]). Gradient
checkpointing enabled via `model.gradient_checkpointing_enable()`.


## 9. Acceptance gate (test must pass before any v11 PJM is run)

[scripts/inpaint/fixtures_adversarial.py](../scripts/inpaint/fixtures_adversarial.py)
+ [test_alignment_parity.py](../scripts/inpaint/test_alignment_parity.py)
+ [test_alignment_behavioral.py](../scripts/inpaint/test_alignment_behavioral.py)
exist as of commit `57021e7`. v10 baseline = parity 12/12, behavioral 26/38.

**v11 must produce, on the same fixtures:**

- Parity: 12/12 (will require fixture updates: the training-side
  alignment now uses padsub too, so fixture `train_phonemes_*` fields
  need to become source-text + `[ph]` markers in the v11 dataset shape).
- Behavioral: ≥ 30/38, with strictly no NEW fail compared to v10's set
  `{yue_dei6_byte_fallback, zh_yinhang_merged, zh_zhongguo_merged,
  zh_wo_singleton_baseline}`. Specifically:
  - `yue_keoi5_byte_fallback`, `yue_dei6_byte_fallback` must PASS the
    M1 + D1 + T1 + E2 asserts. (v10 had the broadcast stutter problem;
    padsub fixes it.)
  - `zh_yinhang_merged`, `zh_zhongguo_merged` must PASS E2 + M1 + D1.
    v10 destabilised the LLM into 200-token loops with wrong overrides;
    v11 should not, because training data exposed the LLM to pad-
    substituted multi-syllable sequences.
  - `zh_wo_singleton_baseline` — the baseline-no-EOS FAIL there is an
    LLM-side OOD issue unrelated to the composer. Can stay failed in v11
    (document it as known LLM-side, not composer).

**Coverage gap accepted:** the gate cannot distinguish "composer responds
in correct direction" from "composer responds in wrong direction with the
SAME magnitude". A v11 that emits `"yin ku"` instead of `"yin xing"` would
still pass all current asserts. Address via a separate ASR-back check
(future work — not blocking v11).


## 10. Implementation checklist

Before the first training step is run, all of these must be true.
Numbered to be checkable on a PR diff.

1. [ ] `PhonemeComposer` rewritten to two heads (`head_cn` for K=2
   Chinese, `head_cmu` for K=6 English). Old mean-pool path gone.
   `slots_per_token` is now 6 (storage) but per-alphabet K is 2 or 6.
   `forward()` takes a per-row `alphabet_id` tensor.
2. [ ] `InpaintDataset.__getitem__` switched to padsub: build
   `text_with_marks`, run `re.sub` for pad substitution, write slot
   blocks only at pad positions. `_align_chinese` and `_align_english`
   are replaced (not modified) by `_align_padsub_chinese` and
   `_align_padsub_english`.
3. [ ] `InpaintInferenceEngine._build_text_and_phone_tokens` rewritten
   to match §5 exactly. The `inference_audio_padsub.py` prototype is
   the starting point; it gets folded back into the engine.
4. [ ] Fixtures updated: each fixture's `train_phonemes_*` field is
   replaced with `train_text_with_marks` (the source-text
   representation that the new dataset alignment consumes).
5. [ ] `test_alignment_parity` updated to compare new training-side
   slot tensors vs new inference-side slot tensors. The byte-equality
   contract is unchanged.
6. [ ] Adversarial fixtures' `expected_inpaint_pads` field correctly
   reflects per-alphabet syllable counts (already does for v10;
   recheck under the new alignment).
7. [ ] PJM file `scripts/inpaint/train_h100_v11_padsub.pjm` written;
   passes a 1000-step smoke run on the 3090 (loss decreases, no NaN,
   gate asserts converge in the right direction).


## 11. Out of scope for v11

These are real issues but explicitly not addressed by this iteration.
Adding any of them mid-training kills the iteration; track separately.

- **English LLM baseline doesn't EOS on short prompts** (`HELLO WORLD
  TODAY`, `BANANA`). LLM-side OOD per
  [[inference-text-must-match-training-format]]. v11 composer can't fix
  this; if anything, it works around it (composer fires emit EOS
  cleaner than the bare LLM). Behavioral test fixtures keep
  `enable_behavioral=False` for en.
- **ASR-back syllable check.** The gate can't tell `yin xing` from
  `yin ku`. Needs an ASR or forced-alignment pass. Future iteration.
- **Multi-alphabet rows.** Spec assumes one alphabet per row at training.
  Inference SSML can mix alphabets per-span, but training data won't
  exercise it. Defer until a real use case appears.
- **Tone classification accuracy as an eval metric.** The aux tone_loss is
  for gradient shaping, not a ship signal. Track it in the train log but
  don't gate v11 release on its value.


## 12. Memories that should be re-read before starting v11

- [[inpaint-retrain-needs-test-gate]] — the gate IS the ship signal, not
  training loss.
- [[frozen-head-loss-shape-first]] — when the small head misbehaves,
  audit the loss + alignment shape before chasing architecture changes.
- [[aishell3-studio-silence-padding]] — keep the filter step; don't try
  to handle silence in the loss.
- [[inference-text-must-match-training-format]] — match the SoulX corpus
  conventions exactly for adversarial fixtures; mismatched text format is
  the most common source of false-negative behavioral failures.
- [[home-3090-power-constraint]] — local smoke tests cap at <2-3 h; the
  full v11 retrain belongs on H100.
- [[early-stop-needs-multi-eval-trend]] — don't kill the v11 H100 run on
  one noisy eval bump.
- [[training-eval-mode-disables-grad-ckpt]] — freeze via
  `requires_grad=False` not `model.eval()`.
