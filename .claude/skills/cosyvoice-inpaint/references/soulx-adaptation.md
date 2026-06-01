# SoulX-Podcast adaptation — design and wire-up

This document is the concrete plan for porting CosyVoice-Inpaint into
SoulX-Podcast on the `experiment/pronunciation-inpaint` branch. The
parent skill page covers the *why*; this page covers the *what / where*.

## 1. SSML surface format

Accepted SSML subset (matches https://docs.cloud.google.com/text-to-speech/docs/ssml#phoneme):

```xml
<speak>
  Hello <phoneme alphabet="cmu" ph="W ER L D">world</phoneme>!
  上 <phoneme alphabet="jyutping" ph="t ong4">堂</phoneme> 終於講到分數。
  请 <phoneme alphabet="pinyin" ph="zh u3 yi4">注意</phoneme>。
</speak>
```

Parser contract (`soulxpodcast/inpaint/ssml.py:parse_ssml`):
- Input: SSML string (root `<speak>` optional but tolerated).
- Output: `plain_text` (`<phoneme>` element replaced by its surface
  word) plus `spans: list[PhonemeSpan]` where:
  ```python
  @dataclass
  class PhonemeSpan:
      char_start: int      # offset into plain_text
      char_end:   int      # exclusive
      alphabet:   Literal['cmu', 'jyutping', 'pinyin']
      ph_tokens:  list[str]  # whitespace-split tokens from ph=
  ```
- Errors are raised (not silently ignored) — unknown `alphabet`,
  malformed XML, empty `ph`, span overlap. The CLI surfaces them.

`alphabet="ipa"` is rejected by v1 (no IPA→native mapper yet);
documented in the parent skill section 9.

## 2. Phoneme tokenizer (shipped)

`soulxpodcast/inpaint/tokenizer.py:PhonemeTokenizer` is implemented.
Sizes derived from the SoulX corpus at
`tmp/dataset.jsonl` (358 611 rows; lang split: yue 73 % / en 24 % /
zh 3 %):

| component | count | id range |
|---|---|---|
| pad | 1 | 0 |
| CMUdict ARPAbet + word boundary `\|` | 40 | 1..40 |
| Jyutping initials (incl. zero-initial `""`) | 20 | 41..60 |
| Jyutping finals-with-tone | 257 | 61..317 |
| Pinyin initials (incl. zero-initial `""`) | 24 | 318..341 |
| Pinyin finals-with-tone | 210 | 342..551 |
| **TOTAL_VOCAB_SIZE** | **552** | |

Initials are hard-coded in `soulxpodcast/inpaint/_vocab.py`; finals
are frozen JSON at `soulxpodcast/inpaint/vocab/finals.json` and
derived from the dataset's whole-syllable inventory via greedy
longest-initial split. Regenerate that JSON if the corpus changes.

Public API:
```python
PhonemeTokenizer.encode_span(alphabet, ph_tokens) -> list[int]
PhonemeTokenizer.vocab_size: int       # 552
PhonemeTokenizer.alphabet_of(id) -> {'pad', 'cmu', 'jyutping', 'pinyin'}
PhonemeTokenizer.split_whole_syllable(alphabet, syl) -> (initial, final)
PhonemeTokenizer.encode_whole_syllable_sequence(alphabet, sylls) -> list[int]
```

The last two helpers are for **offline data prep**: the SoulX dataset
stores Chinese phonemes as whole syllables (`wo3`, `nei1`, `aa3`) but
the model is trained on split-form ids
(`p in1` / `n ei1` / `'' aa3`). Use them to convert
`row['phonemes']` → split-form ids during dataset preprocessing.

### CMUdict notation note

The dataset uses CMUdict 0.7b ARPAbet *without* stress digits. The
"h" phoneme is `HH` (double-H), not `H`. The user-facing example in
the original task prompt used `H` informally; the actual token is
`HH`. Stress digits (`AH0` / `AH1` / `AH2`) are stripped — only the
base phoneme symbol carries an id.

## 3. Frontend

`soulxpodcast/inpaint/frontend.py:InpaintFrontend`:

```python
class InpaintFrontend:
    def __init__(self, qwen_tokenizer, phone_tokenizer, slots_per_token: int = 8):
        self.tok = qwen_tokenizer
        self.pt  = phone_tokenizer
        self.K   = slots_per_token

    def build(self, ssml: str, device) -> InpaintInputs:
        plain, spans = parse_ssml(ssml)
        # 1. tokenize plain text first to get char→subword-index alignment.
        enc = self.tok(plain, return_offsets_mapping=True, add_special_tokens=False)
        text_ids = enc['input_ids']               # (T,)
        offsets  = enc['offset_mapping']          # list[(start, end)]
        # 2. allocate the slot buffer.
        phone_token = torch.zeros(self.K * len(text_ids), dtype=torch.long)
        # 3. for each span, find subword tokens whose offset overlaps the
        #    span's char range, then place phoneme ids into the first
        #    overlapping subword's K slots; overflow into next.
        for span in spans:
            ph_ids = self.pt.encode_span(span.alphabet, span.ph_tokens)
            self._place(phone_token, offsets, span, ph_ids)
        return InpaintInputs(
            text_token  = torch.tensor([text_ids]),
            phone_token = phone_token.unsqueeze(0),
            text_len    = torch.tensor([len(text_ids)]),
        )
```

Placement rule (`_place`): walk the subword tokens whose
`offset_mapping` overlaps `[char_start, char_end)`. Write
`ph_ids[0:K]` into the first such subword's slot block; if
`len(ph_ids) > K`, the remainder goes into the next subword's block,
and so on. Overflow past the end of the span's subwords is an error
(raised loudly — same philosophy as upstream's hard assert).

## 4. Composer module

`soulxpodcast/inpaint/composer.py:PhonemeComposer`:

```python
class PhonemeComposer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, slots_per_token: int,
                 num_alphabets: int = 3, n_tone_finals: int = 0):
        super().__init__()
        self.K = slots_per_token
        self.phone_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.composer  = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # auxiliary heads (training-only)
        self.alphabet_head = nn.Linear(d_model, num_alphabets)  # cmu/jyutping/pinyin

    def forward(self, phone_token: torch.LongTensor) -> tuple[Tensor, BoolTensor]:
        # phone_token: (B, K*L)  →  per-token mean-pool of non-pad slots → MLP
        B, KL = phone_token.shape
        L = KL // self.K
        ids = phone_token.view(B, L, self.K)              # (B, L, K)
        emb = self.phone_emb(ids)                         # (B, L, K, d)
        mask = (ids != 0)                                  # (B, L, K)
        # mean-pool; positions with no slots stay zero (will be masked out)
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=-2) / denom    # (B, L, d)
        composed = self.composer(pooled)
        any_slot = mask.any(dim=-1)                       # (B, L)
        return composed, any_slot
```

**Initialization** mirrors upstream
`init_component_from_text_embed`: `phone_emb.weight` ← `N(0, 0.02)` then
shifted by the mean of `qwen_model.get_input_embeddings().weight`. Row
0 zeroed.

**Auxiliary loss** during training (in the trainer, not the module):
- `alphabet_logits = composer.alphabet_head(composed)` → 3-way CE
  where the label per masked position is the alphabet of the first
  non-pad slot.
- Loss only on positions where `any_slot` is True.
- Weight starts at `0.3` (mirrors upstream `tone_weight`).

## 5. Embedding inject

`soulxpodcast/inpaint/inject.py:apply_phoneme_inpaint`:

```python
def apply_phoneme_inpaint(
    text_emb: Tensor,         # (B, L, d) — from qwen.embed_tokens(text_ids)
    composed: Tensor,         # (B, L, d) — from PhonemeComposer
    mask: BoolTensor,         # (B, L)    — True where composed should replace text
) -> Tensor:
    keep = (~mask).unsqueeze(-1).to(text_emb.dtype)
    take = mask.unsqueeze(-1).to(text_emb.dtype)
    return text_emb * keep + composed * take
```

Identical to upstream — zero text where we're replacing, add the
composed embedding there.

## 6. Training entrypoint

`soulxpodcast/training/train_inpaint.py` — mirrors
`train_lora_trunk.py` structure for stylistic consistency:

```python
def main():
    args = parse_args()
    # 1. build dataset (CSV or HF disk)
    ds = InpaintDataset(args.dataset_path, args.phoneme_keep_prob,
                        args.phoneme_seed, args.dataset_factor)
    # 2. build model
    qwen = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 torch_dtype=torch.bfloat16)
    qwen.requires_grad_(False)                # freeze
    composer = PhonemeComposer(
        vocab_size=phone_tokenizer.vocab_size,
        d_model=qwen.config.hidden_size,
        slots_per_token=8,
    )
    composer.init_from_text_embed(qwen.get_input_embeddings())
    # 3. optimiser + scheduler on composer params only
    opt = torch.optim.AdamW(composer.parameters(), lr=args.lr, ...)
    # 4. loop
    for batch in loader:
        text_emb = qwen.get_input_embeddings()(batch['text_token'])  # (B,L,d)
        composed, mask = composer(batch['phone_token'])
        inputs_embeds = apply_phoneme_inpaint(text_emb, composed, mask)
        out = qwen(inputs_embeds=inputs_embeds,
                   attention_mask=batch['attention_mask'])
        lm_loss  = ce_loss_on_speech_tokens(out.logits, batch['speech_token'])
        aux_loss = alphabet_ce(composer.alphabet_head(composed),
                                batch['alphabet_label'], mask)
        loss = lm_loss + args.aux_weight * aux_loss
        loss.backward(); opt.step(); opt.zero_grad()
```

Frozen backbone means optimiser state is small (composer only) — should
fit comfortably on a single 24 GB card even at bf16. Speech-token CE
mirrors `train_lora_trunk.py`'s `--include_text_loss=False` default.

## 7. Dataset format

CSV columns expected (analogous to upstream):

| column | type | notes |
|---|---|---|
| `text` | str | plain text, no SSML |
| `speech_tokens` | str (space-separated ints) | s3tokenizer output |
| `alphabet` | str | `cmu` / `jyutping` / `pinyin` per sample |
| `phone` | str | space-separated phoneme tokens aligned to `text` |
| `lang` | str (optional) | for stratified sampling |

`alphabet` is per-sample because each training example shows the model
**one** alphabet at a time. The mask logic (`tokenize_add_label`'s
SoulX twin) decides which words/syllables to reveal as inline
phoneme annotations during training.

## 8. Inference path

**Training-only-style** v1: build a `InpaintEngine` that wraps the HF
Qwen3 forward (`Qwen3ForCausalLM(inputs_embeds=...)`). vLLM is
*disabled* for inpaint requests in v1 — the API endpoint accepts an
`ssml` field; presence of an inpaint span forces the HF path.

This is consistent with the existing MTP code path which also forces
HF over vLLM (see CLAUDE.md "Phase 0 inference optimization" table).
Cost: long-content RTF goes from 0.270 (V1+ngram on
AWQ+MeanFlow+chunked-prefill) back to roughly 0.90 (HF baseline). For
inpaint, accept the regression — it's a per-request opt-in feature.

A future v2 can investigate vLLM's `prompt_embeds` extension if/when
the SoulX-AILab vLLM fork supports it.

## 9. Files

Already on this branch:

```
soulxpodcast/inpaint/
  __init__.py            # ✅ public re-exports
  _vocab.py              # ✅ frozen vocab + global-offset layout
  vocab/finals.json      # ✅ data-derived jyutping + pinyin finals
  ssml.py                # ✅ parse_ssml, PhonemeSpan, SSMLParseError
  tokenizer.py           # ✅ PhonemeTokenizer (split form + whole-syll split helper)
  composer.py            # ✅ PhonemeComposer, apply_phoneme_inpaint,
                         #     alphabet_labels (aux supervision)
scripts/inpaint/
  smoke_modules.py       # ✅ end-to-end smoke test for the three modules
```

Still to create (next sub-tasks on this branch):

```
soulxpodcast/inpaint/
  frontend.py            # InpaintFrontend — wraps Qwen3 tokenizer; builds
                         #     text_token + phone_token from SSML or
                         #     bracket-annotated text
soulxpodcast/training/
  inpaint_dataset.py     # HF/JSONL dataset; reads tmp/dataset.jsonl,
                         #     splits whole-syllable phonemes to split form,
                         #     emits text_token + phone_token + speech_token
  train_inpaint.py       # training entrypoint, mirrors train_lora_trunk.py
scripts/inpaint/
  convert_dataset.py     # one-shot: tmp/dataset.jsonl → split-form dataset
  smoke_training_step.py # single-step bwd over a real batch
tests/inpaint/
  test_ssml.py
  test_tokenizer.py
  test_inject.py
```

Skill page `cosyvoice-inpaint` covers the conceptual contract; this
references page covers the concrete file layout. Keep both in sync.
