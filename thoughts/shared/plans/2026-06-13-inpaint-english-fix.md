# Pronunciation-inpaint: English is not working — diagnosis & fix plan

Date: 2026-06-13
Status: Phase A DONE (commit 6fe5b7a). Phase B DONE — base LLM terminates on
English under deployment decoding → retrain greenlit (`v12_en_subst_fix` PJM
ready). zh/yue inpaint already GO (v12).

## Progress log

- **Phase A — DONE (commit 6fe5b7a).** `_substitute_units` whitespace-tolerant
  (`free_chars`); en fallback 96.44% → **0.00%** on 7191 real words; BANANA →
  `['I',' LIKE','PAD','PAD','PAD']`; parity **12/12** (en enabled). Plus honest
  `n_kept` (finding 3) and `slots_per_token` 8→6 (finding 5).
- **Phase B — DONE.** Base SoulX LLM termination on English, existing v12
  checkpoint, `disable_inpaint=True`:
  | decoding | result |
  |---|---|
  | greedy + rep_penalty 1.1 | 5/6 terminate (one 15-word loops) |
  | **sampling + rep_penalty 1.1 (deployment)** | **18/18 (6 prompts × 3 seeds), incl. toy BANANA** |
  | greedy + rep_penalty 1.0 (the gate's A/B mode) | loops on en — this is why the gate "saw" en looping |
  Conclusion: the base model terminates on English under normal decoding; the
  gate's en-looping is a harness artifact (greedy, no rep-penalty), NOT a
  base-model blocker. **Retrain is worthwhile.**
- **Phase 1 (retrain) — READY.** `scripts/inpaint/train_h100_v12_en_subst_fix.pjm`
  (not v12.1 — aux loss off). Submit on the H100 cluster.
Related: `docs/inpaint_v12_spec.md`, memories `inpaint-v12-grapheme-subst`,
`inpaint-v11-fail-pattern`, `inference-text-must-match-training-format`.

## TL;DR

v12 grapheme substitution made zh/yue inpaint work for the first time
(behavioral 33/38; `yue_dei6`/`zh_yinhang` correct↔wrong edit flipped
0.000 → 0.82/0.94; jyutping composed directions de-collapsed cos 0.99 → 0.16–0.41).
**English got none of this.** Two independent problems, neither of which is data
volume (en has 85,270 training rows — more than zh's 5,622):

1. **Mechanism**: the substitution helper silently falls back to v11-style
   *insertion* for English, so the grapheme is never removed and the composer
   stays non-load-bearing (the original v11 collapse persists for en).
2. **Base model**: the SoulX LLM does not terminate on the English test prompts
   — the no-inpaint baseline runs to the 200-token cap with no EOS. This is a
   base-model / OOD-prompt issue, not the composer's.

Both must be addressed, in order: fix (1) so English *can* be steered, then
re-test (2) with realistic prompts before drawing conclusions about en quality.

## Evidence

### Problem 1 — substitution silently degrades to insertion for English

`_substitute_units` (`soulxpodcast/training/inpaint_dataset.py`) removes a
unit's covering BPE(s) only when every char those BPEs cover lies inside the
substituted set — so a removal can never delete a neighbouring un-annotated
char. For English the Qwen3 tokenizer bundles a word's leading space into its
first BPE:

```
'I LIKE BANANA' →  ['I', ' LIKE', ' B', 'AN', 'ANA']
                                    ^^^ char[6,8) = space(6) + 'B'(7)
```

`BANANA` spans chars [7,13), but its first covering BPE `' B'` also covers
char 6 (the space), which is NOT part of the annotated word. So the unit is
"unclean" → fall back to insertion. Verified on the v12 checkpoint:

```
ssml:            I LIKE <phoneme alphabet="cmu" ph="B AH N AE N AH">BANANA</phoneme>
baseline tokens: ['I', ' LIKE', ' B', 'AN', 'ANA']
inpaint  tokens: ['I', ' LIKE', ' B', 'AN', 'ANA', 'PAD', 'PAD', 'PAD']   ← graphemes KEPT
mask True at:    [5, 6, 7]
```

The graphemes `B/AN/ANA` survive and 3 pads are merely appended. Since nearly
every non-sentence-initial English word begins with a space-bundled BPE, almost
all English annotations fall back to insertion → **the composer is never the
sole signal for English** → the exact v11 non-load-bearing collapse persists.

Corroborating behavioral metric: `en_banana` has `edit(base, correct) = 0.985`.
A *working* inpaint should put `correct ≈ baseline` (edit ≈ 0), as it does for
`yue_dei6` (0.065). 0.985 means the en composer is not producing the natural
reading — consistent with insertion fallback.

### Problem 2 — base LLM loops on the English prompts

In the behavioral gate, the English **baseline** (`disable_inpaint=True`, plain
`I LIKE BANANA`) emitted 200 tokens with `eos=False`. The correct and wrong
poles also ran to 200. This is why the fixture carries `enable_behavioral=False`
(and `enable_parity=False`). The model simply does not terminate on these
prompts, independent of inpaint.

This matches the `zh_wo_singleton` failure (`我吃饭`, also a short fragment, also
ran to 200/no-eos) and the `inference-text-must-match-training-format` memory:
isolated short prompts are out-of-distribution for the dialogue-trained model.
`I LIKE BANANA` is correctly uppercase/no-punct, so the *format* is right — the
likely culprit is **length/OOD**, not casing.

## Why English, specifically

| Axis | zh | yue | en |
|---|---|---|---|
| Training rows | 5,622 | 261,338 | 85,270 |
| BPE↔char alignment | whole-char (merged/byte-fallback) | same | **space-bundled, word-internal** |
| Substitution clean? | always | always | **no (space bleed)** |
| Base LLM terminates on test prompt? | mostly | yes | **no (loops)** |

English's tokenization is the qualitatively different one: zh/yue BPEs cover
whole CJK chars (clean to excise), English BPEs straddle the space→word
boundary. That single fact routes en into the insertion fallback.

## Fix plan

### Phase A — make English substitutable (mechanism)  ·  ~½ day, low risk

Goal: English words get true grapheme substitution like zh/yue.

**A1. Whitespace-tolerant removal in `_substitute_units`.**
Change the "clean" predicate so a covering BPE is removable when every
*non-whitespace* char it covers is in `covered_chars` (whitespace is "free" to
delete along with the word). Then `' B'` (space + `B`) becomes removable when
`B` is annotated; the leading space is absorbed into the substitution.

- Mirror automatically holds: training (`_align_grapheme_subst_english`) and
  inference both call the same `_substitute_units`, so parity is preserved by
  construction.
- Edge: keep the existing guard against deleting a *non-whitespace* neighbour
  (e.g. a partial merged CJK BPE) — only whitespace is exempted.
- Decide: when the leading space is removed, do we need to preserve a word
  boundary for the LLM? Options: (i) drop it (the pad implies a token break);
  (ii) keep one space token before the first pad. Prefer (i) first; A/B if en
  prosody/segmentation degrades.

**A2. Re-enable English parity.**
With A1, training and inference produce the same layout for en. Flip
`EN_BANANA_MULTIBPE.enable_parity = True` in
`scripts/inpaint/fixtures_adversarial.py` and confirm `test_alignment_parity.py`
stays green for en (it currently skips en). This locks the en contract.

**A3. Verify the grapheme is actually removed.**
Re-run the BANANA dump from the evidence section; assert the inpaint tokens no
longer contain `B/AN/ANA` and that pads replace them.

### Phase B — establish whether the LLM can be steered in English  ·  ~½ day

Goal: separate the mechanism fix (Phase A) from the base-LLM looping.

**B1. Realistic-prompt en behavioral fixtures.**
Add 2–3 en fixtures with *full, natural* sentences (not 3-word fragments),
matching the en training distribution (uppercase, no punctuation, sentence
length ≥ ~10 words). Pick words with genuine pronunciation ambiguity
(homographs: `READ`, `LEAD`, `LIVE`, `TEAR`, `BOW`, `WIND`) so correct vs wrong
phonemes are meaningfully different. Keep `en_banana` as the byte-structure
regression canary.

**B2. Re-run behavioral on the v12 checkpoint with B1 fixtures.**
- If the baseline now terminates (eos=True) → the looping was OOD-prompt, and
  we can enable `enable_behavioral=True` for the realistic fixtures.
- If the baseline still loops on full sentences → it is a base-model en quality
  issue (Phase D), and en inpaint cannot be validated on this checkpoint
  regardless of the composer.

**B3. Diagnostic on en.**
Add the en fixtures to `scripts/inpaint/diagnose_composer_noop.py` `PROBES` and
check cross-fixture cosine. Note the pad0-only limitation (compare per-pad, or
extend the diagnostic to average over all pads of a unit) — English words are
multi-pad (one per syllable), so pad0-only is especially misleading here; fix
the diagnostic to compare the *differing* pad or the mean over pads.

### Phase C — strengthen the English composer if needed  ·  conditional

Only if Phase B shows the LLM terminates but correct↔wrong still doesn't steer
(en composer still collapsed, like v11 zh):

**C1. Turn on the aux phoneme-recovery loss** (already implemented, default-off):
`--aux_phoneme_loss_weight 0.2`. Forces the composed embedding to be decodable
to its ARPAbet ids — directly anti-collapse. Verified to run (aux loss
decreases on smoke).

**C2. Audit `head_cmu`.** English uses the 6-slot `head_cmu`
(`soulxpodcast/inpaint/composer.py`). Confirm the per-syllable ARPAbet packing
(`encode_arpabet_per_syllable`) fills the 6 slots sensibly and that K=6 is not
truncating multi-phone syllables (the dataset note claims K=8 covered 97.3% of
en words — re-check at K=6).

### Phase D — base-model English quality  ·  out of scope for the composer

If the baseline loops even on realistic sentences, the SoulX checkpoint is
weak at en speech-token generation. This is a base-model/training-data problem,
not inpaint. Options (deferred): fine-tune the trunk on more en dialogue, or
accept that en inpaint ships only once the base model terminates reliably.

## Recommended order & gate

1. Phase A (mechanism) — cheap, unblocks everything. Gate: BANANA grapheme
   removed + en parity green.
2. Phase B (realistic prompts) — the decisive test. Gate: does the en baseline
   terminate? This bifurcates the remaining work into C (composer) vs D
   (base model).
3. Phase C only if B shows terminate-but-no-steering.

Do NOT conclude "English inpaint is broken/working" until Phase B — the current
`I LIKE BANANA` result conflates the mechanism gap (A) and the OOD-prompt loop
(B/D) and tells us nothing clean about the composer's English ability.

## Code touch-points

- `soulxpodcast/training/inpaint_dataset.py` — `_substitute_units` (A1),
  `_align_grapheme_subst_english` (consumes A1 via shared helper).
- `scripts/inpaint/fixtures_adversarial.py` — en parity flag (A2), new realistic
  en fixtures (B1).
- `scripts/inpaint/test_alignment_parity.py` — picks up A2 automatically.
- `scripts/inpaint/test_alignment_behavioral.py` — B2.
- `scripts/inpaint/diagnose_composer_noop.py` — en probes + per-pad fix (B3).
- `soulxpodcast/training/train_inpaint.py` — `--aux_phoneme_loss_weight` (C1,
  already wired).
- `soulxpodcast/inpaint/composer.py` — `head_cmu` audit (C2).
