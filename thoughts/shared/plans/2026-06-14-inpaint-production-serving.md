# Inpaint production serving — wiring the composer into the live path

Date: 2026-06-14
Status: capability bundling DONE. **Option A DONE** for the non-streaming
`/generate` endpoint — inpaint serves via the shared HF trunk (no second model
copy). Streaming + vLLM + multi-turn-with-history remain gated (see below).

## DONE (Option A, non-streaming /generate)

The shared-trunk approach worked — no second trunk, just the +86 MB composer:

- `SoulXPodcast._load_composer` loads the bundled composer onto the existing HF
  trunk at model load (HF engine + inpaint-capable → `self.inpaint_ready=True`).
- `SoulXPodcast.generate_speech_tokens_inpaint(ssml, lang, …)` builds
  `inputs_embeds` (text emb + composer inject) on `self.llm.model` and generates
  speech tokens. **Verified bit-identical to the standalone InpaintInferenceEngine.**
- `SoulXPodcast.synthesize_from_tokens(tokens, prompt_feats…)` vocodes via the
  same prompt-align + flow + HiFT blocks as `forward_longform`. **Verified: 35
  tokens → finite 1.40 s wav.**
- `SoulXPodcastService.generate` detects `<phoneme>`/`<speak>` in any segment and
  routes to `_generate_inpaint_wavs` (per-segment, independent; lang inferred
  from the SSML alphabet). `/generate` passes `dialogue_text` straight through —
  no route change; `validate_dialogue_format` doesn't reject SSML.
- **Gates:** `<phoneme>` with a non-inpaint-ready model (no composer / vLLM
  engine) → clear `RuntimeError`. `/generate-stream` with `<phoneme>` → clear
  `RuntimeError` ("use /generate").

Trade accepted: inpaint segments are generated independently (no cross-turn
history that `forward_longform` provides) — speaker still comes from the flow
prompt conditioning. Fine for the "fix this word" use case.

## Still NOT built

## Where we are

The composer is now **bundled** into the model dir (`composer.pt` sibling +
`inpaint` block in `soulxpodcast_config.json`), and the stack auto-detects it:

- `soulxpodcast.inpaint.inpaint_capability(model_path)` → capability dict / None.
- `SoulXPodcast.inpaint_supported` flag set + logged at model load (mirrors
  MeanFlow detection).
- `InpaintInferenceEngine(model_path)` auto-loads the bundled composer when no
  explicit path is given.
- Tool: `scripts/inpaint/bundle_composer.py` (run after each retrain).

This makes "the model dir supports inpaint" a real, self-describing signal. It
does **not** make the production API serve inpaint — that's this plan.

## The gap

Inpaint injects its embedding at **`inputs_embeds`**. Both production engines
(`HFLLMEngine`, `VLLMEngine` in `soulxpodcast/engine/llm_engine.py`) take
**`prompt: list[int]`** (token IDs). The composer only runs through the
standalone `InpaintInferenceEngine` (HF, `inputs_embeds`), which is separate
from `api/service.py` / `SoulXPodcast.forward_longform`. So:

- `/generate-stream` cannot honour `<phoneme>` SSML today.
- vLLM (production default, RAS-patched) is the harder case — token-id only.

## Options

| Option | What | Cost | Verdict |
|---|---|---|---|
| **A. Route inpaint → HF embeds path** | When a request carries `<phoneme>`, run that request's LLM stage via the HF `inputs_embeds` forward (the existing `InpaintInferenceEngine` machinery), then hand the speech tokens to the normal flow+HiFT path. Non-inpaint requests stay on vLLM. | Medium — share flow/HiFT, add an HF LLM path in the service, route by SSML presence. | **Recommended first.** Inpaint is a low-QPS "fix this word" feature; losing vLLM batching on those requests is acceptable. |
| B. vLLM prompt-embeds | Teach the vLLM path to accept per-position input embeddings (vLLM has limited `prompt_embeds` support; our RAS patches target token-id flow). | High — engine port + RAS-patch interaction. | Later, only if inpaint QPS justifies it. |
| C. Pre-compose to token space | Approximate the composed embedding by nearest text tokens. | — | No — defeats the point (composed vectors are off the text manifold). |

## Plan (Option A)

1. **Service capability + routing.** In `api/service.py`: on load, read
   `model.inpaint_supported`. Add a lazily-constructed `InpaintInferenceEngine`
   (reusing the already-loaded trunk weights if possible, else a second HF copy
   — measure VRAM). Route a request to it iff its text contains `<phoneme>`/
   `<speak>` AND `inpaint_supported`; else the normal vLLM path.
2. **Share flow + HiFT.** `InpaintInferenceEngine` is LLM-only; it already
   returns speech tokens. Feed those into the existing `forward_longform`
   flow+HiFT + the `FlowHiftBatcher` so audio synthesis is unchanged.
3. **Decoding.** Use sampling + `repetition_penalty=1.1` for inpaint requests
   (en loops under greedy/no-penalty — see docs/pronunciation_inpaint.md).
4. **Multi-turn / streaming.** Map the SSML→composed-embeds build per turn into
   the streaming chunker. The composer fires only at the annotated pads; the
   rest of the turn is normal text — so chunking is unaffected except the
   prompt assembly.
5. **Graceful gate.** If `<phoneme>` arrives and `inpaint_supported` is False
   (or engine can't inject), return a clear 4xx ("model not inpaint-capable")
   rather than silently dropping the override.
6. **VRAM.** A second HF trunk copy alongside vLLM may not fit on a 3090.
   Options: (a) free the vLLM engine's KV headroom; (b) share the HF weights
   with the InpaintInferenceEngine (it already loads its own copy today);
   (c) make inpaint an opt-in deployment mode (HF engine only) rather than
   coexisting with vLLM. Decide by measuring.

## Open questions

- Can `InpaintInferenceEngine` and the production `SoulXPodcast` **share one HF
  trunk** instead of loading two copies? (Biggest VRAM lever.)
- AWQ-INT4: the composer was trained on the bf16/fp16 trunk's input-embedding
  table (NOT quantized by AWQ — only linear/attn weights are). Confirm the
  composed embeddings still land correctly when the trunk is AWQ. Likely fine;
  verify before shipping inpaint on the quantized model.
- Streaming TTFA impact of the HF path vs vLLM for inpaint requests.

## Remaining (follow-ups)

- **Streaming inpaint** (`/generate-stream`): inject composed embeds per chunk in
  `forward_longform_streaming`. Currently gated with a clear error.
- **vLLM coexistence**: inpaint needs the HF trunk; under `LLM_ENGINE=vllm` the
  model advertises capability but `inpaint_ready=False`. Either a small HF trunk
  alongside vLLM (VRAM permitting) or an HF-only inpaint deployment mode.
- **Multi-turn history**: inpaint segments currently generate independently. If
  cross-turn coherence matters for inpaint dialogues, thread the composer inject
  into the KV-cached `forward_longform` loop (harder; RAS + cache interaction).
- **AWQ trunk**: the composer was trained on the bf16 input-embedding table
  (not quantized by AWQ). Confirm composed embeds land correctly if inpaint runs
  on an AWQ HF trunk before shipping that combo.
