# Inpaint production serving — wiring the composer into the live path

Date: 2026-06-14
Status: capability bundling DONE (composer.pt sibling + auto-detect). The
embeds-injection integration below is NOT built.

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

## Not doing now

The bundling (this commit) is the clean, low-risk capability layer. Option A is
the next project; it's gated on the VRAM/shared-trunk question above.
