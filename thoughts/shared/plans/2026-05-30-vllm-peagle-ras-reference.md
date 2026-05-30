---
title: "vLLM P-EAGLE RAS Reference from vllm-omni"
status: implemented_in_repo
created_date: 2026-05-30
updated_date: 2026-05-30
ticket: null
author: codex
tags: [vllm, peagle, ras, sampling, cosyvoice3]
---

# vLLM P-EAGLE RAS Reference from vllm-omni

## Source References

- Shared helper: https://github.com/vllm-project/vllm-omni/blob/f43bb124bc8ed53255b4bf933bb2d128fd19f744/vllm_omni/model_executor/models/common/nucleus_ras_sampling.py
- CosyVoice3 model sampler: https://github.com/vllm-project/vllm-omni/blob/f43bb124bc8ed53255b4bf933bb2d128fd19f744/vllm_omni/model_executor/models/cosyvoice3/cosyvoice3.py
- vLLM-Omni API docs snapshot: https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/model_executor/models/cosyvoice3/cosyvoice3/

## Why This Matters

The current SoulX production image uses patched vLLM `0.10.1` with custom
`SamplingParams` fields:

- `use_ras`
- `win_size`
- `tau_r`

P-EAGLE requires a newer vLLM/speculators runtime. That newer runtime is not
expected to accept the old SoulX-patched sampling fields, so
`VLLMEngine._make_sampling_params()` falls back to base vLLM sampling and RAS is
lost.

vllm-omni avoids this problem by moving TTS-specific sampling into the model
implementation instead of extending public `SamplingParams`.

## vllm-omni Pattern

CosyVoice3 sets:

```python
prefer_model_sampler = True
```

and implements:

```python
def sample(self, logits, sampling_metadata) -> SamplerOutput | None:
    ...
```

The sample override:

- delegates to vLLM's default `Sampler` when the request uses unsupported
  features such as logprobs or bad-word filters;
- applies vLLM logits processors before custom sampling. CosyVoice3 disables
  penalties there; SoulX keeps repetition penalty for the filtered candidate to
  match the existing HF/vLLM sampler behavior;
- reads per-request `temperature`, `top_p`, and `top_k` from
  `SamplingMetadata`;
- reads recent generated tokens from `sampling_metadata.output_token_ids`;
- runs RAS in the model sampler and returns `SamplerOutput` directly.

The helper file factors out two primitives:

- nucleus top-p/top-k sampling;
- RAS: sample once from the filtered distribution, check recent repetition,
  then sample once from the original full distribution if the repetition
  threshold fires.

## Behavior Difference to Watch

The shared helper at the pinned commit says the fallback should resample from
the original full distribution when RAS fires.

The CosyVoice3 class-local implementation at the same commit masks the repeated
token before fallback sampling. The shared helper is closer to the current SoulX
HF sampler and the VALL-E 2/RAS behavior used by SoulX:

```text
candidate from filtered distribution -> if repeated too often -> resample from
raw/full distribution
```

For SoulX, prefer the shared helper behavior unless audio A/B proves masking the
candidate is better.

## SoulX Integration Direction

Implemented path for newer vLLM + P-EAGLE:

1. Stop relying on patched `SamplingParams` fields in the P-EAGLE runtime.
2. Install a small runtime monkeypatch for vLLM's Qwen3 model class from
   `soulxpodcast.engine.vllm_ras`.
3. Set `prefer_model_sampler = True` on `Qwen3ForCausalLM`.
4. Implement `sample(logits, sampling_metadata)` with SoulX RAS semantics:
   filtered candidate uses temperature, repetition penalty, top-k, top-p;
   repeated candidate falls back to raw logits.
5. Keep `VLLMEngine._make_sampling_params()` passing only stock vLLM fields for
   the newer runtime.
6. Verify P-EAGLE acceptance and RAS audio behavior separately:
   P-EAGLE tests token throughput and accepted tokens; RAS tests repetition and
   audio quality.

## Compatibility Notes

- This likely belongs in the newer vLLM/speculators environment, not the
  current `vllm/vllm-openai:v0.10.1` production image.
- The implementation monkeypatches vLLM's generic Qwen3 model class before
  `LLMEngine.from_engine_args()` loads the model. This is intended for the
  repo's in-process `LLMEngine` path.
- A direct vLLM patch remains possible, but vllm-omni indicates the cleaner V1
  path is model-owned sampling.
- The public `SamplingParams` should remain stock for P-EAGLE so speculative
  decoding can use the newer vLLM engine without fighting unknown custom fields.

## Open Questions

- Which exact vLLM version will be used for P-EAGLE serving?
- Does P-EAGLE validation call the model sampler for target verification, or
  does it bypass sampling in a way that requires a deeper sampler integration?
- Does the target serving setup ever move Qwen3 execution into worker processes
  that do not inherit the in-process monkeypatch? If yes, convert this into a
  site-package patch or vLLM plugin.
