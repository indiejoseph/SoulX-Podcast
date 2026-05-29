---
title: "Non-MTP LLM Speedups Inspired by cosyvoice.cpp"
status: implemented_pending_gpu_verification
created_date: 2026-05-29
updated_date: 2026-05-29
ticket: null
author: codex
tags: [llm, inference, performance, cosyvoice-cpp, hf, vllm]
---

# Non-MTP LLM Speedups Inspired by cosyvoice.cpp

## Problem Statement

SoulX-Podcast RTF is still dominated by the Qwen3 LLM decode path. MTP remains the larger speculative-decoding project, but there are non-MTP optimizations worth isolating first: reduce per-token logits/sampling overhead, avoid repeated sampler work, reuse prompt-prefix KV outside the MTP path, and make vLLM graph execution easier to benchmark in Docker.

The `cosyvoice.cpp` review suggests useful implementation ideas, but not a direct port. Its strongest transferable ideas are speech-vocabulary-only decoding, single-pass RAS candidate reuse, prompt-prefix KV reuse, and explicit runtime knobs for cache/buffer behavior.

## Proposed Solution

Implement a sequence of small, benchmarkable LLM-only changes behind explicit config flags. The first target is the HF trunk path because it is the easiest place to preserve current token distribution exactly and compare against the existing sampler. vLLM and API serving changes come after the HF path has a measured win.

The main technical shape is:

- Add a restricted speech-token projection path that samples only from valid speech-token IDs plus EOS, while mapping sampled compact IDs back to full tokenizer IDs.
- Patch the HF RAS sampler to reuse its first sampled candidate when RAS does not trigger.
- Extend API prompt-prefix KV reuse to the non-MTP HF path.
- Expose vLLM eager/graph mode in serving config and Docker so CUDA graph behavior can be measured rather than hardcoded.

## Implementation Notes

Implemented on branch `plan/llm-speedups-cosyvoice-cpp` on 2026-05-29. This environment does not have `torch` or CUDA, so only syntax-level verification was run locally:

```bash
python3 -m py_compile soulxpodcast/config.py soulxpodcast/engine/llm_engine.py soulxpodcast/models/modules/sampler.py soulxpodcast/utils/infer_utils.py api/config.py api/service.py scripts/inference/profile_latency.py scripts/inference/inference_test.py
```

GPU/model verification remains pending on a CUDA machine.

## Phases

### Phase 1: Baseline And Guardrails

**Goal:** Establish a reproducible non-MTP LLM benchmark before changing sampler behavior.

**Changes:**
- `scripts/inference/profile_latency.py:73` - extend the existing LLM timing wrapper to report generated tokens/sec, prefill time, decode time, and sampler mode.
- `scripts/inference/inference_test.py:25` - add optional flags for `--engine hf|vllm`, `--max-new-tokens`, `--no-dialect-prompt`, and output JSON for before/after comparisons.
- `soulxpodcast/config.py:107` - add planned sampling flags with conservative defaults disabled, so experiments can be turned on without changing current behavior.

**Success Criteria:**
- [ ] `python scripts/inference/profile_latency.py` reports LLM tokens/sec and end-to-end RTF.
- [ ] `python scripts/inference/inference_test.py hf` still produces audio and a JSON summary.
- [ ] Baseline output tokens match the pre-change HF path when new flags are disabled.

**Dependencies:** None.

### Phase 2: HF RAS Candidate Reuse

**Goal:** Remove duplicated full-vocab softmax/multinomial work in the common RAS-not-triggered case.

**Changes:**
- `soulxpodcast/models/modules/sampler.py:142` - keep the candidate sampled for the RAS check and commit it directly when repetition is below threshold.
- `soulxpodcast/models/modules/sampler.py:171` - only run the second sampling step when RAS fires or when `use_ras=False`.
- `soulxpodcast/training/mtp_inference.py:592` - use the existing MTP `_sample_with_ras` behavior as the reference implementation, because it already reuses the candidate when RAS does not trigger.

**Success Criteria:**
- [ ] With fixed seed, generated token IDs remain distribution-compatible and do not regress EOS handling.
- [ ] Streaming still calls `streamer.put()` once per accepted generated token.
- [ ] LLM decode time improves or remains neutral on `scripts/inference/profile_latency.py`.

**Dependencies:** Phase 1.

### Phase 3: Restricted Speech-Vocab HF Sampler

**Goal:** Avoid full-vocab logits processing during speech-token generation.

**Changes:**
- `soulxpodcast/config.py:42` - read `speech_token_offset` from loaded config and derive valid speech IDs as `[offset, offset + 6560]`.
- `soulxpodcast/config.py:107` - add config fields such as `restrict_speech_vocab: bool = False` and `speech_vocab_size: int = 6561`.
- `soulxpodcast/engine/llm_engine.py:61` - pass speech-vocab restriction metadata into the custom HF sampler only when enabled.
- `soulxpodcast/models/modules/sampler.py:136` - replace full-vocab `outputs.logits[:, -1, :]` sampling with compact logits for allowed speech IDs plus EOS where feasible.
- `soulxpodcast/models/modules/sampler.py:140` - adapt repetition penalty, top-k, top-p, min-EOS masking, and RAS to operate in compact space while gathering context counts from full token IDs.

**Success Criteria:**
- [ ] Disabled flag gives byte-for-byte identical generated token IDs to current HF for a fixed seed.
- [ ] Enabled flag emits only valid speech token IDs or EOS after `<|semantic_token_start|>`.
- [ ] Generated 0-based speech tokens are still valid after subtracting `speech_token_offset`.
- [ ] Audio A/B check has no obvious degradation on the demo prompt.
- [ ] LLM decode tokens/sec improves measurably on RTX 3090.

**Dependencies:** Phases 1 and 2.

### Phase 4: Prompt-Prefix KV Reuse For Non-MTP HF

**Goal:** Reuse voice prompt KV for repeated API requests even when MTP is disabled.

**Changes:**
- `api/service.py:645` - keep `_build_prompt_prefix_kv()` as the shared prompt-prefix builder for direct HF trunk access.
- `api/service.py:731` - store prompt-prefix KV/cache data for non-MTP HF when an opt-in flag is enabled.
- `api/service.py:797` - route cached prefix plus per-request target text into `HFLLMEngine.generate(..., past_key_values=...)` or a dedicated helper that mirrors the current MTP prefix-tail flow.
- `soulxpodcast/engine/llm_engine.py:53` - verify `past_key_values` works for non-MTP generation and that returned token slicing still excludes only the uncached target input, not the prefix.

**Success Criteria:**
- [ ] Repeated requests with the same `prompt_cache_id` skip prompt-prefix prefill in the HF non-MTP path.
- [ ] Generated token IDs match the uncached path for equivalent random seed and sampling state, allowing expected sampling nondeterminism when no seed is fixed.
- [ ] Prompt cache memory remains bounded by existing prompt cache TTL/size controls.

**Dependencies:** Phase 1.

### Phase 5: vLLM Runtime Knobs

**Goal:** Make vLLM graph execution and quantized model experiments selectable in Docker/API serving.

**Changes:**
- `api/config.py:35` - expose `VLLM_ENFORCE_EAGER` or equivalent and avoid silently forcing eager mode unless a known incompatibility requires it.
- `soulxpodcast/engine/llm_engine.py:120` - keep `enforce_eager=config.enforce_eager`, then verify the API config actually sets it from environment.
- `docker-compose.yml` - add the environment variable with a default suitable for benchmarking CUDA graph mode.
- `scripts/inference/inference_test.py:108` - record whether vLLM is running eager or graph mode in benchmark output.

**Success Criteria:**
- [ ] `LLM_ENGINE=vllm VLLM_ENFORCE_EAGER=false` starts successfully.
- [ ] Warm vLLM inference is benchmarked separately from cold CUDA graph compilation.
- [ ] HF remains the fallback path for MTP-serving mode.

**Dependencies:** Phase 1.

## Risks & Considerations

- Restricted-vocab sampling must not hardcode `153595`; it must read `model.config.hf_config.speech_token_offset`.
- Some special control tokens live near but outside the actual speech-token range. The sampler must allow EOS but not the placeholder `[add_token_*]` range.
- HF `generate()` owns prompt token streaming semantics. Any custom sampler path must preserve the existing `SpeechTokenStreamer` prompt-skip behavior.
- Directly bypassing `AutoModelForCausalLM.forward()` to avoid full `lm_head` would be faster, but it is riskier because it must manually maintain cache position, attention masks, and generation outputs. Start with a low-risk sampler patch first.
- vLLM may remain faster for long multi-turn contexts because prefix caching and paged attention solve a different part of the bottleneck.
- `cosyvoice.cpp` is not a direct compatibility target. It is an implementation reference for speech-only output space and cache policy, not a source dependency.

## File Impact

- `soulxpodcast/models/modules/sampler.py` - RAS candidate reuse and optional restricted speech-vocab sampling.
- `soulxpodcast/engine/llm_engine.py` - pass sampler feature flags and verify non-MTP `past_key_values` behavior.
- `soulxpodcast/config.py` - add opt-in sampler/runtime flags.
- `api/service.py` - extend prompt-prefix KV reuse to non-MTP HF serving.
- `api/config.py` - expose runtime flags for vLLM eager/graph behavior.
- `docker-compose.yml` - add corresponding environment defaults.
- `scripts/inference/profile_latency.py` - stronger LLM stage benchmark output.
- `scripts/inference/inference_test.py` - reproducible JSON benchmark harness.

## Questions for Reviewer

- Should restricted speech-vocab sampling be HF-only first, or should we also plan a vLLM `logits_processor` equivalent?
- Should prompt-prefix KV reuse be enabled by default for non-MTP HF, or guarded behind an environment flag until memory behavior is measured?
- What is the minimum audio A/B set for accepting the restricted sampler: single Mandarin turn, four-turn Cantonese, and one long English turn?

## GPU Verification Commands

Run these on a CUDA machine with the normal SoulX environment:

```bash
python scripts/inference/profile_latency.py \
  --engine hf \
  --json-output outputs/bench/profile_hf_baseline.json

python scripts/inference/profile_latency.py \
  --engine hf \
  --restrict-speech-vocab \
  --json-output outputs/bench/profile_hf_restrict_vocab.json

python scripts/inference/inference_test.py hf \
  --no-dialect-prompt \
  --json-output outputs/bench/inference_hf_baseline.json

python scripts/inference/inference_test.py hf \
  --no-dialect-prompt \
  --restrict-speech-vocab \
  --json-output outputs/bench/inference_hf_restrict_vocab.json

python scripts/inference/inference_test.py vllm \
  --no-vllm-enforce-eager \
  --json-output outputs/bench/inference_vllm_graph.json
```

For Docker/API tests:

```bash
ENABLE_MTP=false RESTRICT_SPEECH_VOCAB=true HF_PROMPT_PREFIX_CACHE=true docker compose up --build
LLM_ENGINE=vllm VLLM_ENFORCE_EAGER=false ENABLE_MTP=false docker compose up --build
```

## CUDA Handoff Note

**Branch:** `plan/llm-speedups-cosyvoice-cpp`

**Base commit:** `e834dd2`

**Current status:** implementation is present but only syntax-checked locally. The local environment used for this handoff does not have `torch` or CUDA, so no model load, no audio generation, and no API smoke test has been run.

### What Changed

- `soulxpodcast/models/modules/sampler.py` - added HF RAS candidate reuse and an experimental compact speech-vocab sampler. The compact path bypasses the full `AutoModelForCausalLM.forward()` logits projection and calls the Qwen backbone plus selected `lm_head` rows for EOS + speech token IDs.
- `soulxpodcast/engine/llm_engine.py` - passes `restrict_speech_vocab`, `speech_token_offset`, `speech_vocab_size`, and EOS metadata into the custom HF sampler.
- `soulxpodcast/config.py` - added `SamplingParams.restrict_speech_vocab` and `SamplingParams.speech_vocab_size`, defaulting to disabled/6561.
- `api/config.py`, `docker-compose.yml` - added `RESTRICT_SPEECH_VOCAB`, `SPEECH_VOCAB_SIZE`, `HF_PROMPT_PREFIX_CACHE`, and `VLLM_ENFORCE_EAGER`.
- `api/service.py` - added opt-in non-MTP HF prompt-prefix KV reuse for one-shot `/v1/audio/speech` requests when `HF_PROMPT_PREFIX_CACHE=true`.
- `scripts/inference/profile_latency.py`, `scripts/inference/inference_test.py` - added benchmark flags and JSON output.

### First Commands On CUDA Machine

Use the project environment from `AGENTS.md` before testing:

```bash
conda deactivate || true
source .venv/bin/activate
python -m py_compile soulxpodcast/config.py soulxpodcast/engine/llm_engine.py soulxpodcast/models/modules/sampler.py soulxpodcast/utils/infer_utils.py api/config.py api/service.py scripts/inference/profile_latency.py scripts/inference/inference_test.py
git diff --check
```

Then run the non-API benchmark sequence:

```bash
mkdir -p outputs/bench

python scripts/inference/profile_latency.py \
  --engine hf \
  --json-output outputs/bench/profile_hf_baseline.json

python scripts/inference/profile_latency.py \
  --engine hf \
  --restrict-speech-vocab \
  --json-output outputs/bench/profile_hf_restrict_vocab.json

python scripts/inference/inference_test.py hf \
  --no-dialect-prompt \
  --json-output outputs/bench/inference_hf_baseline.json

python scripts/inference/inference_test.py hf \
  --no-dialect-prompt \
  --restrict-speech-vocab \
  --json-output outputs/bench/inference_hf_restrict_vocab.json

python scripts/inference/inference_test.py vllm \
  --no-vllm-enforce-eager \
  --json-output outputs/bench/inference_vllm_graph.json
```

### API/Docker Follow-Up

Test API flags separately so failures are easy to isolate:

```bash
# Restricted sampler only; MTP disabled so the HF trunk path is exercised directly.
ENABLE_MTP=false RESTRICT_SPEECH_VOCAB=true docker compose up --build

# Prompt-prefix cache path. Reuse the same Prompt-Cache-Id across at least two requests.
ENABLE_MTP=false HF_PROMPT_PREFIX_CACHE=true docker compose up --build

# vLLM graph-mode experiment. Keep MTP disabled because MTP requires direct HF trunk access.
LLM_ENGINE=vllm VLLM_ENFORCE_EAGER=false ENABLE_MTP=false docker compose up --build
```

Use `scripts/api/measure_ttfa.py` if you want an API-level TTFA/RTF comparison; run once without prompt-cache reuse and once with the returned `Prompt-Cache-Id`.

### What To Validate

- Baseline branch behavior with `RESTRICT_SPEECH_VOCAB=false` should still generate valid audio.
- Restricted sampler output must contain only actual speech-token IDs in `[speech_token_offset, speech_token_offset + 6560]` plus EOS. Do not allow placeholder IDs from `152927-153594` except the configured EOS.
- Compare branch baseline vs branch restricted-vocab for LLM decode tok/s, total RTF, generated audio duration, and obvious audio quality. This is the main apples-to-apples comparison.
- Compare vLLM graph mode as a warm benchmark. The first vLLM request can include CUDA graph compilation overhead; run at least twice and compare the second run.
- For `HF_PROMPT_PREFIX_CACHE=true`, verify the second request with the same `prompt_cache_id` is faster in prefill/TTFA and still sounds equivalent.

### Important Caveats

- Do not expect fixed-seed token IDs to match `e834dd2` exactly when RAS is enabled. Phase 2 intentionally reuses the RAS candidate instead of drawing a second sample, so the RNG draw count changes while preserving the intended sampling distribution.
- The compact sampler only activates when its logits processors are recognized. If Transformers injects an unsupported processor, it should fall back to the full-vocab sampler. Inspect `_compact_processors_supported()` in `soulxpodcast/models/modules/sampler.py` if `--restrict-speech-vocab` appears to have no speed effect.
- The compact path calls the Qwen backbone directly. If it fails, first inspect `_backbone_forward_for_restricted_logits()` and whether the installed Transformers/Qwen3 version expects extra arguments such as `position_embeddings` or different cache-position handling.
- The non-MTP prompt-prefix cache path depends on `HFLLMEngine.generate(..., past_key_values=prefix_cache)` handling cache positions correctly. If repeated prompt-cache requests produce bad tokens or shape errors, compare against the MTP prefix-cache plumbing in `api/service.py` and `soulxpodcast/training/mtp_inference.py`.
- If any experimental flag fails, rerun with only that flag disabled before changing shared serving code. Defaults are intentionally conservative: restricted sampler off, HF prompt-prefix cache off, vLLM eager false only when `LLM_ENGINE=vllm`.
