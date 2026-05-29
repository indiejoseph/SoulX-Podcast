---
title: "Non-MTP LLM Speedups: vLLM CUDA Graph Pivot"
status: vllm_graph_verified_phase3_dropped
created_date: 2026-05-29
updated_date: 2026-05-29
ticket: null
author: codex
tags: [llm, inference, performance, hf, vllm]
---

# Non-MTP LLM Speedups: vLLM CUDA Graph Pivot

## Decision

The LLM is still the main contributor to slow generation, but the measured bottleneck is the Qwen backbone, not `lm_head` or sampler overhead.

Measured result from GPU verification:

- Restricted-vocab `lm_head` saving is real in isolation: about `0.56 ms/token`.
- At roughly `750` tokens, that is only about `420 ms`, or about `1.5%` of a `28 s` wall time.
- The Qwen backbone costs roughly `35 ms/token`, so it dominates decode time.
- The compact sampler also had a hidden per-token cost from `weight.index_select(0, allowed_ids)`, which gathers sliced `lm_head` rows every step.
- Caching the sliced rows would improve the isolated `lm_head` microbenchmark, but not the real end-to-end bottleneck.
- vLLM CUDA graphs are the real non-MTP win: measured RTF `0.418` versus HF RTF `0.911`.

Therefore: **drop Phase 3 restricted-vocab sampling from the merge path**. Keep the branch focused on vLLM graph mode, benchmark guardrails, and small low-risk HF cleanup.

## Implementation Direction

Use vLLM graph mode as the default fast path for non-MTP serving and benchmarking. Keep HF as the direct-trunk path for MTP work and as a fallback when vLLM is unavailable.

The code should avoid carrying the compact speech-vocab sampler because its speedup is below measurement noise for normal end-to-end runs and the implementation touches fragile generation internals.

## Current Branch Scope

Keep:

- Benchmark script improvements for `--engine hf|vllm`, JSON output, `--max-new-tokens`, and `--no-dialect-prompt`.
- API/Docker `VLLM_ENFORCE_EAGER` control so graph mode is explicit.
- vLLM AWQ auto-detection in `VLLMEngine`.
- vLLM token streaming through the engine loop.
- HF RAS candidate reuse, because it is small and avoids one duplicate sample in the common non-repetition case.
- Optional `HF_PROMPT_PREFIX_CACHE` for repeated HF API requests.

Remove or do not merge:

- `restrict_speech_vocab` config.
- `RESTRICT_SPEECH_VOCAB` / `SPEECH_VOCAB_SIZE` environment flags.
- Direct Qwen-backbone compact sampler path.
- Restricted-vocab benchmark commands.

## Phases

### Phase 1: Benchmark Guardrails

**Goal:** Keep reproducible LLM benchmarks before and after runtime changes.

**Success Criteria:**
- [ ] `python scripts/inference/profile_latency.py --engine hf` reports LLM tokens/sec and end-to-end RTF.
- [ ] `python scripts/inference/inference_test.py hf --json-output ...` still produces audio and JSON.
- [ ] `python scripts/inference/inference_test.py vllm --no-vllm-enforce-eager --json-output ...` captures graph-mode settings.

### Phase 2: HF RAS Candidate Reuse

**Goal:** Remove duplicated full-vocab softmax/multinomial work in the common RAS-not-triggered case without changing the model path.

**Success Criteria:**
- [ ] Generated tokens remain distribution-compatible for fixed seed comparisons.
- [ ] Streaming still calls `streamer.put()` once per accepted generated token.
- [ ] Decode time is neutral or slightly better.

This is not expected to be the main speedup.

### Phase 3: Restricted Speech-Vocab HF Sampler

**Status:** Dropped.

Reason: measured `lm_head` savings are about `1.5%` end-to-end while backbone decode dominates. The extra generation-internal complexity is not justified for Qwen3-1.7B in this pipeline.

### Phase 4: vLLM CUDA Graph Fast Path

**Goal:** Make vLLM graph mode the default non-MTP speed path and keep eager mode selectable only for debugging.

**Changes:**
- `api/config.py` exposes `VLLM_ENFORCE_EAGER`.
- `docker-compose.yml` defaults `VLLM_ENFORCE_EAGER=false`.
- `soulxpodcast/engine/llm_engine.py` passes `enforce_eager=config.enforce_eager`.
- Benchmark scripts record the vLLM eager/graph setting.

**Success Criteria:**
- [ ] `LLM_ENGINE=vllm VLLM_ENFORCE_EAGER=false ENABLE_MTP=false docker compose up --build` starts successfully.
- [ ] Warm graph-mode inference reproduces the measured RTF advantage over HF.
- [ ] Cold-start CUDA graph compilation is reported separately from warm inference.
- [ ] HF remains forced for MTP serving because MTP needs direct trunk access.

### Phase 5: HF Prompt-Prefix Cache

**Goal:** Reduce repeated-request prefill in HF serving.

This helps TTFA and repeated same-speaker API calls, but it does not solve decode speed. Keep it opt-in with `HF_PROMPT_PREFIX_CACHE=true`.

**Success Criteria:**
- [ ] Repeated requests with the same `prompt_cache_id` reuse prefix KV.
- [ ] Prompt cache memory remains bounded by existing cache TTL/size controls.

### Phase 6: Structural Decode Speedup

**Goal:** Move beyond runtime tuning if vLLM graph mode is still not enough.

Next real speedup path is MTP/speculative decoding, because it reduces the number of Qwen backbone steps. Non-MTP sampler optimizations should not be expected to beat the backbone bottleneck.

## Verification Commands

Use the project environment from `AGENTS.md`:

```bash
conda deactivate || true
source .venv/bin/activate
python -m py_compile soulxpodcast/config.py soulxpodcast/engine/llm_engine.py soulxpodcast/models/modules/sampler.py soulxpodcast/utils/infer_utils.py api/config.py api/service.py scripts/inference/profile_latency.py scripts/inference/inference_test.py
git diff --check
```

Benchmark sequence:

```bash
mkdir -p outputs/bench

python scripts/inference/profile_latency.py \
  --engine hf \
  --json-output outputs/bench/profile_hf.json

python scripts/inference/inference_test.py hf \
  --no-dialect-prompt \
  --json-output outputs/bench/inference_hf.json

python scripts/inference/inference_test.py vllm \
  --no-vllm-enforce-eager \
  --no-dialect-prompt \
  --json-output outputs/bench/inference_vllm_graph.json
```

API/Docker:

```bash
LLM_ENGINE=vllm VLLM_ENFORCE_EAGER=false ENABLE_MTP=false docker compose up --build

ENABLE_MTP=false HF_PROMPT_PREFIX_CACHE=true docker compose up --build
```

## Merge Recommendation

Merge only the vLLM graph/runtime knobs, benchmark improvements, vLLM streaming loop, AWQ auto-detection, and small HF RAS candidate reuse after syntax and GPU smoke verification. Do not merge restricted-vocab sampling.
