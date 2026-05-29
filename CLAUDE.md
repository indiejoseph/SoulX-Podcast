# SoulX-Podcast — Claude project notes

Multi-speaker dialogue TTS: Qwen3-1.7B → discrete speech tokens (s3tokenizer, 25 Hz) → CausalMaskedDiffWithXvec flow (15-step CFM diffusion) → HiFTGenerator vocoder → 24 kHz audio. Supports voice cloning, dialect (`<|Yue|>`, `<|Sichuan|>`, `<|Henan|>`), and paralinguistic tokens.

For full architecture details read `.claude/skills/soulx-model/SKILL.md` and `references/architecture.md` before any non-trivial implementation work.

## Environment

- Project venv: `.venv` built from `/opt/conda/bin/python` (Python 3.11.5). System `python3` is 3.10 and lacks `python3-venv`, so do NOT use `python3 -m venv` — use `/opt/conda/bin/python -m venv .venv`.
- Always `conda deactivate` before activating `.venv`.
- vLLM 0.10.1 binary wheel is ABI-compatible with torch 2.7.1. The Soul-AILab RAS patches (4 Python files from `Soul-AILab/vllm@v0.10.1.1-soulxpodcast`) must be copied over the installed package — these enable RAS sampling inside vLLM.

## Performance baselines (RTX 3090, 24 GB, bf16)

### Direct model calls (no HTTP overhead)

Measured via `scripts/inference/inference_test.py`, `scripts/inference/bistream_test.py`, `scripts/inference/multi_turn_bistream_test.py`. Dialogue: 4-turn Cantonese dialect.

| Mode | TTFA | Total wall | RTF | Notes |
|------|------|-----------|-----|-------|
| HF one-shot, single-turn | 10.2s | 10.2s | 0.74 | nothing emitted until done |
| HF one-shot, 4-turn dialect | — | 48.6s | 1.02 | full multi-turn baseline |
| vLLM one-shot, 4-turn | — | 44.7s | 0.79 | ~22% faster than HF on multi-turn |
| bi-stream chunk=50, single CUDA stream | 2.93s | 15.4s | 1.01 | 3.5× faster TTFA |
| bi-stream chunk=50, dual CUDA streams (B3) | 2.80s | 14.3s | 0.94 | B3 always active in forward_longform_streaming |
| bi-stream chunk=150, single stream | 5.52s | 11.9s | 0.78 | best wall, modest TTFA |

### API streaming via Docker (dual CUDA streams always active)

Measured via `scripts/inference/bench_stream.py` against `/generate-stream` endpoint. RTX 3090, Docker `tts:latest`, `vllm/vllm-openai:v0.10.1` base. B3 dual streams always active.

Short dialogue = 3-turn Mandarin (~6s audio). Long dialogue = 6-turn Mandarin (~30s audio).

| Engine | MTP | CUDA graphs | flow steps | chunk | Dialogue | TTFA | Wall | RTF | Notes |
|--------|-----|------------|-----------|-------|----------|------|------|-----|-------|
| HF | on | n/a | 8 | 50 | short | 1.32s | 8.38s | 1.332 | |
| HF | on | n/a | 8 | 100 | short | 0.95s | 7.36s | 1.219 | |
| HF | on | n/a | 8 | 150 | short | 0.94s | 7.62s | 1.197 | |
| HF | on | n/a | 8 | 150 | long | 0.97s | 28.56s | 0.904 | |
| vLLM | off | off (eager) | 8 | 150 | long | 1.17s¹ | 28.28s | 0.904 | same as HF+MTP |
| vLLM | off | on | 8 | 100 | short | 0.88s | 4.53s | 0.680 | RTF < 1 on short |
| vLLM | off | on | 8 | 150 | long | 0.86s² | 12.56s | 0.418 | |
| vLLM | off | on | 4 | 150 | short | 0.65s | 3.27s | 0.486 | |
| **vLLM** | **off** | **on** | **4** | **150** | **long** | **0.65s²** | **11.83s** | **🏆 0.374** | **current best** |

¹ vLLM `enforce_eager=True`: CUDA graphs disabled, same RTF as HF+MTP.  
² Warm runs; first request (cold graph) TTFA ~1.6s.

**Key finding:** vLLM CUDA graphs + FLOW_STEPS=4 is the dominant configuration. RTF=0.374 on long content (2.7× real-time), RTF=0.486 on short (~6s audio). Default config: `LLM_ENGINE=vllm VLLM_ENFORCE_EAGER=false FLOW_STEPS=4`.

### Inference runtime comparison summary

| Configuration | Long RTF | Short RTF | TTFA | Bottleneck |
|--------------|---------|---------|------|-----------|
| HF + MTP, 8 steps | 0.904 | ~1.2 | ~0.97s | LLM (HF decode) |
| vLLM eager, 8 steps | 0.904 | ~1.2 | ~1.17s | LLM (no CUDA graphs) |
| vLLM CUDA graphs, 8 steps | 0.418 | 0.680 | 0.86s | flow+HiFT |
| **vLLM CUDA graphs, 4 steps** | **0.374** | **0.486** | **0.65s** | **flow+HiFT (fixed overhead)** |

**Why FLOW_STEPS=4 gives diminishing returns:** halving steps from 8→4 saved only ~0.10s/call (4%) on warm flow calls. Most of the 2.2s/call is fixed overhead — encoder conditioning, mel feature extraction, HiFT vocoder — not the ODE step count. Further step reduction (2 steps) would give minimal benefit. The next lever is either fewer flow calls (larger chunk, higher TTFA) or flow architecture changes.

## Key findings — vLLM CUDA graphs dominate; flow+HiFT is the new bottleneck

**vLLM with CUDA graphs (`VLLM_ENFORCE_EAGER=false`) is the dominant mode.** The LLM runs so fast under CUDA graphs that flow+HiFT is the new bottleneck, same as MTP but 2.2× cheaper to achieve.

1. **`VLLM_ENFORCE_EAGER=false` is the key unlock.** All prior vLLM tests had `enforce_eager=True` hardcoded in service.py — they tested vLLM eager mode, not CUDA-graph mode. With graphs enabled, vLLM (no MTP) achieves RTF=0.418 vs HF+MTP RTF=0.904. MTP is no longer needed for RTF < 1.

2. **Bi-streaming reduces TTFA dramatically (3.5×) but adds 7-50% total-wall regression** depending on chunk size. The regression is from per-chunk diffusion overhead (~1s fixed cost per flow call) compounding across chunks.

3. **The flow architecture has no inter-call state cache.** The existing `streaming=True` flag only enables chunk-masked attention within a single call. Building true incremental streaming requires modifying `flow.py` + `flow_components/estimator.py` to add encoder KV cache + decoder feature cache + `solve_euler` state — a multi-day refactor with audio-quality risk. Not worth it given the flow call count is already minimised by large chunk sizes.

4. **Separate CUDA streams (B3) give ~7% wall improvement** by letting LLM and flow overlap on the GPU. Win is modest because both are memory-bandwidth bound on the 3090. B3 is always active in `forward_longform_streaming` — no action needed. Would scale better on H100/H200.

5. **Chunk size is the main TTFA/wall lever:** chunk=50 minimises TTFA, chunk=150 minimises wall. Per-chunk flow overhead is roughly constant, so fewer chunks → less total overhead. Default of 100 balances both.

6. **MTP is now optional** — it helps HF-engine RTF (from ~0.9 to ~0.9 on long content), but vLLM+CUDA graphs already achieves RTF=0.418 without MTP. MTP cannot be combined with vLLM (service enforces HF engine when MTP_CHECKPOINT is set).

7. **RTF < 1 requires long content to amortise flow overhead.** Short dialogues (~6s) yield RTF ~0.49 (vLLM CUDA graphs, 4 steps) to ~1.2 (HF). Long dialogues (~30s) reach RTF ~0.90 (HF) or 0.374 (vLLM CUDA graphs, 4 steps).

10. **FLOW_STEPS=4 gives ~11% RTF improvement over 8 steps** with no audible quality difference on the test dialogue. Audio quality was A/B checked — both 19s samples identical in content. Halving steps saves only ~0.10s/call on warm flow calls (4% per-call improvement) because flow time is dominated by fixed per-call overhead (encoder, vocoder), not ODE step count. Reducing steps further would give negligible gain. `FLOW_STEPS=4` is now the recommended default.

8. **Restricted speech-vocab sampler (RESTRICT_SPEECH_VOCAB) gives ~1.5% HF speedup — not worth it.** Measured: ~0.56ms/token saving on the lm_head projection (160K→6561 vocab), totalling ~420ms on a 30s dialogue. This is unmeasurable noise relative to backbone computation (~35ms/token). Root cause: Qwen3-1.7B is memory-bandwidth bound; the backbone FFN/attention dominates, not the lm_head. `RESTRICT_SPEECH_VOCAB` is disabled by default.

9. **TRT_ESTIMATOR is NOT beneficial for this model.** TRT 11 removed global `FP16`/`EXPLICIT_BATCH` flags; per-layer FP16 insertion adds type-conversion overhead. Measured: TRT TF32 RTF=1.048, TRT FP16 (per-layer) RTF=1.021 — both worse than PyTorch native FP16 (RTF=0.904). 285MB plan, 23.6s build. Root cause: PyTorch's cuBLAS FP16 path is already well-optimised for this network; TRT adds per-call address-binding overhead that dominates for small batch (B=2) inference. `TRT_ESTIMATOR` is disabled in `docker-compose.dev.yml`.

11. **`TORCH_COMPILE=true` is NOT beneficial (makes things slower).** `torch.compile(flow.decoder.estimator, mode="default", dynamic=True)` hits two blockers in PyTorch 2.7.1: (a) Inductor bounds analysis crashes on a `torch.bool` tensor of symbolic shape `[s5, s3, s3]` from the streaming attention mask (`TypeError: Invalid NaN comparison`); (b) `LoRACompatibleLinear.attn1.processor` object-identity guards cause 127+ Dynamo recompilations per request until the 128-recompile limit is hit and the estimator permanently falls back to eager. Net result: run-1 RTF ≈ 10 (compilation cost), run-2+ RTF ≈ 0.56 (eager fallback, worse than no-compile 0.486). Root cause: the estimator is only ~5% of total flow+HiFT time (~0.1s of 2.2s), so even a perfect 2× speedup would save 0.05s. Not worth fighting compiler bugs. `TORCH_COMPILE` is set to `false` in both compose files.

## Phase 0 inference optimization — exhausted

All practical vLLM/inference-level speedups have been tried. Summary:

| Optimization | Outcome | Active? |
|---|---|---|
| vLLM CUDA graphs (`ENFORCE_EAGER=false`) | RTF 0.90 → 0.42 | ✅ yes |
| FLOW_STEPS 8→4 | RTF 0.42 → 0.37 | ✅ yes |
| Dual CUDA streams (B3) | ~7% wall improvement | ✅ always on |
| MTP removed | No regression, simpler | ✅ done |
| TRT_ESTIMATOR | Slower (RTF 1.02 vs 0.90) | ❌ disabled |
| TORCH_COMPILE | Slower (compiler bugs, 128 recompilations) | ❌ disabled |
| RESTRICT_SPEECH_VOCAB | ~1.5% HF-only, backbone dominates | ❌ disabled |
| Flow state cache | Multi-day refactor, audio quality risk | ❌ not attempted |
| FLOW_STEPS 4→2 | ODE steps are only ~5% of call time | ❌ negligible |

**Structural ceiling:** flow+HiFT is the bottleneck at ~2.2s/call. HiFT accounts for ~82% of that, encoder ~9%, ODE steps ~9%. None of these are cheaply optimizable without model architecture changes.

**Current best:** RTF=0.374 (long, ~30s audio) / RTF=0.486 (short, ~6s audio), TTFA=0.65s. Config: `LLM_ENGINE=vllm VLLM_ENFORCE_EAGER=false FLOW_STEPS=4`.

## Implications for PLAN.md phases

- **Phase 0 inference optimization is complete.** The RTF=0.374 ceiling is structural — it requires changing the model, not the runtime.
- **The path to lower RTF goes through reducing flow+HiFT calls or making them cheaper.** Larger chunk sizes (>150) reduce call count at the cost of higher TTFA. Architecture changes (smaller vocoder, fewer mel frames) would be more impactful.
- **The path to faster LLM goes through training.** Phase 1 (token-interleaved bi-streaming) and Phase 2 (Sequential MTP) both require retraining/fine-tuning.
- **Phase 2 (Sequential MTP) is the cheaper next move** — adds K-1 lightweight mixing layers, trains heads-only Medusa-1 style with base frozen. Realistic ~1.8× per-step speedup with prosody preserved.
- **Phase 1 (token-interleaved grammar) is the bigger swing** — needs forced-alignment data pipeline and full base-model retraining. Months of work. Only do it after Phase 2 proves insufficient.

## Critical implementation gotchas

1. **HF transformers `_extract_generation_mode_kwargs` silently drops `streamer` when `custom_generate` is a Callable.** The streamer is filtered out because it's shared with `_sample`. Fix: bind it into the `partial(_ras_sample_hf_engine, ...)` so the custom sampler receives it directly. Any future MTP work that touches the custom sampler must preserve this binding. See `soulxpodcast/engine/llm_engine.py:HFLLMEngine.generate`.

2. **HF `model.generate()` calls `streamer.put(prompt_tensor)` once at start** with the full prompt before sampling. The `SpeechTokenStreamer` skips this via a `_got_prompt` flag (matches `TextStreamer` pattern).

3. **Two different mel dims:** `prompt_mels_for_llm` is 128-dim (LLM prefix), `prompt_mels_for_flow_ori` is 80-dim (flow conditioning). Never swap them.

4. **Speech token offset math:** s3tokenizer outputs are 0-based; LLM inputs/outputs are offset by `speech_token_offset` (152,927). Apply/strip the offset at every text↔speech boundary.

5. **vLLM streaming is a post-hoc replay** in `VLLMEngine.generate` — tokens stream out only after `model.generate()` returns. True vLLM token streaming would require the async vLLM API.

## File map

- `soulxpodcast/models/soulxpodcast.py` — main model. `forward_longform` (batch) + `forward_longform_streaming` (generator, bi-stream, B3 dual streams always active).
- `soulxpodcast/engine/llm_engine.py` — `HFLLMEngine` (with streamer hook) + `VLLMEngine`.
- `soulxpodcast/utils/streaming.py` — `SpeechTokenStreamer` + `run_llm_in_thread` (supports `cuda_stream=` for B3).
- `soulxpodcast/utils/infer_utils.py` — `initiate_model`, `process_single_input`.
- `soulxpodcast/utils/parser.py` — `podcast_format_parser` (user dict → internal format).
- `soulxpodcast/models/modules/flow.py` — `CausalMaskedDiffWithXvec` (encoder + CFM decoder).
- `soulxpodcast/models/modules/hifigan.py` — `HiFTGenerator` vocoder.
- `api/main.py` — FastAPI app. `/generate` (sync), `/generate-stream` (chunked streaming, accepts `chunk_size`/`first_chunk_size` per-request), `/generate-async` (Redis task queue), `/task/{id}`, `/download/{filename}`.
- `api/service.py` — `SoulXPodcastService`. `generate_speech_podcast` (one-shot) + `stream_speech_podcast` (streaming generator).
- `Dockerfile.serve` — production image (`vllm/vllm-openai:v0.10.1` base, RAS-patched vLLM, `python3 run_api.py` entrypoint).
- `docker-compose.yml` — production stack (tts + redis). GPU reservation via `deploy.resources`.
- `docker-compose.dev.yml` — local dev overlay: exposes Redis, mounts external model paths, symlink resolution volume, overrides `REDIS_URL`/`MODEL_PATH`/`MTP_CHECKPOINT`.

## Scripts layout

All dev scripts live under `scripts/` (organized by topic). Each has a small
`sys.path` shim at the top so it can be run directly from project root, e.g.
`python scripts/lora/lora_sweep.py`. Logs go to `logs/` (gitignored).

- `scripts/inference/` — Phase 0 benchmarks + bi-stream tests
    - `inference_test.py` — A/B HF vs vLLM on single/multi-turn.
    - `streaming_hook_test.py` — validates LLM token streaming hook.
    - `bistream_test.py` — single-turn bi-streaming with per-chunk wavs.
    - `multi_turn_bistream_test.py` — full 4-turn Cantonese dialect dialogue with speaker switching.
    - `profile_latency.py` — per-stage latency profiling.
    - `bench_stream.py` — HTTP API streaming benchmark (TTFA/wall/RTF); `--chunk N`, `--first-chunk N`, `--long` flags.
- `scripts/lora/` — LoRA sweep, averaging, and tests
    - `lora_sweep.py` — generate audio for every adapter checkpoint via `set_adapter()` (no merge).
    - `lora_average.py` — element-wise weighted average of N adapter `.safetensors` (incl. `lm_head` from `modules_to_save`).
    - `smoke_test_eos_mask.py` — verify EOS-in-mask fix lets LoRA learn termination.
    - `test_lora_{greedy_match,merged,overfit_regen}.py` — LoRA correctness tests.
- `scripts/mtp/` — MTP fine-tune tests + training-pipeline diagnostics
    - `diagnose_mtp.py`, `mtp_audio_ab.py`, `test_mtp_{dataloader,inference,train_smoke}.py`
    - `diagnose_training_pipeline.py` — self-consistency CE check (training vs inference assembly).
- `scripts/dataset/` — dataset verification
    - `verify_dataset_tokens.py`, `test_dataset_quality.py`, `test_fresh_tokens_audio.py`

## Outputs

- `outputs/engine_{hf,vllm}/turn_*.wav` — baseline one-shot reference audio.
- `outputs/streaming/streamed_chunk50.wav` — LLM-streamed tokens synthesized at end.
- `outputs/bistream/chunk{50,100,150}/` — bi-stream chunk wavs + concatenated.
- `outputs/bistream_multiturn/chunk50/` — multi-turn bi-stream per-turn + full dialogue.
