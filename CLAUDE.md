# SoulX-Podcast — Claude project notes

Multi-speaker dialogue TTS: Qwen3-1.7B → discrete speech tokens (s3tokenizer, 25 Hz) → CausalMaskedDiffWithXvec flow (15-step CFM diffusion) → HiFTGenerator vocoder → 24 kHz audio. Supports voice cloning, dialect (`<|Yue|>`, `<|Sichuan|>`, `<|Henan|>`), and paralinguistic tokens.

For full architecture details read `.claude/skills/soulx-model/SKILL.md` and `references/architecture.md` before any non-trivial implementation work.

## Environment

- Project venv: `.venv` built from `/opt/conda/bin/python` (Python 3.11.5). System `python3` is 3.10 and lacks `python3-venv`, so do NOT use `python3 -m venv` — use `/opt/conda/bin/python -m venv .venv`.
- Always `conda deactivate` before activating `.venv`.
- vLLM 0.10.1 binary wheel is ABI-compatible with torch 2.7.1. The Soul-AILab RAS patches (4 Python files from `Soul-AILab/vllm@v0.10.1.1-soulxpodcast`) must be copied over the installed package — these enable RAS sampling inside vLLM.

## Performance baselines (RTX 3090, 24 GB)

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
| vLLM | off | on | 4 | 150 | long | 0.65s² | 11.83s | 0.374 | before turn-0-only first_chunk fix |
| vLLM bf16 | off | on | 4 | 150 | short | 0.64s | 2.62s | 0.405 | post both first-chunk + final-partial fixes |
| vLLM bf16 | off | on | 4 | 150 | long | 0.64s | 8.49s | 0.265 | post both fixes — pre-AWQ best |
| vLLM AWQ-INT4 | off | on | 4 | 150 | short | 0.66s | 2.26s | 0.343 | `awq_marlin` kernel, dtype=fp16 |
| vLLM AWQ-INT4 | off | on | 4 | 150 | long | 0.67s | 6.53s | 0.208 | pre-MeanFlow best |
| **vLLM AWQ + MeanFlow** | off | on | 1 | 150 | short | **0.49s** | **1.68s** | **0.261** | Chatterbox drop-in, no CFG, 1 step |
| vLLM AWQ + MeanFlow | off | on | 1 | 150 | long | 0.49s | 5.36s | 0.164 | pre-chunked-prefill |
| **vLLM AWQ + MeanFlow + chunked prefill** | off | on | 1 | 150 | long | **0.49s** | **5.29s** | **0.162** | **🏆 current best** |

¹ vLLM `enforce_eager=True`: CUDA graphs disabled, same RTF as HF+MTP.  
² Warm runs; first request (cold graph) TTFA ~1.6s.

**Key finding:** AWQ-INT4 LLM + Chatterbox MeanFlow flow is the dominant configuration. RTF=**0.164** on long content (**6.1× real-time**), RTF=0.261 on short (~6.5 s audio), TTFA=**0.49 s**. Improvement over AWQ + CFM-4 baseline: -21% long RTF, -24% short RTF, -26% TTFA. MeanFlow drops CFG entirely (basic_euler does single forward per step, no batch doubling), which dwarfs the step-count reduction as the source of speedup. Default config: `LLM_ENGINE=vllm VLLM_ENFORCE_EAGER=false FLOW_STEPS=1 MODEL_PATH=<awq+meanflow-checkpoint>`.

### Inference runtime comparison summary

| Configuration | Long RTF | Short RTF | TTFA | Bottleneck |
|--------------|---------|---------|------|-----------|
| HF + MTP, 8 steps | 0.904 | ~1.2 | ~0.97s | LLM (HF decode) |
| vLLM eager, 8 steps | 0.904 | ~1.2 | ~1.17s | LLM (no CUDA graphs) |
| vLLM CUDA graphs, 8 steps | 0.418 | 0.680 | 0.86s | flow+HiFT |
| vLLM CUDA graphs, 4 steps | 0.374 | 0.486 | 0.65s | flow+HiFT (fixed overhead) |
| vLLM CUDA graphs, 4 steps + turn-0-only first_chunk | 0.303 | 0.445 | 0.65s | intermediate |
| vLLM CUDA graphs, 4 steps + skip final-partial finalize=False | 0.265 | 0.405 | 0.64s | flow+HiFT (pre-AWQ bf16 best) |
| vLLM AWQ-INT4 + CUDA graphs, 4 steps, both fixes | 0.208 | 0.343 | 0.66s | flow+HiFT (pre-MeanFlow best) |
| **vLLM AWQ + MeanFlow (Chatterbox drop-in), 1 step, both fixes** | **0.164** | **0.261** | **0.49s** | **🏆 flow+HiFT collapsed (no CFG)** |

**Why FLOW_STEPS=4 gives diminishing returns:** halving steps from 8→4 saved only ~0.10s/call (4%) on warm flow calls. Most of the 2.2s/call is fixed overhead — encoder conditioning, mel feature extraction, HiFT vocoder — not the ODE step count. Further step reduction (2 steps) would give minimal benefit. The next lever is either fewer flow calls (larger chunk, higher TTFA) or flow architecture changes.

### Per-chunk flow vs HiFT breakdown (chunk=150, vLLM CUDA graphs, FLOW_STEPS=4)

Measured via `PROFILE_FLOW_STAGES=1` on the 6-turn long dialogue (31s audio, 20 flow calls):

| Component | Avg per call | Share | Notes |
|-----------|-------------|-------|-------|
| Flow (encoder + CFM) | 0.260s | **84%** | scales weakly with sequence length |
| HiFT vocoder | 0.049s | **16%** | **constant regardless of mel_frames** |

**HiFT is memory-bandwidth saturated, not compute-bound.** Measured HiFT time: 0.046–0.049s for 172 mel frames, 0.046–0.048s for 344–390 mel frames — completely flat. The GPU finishes convolutions faster than memory can feed them; doubling input size doesn't double time.

**Implication for next-step strategies:**
- Windowed/incremental HiFT: saves ~0.046s × 20 calls = ~0.9s total (**9%** of wall) — not worth the complexity
- HiFT fp16: halving HiFT time saves ~0.023s × 20 calls = ~0.5s (**5%**) — marginal
- Larger chunk size (250–400 tokens): halves flow call count, saves ~2–3s (**20–30%**) — free config change
- Incremental flow (encoder KV cache + ODE state reuse): targets the 84% share — high value, but **attempted and rejected**: introduced ~2.5 dB extra dynamic range on streaming output vs the bidirectional sync path. See [Flow chunk cache experiment, rejected](#flow-chunk-cache-experiment-rejected).

### MeanFlow flow — direct drop-in from Chatterbox (no distillation needed)

The user-flagged hypothesis turned out correct: Chatterbox's `s3gen_meanflow.safetensors` is **architecturally + bitwise compatible** with our `CausalMaskedDiffWithXvec` flow. No distillation, no retraining — just convert the safetensors into our `flow.pt` format and drop it in.

**Key compatibility:** their `flow.*` subset of the safetensors has 1121 keys all matching ours exactly with zero shape mismatches, plus one extra key `decoder.estimator.time_embed_mixer.weight (1024, 2048)` — the MeanFlow time-embedding fuser, a single bias-free `nn.Linear(2*time_embed_dim, time_embed_dim)`. The earlier port had this module as a 2-layer SiLU MLP (wrong guess); fixing it to match the upstream single-Linear layout makes `strict=True` loading work.

**End-to-end TTFA / RTF via `/generate-stream` (vLLM AWQ + CUDA graphs, RTX 3090, chunk=150, first_chunk=4):**

| Config | Dialogue | TTFA | Wall | RTF |
|---|---|---|---|---|
| AWQ + CFM, FLOW_STEPS=4 (previous best) | short ~6.5 s | 0.66 s | 2.26 s | 0.343 |
| **AWQ + MeanFlow, FLOW_STEPS=1** | short ~6.5 s | **0.49 s** | **1.68 s** | **0.261** |
| AWQ + CFM, FLOW_STEPS=4 (previous best) | long ~32 s | 0.67 s | 6.53 s | 0.208 |
| **AWQ + MeanFlow, FLOW_STEPS=1** | long ~32 s | **0.49 s** | **5.36 s** | **0.164** |

**Improvement:** -24% short RTF, -21% long RTF, -26% TTFA. Long content now runs **6.1× real-time** with sub-500 ms TTFA. All measured via `scripts/inference/bench_stream.py` against the production `/generate-stream` endpoint with the same dialogue and seed used for previous baselines.

**Flow+HiFT direct timing** (no LLM, same speech tokens replayed through each flow, fp16):

| Config | Total flow+HiFT | Speedup vs CFM-4 |
|---|---|---|
| CFM, steps=4 (previous default) | 1070 ms | 1.00× |
| **MeanFlow, steps=1** | **374 ms** | **2.86× (-65%)** |
| MeanFlow, steps=4 | 825 ms | 1.30× (-23%) |

The 2.86× flow-only win is much larger than the ~10% I initially predicted because **MeanFlow drops CFG entirely**. CFM's `solve_euler` doubles the batch on every estimator call for classifier-free guidance (uncond + cond). MeanFlow's `basic_euler` does a single forward per step with no doubling — distilled students bake CFG in during training. Net per-flow-call: 4×2=8 estimator forwards under CFM-4, 1 under MeanFlow-1. End-to-end the ~20% wall reduction is less than the 65% flow-only number because LLM, encoder, and HiFT are unchanged.

**Cross-lingual transfer works.** Chatterbox-Turbo is English-only by their README, yet the converted weights produced intelligible Cantonese on our 4-turn dialect dialogue (user-confirmed listening test). The s3tokenizer is content-agnostic enough that flow weights trained on English speech still map to reasonable Chinese mels — same phonemes, same speaker conditioning interface, same mel scale.

**Quality caveats (real, need follow-up):**
- MeanFlow output is ~1.5 dB hotter overall than CFM baseline on the same tokens.
- At FLOW_STEPS=1, turn 0 peaks at 0.99 — near clipping. The other turns peak 0.58-0.70 (sane). Per-turn peak spread is 4.62 dB vs CFM's 2.10 dB.
- At FLOW_STEPS=4, MeanFlow output matches CFM levels closely (peaks 0.55-0.71, RMS within 0.5 dB) but only delivers 1.30× speedup.
- Speaker similarity not yet measured against a ground-truth reference — Chatterbox's training data used different speakers and possibly a different CAMPPlus checkpoint. Our `campplus.onnx` produces embeddings that the MeanFlow flow conditions on, but whether the resulting voice matches the prompt speaker tightly needs a SECS / cosine-sim test.

**Recommended config for shipping:**
- `FLOW_STEPS=4` with MeanFlow weights (1.30× faster, audio matches CFM, no post-processing needed), OR
- `FLOW_STEPS=1` with a 0.85 peak limiter applied post-vocoder (2.86× faster, audio intelligible, transient peaks tamed). The +1.5 dB overall loudness can be normalised the same way.

Either way, the per-piece streaming loudness artifacts (turn-0 soft-hard, end-of-turn hot flush) **still apply** — they're orchestration-level (chunked finalize=False + flush mismatch), independent of the underlying flow model. Bumping `STREAM_CHUNK_SIZE` 150 → 250 + dropping the turn-0 first piece are still the right separate fixes.

**Files:**
- [scripts/inference/convert_chatterbox_meanflow.py](scripts/inference/convert_chatterbox_meanflow.py) — extracts `flow.*` keys from the safetensors, strips the `flow.` prefix, saves as `flow.pt`-compatible torch checkpoint. Verifies meanflow marker + cross-checks against existing CFM checkpoint.
- [scripts/inference/test_meanflow_inference.py](scripts/inference/test_meanflow_inference.py) — end-to-end test on the 4-turn Cantonese dialogue. Asserts auto-detected meanflow=True, reports per-turn peak/RMS.
- [scripts/inference/meanflow_smoke_test.py](scripts/inference/meanflow_smoke_test.py) — unit-level: module construction, backward-compat CFM load, basic_euler no-CFG batch check.

**To deploy MeanFlow as default:**
```bash
# Download Chatterbox MeanFlow weights (one-time)
huggingface-cli download ResembleAI/chatterbox-turbo s3gen_meanflow.safetensors --local-dir tmp/chatterbox/

# Convert
python scripts/inference/convert_chatterbox_meanflow.py \
  --src tmp/chatterbox/s3gen_meanflow.safetensors \
  --dst <model_dir>/flow.pt \
  --reference <original-model_dir>/flow.pt  # sanity check

# Restart service — auto-detect kicks in via SoulXPodcastService._load_model
```

### AWQ-INT4 quantization — shipped

**Result: -22% long RTF, -15% short RTF, TTFA unchanged.** Same dialogue + seed, vLLM `awq_marlin` kernel, fp16 dtype (auto-detected from `quantization_config.quant_method` in `config.json`).

| Metric | bf16 | AWQ-INT4 | Δ |
|---|---|---|---|
| Long RTF (~32s audio) | 0.265 | **0.208** | **-22%** |
| Short RTF (~6.5s audio) | 0.405 | **0.343** | **-15%** |
| Wall long | 8.49 s | 6.53 s | -23% |
| Wall short | 2.62 s | 2.26 s | -14% |
| TTFA (warm) | 0.64 s | 0.66 s | +0.02s (noise) |
| Model weight GPU memory | 3.86 GiB | **1.91 GiB** | **-50%** |
| KV cache headroom | 10.93 GiB | 12.88 GiB | +18% |
| vLLM engine init (incl. graph capture) | 42.5 s | 29.8 s | -30% |

vLLM logs `awq_marlin.py:117 The model is convertible to awq_marlin during runtime. Using awq_marlin kernel`. No code change beyond the auto-detect block already present in [llm_engine.py](soulxpodcast/engine/llm_engine.py) (commit b0ef599+). Calibration: 256 SoulX-formatted prompts (text + speech tokens) from the local training dataset, q_group_size=128, GEMM version. See [scripts/inference/quantize_awq.py](scripts/inference/quantize_awq.py).

**Audio quality A/B (Cantonese, seed=42, /generate-stream):** per-segment loudness spread is **identical** to bf16:

| Path | segs | rms σ/μ | max/min rms | overall peak |
|---|---|---|---|---|
| bf16 Cantonese stream | 7 | 0.23 | 6.2 dB | 0.523 |
| AWQ  Cantonese stream | 8 | 0.24 | 6.2 dB | **0.990** |

AWQ does NOT regress streaming dynamic range. The peak being ~5.5 dB hotter (0.99 vs 0.52) is the only audible difference — same relative loudness variation rides on a louder signal, which makes the variation feel more pronounced. This is the LLM's logit distribution shifting slightly under int4 rounding so the emitted speech tokens drive the (unchanged) bf16 flow harder. Listening A/B at matched peak is essentially identical.

**Reproducibility note:** the AWQ checkpoint's heavy-file symlinks (`flow.pt`, `hift.pt`, `campplus.onnx`, `flow.cache.pt`, `flow.decoder.estimator.fp32.onnx`) point at the absolute path they were created under (`/notebooks/projects/SoulX-Podcast/pretrained_models/...`). The [docker-compose.bench.yml](docker-compose.bench.yml) overlay bind-mounts the host `pretrained_models/` tree at that absolute path so the symlinks resolve inside the container. Rolling AWQ into `docker-compose.yml` as the default needs the same mount or relative symlinks.

### vLLM-side LLM optimization — exhausted on Ampere

After shipping AWQ + MeanFlow, the LLM is now ~75-85% of long-content wall (~3-4 s of the 5.36 s total). Standard vLLM tuning knobs were exercised against the prod stack (AWQ + MeanFlow + CUDA graphs + chunk=150, RTX 3090). **None deliver a measurable win on this hardware.**

Baseline reference (this section, fresh container instance, N=4 warm avg):

| Path | Long RTF | Short RTF | TTFA | Cold run 1 |
|---|---|---|---|---|
| FlashAttention + fp16 KV (default) | **0.169** | **0.263** | **0.49 s** | TTFA 1.06 s |
| fp8_e5m2 KV + XFormers (FA falls back) | 0.169 | 0.250 | 0.48 s | wall 6.41 s |
| fp8_e5m2 KV + FlashInfer (vLLM's recommended fp8 path) | **0.180** | 0.259 | 0.49 s | **TTFA 28.3 s** |

**Three structural reasons no win is available on Ampere:**

1. **FlashAttention is already auto-selected.** vLLM logs `Using Flash Attention backend.` on Ampere fp16 with `head_dim=128`. Nothing to switch to that's faster.

2. **fp8 KV cache is incompatible with FlashAttention.** vLLM explicitly errors: `Cannot use FlashAttention backend for FP8 KV cache.` Falls back to either XFormers (wash on warm, much worse cold) or FlashInfer (slower on warm, **catastrophic 28 s cold TTFA** from FlashInfer JIT compilation). On Hopper/Ada (sm_89+) the FA-fp8 path exists with native hardware support; **RTX 3090 (sm_86) has no fp8 hardware** so the dequant kernel overhead eats the bandwidth saving.

3. **AWQ INT4 already extracted the weight-read bandwidth win.** Current ~3.0 ms/token is within ~17% of the theoretical floor (1.91 GiB weights / 936 GB/s mem bandwidth = 2.0 ms minimum). The remaining 17% gap is Python + scheduler + sampling overhead — not addressable by vLLM env flags.

**Chunked prefill — measured, kept (-4% long RTF):**

Benched after the earlier "no win" conclusion when the user pushed back. With `enable_chunked_prefill=True` (vLLM default `max_num_batched_tokens=2048`), N=6 long-content runs (1 cold + 5 warm):

| Config | Long RTF | Long TTFA | Long wall | Notes |
|---|---|---|---|---|
| Chunked prefill OFF (V0 default) | 0.169 | 0.51 s | 5.32 s | baseline |
| **Chunked prefill ON, max_batch=2048 (default)** | **0.162** | **0.49 s** | 5.29 s | **-4% RTF, -4% TTFA** |
| Chunked prefill ON, max_batch=8192 | 0.171 | 0.50 s | 5.33 s | regression — too few scheduler steps |

The win comes from the scheduler being able to interleave a new-turn prefill with the previous turn's decode tail across the multi-turn dialogue. Single-request, but multi-turn → real overlap potential the V0 scheduler exploits when chunked. Larger `max_num_batched_tokens` removes the chunking benefit (back to baseline) — vLLM docs confirm: *"Smaller values (e.g., 2048) achieve better inter-token latency."* Default is optimal.

**Now shipped** as `VLLM_ENABLE_CHUNKED_PREFILL=true` in `docker-compose.yml` and `.env.serve.example`. Audio quality unchanged (Cantonese sample at [outputs/chunked_prefill_bench/awq_meanflow_chunked_cantonese.wav](outputs/chunked_prefill_bench/awq_meanflow_chunked_cantonese.wav), peak 0.70, finite, intelligible).

**Other knobs that were considered and skipped without measurement:**

| Knob | Why not | Notes |
|---|---|---|
| `block_size` (KV cache block size) | Single-request decode; default 16 has negligible indirection overhead | Larger blocks waste memory at sequence ends |
| `max_num_seqs` | Affects concurrent throughput, not single-request latency | Already at default |
| `--num-scheduler-steps` | V0 engine; not available in our V0 path | Would need V1 engine + RAS patch rewrite |
| FlashInfer attention (with fp16 KV) | V1 backend; our RAS patches target V0 | Switching engine versions is a multi-day port |

**Infrastructure added anyway** (so the knobs are available when the hardware is):
- `VLLMEngine.__init__` reads `VLLM_KV_CACHE_DTYPE` env (`auto` / `fp8_e5m2` / `fp8_e4m3` / `int8`) and threads it into `EngineArgs(kv_cache_dtype=...)`. Empty/unset = `auto` (matches model dtype).
- `docker-compose.yml` exposes the env var passthrough. **No default value** so vLLM auto-picks correctly.
- `docker-compose.bench.yml` adds `VLLM_ATTENTION_BACKEND` passthrough (bench-only — vLLM rejects an empty-string value, so production compose deliberately omits it).

When the deployment GPU moves to Ada/Hopper, `VLLM_KV_CACHE_DTYPE=fp8_e5m2` is expected to deliver a real ~10-20% decode-time speedup with native FA-fp8 — re-run the bench above to confirm.

### Next LLM lever: speculative decoding

The only remaining LLM-side lever that doesn't require new hardware is **speculative decoding** with a small draft model (e.g. Qwen3-0.6B). vLLM 0.10 supports it natively via `speculative_config=`. Realistic gain: **1.5-2× LLM throughput → RTF ~0.10-0.13 on long content.** Stacks on AWQ + MeanFlow.

The codebase already has a P-EAGLE speculator experiment (see `experiment/vllm-peagle-speculator` branch + `peagle-runtime-contract-mismatch.md` memory). Runtime acceptance was 24% vs train/val 65%; the cached-hidden-state A/B between PyTorch and vLLM in-process drafter forward is the decisive next test. Alternatively, train a fresh small draft from scratch — single-day job once data + recipe are in place.

After speculative decoding lands, Phase 0 is genuinely exhausted: the next wall reductions require either training (Phase 1 token-interleaved bi-streaming, Phase 2 sequential MTP) or hardware (sm_89+ for fp8 path).

### Residual loudness inconsistency — two streaming-specific artifacts, both in piece boundaries

After ripping out the flow chunk cache AND switching to AWQ, listening A/B on the streamed output exposed two specific complaints:

> Loudness problem only on turn 0; almost all streams have some extent of louder ending.

Per-piece RMS diagnostic ([`scripts/inference/flow_piece_diagnostic.py`](scripts/inference/flow_piece_diagnostic.py)) on a 4-turn Cantonese dialogue with the same LLM-generated tokens replayed through both paths (chunk=150, first_chunk=4) confirms both:

**Per-piece loudness within each turn — turn 0 stands out:**

| Turn | piece | dur | peak | rms_dB | comment |
|---|---|---|---|---|---|
| **0** | c0_nf4 | 0.04 s | **0.183** | **-33.4** | ← near-silent blip (4 tok, 3 dropped as lookahead) |
| **0** | c1_nf150 | 6.00 s | 0.501 | -25.6 | normal body |
| **0** | flush | 5.48 s | **0.689** | -25.6 | ← hot peak (+3 dB vs body) |
| 1 | c0_nf150 | 5.88 s | 0.441 | -24.5 | |
| 1 | c1_nf150 | 6.00 s | 0.521 | -25.5 | |
| 1 | flush | 0.40 s | 0.127 | -36.3 | trailing silence |
| 2 | c0_nf150 / c1 / flush | — | 0.34-0.48 | -26.3 ↔ -25.9 | flat ✓ |
| 3 | c0_nf150 | 5.88 s | 0.380 | -25.6 | |
| 3 | flush | 0.80 s | 0.293 | **-23.1** | ← hot rms (+2.8 dB) |

Turn 0 has **11.5 dB peak swing across pieces** (0.183 → 0.689). Other turns: 1-3 dB.

**Two artifacts, two separate root causes:**

1. **Turn-0 soft-hard pattern is caused by `first_chunk_size=4`.** It was added (fix #12 above) to keep TTFA at 0.64 s, but emits a 40 ms piece with only 4 tokens (3 of which are lookahead-dropped), then jumps straight to a normal-loudness 6 s body. Perceptually: speech starts soft then suddenly becomes loud. **Only turn 0 is affected** because `effective_first_chunk_size = first_chunk_size if turn_i == 0 else chunk_size`.

   Sweeping `first_chunk_size` on turn 0:
    | first_chunk | turn-0 peak | turn-0 rms | piece0 peak | piece0 rms | verdict |
    |---|---|---|---|---|---|
    | (sync) | 0.859 | -24.4 | — | — | reference |
    | **4** (current) | 0.655 | -25.5 | 0.116 | **-37.4** | silent first piece |
    | 16 | **0.935** | -24.1 | **0.703** | **-19.1** | ⚠️ catastrophic — piece0 +5 dB hotter than body |
    | 50 | 0.796 | -25.0 | 0.491 | -24.1 | ✓ piece0 matches body |
    | 100 | 0.607 | -25.4 | 0.456 | -26.2 | ✓ |
    | 150 | 0.691 | -25.0 | 0.585 | -25.7 | ✓ (no special first chunk) |

   first_chunk=16 is **worse than the current 4** — piece0 lands inside a flow-internal boundary the chunk-masked attention wasn't built for. first_chunk ≥ 50 normalises turn 0.

2. **End-of-turn hot flush is real for some turns, not all.** Body-vs-flush per turn:
    | Turn | flush dur | body→flush Δrms dB | body→flush Δpeak dB |
    |---|---|---|---|
    | 0 | 5.5 s | +0.86 | **+3.09** |
    | 1 | 0.4 s | -11.31 (silence) | -10.98 |
    | 2 | 0.7 s | +0.62 | -4.15 |
    | 3 | 0.8 s | **+2.77** | -1.74 |

   The `finalize=True` flush re-runs flow on the full sequence with a different attention mask than the chunked `finalize=False` body. When the flush size is large (turn 0: 5.5 s of audio because chunk math left a 134-token trailing partial), the mismatch in mel energy is audible. When the flush is trailing silence (turn 1), it's invisible.

**Earlier finding (chunk-size sweep, sync vs stream peak range) was directionally right but understated:**

| Path | per-turn RMS spread | per-turn peak spread |
|---|---|---|
| SYNC (single flow call/turn) | 0.89 dB | 2.10 dB |
| STREAM chunk=50 | 1.35 dB | 6.24 dB |
| STREAM chunk=150 (default) | 0.59 dB | 5.86 dB |
| STREAM chunk=250 | 0.55 dB | 2.98 dB |

The peak-range improvement at larger chunks is real but partly because larger chunks shift the partial/flush balance — not a clean "per-chunk finalize=False is the only cause" story.

**Other evidence (still valid):**
- bf16 and AWQ produce identical per-segment streaming spread (rms σ/μ ≈ 0.23). Quantization not the cause.
- Within-turn loudness variation (6-12 dB segment-level spread on `gen_stream_vs_sync.py`) is intrinsic to the LLM's token sequence — prosody, present in both sync and stream.
- Cache fully removed: `grep` for `flow_chunk_cache | FLOW_CHUNK_CACHE | chunk_cache | encoder_kv_cache | conv_cache | ode_state` in [soulxpodcast/](soulxpodcast/) + [api/](api/) → 0 matches. Commit 8d80db0 deleted 125 lines from [flow.py](soulxpodcast/models/modules/flow.py), 280 from [estimator.py](soulxpodcast/models/modules/flow_components/estimator.py), 113 from [upsample_encoder.py](soulxpodcast/models/modules/flow_components/upsample_encoder.py).

**Fix candidates:**

| Fix | Addresses | Cost | Effort |
|---|---|---|---|
| (A) `STREAM_FIRST_CHUNK_SIZE` 4 → 50 | turn-0 soft-hard | TTFA 0.64 s → ~1.5 s | env flip |
| (B) Drop/silence first piece on turn 0 in `forward_longform_streaming` | turn-0 soft-hard | TTFA unchanged | small code change |
| (C) `STREAM_CHUNK_SIZE` 150 → 250 | reduces flush size (no large trailing partial) + general peak spread | TTFA unchanged, RTF likely improves (fewer flow calls) | env flip |
| (D) Post-vocoder peak limiter on flush piece | end-of-turn hot peak | adds 10-20 ms processing | code change |
| (E) Replace flow architecture (e.g. MeanFlow / 1-step CFM) | both, indirectly | retraining + audio quality risk | research |

**Recommendation:** combine **(B) + (C)** — drop the first piece's audio on turn 0 (it's only 40 ms of mostly-lookahead-dropped garbage; nobody hears anything useful in it) AND bump `STREAM_CHUNK_SIZE` to 250 (TTFA-neutral, eliminates the 5 s turn-0 trailing partial). Then listen — if the end-of-turn hotness still bothers, add (D).

### Flow chunk cache experiment, rejected

We implemented per-chunk K/V + causal-conv caching across the encoder, U-Net decoder, and per-ODE-step estimator (commits 8aeb00f / 8f78568, reverted by [this rip-out commit]). The standalone flow+HiFT bench showed -50.6% wall on long content. End-to-end gain was only ~6.5% because B3 already overlapped LLM and flow.

**Audio quality regression discovered after deployment.** A/B comparison of `/generate` (sync, full bidirectional flow) vs `/generate-stream` (chunked + cached) on the same dialogue + seed (`scripts/inference/gen_stream_vs_sync.py`):

| Path | p90-p10 dB spread | max-min dB |
|---|---|---|
| zh_sync | 5.51 | 9.69 |
| zh_stream (cache **on**) | **8.05** | **14.75** |
| zh_stream (cache **off**) | 5.72 | 9.52 — matches sync ✓ |
| yue_sync | 5.45 | 12.05 |
| yue_stream (cache **on**) | **6.79** | **15.71** |
| yue_stream (cache **off**) | 5.90 | 12.93 — matches sync ✓ |

The cached chunked flow produces mels with ~2.5 dB extra dynamic range vs the bidirectional reference. Root cause: the cached path approximates bidirectional self-attention with chunk-causal K/V accumulation, an attention pattern the model was never trained on. The 0.991 cosine similarity smoke test was vs `streaming=True` (chunk-masked) — both diverge from the actual bidirectional reference. **Trade-off: 6.5% end-to-end RTF win for audibly worse audio is not worth it.** Ripped out.

## Key findings — AWQ + vLLM CUDA graphs dominate; flow+HiFT is the only remaining bottleneck

**AWQ-INT4 served under vLLM `awq_marlin` is the dominant mode.** Long-content RTF dropped from 0.265 (bf16) to 0.208 with no audio quality regression. LLM is now a thin slice of total wall; flow+HiFT is the only remaining lever.

0. **AWQ-INT4 quantization is shipped.** -22% long RTF, -15% short RTF, TTFA unchanged. Halves model VRAM (3.86 → 1.91 GiB). vLLM auto-detects the `quantization_config` marker and picks `awq_marlin` + fp16 — no code change. Audio quality A/B on Cantonese (same seed) shows identical per-segment loudness spread to bf16 (rms σ/μ ≈ 0.23). The only audible difference is a hotter overall peak (0.99 vs 0.52); apply post-vocoder gain if matching bf16 levels is important. See [AWQ-INT4 quantization — shipped](#awq-int4-quantization--shipped).

1. **`VLLM_ENFORCE_EAGER=false` is the key unlock.** All prior vLLM tests had `enforce_eager=True` hardcoded in service.py — they tested vLLM eager mode, not CUDA-graph mode. With graphs enabled, vLLM (no MTP) achieves RTF=0.418 vs HF+MTP RTF=0.904. MTP is no longer needed for RTF < 1.

2. **Bi-streaming reduces TTFA dramatically (3.5×) but adds 7-50% total-wall regression** depending on chunk size. The regression is from per-chunk flow+HiFT overhead (~0.3s/call at chunk=150) compounding across chunks.

3. **The flow architecture has no inter-call state cache.** The existing `streaming=True` flag only enables chunk-masked attention within a single call. Building true incremental streaming requires modifying `flow.py` + `flow_components/estimator.py` to add encoder KV cache + decoder feature cache + `solve_euler` state — a multi-day refactor with audio-quality risk. Not worth it given the flow call count is already minimised by large chunk sizes.

4. **Separate CUDA streams (B3) give ~7% wall improvement** by letting LLM and flow overlap on the GPU. Win is modest because both are memory-bandwidth bound on the 3090. B3 is always active in `forward_longform_streaming` — no action needed. Would scale better on H100/H200.

5. **Chunk size is the main TTFA/wall lever:** chunk=50 minimises TTFA, chunk=150 minimises wall. Flow cost per call is ~0.26s (84% of call time) and grows weakly with sequence length; HiFT is constant at ~0.05s/call regardless of chunk size. Fewer chunks → less total overhead. Default of 100 balances TTFA and wall time.

6. **MTP is now optional** — it helps HF-engine RTF (from ~0.9 to ~0.9 on long content), but vLLM+CUDA graphs already achieves RTF=0.418 without MTP. MTP cannot be combined with vLLM (service enforces HF engine when MTP_CHECKPOINT is set).

7. **RTF < 1 requires long content to amortise flow overhead.** Short dialogues (~6s) yield RTF ~0.49 (vLLM CUDA graphs, 4 steps) to ~1.2 (HF). Long dialogues (~30s) reach RTF ~0.90 (HF) or 0.374 (vLLM CUDA graphs, 4 steps).

10. **FLOW_STEPS=4 gives ~11% RTF improvement over 8 steps** with no audible quality difference on the test dialogue. Audio quality was A/B checked — both 19s samples identical in content. Halving steps saves only ~0.10s/call on warm flow calls (4% per-call improvement) because flow time is dominated by fixed per-call overhead (encoder, vocoder), not ODE step count. Reducing steps further would give negligible gain. `FLOW_STEPS=4` is now the recommended default.

8. **Restricted speech-vocab sampler (RESTRICT_SPEECH_VOCAB) gives ~1.5% HF speedup — not worth it.** Measured: ~0.56ms/token saving on the lm_head projection (160K→6561 vocab), totalling ~420ms on a 30s dialogue. This is unmeasurable noise relative to backbone computation (~35ms/token). Root cause: Qwen3-1.7B is memory-bandwidth bound; the backbone FFN/attention dominates, not the lm_head. `RESTRICT_SPEECH_VOCAB` is disabled by default.

9. **TRT_ESTIMATOR is NOT beneficial for this model.** TRT 11 removed global `FP16`/`EXPLICIT_BATCH` flags; per-layer FP16 insertion adds type-conversion overhead. Measured: TRT TF32 RTF=1.048, TRT FP16 (per-layer) RTF=1.021 — both worse than PyTorch native FP16 (RTF=0.904). 285MB plan, 23.6s build. Root cause: PyTorch's cuBLAS FP16 path is already well-optimised for this network; TRT adds per-call address-binding overhead that dominates for small batch (B=2) inference. `TRT_ESTIMATOR` is disabled in `docker-compose.dev.yml`.

13. **Skip `finalize=False` on the final partial chunk for turns > 0.** For any turn whose total tokens are not a multiple of `chunk_size`, `iter_chunks` yields a trailing partial chunk. The loop synthesises it with `finalize=False` (paying a full ~0.23s Flow call for 3-token-lookahead-stripped audio), then immediately the `finalize=True` flush synthesises the same tokens again including the 3 lookahead tokens. Two Flow calls for one partial. Fix: `iter_chunks(yield_final_flag=True)` signals which yield is the trailing partial; for turns > 0 we `break` on the final partial without calling `finalize=False` — `finalize=True` absorbs the partial in one call. Saves ~1 Flow call per turn for turns > 0 whenever tokens % chunk_size ≠ 0. Combined with fix #12: 20→10 Flow calls on a 6-turn long dialogue; RTF 0.374→0.262 (long), 0.486→0.391 (short), TTFA unchanged at 0.64s. See `soulxpodcast/models/soulxpodcast.py` + `soulxpodcast/utils/streaming.py`.

12. **`first_chunk_size` should only apply to turn 0.** With `first_chunk_size=4` applied to every turn, turns 2–N each pay a full Flow call (~0.23s) to emit only 2 mel frames (~40ms audio). These calls are pure waste — later turns are already playing previous-turn audio so there is no TTFA to optimise. Fix: `effective_first_chunk_size = first_chunk_size if turn_i == 0 else chunk_size`. Result: 20→15 Flow calls on a 6-turn long dialogue, RTF 0.374→0.303 (long), 0.486→0.445 (short), TTFA unchanged at 0.65s. Implemented in `forward_longform_streaming`. See `soulxpodcast/models/soulxpodcast.py`.

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
| Skip tiny first-chunk on turns 2+ | RTF 0.374→0.303 long, 0.486→0.445 short | ✅ yes |
| Skip final-partial finalize=False on turns 2+ | RTF 0.303→0.262 long, 0.445→0.391 short | ✅ yes |
| AWQ-INT4 (`awq_marlin` kernel) | RTF 0.265→0.208 long, 0.405→0.343 short, -50% model VRAM | ✅ yes — new default |
| MeanFlow flow weights (Chatterbox drop-in) | end-to-end RTF 0.208→0.164 long, 0.343→0.261 short, TTFA 0.66s→0.49s; flow+HiFT 2.86× faster at FLOW_STEPS=1 | ✅ yes — new default; audio intelligible cross-lingually; ~1.5 dB hotter, recommend post-vocoder peak limiter |
| fp8_e5m2 KV cache (Ampere) | wash on warm (XFormers fallback) or -7% (FlashInfer); cold TTFA 28s with FlashInfer | ❌ not on Ampere — fp8 + FA incompatible, no native fp8 hw on sm_86. Plumbed via `VLLM_KV_CACHE_DTYPE` for future Ada/Hopper deploy. |
| Chunked prefill (`enable_chunked_prefill=True`, default `max_num_batched_tokens=2048`) | RTF 0.169→0.162 long (-4%), TTFA 0.51s→0.49s (-4%), short unchanged | ✅ yes — new default. Multi-turn dialogue lets the scheduler interleave new-turn prefill with previous-turn decode tail. |
| Flow chunk cache (per-chunk K/V + conv cache) | -50.6% flow+HiFT wall long, but +2.5 dB output dynamic range vs sync | ❌ rejected — audio quality regression |
| FLOW_STEPS 4→2 | ODE steps are only ~5% of call time | ❌ negligible |

**Current best:** RTF=**0.162** (long, ~32 s audio, **6.2× real-time**) / RTF=**0.261** (short, ~6.5 s audio) / TTFA=**0.49 s**. Config: `LLM_ENGINE=vllm VLLM_ENFORCE_EAGER=false FLOW_STEPS=1 STREAM_CHUNK_SIZE=150 VLLM_ENABLE_CHUNKED_PREFILL=true MODEL_PATH=<awq+meanflow-checkpoint>`. Auto-detected MeanFlow path: the SoulXPodcastService sniffs `flow.pt` keys at startup; presence of `decoder.estimator.time_embed_mixer.weight` switches the model to MeanFlow inference automatically (no env var, just point MODEL_PATH at a checkpoint converted via `scripts/inference/convert_chatterbox_meanflow.py`).

## Implications for PLAN.md phases

- **Phase 0 inference optimization has hit a quality floor.** Flow chunk cache was tried and rejected. AWQ-INT4 is now shipped (-22% long RTF, no quality regression). Further flow-side optimizations are unlikely to yield wall savings without quality regression.
- **LLM is now ~75-85% of wall** after MeanFlow collapsed the flow share to ~20% (mostly hidden under LLM via B3 dual streams). vLLM-side env knobs are exhausted on Ampere — fp8 KV / FlashInfer benched and rejected (see [vLLM-side LLM optimization — exhausted on Ampere](#vllm-side-llm-optimization--exhausted-on-ampere)).
- **Concrete next steps (must stay vLLM-compatible).** MTP forces a fallback to HF (RTF 0.904 long), which is a 5.5× regression on current AWQ+MeanFlow RTF 0.164 — net loss even after MTP's ~1.8×. The viable moves are:
  - **Speculative decoding** with a smaller draft model (e.g. Qwen3-0.6B). vLLM 0.10 supports it natively; typical 1.5–2× on memory-bandwidth-bound decode. Stacks on top of AWQ+MeanFlow → projected RTF ~0.10-0.13 long. The existing P-EAGLE branch is partially built but stuck on a runtime/train acceptance gap; either fix that or train a fresh tiny draft.
  - **Larger chunks (250–400 tokens):** halves flow call count, saves ~5-10% wall at the cost of higher TTFA. Free config change. Lower-priority now that flow is no longer the bottleneck.
  - **Token-interleaved bi-streaming (Phase 1)** — biggest swing but requires retraining; lets LLM and flow run at sub-token granularity instead of needing the B3 overlap to hide flow.
  - Larger vLLM batch sizes help concurrent-request throughput, not single-dialogue latency.
  - **Hardware upgrade to sm_89+ (Ada/Hopper)** unlocks native fp8 KV cache + fp8 GEMM. Plumbing already in place via `VLLM_KV_CACHE_DTYPE`.
- **The path to faster LLM goes through training.** Phase 1 (token-interleaved bi-streaming) and Phase 2 (Sequential MTP) both require retraining/fine-tuning.
- **Phase 2 (Sequential MTP) is the cheaper next move** — adds K-1 lightweight mixing layers, trains heads-only Medusa-1 style with base frozen. Realistic ~1.8× per-step speedup with prosody preserved. **Caveat:** MTP currently forces HF engine, losing the AWQ win. Would need vLLM MTP integration (or accept HF + MTP + AWQ-quantized HF weights) to net out positive.
- **Phase 1 (token-interleaved grammar) is the bigger swing** — needs forced-alignment data pipeline and full base-model retraining. Months of work. Only do it after Phase 2 proves insufficient.
- **Open quality work: two streaming-specific loudness artifacts.** AWQ ships clean. Per-piece RMS diagnostic isolates two artifacts: (1) turn-0 soft-hard transition caused by `first_chunk_size=4` emitting a 40 ms near-silent piece; (2) end-of-turn hot flush when the `finalize=True` re-run renders >1 s of trailing audio with a different attention mask than the chunked body. See [Residual loudness inconsistency — two streaming-specific artifacts, both in piece boundaries](#residual-loudness-inconsistency--two-streaming-specific-artifacts-both-in-piece-boundaries) for the full table and fix candidates. Recommended combo: drop/silence the turn-0 first piece (40 ms code change, TTFA-neutral) + bump `STREAM_CHUNK_SIZE` 150 → 250 (env flip, removes the 5 s flush case).

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
    - `gen_stream_vs_sync.py` — A/B `/generate` (sync) vs `/generate-stream` on the same dialogue+seed; flags any streaming-path audio regression (used to detect the flow-cache loudness issue).
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
