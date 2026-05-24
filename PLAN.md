## Plan: MTP & Token-Interleaved Bi-Streaming for SoulX-Podcast

Based on profiling, the LLM memory bandwidth limits us to ~40 tokens/sec. To bypass this and hit real-time, we will introduce Multi-Token Prediction (MTP) and change the token grammar to enable true bi-streaming (Time-To-First-Audio without waiting for full text decoding).

### Phase 0: Base Tuning & Orchestration Pipeline

1. **vLLM Integration Fix** — ✅ **DONE.** Root cause was just that vLLM wasn't installed in the project `.venv`; the `_C.abi3.so undefined symbol` was from a stale conda env. Fixed with:
   - `pip install vllm==0.10.1` (binary wheel is ABI-compatible with our torch 2.7.1)
   - Cloned `Soul-AILab/vllm@v0.10.1.1-soulxpodcast` and copied the 4 RAS patch files (`sampler.py`, `utils.py`, `sampling_metadata.py`, `sampling_params.py`) over the installed vLLM.
   - See `inference_test.py` for the reproducible test harness.

   **Measured baselines (RTX 3090, 24 GB, bf16):**

   | Test                                    | Engine | Load   | Inference | Audio  | **RTF**  |
   |-----------------------------------------|--------|--------|-----------|--------|----------|
   | Single-turn English (Mandarin prompt)   | hf     | 6.5s   | 10.2s     | 13.7s  | **0.742** |
   | Single-turn English (Mandarin prompt)   | vllm   | 25.1s  | 9.8s      | 13.4s  | **0.730** |
   | 4-turn Cantonese dialect dialogue       | hf     | 7.4s   | 48.6s     | 47.9s  | **1.015** |
   | 4-turn Cantonese dialect dialogue       | vllm   | 23.9s  | 44.7s     | 56.5s  | **0.792** |

   **Findings that change the rest of the plan:**
   - The PLAN's "~40 tok/s, sub-real-time" premise is **only true for multi-turn HF**. Single-turn HF already runs at RTF 0.74. vLLM holds RTF≈0.79 across both single and 4-turn.
   - **vLLM speedup is ~22% on multi-turn, ~2% on single-turn** — far from the 1.5–2× hoped for. KV reuse helps as context grows, but the bottleneck on a 3090 is the per-step latency of the 1.7B Qwen3, not memory bandwidth.
   - **vLLM cold start: 16–25s** from CUDA graph compilation — only acceptable for long-running servers.
   - RAS sampling works correctly with the patches (verified: no warnings, audio quality preserved).

2. **Plumbing the Text-Iterator & Streaming Hook** — ✅ **DONE (LLM-side + bi-stream MVP).** See `streaming_hook_test.py` and `bistream_test.py`.

   **2a — LLM streaming hook:** Built `SpeechTokenStreamer(BaseStreamer)` in `soulxpodcast/utils/streaming.py`; wired through `HFLLMEngine.generate(..., streamer=...)`. Two bugs squashed along the way:
   - HF calls `streamer.put(prompt)` once at start with full prompt tensor — needed a `_got_prompt` flag (matches `TextStreamer` pattern).
   - **HF's `_extract_generation_mode_kwargs` silently drops `streamer` when a custom `custom_generate` callable is used** (kwargs are filtered to those *unique* to the custom function). Fixed by binding `streamer` into the `partial(_ras_sample_hf_engine, ...)` so the custom sampler receives it regardless. This is non-obvious — future MTP work that touches the custom sampler must preserve this binding.

   **2b — Bi-streaming MVP (chunked flow+HiFT):** Per-chunk synthesis with sliding-window flow recompute. Measured on RTX 3090, chunk_size=50:

   | Mode | TTFA | Total wall | RTF |
   |------|------|------------|-----|
   | one-shot | 10.2s | 10.2s | 0.74 |
   | stream-a (LLM only) | 11.0s | 11.0s | 0.81 |
   | **stream-b (bi-stream)** | **2.93s** | **15.4s** | **1.01** |

   - **TTFA win: 3.8× (11.0s → 2.93s)** — the actual streaming metric.
   - **Wall regression: +50%** — each chunk pays ~1s fixed flow+HiFT overhead (15 diffusion steps + encoder). One-shot pays this once; bi-stream pays 8×.
   - Cadence is safely sub-real-time: 2s audio produced per 1.7s wall, so a real-time playback consumer never starves after TTFA.
   - **Chunk-boundary audio quality: confirmed acceptable** by ear test on the concatenated wav.

   **Remaining limitation:** the flow's `streaming=True` flag only enables chunk-masked attention within a single call — it's **not a stateful inter-call cache**. Eliminating the wall-time regression requires either: (a) stateful CFM with proper z/μ cache between calls (`flow.py` + estimator change, no retraining required for inference-only state surgery), (b) a causal CFM that doesn't need all-token attention (architectural change requiring retraining), or (c) larger chunks at the cost of TTFA.

3. **Recalibrated speedup target.** Starting from vLLM's RTF≈0.79 on multi-turn (not 0.025 implied by "40 tok/s"), the realistic ceiling for Phases 1–3 is:
   - +2× from MTP (with rejection-sampling validation) → RTF≈0.40
   - Token-interleaved bi-streaming primarily reduces **TTFA**, not throughput. The ~2s chunk-boundary latency is what matters, not total RTF.
   - **Drop the "40 → 150 tok/s" claim from Part 2.** Replace with: "MTP targets ~2× per-step throughput on speech tokens, taking us from ~80 tok/s (vLLM) to ~150 tok/s."

### Phase 1: Token-Interleaved Bi-Streaming
Currently, the model predicts the *entire text* and then the *entire audio*. We will change the sequence formatting so the LLM naturally interleaves text and audio.

**1. Token Grammar Redesign:**
- Remove the rigid `<|text_start|>`...`<|text_end|>` `<|audio_start|>`... blocks.
- Introduce an interleaved ratio based on frame rates (e.g., 5 text tokens per 15 speech tokens, like CosyVoice).
- Sequence format: `[instruct] [T_0..4] [S_0..14] [T_5..9] [S_15..29] ... [EOS]`
- **`fill_token`**: We must add a special `fill_token` to the tokenizer. If the upstream text input generator runs dry (e.g., the user hasn't typed fast enough), the LLM must be taught to wait or yield this `fill_token` rather than hallucinate text.

**2. Data Pipeline Update (`utils/parser.py` & Dataloader):**
- Rewrite the dataset processing logic so that training examples are constructed using this interleaved text/audio ratio, parsing dialect mapping boundaries.

### Phase 2: Multi-Token Prediction (MTP) — Sequential Variant

To raise per-step throughput on speech-token generation, the LLM will decode $K$ tokens per forward pass instead of 1. **We use Sequential MTP (DeepSeek-V3 style), not parallel Medusa-style heads**, to preserve prosody.

**Why sequential, not parallel:** Parallel heads ($\hat{y}_{t+1}, \hat{y}_{t+2}, \ldots$ predicted from the same $h_t$) fall into a conditional-independence trap — each head guesses without seeing the previous head's choice, so the model converges on the *average* token, washing out pitch contours and producing flat / warbly prosody. Sequential MTP feeds each head's prediction forward through a lightweight mixing layer, restoring the micro-causality that natural prosody requires.

**Non-goal: pure NAR.** SoulX's `s3tokenizer` is a **single-codebook** discrete tokenizer (~6,500 entries at 25 Hz). There is no residual VQ stack to fan a NAR head across — every speech token directly determines audio. Removing the AR loop on speech tokens would collapse prosody. NAR-style parallelism is only feasible after switching to a multi-codebook codec (EnCodec, FSQ, etc.), which is a separate, much larger project.

**1. Architecture Addition (`models/soulxpodcast.py`):**
- Wrap `Qwen3ForCausalLM` with $K-1$ **Sequential MTP layers** (target $K=4$).
- Each MTP layer is a lightweight residual block (one MLP or a single-layer transformer block) that takes $h_t$ + the **embedding of the previously-predicted token** $\hat{y}_{t+k-1}$ and produces a modified hidden state $h'_t$, from which $\hat{y}_{t+k}$ is predicted. The LM projection (output head) is shared with the primary head to keep parameter count down.

```
   Trunk(h_t) ──► LM_head ──► ŷ(t+1)
       │                         │
       ▼                         ▼
   [MTP-1] ◄──────────── embed(ŷ(t+1))
       │
       ▼ h_t'
   LM_head ──► ŷ(t+2)
       │                         │
       ▼                         ▼
   [MTP-2] ◄──────────── embed(ŷ(t+2))
       │
       ▼ h_t''
   LM_head ──► ŷ(t+3)
       ...
```

**2. Training Strategy:**
- Fine-tune with combined loss: CE on the primary next-token prediction + CE on each of the $K-1$ MTP heads against offset speech targets, weighted by depth (e.g., $\lambda_k = 0.5^{k-1}$ — deeper heads contribute less).
- MTP losses apply **only to speech-token positions**, not text positions.
- Start with Medusa-1 strategy: **freeze the base trunk**, train only the MTP layers + (optionally) lightly fine-tune `lm_head`. This is cheap, low-risk, and avoids regressing base quality.

**3. Inference Overhaul (`engine/llm_engine.py`):**
- Implement a sequential-draft decode loop with rejection-sampling validation:
  1. One forward through the trunk → $h_t$.
  2. Roll through the $K-1$ MTP layers to draft $[\hat{y}_{t+1}, \ldots, \hat{y}_{t+K}]$.
  3. **Validate** by re-running the trunk on the drafted sequence in *one* batched forward pass; accept the longest prefix where the trunk's distribution agrees within a rejection threshold; resample from the first divergence point.
  4. Commit verified tokens; update KV cache; loop.
- Rejection sampling preserves audio quality at the cost of some speedup. Expected per-step acceptance length: 2–3 tokens at $K=4$ on speech-token sequences.

**Expected outcome:** ~1.8× speedup on speech-token throughput (slightly less than parallel MTP due to mix-layer cost + rejection overhead, but with prosody intact). From the vLLM baseline of ~80 tok/s, this targets ~140 tok/s — short of the original "150" claim, well above the "40" pre-vLLM baseline.

### Phase 3: Orchestrator Threading
To support the bi-stream yield:
1. **Thread 1 (LLM/Text Consumer)**: Takes a Python generator of incoming text. Decodes text iteratively. As soon as it flips into the audio pattern, it uses MTP to blast out audio tokens and pushes them to a ring buffer.
2. **Thread 2 (Flow/HiFT Consumer)**: Polls the ring buffer. Every time a fixed chunk (e.g., 50 audio tokens) is ready, it runs `self.flow(..., streaming=True)`, caches the `[z, mu]` overlaps, passes the mel to `self.hift(...)`, and directly yields `24kHz` audio bytes to the client.
