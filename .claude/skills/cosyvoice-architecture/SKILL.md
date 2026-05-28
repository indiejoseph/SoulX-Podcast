---
name: cosyvoice-architecture
description: Deep architectural reference for CosyVoice / CosyVoice2 / CosyVoice3 TTS systems. Covers the full pipeline (Frontend → Qwen2 LLM → Flow/CFM → HiFi-GAN), tensor shapes at every boundary, special token semantics, streaming inference, bi-stream mode, and version differences. Use when implementing, debugging, extending, or integrating with CosyVoice models.
when_to_use: When working on anything related to CosyVoice TTS — model architecture, speech token generation, flow matching, vocoder integration, streaming inference, continuous speech latents, multi-token prediction for speech, or any CosyVoice-derived system.
user-invocable: true
---

# CosyVoice Architecture Reference

This skill provides comprehensive architectural knowledge of the CosyVoice family of TTS models (v1, v2, v3). Use this as ground truth when implementing, extending, or debugging CosyVoice-based systems.

---

## End-to-End Pipeline

```
Text ──► Frontend ──► Qwen2 LLM ──► speech tokens ──► Flow/CFM ──► mel ──► HiFi-GAN ──► waveform
              │              │                            │
       text normalize   spk embed (CAM++)           prompt mel + spk
       BPE tokenize     prompt speech tokens         (voice cloning)
       (Qwen2 tok)      (speech tokenizer ONNX)
```

Four stages run as separate neural networks. The LLM is the latency bottleneck.

---

## Stage 0: Frontend

**Class:** `CosyVoiceFrontEnd` in `cosyvoice/cli/frontend.py`

| Component | Implementation | Output |
|---|---|---|
| Text normalization | `ttsfrd` → `wetext` → raw fallback | Normalized UTF-8 |
| Text tokenizer | Qwen2 BPE tokenizer | `(B, L_text)` int token IDs |
| Speaker embedding | CAM++ ONNX (`campplus.onnx`) | `(B, 192)` x-vector |
| Speech tokenizer | `speech_tokenizer_v2.batch.onnx` (`s3tokenizer v2_25hz`) — Whisper 128-mel → discrete tokens | Token IDs @ **25 Hz**, range `[0, 6560]` |

The speech tokenizer is an opaque ONNX model (likely VQ-VAE internally). It maps Whisper 128-bin log-mel spectrograms to discrete tokens at **25 Hz** (confirmed in `cosyvoice2.yaml`: `token_frame_rate: 25`). There is no exposed `quantize()`/`dequantize()` in the Python code — quantization happens inside the ONNX graph.

---

## Stage 1: LLM — Text → Speech Tokens

**Class:** `Qwen2LM` (CosyVoice2), `CosyVoice3LM` (v3) in `cosyvoice/llm/llm.py`
**Backbone:** `Qwen2ForCausalLM` from HuggingFace `transformers`

### Dimensions

| Tensor | Shape | Notes |
|---|---|---|
| `llm_input_size` | **896** | Embedding dim fed into Qwen2 |
| `llm_output_size` | **3584** | Qwen2 hidden state dim |
| `speech_token_size` | **6561** | Discrete speech vocab (s3tokenizer v2, 25 Hz) |
| LM head output (v2) | **6564** | `speech_token_size + 3` |
| LM head output (v3) | **6761** | `speech_token_size + 200` |

### Input Sequence Construction

```
[SOS_emb] [spk_embed_proj] [text_token_emb] [TASK_ID_emb] [prompt_speech_token_emb]
```

| Segment | Embedding source | Shape |
|---|---|---|
| SOS, TASK_ID | `nn.Embedding(2, 896)` | `(B, 1, 896)` each |
| Text tokens | `Qwen2.model.embed_tokens` (tied to LM head) | `(B, L_text, 896)` |
| Speaker embed | 192-dim → `nn.Linear(192, 896)` | `(B, 1, 896)` |
| Speech tokens | `nn.Embedding(speech_token_size + 3, 896)` | `(B, L_speech, 896)` |

### Special Token IDs

**CosyVoice2 (`Qwen2LM`):**
- `SOS = 0`
- `TASK_ID = 1`
- `EOS = speech_token_size` (6561)
- `fill_token = speech_token_size + 2` (6563) — bi-stream padding
- LM decoder outputs `speech_token_size + 3 = 6564` classes

**CosyVoice3 (`CosyVoice3LM`):**
- `SOS = speech_token_size + 0` (6561)
- `EOS = speech_token_size + 1` (6562)
- `TASK_ID = speech_token_size + 2` (6563)
- `fill_token = speech_token_size + 3` (6564)
- LM decoder outputs `speech_token_size + 200 = 6761` classes
- `stop_token_ids = [6561 .. 6760]`

### Decoding

- Autoregressive, token-by-token
- Sampling: top-k with `k=25`
- Length bounds: `min_len = L_text × 2`, `max_len = L_text × 20`
- Tokens yielded via Python generator (streaming-compatible)
- KV-cache used for incremental decoding (`forward_one_step`)

### Bi-Stream Mode (CosyVoice2+)

When text arrives as a generator (streaming from upstream LLM), text and speech tokens are interleaved at mix ratio `[5, 15]`:

```
[SOS] [instruct] [text_0..4] [speech_0..14] [text_5..9] [speech_15..29] ... [TASK_ID] [tail] [EOS]
```

Every 5 text tokens predict 15 speech tokens, reducing time-to-first-audio.

### vLLM Support (CosyVoice2+)

CosyVoice2 has a native vLLM integration in `cosyvoice/vllm/cosyvoice2.py`. `CosyVoice2ForCausalLM` is registered with vLLM's `ModelRegistry` and extends the Qwen2 vLLM model. Enabled via `load_vllm=True` in `AutoModel(...)` or `CosyVoice2(...)`.

### LLM Bottleneck

The LLM stage is the dominant latency source. At token rate 25 Hz and near-matched decode throughput, `RTF_min = f_tok / v_llm`. Meaningful speedups require architectural changes:
1. Multi-token prediction per step (MTP)
2. Smaller dedicated speech decoder
3. Better batching via vLLM (already supported)

---

## Stage 2: Flow — Speech Tokens → Mel Spectrogram

**Class (v2):** `CausalMaskedDiffWithXvec` in `cosyvoice/flow/flow.py`
**Class (v1):** `MaskedDiffWithXvec` (non-causal)

### Pipeline

```
speech tokens (int32, 25 Hz)
  └─► nn.Embedding(6561, 512)           input_embedding
  └─► Conformer encoder (causal)        streaming-compatible
  └─► nn.Linear(encoder_dim, 80)        encoder_proj
  └─► LengthRegulator                   50 Hz → ~86 Hz (22050/256)
  └─► ConditionalCFM                    ODE: noise → 80-dim mel
```

### Dimensions

| Tensor | Shape | Notes |
|---|---|---|
| Token embedding | `(B, L_tokens, 512)` | `input_size = 512` |
| Encoder output | `(B, L_tokens, 80)` | `output_size = 80` |
| After length regulation | `(B, L_mel, 80)` | `token_mel_ratio ≈ 2` |
| Speaker embed projection | `(B, 80)` | 192-dim → 80-dim |
| Final mel output | `(B, 80, L_mel)` | 80 mel bins |

### Frame Rate Math

- Input token rate: **25 Hz** (`input_frame_rate = 25`, confirmed in `cosyvoice2.yaml`)
- `token_mel_ratio = 2` → each token maps to 2 mel frames
- Mel frame rate: 25 × 2 = **50 Hz** (or `22050 / 256 ≈ 86 Hz` at the final hop size)
- Formula: `mel_len = int(token_len / input_frame_rate * sample_rate / hop_size)`

### ConditionalCFM (Flow Matching)

Based on Matcha-TTS `BASECFM`. Solves probability-flow ODE:

| Parameter | Value |
|---|---|
| Solver | Euler |
| Steps | 10 |
| Time schedule | Cosine |
| Estimator | DiT (Diffusion Transformer) |
| CFG rate (inference) | 0.7 |
| CFG training drop rate | 0.2 |
| σ_min | 1e-6 |

**ODE step:** `z_{t+1} = z_t + dt × estimator(z_t, mask, mu, t, spks, cond)`

Starting from `z ~ N(0, I)` with shape `(B, 80, L_mel)`, iterating t from 0 to 1.

**Classifier-Free Guidance:** `output = (1 + cfg_rate) × pred - cfg_rate × pred_uncond`

**Voice cloning:** Prompt mel frames prepended as prefix condition. Generated frames only are kept in output.

**Streaming cache:** `(1, 80, cache_len, 2)` storing `[z, mu]` pairs for overlap region.

### DiT Estimator

- Time embedding: sinusoidal + MLP
- Input: concatenate `[x, cond, text_embed, spks]` → linear projection
- Transformer blocks with rotary embeddings
- Causal convolution for streaming
- Output: velocity estimate, same shape as input

---

## Stage 3: Vocoder — Mel → Waveform

**Class:** HiFi-GAN generator in `cosyvoice/hifigan/generator.py`
**Architecture:** BigVGAN-style source-filter neural vocoder

### Components

| Module | Role |
|---|---|
| `F0Predictor` | Estimates fundamental frequency from mel |
| `SineGen2` | F0 → harmonic sinusoidal excitation |
| `ResBlock` | Dilated conv + Snake activation |
| `CausalConv1dUpsample` | Progressive upsampling to sample rate |

### Upsampling

Input: `(B, 80, L_mel)` → Conv1D → `(B, 512, L_mel)` → 6 transpose conv stages → `(B, 1, L_samples)`

Total upsampling factor: `8 × 8 × 2 × 2 × 2 × 2 = 512` (but actual varies by config)

### Training Losses

`L = L_gen + 2·L_fm + 45·L_mel + L_tpr + L_f0`

### Output

**22050 Hz** (CosyVoice2) or **24000 Hz** (CosyVoice3) waveform

---

## Streaming & Threading Model

**Class:** `CosyVoice2Model` in `cosyvoice/cli/model.py`

Each request gets a `uuid`. Per-request state:

```python
tts_speech_token_dict[uuid]   # ring buffer filled by LLM thread
llm_end_dict[uuid]            # flag: LLM finished
flow_cache_dict[uuid]         # [z, mu] overlap cache
mel_overlap_dict[uuid]        # mel Hamming-window overlap
hift_cache_dict[uuid]         # vocoder source/mel cache
```

### Execution

```
Thread 1 (LLM):  generates tokens → appends to buffer
Main thread:     polls buffer → flow(tokens) → hift(mel) → yield wav chunk
```

### Hop Lengths

- `token_min_hop_len = 2 × input_frame_rate` ≈ 50 tokens
- `token_max_hop_len = 4 × input_frame_rate` ≈ 100 tokens
- `token_overlap_len = 20` tokens
- Chunks joined with Hamming-window fade-in/out

---

## Version Comparison

| | CosyVoice1 | CosyVoice2 | CosyVoice3 |
|---|---|---|---|
| LLM backbone | Custom TransformerLM | Qwen2ForCausalLM | Qwen2ForCausalLM (extended) |
| LLM class | `TransformerLM` | `Qwen2LM` | `CosyVoice3LM` |
| Speech tokenizer | v1 ONNX | v2 batch ONNX, **25 Hz**, vocab 6561 | v2 batch ONNX, 25 Hz, vocab 6561 |
| Flow | `MaskedDiffWithXvec` (non-causal) | `CausalMaskedDiffWithXvec` | `CausalMaskedDiffWithDiT` |
| Bi-stream | No | Yes `[5:15]` | Yes |
| Instruct tokens | No | No | Yes |
| vLLM support | No | Yes (`cosyvoice/vllm/cosyvoice2.py`) | Yes |
| Checkpoint size | ~300M | ~0.5B | ~0.5B |
| Sample rate | 22050 Hz | 22050 Hz | 24000 Hz |

---

## Complete Tensor Shape Flow

```
INPUT:
  text_token:          (B, L_text)              int64, Qwen2 BPE IDs
  speech_token:        (B, L_prompt_speech)     int64, [0..6560]
  speech_feat:         (B, L_prompt_mel, 80)    float, mel from prompt audio
  embedding:           (B, 192)                 float, CAM++ speaker x-vector

LLM STAGE:
  text_emb:            (B, L_text, 896)         Qwen2 embed_tokens
  spk_proj:            (B, 1, 896)              nn.Linear(192, 896)
  sos_emb:             (B, 1, 896)              nn.Embedding(2, 896)
  lm_input:            (B, L_total, 896)        concatenated
  qwen2_hidden:        (B, L_total, 3584)       last hidden state
  logits:              (B, L_total, 6564)        speech_token_size + 3
  sampled_token:       scalar                    top-k=25

FLOW STAGE:
  token_embed:         (B, L_tokens, 512)       nn.Embedding(6561, 512)
  encoder_out:         (B, L_tokens, 80)        Conformer → Linear
  after_repeat:        (B, L_tokens×2, 80)      token_mel_ratio=2
  z (noise):           (B, 80, L_mel)           N(0, I)
  mu (condition):      (B, 80, L_mel)           from encoder
  velocity:            (B, 80, L_mel)           DiT output per ODE step
  mel_output:          (B, 80, L_mel)           final denoised mel

VOCODER:
  mel_input:           (B, 80, L_mel)
  waveform:            (B, L_samples)           22050 or 24000 Hz
```

---

## File Map

```
cosyvoice/
├── cli/
│   ├── cosyvoice.py        Entry points: CosyVoice, CosyVoice2, AutoModel
│   ├── model.py            Runtime orchestration, streaming, threading
│   └── frontend.py         Text norm, tokenization, speaker embedding
├── llm/
│   └── llm.py              TransformerLM, Qwen2Encoder, Qwen2LM, CosyVoice3LM
├── flow/
│   ├── flow.py             MaskedDiffWithXvec, CausalMaskedDiffWithXvec
│   ├── flow_matching.py    ConditionalCFM (Euler ODE solver)
│   ├── length_regulator.py Token-to-mel length upsampling
│   └── DiT/                Diffusion Transformer estimator
├── hifigan/
│   ├── generator.py        SineGen2, ResBlock, HiFi-GAN generator
│   ├── discriminator.py    Multi-period / multi-scale discriminator
│   ├── f0_predictor.py     Pitch estimator
│   └── hifigan.py          Training wrapper with losses
├── transformer/            Conformer encoder/decoder building blocks
├── tokenizer/              Speech tokenizer wrapper
└── utils/                  Masking, losses, schedulers, ONNX helpers
```

---

## Inference Reference

### Setup

```python
import sys
sys.path.append('third_party/Matcha-TTS')

import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

cosyvoice = AutoModel(model_dir='/path/to/cosyvoice2/hf')
# cosyvoice.sample_rate → 22050
```

### Zero-Shot Voice Cloning (batch)

Clones a voice from a short reference clip without fine-tuning.

```python
for i, result in enumerate(cosyvoice.inference_zero_shot(
    tts_text='Hello！Good afternoon。收到好友从远方寄来的生日礼物。',
    prompt_text='希望你以后能够做的比我还好呦。',   # transcript of prompt_wav
    prompt_wav='./asset/zero_shot_prompt.wav',
    stream=False
)):
    torchaudio.save('output.wav', result['tts_speech'], cosyvoice.sample_rate)
```

Observed RTF: **~0.47** on a single GPU (A100). Latency dominated by LLM stage (~12s for ~12.5s of audio).

### Streaming / Bi-Stream Input

Pass a text **generator** instead of a string to enable bi-stream mode. Each text chunk is consumed as it arrives; audio chunks start yielding before text is fully received.

```python
import torch

def text_generator():
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'

chunks = []
for result in cosyvoice.inference_zero_shot(
    text_generator(),                            # generator, not string
    '希望你以后能够做的比我还好呦。',
    './asset/zero_shot_prompt.wav',
    stream=True
):
    chunks.append(result['tts_speech'])

torchaudio.save('output.wav', torch.cat(chunks, dim=-1), cosyvoice.sample_rate)
```

**What happens internally (from logs):**
- Detects generator input → skips text normalization (`get tts_text generator, will skip text_normalize!`)
- Bi-stream interleaving: every **5 text tokens → 15 speech tokens** predicted (`[5:15]` ratio)
- When speech decoding runs ahead of text: emits `fill_token` and waits (`get fill token, need to append more text token`)
- First audio chunk latency is high (~RTF 1.1), subsequent chunks settle to ~0.48
- Final chunk: `no more text token, decode until met eos`

**Chunk sizes observed:**
```
chunk 1: 1.36s audio,  RTF 1.097   ← cold start
chunk 2: 2.00s audio,  RTF 0.573
chunk 3: 4.00s audio,  RTF 0.483
chunk 4: 3.36s audio,  RTF 0.462
```

### Loading a Fine-Tuned Checkpoint

```python
# HuggingFace-format checkpoint (saved with model.save_pretrained or equivalent)
cosyvoice = AutoModel(model_dir='/path/to/finetuned/hf')
# AutoModel auto-detects CosyVoice2 vs CosyVoice3 from config
```

---

## Key Design Decisions to Know

1. **Speech tokenizer is a black box** — the ONNX model has no exposed continuous latent space. To get pre-quantization latents, you would need to either (a) crack the ONNX graph, (b) find the original training code, or (c) use the Flow encoder's intermediate representations as a proxy.

2. **The LLM predicts speech tokens, not mel frames** — the mapping from discrete tokens to continuous audio is handled entirely by the Flow stage. Any continuous-prediction modification must replace or augment the LLM head output.

3. **Speaker identity is injected at two points** — as a projected embedding in the LLM input sequence, AND as a condition in the Flow/CFM decoder. Both must be consistent for voice cloning.

4. **Streaming requires cache management at every stage** — KV-cache in LLM, flow cache `[z, mu]` in CFM, mel overlap + vocoder cache in HiFi-GAN. Missing any one breaks streaming continuity.

5. **The `+3` or `+200` special tokens in the LM head** are critical — they encode SOS, EOS, TASK_ID, and (in v3) instruction markers. Any architecture modification must preserve these control tokens.

---

## Repository & Models

**GitHub:** [https://github.com/FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

### Available Models

| Model | Parameters | ModelScope ID |
|---|---|---|
| Fun-CosyVoice 3.0 (latest) | 0.5B | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| CosyVoice 2.0 | 0.5B | `FunAudioLLM/CosyVoice2-0.5B` |
| CosyVoice 1.0 | 300M | `FunAudioLLM/CosyVoice-300M` |

### Language Support

9 languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
18+ Chinese dialects/accents

### Installation

```sh
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
conda create -n cosyvoice python=3.10
conda activate cosyvoice
pip install -r requirements.txt
```

### Model Download (ModelScope)

```python
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

Local reference copy: `/home/joseph/projects/CosyVoice`
