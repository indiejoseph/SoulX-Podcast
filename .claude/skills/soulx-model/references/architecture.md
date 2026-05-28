# SoulX-Podcast Architecture Reference

## Model overview

Three cascaded components:
1. **LLM** (`Qwen3ForCausalLM`, 1.7B) — generates discrete speech tokens from text + audio prompts
2. **Flow model** (`CausalMaskedDiffWithXvec`) — diffusion, speech tokens → 80-dim mel-spectrograms (15 steps)
3. **Vocoder** (`HiFTGenerator`) — mel → 24 kHz waveforms

Config loaded from: `pretrained_models/SoulX-Podcast-1.7B-dialect/soulxpodcast_config.json`

---

## Entry points

### `initiate_model(seed, model_path, llm_engine, fp16_flow)`
**File:** `soulxpodcast/utils/infer_utils.py`

Returns `(model: SoulXPodcast, dataset: PodcastInferHandler)`.

- `llm_engine`: `"hf"` or `"vllm"`
- `fp16_flow`: cast flow model to fp16

### `podcast_format_parser(data, output_dir="outputs")`
**File:** `soulxpodcast/utils/parser.py`

### `process_single_input(dataset, target_text_list, prompt_wav_list, prompt_text_list, use_dialect_prompt, dialect_prompt_text_list)`
**File:** `soulxpodcast/utils/infer_utils.py`

### `model.forward_longform(**data)`
**File:** `soulxpodcast/models/soulxpodcast.py`

Returns `{"generated_wavs": list[torch.Tensor]}` — one tensor per dialogue turn, 24 kHz.

---

## User-facing input format

```python
data = {
    "speakers": {
        "S1": {
            "prompt_audio": Path("example/audios/female_mandarin.wav"),
            "prompt_text": "Transcription of the reference clip",
            "dialect_prompt": "<|Yue|>Cantonese version (optional)",
        },
        "S2": { ... }
    },
    "text": [
        ["S1", "First dialogue line"],
        ["S2", "Second dialogue line"],
    ]
}
```

Up to 4 speakers (`S1`–`S4`). `dialect_prompt` is optional; omit to use Mandarin synthesis.

---

## `podcast_format_parser` output

```python
{
    "key": "YYYYMMDD-HHMMSS",
    "prompt_text": list[str],           # one per speaker
    "prompt_wav":  list[str],           # one audio path per speaker
    "text":        list[str],           # lines with "[S#]" prefix, e.g. "[S1]Hello..."
    "spk":         list[int],           # 0-indexed speaker IDs per line
    "wav":         str,                 # output file path
    "use_dialect_prompt": bool,
    "dialect_prompt_text": list[str],   # one per speaker
}
```

---

## `process_single_input` output (= `forward_longform` kwargs)

```python
{
    "prompt_mels_for_llm":               torch.Tensor,       # [B, 128, T]  ← 128-dim
    "prompt_mels_lens_for_llm":          torch.Tensor,
    "prompt_text_tokens_for_llm":        list[list[int]],
    "text_tokens_for_llm":               list[list[int]],
    "prompt_mels_for_flow_ori":          torch.Tensor,       # [B, T', 80]  ← 80-dim
    "prompt_mels_lens_for_flow":         torch.Tensor,
    "spk_emb_for_flow":                  torch.Tensor,       # [B, 192]
    "sampling_params":                   SamplingParams,
    "spk_ids":                           list[list[int]],
    "infos":                             list,
    "use_dialect_prompt":                bool,
    # only when use_dialect_prompt=True:
    "dialect_prompt_text_tokens_for_llm": list[list[int]],
    "dialect_prefix":                    list[list[int]],
}
```

---

## LLM token format

Each dialogue turn is formatted as:

```
<|task_podcast|><|SPEAKER_0|><|text_start|>{text}<|text_end|><|semantic_token_start|>
<|1594|><|5352|>...<|2112|>
<|semantic_token_end|>
```

`<|task_podcast|>` appears once at the start of the full sequence.

### Special tokens

| Token | Purpose |
|-------|---------|
| `<\|task_podcast\|>` | Task prefix |
| `<\|SPEAKER_0\|>` .. `<\|SPEAKER_3\|>` | Speaker identity |
| `<\|text_start\|>` / `<\|text_end\|>` | Text span delimiters |
| `<\|semantic_token_start\|>` | Begin speech token sequence |
| `<\|semantic_token_end\|>` | EOS for a turn (token ID 151675 or 153478) |
| `<\|Yue\|>`, `<\|Sichuan\|>`, `<\|Henan\|>` | Dialect conditioning prefix |
| `<\|laughter\|>`, `<\|sigh\|>`, `<\|breathing\|>`, `<\|coughing\|>`, `<\|throat_clearing\|>` | Paralinguistic events — inline in text to trigger naturalistic sounds |

### Vocabulary layout

The Qwen3 tokenizer is heavily customized. Critical layout you MUST get right:

| Range | Contents | Notes |
|-------|----------|-------|
| `0 – 151,642` | Qwen3 BPE text tokens | normal text |
| `151,643 – 152,926` | Qwen3 special tokens + SoulX additions | `<\|task_podcast\|>`, `<\|SPEAKER_0\|>`, dialect tags, etc. |
| `152,927 – 153,594` | **`[add_token_*]` + `[silence_time_*]` placeholders** | **NOT speech tokens** — reserved/silence markers |
| `153,477` | `<\|semantic_token_start\|>` | embedded inside the placeholder range |
| `153,478` | `<\|semantic_token_end\|>` (= speech EOS) | embedded inside the placeholder range |
| `153,595 – 160,155` | **`<\|0\|>` through `<\|6560\|>` — ACTUAL speech tokens** | 6,561 tokens at 25 Hz |

**`speech_token_offset = 153,595`** (verify with `tokenizer.encode("<|0|>") → [153595]`).

This was a real bug — the project originally hardcoded `152,927` (the start of the placeholder range) as the offset, which mapped raw speech_token 0 to `[add_token_450]` instead of `<|0|>`. Inference works because `soulxpodcast_config.json` overrides to `153595` at load time, but training scripts using the dataclass default were broken until `2025-05`.

### Token offset math

```python
# s3tokenizer (0-based, range [0, 6560]) → LLM input
llm_id = s3_id + speech_token_offset    # = s3_id + 153_595

# LLM output → speech token (for flow / vocoder)
s3_id = llm_id - speech_token_offset
```

**Always read `speech_token_offset` from `model.config.hf_config.speech_token_offset`** — never hardcode it.

---

## Model configuration (`soulxpodcast_config.json`)

```json
{
    "architectures": ["Qwen3ForCausalLM"],
    "bos_token_id": 151643,
    "eos_token_id": 151675,
    "hidden_size": 2048,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "vocab_size": 159488,        // declared; the loaded model is actually 160_156 (tokenizer)
    "speech_token_offset": 153595,
    "torch_dtype": "bfloat16"
}
```

---

## Subcomponents

### Audio tokenizer (`s3tokenizer`)
- 25 Hz frame rate (1 token = 40 ms)
- Encodes reference audio → 0-based speech token IDs
- Must apply `+ speech_token_offset` before feeding to LLM

### Flow model (`CausalMaskedDiffWithXvec`)
**File:** `soulxpodcast/models/modules/flow.py`

```python
CausalMaskedDiffWithXvec(
    input_size=512,
    output_size=80,          # 80-dim mel output
    spk_embed_dim=192,       # campplus speaker embedding
    vocab_size=6561,         # max speech token ID + 1
    input_frame_rate=25,
    token_mel_ratio=2,       # 1 speech token → 2 mel frames
)
```

15 diffusion steps. Inputs: speech tokens + campplus speaker embedding.

### Speaker embedding (`campplus.onnx`)
- Located in `pretrained_models/SoulX-Podcast-1.7B-dialect/`
- Outputs 192-dim embedding from reference audio
- Used by flow model for voice cloning

### Vocoder (`HiFTGenerator`)
**File:** `soulxpodcast/models/modules/hifigan.py`

```python
HiFTGenerator(
    in_channels=80,
    base_channels=512,
    sampling_rate=24000,
    upsample_rates=[8, 5, 3],
    istft_params={"n_fft": 16, "hop_len": 4},
)
```

---

## Sampling parameters

```python
@dataclass
class SamplingParams:
    temperature: float = 0.6
    repetition_penalty: float = 1.25
    top_k: int = 100
    top_p: float = 0.9
    min_tokens: int = 8
    max_tokens: int = 3000
    stop_token_ids: list[int] = [151675]
    use_ras: bool = True      # Rectified Autoregressive Sampling
    win_size: int = 25        # RAS sliding window
    tau_r: float = 0.2        # RAS temperature
```

---

## Context management (`config.py`)

```python
prompt_context: int = 2         # keep first N speakers in KV-cache prefix
history_context: int = 2        # include last N generated turns as context
max_turn_size: int = 10         # reset KV cache after N turns
turn_tokens_threshold: int = 6192
```

---

## LLM backends

### HF engine (`"hf"`)
**File:** `soulxpodcast/engine/llm_engine.py`
- `AutoModelForCausalLM` + `model.generate()`, dtype `bfloat16`
- Custom RAS sampling via `_ras_sample_hf_engine()`

### vLLM engine (`"vllm"`)
- Faster, prefix caching enabled, max model length 8192
- Same interface as HF engine

---

## Tokenizer

Identical to `Qwen/Qwen3-0.6B-Base`. Can be loaded independently:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
```

---

## Common gotchas

- **Two mel dims**: `prompt_mels_for_llm` = 128-dim (LLM prefix), `prompt_mels_for_flow_ori` = 80-dim (flow model). Never swap them.
- **Token offset**: always apply/reverse `speech_token_offset` when crossing the text↔speech boundary. Read it from `model.config.hf_config.speech_token_offset` — do NOT hardcode 152,927 (that's the `[add_token_*]` placeholder range, not speech tokens).
- **Output sample rate**: always 24,000 Hz. Use `torchaudio.save(path, wav, 24000)`.
- **Speaker limit**: max 4 speakers (`S1`–`S4` → `SPEAKER_0`–`SPEAKER_3`).
- **Dialect mode** (base model only): set `use_dialect_prompt=True` and include `"dialect_prompt"` with `<|Yue|>` prefix per speaker. See "dialect_prompt mechanism" below — this is a base-model warm-up trick that becomes counterproductive when a dialect LoRA is loaded.

---

## dialect_prompt mechanism (when to use, when to skip)

`dialect_prompt` (with `use_dialect_prompt=True`) implements **two-stage cross-lingual voice cloning** at inference time. Mechanism:

1. **Warm-up generation** — `self.llm.generate(prompt_text + prompt_speech_tokens + dialect_prompt_text, ...)` produces **synthetic** speech tokens for the dialect_prompt_text in the speaker's voice. Costs one extra LLM pass.
2. **Main generation** — uses those synthetic dialect tokens as the voice-clone context (instead of the speaker's real Mandarin tokens), then generates target speech.

The point is to anchor a Mandarin-only speaker prompt into Cantonese phoneme space before the actual generation.

### When to OMIT dialect_prompt

A loaded dialect-LoRA already knows the target dialect's speech-token distribution directly. The warm-up step then:
- Runs the LLM twice (~2× wall time)
- Generates synthetic tokens that may drift from the LoRA's clean distribution (warm-up uses a 2-turn context the LoRA wasn't trained on)
- Feeds drifted tokens as the voice-clone reference for stage 2 → compounds drift

**Empirically with the HK Cantonese LoRA (May 2025): omitting `dialect_prompt` gives BOTH cleaner audio AND ~3× faster Cantonese inference.** Implementations exposing dialect mode (WebUI / API) should drop `dialect_prompt` automatically when a dialect-LoRA is loaded.

---

## Training pipeline (LoRA fine-tuning)

The SoulX-Podcast repo doesn't ship training code; it was added in-house following CosyVoice2's working recipe. Entry points:

- `soulxpodcast/training/mtp_dataset.py` — dataset + collator (single-turn `task_prefix + text + speech + EOS` per sample)
- `soulxpodcast/training/train_lora_trunk.py` — LoRA fine-tune of the trunk LLM
- `soulxpodcast/training/train_mtp.py` — MTP-head distillation (Phase 2 spec decoding)

### Proven LoRA recipe (HK Cantonese style transfer)

```bash
python -m soulxpodcast.training.train_lora_trunk \
    --dataset_path /path/to/dataset_with_tokens_16khz \
    --model_path pretrained_models/SoulX-Podcast-1.7B-dialect \
    --output_dir runs/lora_hk \
    --batch_size 16 --grad_accum_steps 2 \
    --num_epochs 2 \
    --lr 2e-4 --warmup_steps 200 \
    --save_every 500 \
    --wandb --wandb_project soulxpodcast-lora
```

Defaults (in `TrainConfig`) match CosyVoice2's proven LoRA recipe:
- `lora_rank=8, lora_alpha=16` (scale=2.0, enough to steer style without collapsing hidden states)
- `lora_target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"` (all attention + MLP)
- `lora_modules_to_save = "lm_head"` — **CRITICAL**: lm_head must train alongside the LoRA to follow the shifted hidden state distribution. Without this, the frozen lm_head can't decode the LoRA'd hidden states and output is garbled.
- `label_smoothing = 0.1` (CosyVoice2 used 0.0; for TTS, 0.0 may produce sharper output — worth ablating)
- `gradient_checkpointing = True` (required: 326M trainable lm_head + LoRA needs the memory headroom)
- `include_eos_in_speech_mask = True` — speech EOS (`<|semantic_token_end|>`) target IS in the loss mask for trunk training. (For MTP training, default `False` is correct because MTP heads predict K-step-ahead and shouldn't try to predict past the last speech token.)

### Bugs to avoid (all encountered & fixed in May 2025)

1. **EOS excluded from loss mask** — `speech_mask` originally stopped at the last speech token. Trunk training never received gradient on the EOS prediction → model failed to terminate → runaway generation. Fix: `MtpDatasetConfig(include_eos_in_speech_mask=True)` for trunk LoRA.

2. **`lm_head` frozen** — `LoraConfig(modules_to_save=None)` leaves lm_head fixed; LoRA shifts hidden states but lm_head can't decode them → garbled phonemes. Fix: `modules_to_save=["lm_head"]`.

3. **Wrong `speech_token_offset`** — hardcoded `152927` (placeholder range start) instead of `153595` (`<|0|>` id). Training maps speech_token 0 → `[add_token_450]`. Fix: default 153595 in `MtpDatasetConfig` + `SoulXPodcastLLMConfig`.

4. **Dataset extracted at 24 kHz fed to `whisper.log_mel_spectrogram`** (which hardcodes 16 kHz). Produces 1.5× too many mel frames → 1.5× too many speech tokens. The stored tokens then "claim" the audio is 1.5× longer than it is, and flow + vocoder produces garbled output. Fix: resample to 16 kHz **before** `whisper.log_mel_spectrogram`. Reference impl: `tmp/extract_speech_token.py`.

5. **Aggressive LoRA (r=32, α=64)** — too much capacity for style transfer, drives lm_head into degenerate minimum on 367K samples. Fix: r=8, α=16 (CosyVoice2 recipe).

### Training vs inference format mismatch (known limitation)

Training samples are **single-turn**: `task_podcast + speaker + text_start + text + text_end + sem_start + speech + sem_end`.

Inference is **multi-turn with voice clone**: `task_podcast + speaker + text_start + prompt_text + ... + sem_end` (first turn) `+ speaker + text_start + target_text + ... + sem_start` (second turn) `+ generate`.

The base model handles this because it was pretrained on multi-turn data. The LoRA generalizes from the single-turn training thanks to the diversity of 300K+ samples — but small overfit subsets (e.g., 16 samples × 25 epochs) will collapse to format-specific memorization and fail at multi-turn inference. A more robust future fix would be to build multi-turn training samples that match the inference grammar.

### MTP (Phase 2) training notes

`train_mtp.py` distills `K` MTP heads against the trunk's KL distribution. Heads-only (Medusa-1 style), base trunk frozen. The MTP dataset uses `include_eos_in_speech_mask=False` (default) — deeper heads predict K-step-ahead and shouldn't try to predict past the last speech token.

After a trunk LoRA, refresh MTP on top of the LoRA'd trunk (Phase C) to restore spec-decode acceptance length — the original MTP was distilled against base hidden states and will see distribution shift under a LoRA.
