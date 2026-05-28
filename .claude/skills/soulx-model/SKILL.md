---
name: soulx-model
description: Reference guide for the SoulX-Podcast model architecture and inference pipeline. Use this skill whenever implementing, refactoring, debugging, or extending any part of the SoulX-Podcast codebase — including inference utilities, tokenization, model components (LLM/flow/vocoder), CLI, API, or WebUI. Trigger on any task touching soulxpodcast/ source files, demo.ipynb, or the pretrained model.
---

# SoulX-Podcast Model Skill

When working on this codebase, read `references/architecture.md` first. It contains the complete inference pipeline, all component signatures, token formats, and implementation gotchas.

## Resources

- **Repo**: https://github.com/Soul-AILab/SoulX-Podcast
- **Models (HuggingFace)**: https://huggingface.co/collections/Soul-AILab/soulx-podcast
- **Demo page**: https://soul-ailab.github.io/soulx-podcast/
- **Technical report**: https://arxiv.org/pdf/2510.23541

## What this codebase does

SoulX-Podcast generates multi-speaker podcast audio from a dialogue script. Given reference audio clips per speaker + a dialogue text, it outputs a 24 kHz waveform for each turn. It supports voice cloning, dialect synthesis (Cantonese, Sichuan, Henan), and paralinguistic events (laughter, sighs, breathing, coughing, throat-clearing).

## Core call chain

```
podcast_format_parser(data)           # parse user-facing dict → internal format
    ↓
process_single_input(dataset, ...)    # tokenize + extract mel/speaker embeddings
    ↓
model.forward_longform(**data)        # LLM → flow → vocoder, returns {"generated_wavs": [...]}
```

All three steps are defined in `soulxpodcast/utils/`. The model class is `soulxpodcast/models/soulxpodcast.py`.

## When implementing or refactoring

1. **Read `references/architecture.md`** — covers every function signature, tensor shape, token ID, config value, and the training/fine-tuning recipe.
2. **Identify the layer** you're working in:
   - Input parsing → `utils/parser.py`
   - Data preprocessing → `utils/infer_utils.py`, `utils/dataloader.py`
   - LLM inference → `engine/llm_engine.py`
   - Speech synthesis → `models/modules/flow.py`, `models/modules/hifigan.py`
   - Top-level orchestration → `models/soulxpodcast.py`
   - LoRA / MTP fine-tuning → `training/train_lora_trunk.py`, `training/train_mtp.py`, `training/mtp_dataset.py`
3. **Check token math** before touching anything that crosses text↔speech boundaries. **The speech token offset is `153,595`** (id of `<|0|>` in the tokenizer). Read it from `model.config.hf_config.speech_token_offset` — never hardcode. The earlier hardcoded `152,927` was a real bug that maps speech tokens into the `[add_token_*]` placeholder range.
4. **Mind the two mel dimensions**: `prompt_mels_for_llm` is 128-dim; `prompt_mels_for_flow_ori` is 80-dim. They serve different subcomponents and must not be swapped.
5. **Mind the training/inference format mismatch**: training samples are single-turn; inference is multi-turn with a voice-clone prefix. The base model handles both; LoRA generalizes when trained on diverse data, but overfit subsets break at multi-turn inference.
6. **For LoRA inference of a dialect-fine-tuned trunk: SKIP `dialect_prompt`.** The dialect_prompt warm-up is a base-model trick that becomes counterproductive (drift + 2× slower) when the LoRA already knows the target dialect. See `references/architecture.md` § "dialect_prompt mechanism".
7. **Test with the demo notebook** (`demo.ipynb`) after changes — it exercises the full pipeline end-to-end with a real model checkpoint and includes a LoRA-merged inference section.

## Training quick start

Proven LoRA recipe (CosyVoice2-compatible, fixes all the May-2025 bugs):

```bash
python -m soulxpodcast.training.train_lora_trunk \
    --dataset_path /path/to/dataset_with_tokens_16khz \
    --model_path pretrained_models/SoulX-Podcast-1.7B-dialect \
    --output_dir runs/lora_X \
    --batch_size 16 --grad_accum_steps 2 --num_epochs 2 \
    --lr 2e-4 --warmup_steps 200 --save_every 500
```

Defaults that matter and why — see `references/architecture.md` § "Training pipeline":
- `r=8, α=16` (small enough to steer without collapsing hidden states)
- `modules_to_save=["lm_head"]` (must train alongside LoRA — otherwise garbled output)
- `include_eos_in_speech_mask=True` (trunk learns to terminate)
- `gradient_checkpointing=True` (fits 326M trainable lm_head on H100)
- `speech_token_offset=153595` (correct vocab range)
- Dataset extraction MUST resample to 16 kHz before `whisper.log_mel_spectrogram` — see `tmp/extract_speech_token.py`.
