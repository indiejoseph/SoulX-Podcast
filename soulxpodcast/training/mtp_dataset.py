"""Dataset + collator for Sequential MTP training (PLAN.md Phase 2).

Wraps a HuggingFace `datasets.Dataset` (Arrow shards) that has these columns:
    - text          : str   — transcript
    - speech_tokens : str   — space-delimited 0-based s3tokenizer ids
                              ("1 50 102 1205 ...")
    - lang          : str   — "en" / "zh" / "yue"
    - (audio column is ignored — we never decode it)

For each example we build the per-turn LLM input format used at inference time
(without the voice-clone prompt prefix, which isn't needed for MTP head training):

    <|task_podcast|><|SPEAKER_0|>
        <|text_start|>{dialect_prefix}{text}<|text_end|>
        <|semantic_token_start|>{speech_tokens_offset}<|semantic_token_end|>

The collator pads to longest in batch and returns:
    {
        "input_ids"    : LongTensor [B, T]   — full LLM input
        "attention_mask": LongTensor [B, T]
        "speech_mask"  : LongTensor [B, T]   — 1 where token is a speech token
                                                (MTP loss only computed here)
        "lengths"      : LongTensor [B]      — unpadded length per sample
    }

The training loop is responsible for the MTP-head shifting + depth-weighted
loss (so the dataloader stays simple and stack-agnostic).
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


# Special tokens — these MUST exist in the SoulX-Podcast tokenizer's
# added_tokens. Verified at runtime by `MtpDataset._resolve_special_tokens`.
SPECIAL_TOKENS = {
    "task_podcast": "<|task_podcast|>",
    "speaker_0": "<|SPEAKER_0|>",
    "text_start": "<|text_start|>",
    "text_end": "<|text_end|>",
    "semantic_token_start": "<|semantic_token_start|>",
    "semantic_token_end": "<|semantic_token_end|>",  # this is the speech EOS
}

# Per `lang` field. en/zh use no dialect prefix; yue/sichuan/henan tag the
# text content. Mapping is straightforward — extend here if more dialects
# are added to the dataset.
DIALECT_PREFIX = {
    "en": "",
    "zh": "",
    "yue": "<|Yue|>",
    "sichuan": "<|Sichuan|>",
    "henan": "<|Henan|>",
}


@dataclass
class MtpDatasetConfig:
    """Knobs for `MtpDataset`. Keep small + serializable."""
    speech_token_offset: int = 152927      # from soulxpodcast_config.json
    max_total_tokens: int = 2048           # drop full sequences longer than this
    min_speech_tokens: int = 8             # filter trivially-short clips, also catches None/empty speech_tokens
    max_speech_tokens: int = 750           # drop clips with >this many speech tokens
                                            # (750 tokens @ 25 Hz = 30 s — matches s3tokenizer cap;
                                            #  samples without speech_tokens were skipped upstream
                                            #  because they exceeded this duration too)
    skip_dialect_prefix: bool = False      # for ablations
    text_field: str = "text"
    speech_tokens_field: str = "speech_tokens"
    lang_field: str = "lang"
    # Fallback path: decode audio bytes → s3tokenizer.quantize when the
    # speech_tokens column is missing (e.g. on a debug dataset). For the
    # full dataset, precompute speech_tokens upstream — it's much faster
    # than per-sample tokenization.
    audio_field: str = "audio"
    s3tokenizer_model: str = "speech_tokenizer_v2_25hz"
    s3tokenizer_device: str = "cuda"       # "cuda" | "cpu"
    s3tokenizer_sr: int = 16000            # s3tokenizer expects 16 kHz


class MtpDataset(Dataset):
    """Adapter from a HF `datasets.Dataset` to per-sample tensor dicts.

    Does NOT load audio. Reads only the three columns it needs, so the
    underlying HF dataset can be opened with `with_format(columns=...)`
    upstream to avoid touching the audio bytes at all.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        config: Optional[MtpDatasetConfig] = None,
    ):
        self.config = config or MtpDatasetConfig()
        self.tokenizer = tokenizer
        self._special = self._resolve_special_tokens(tokenizer)

        # Decide once at construction whether we use the precomputed
        # `speech_tokens` column or fall back to on-the-fly s3tokenizer.
        cols = set(hf_dataset.column_names)
        self._has_speech_tokens = self.config.speech_tokens_field in cols
        if self._has_speech_tokens:
            self.ds = hf_dataset
            self._audio_tokenizer = None
            logger.info(
                f"using precomputed '{self.config.speech_tokens_field}' column"
            )
        else:
            if self.config.audio_field not in cols:
                raise ValueError(
                    f"dataset has neither '{self.config.speech_tokens_field}' "
                    f"nor '{self.config.audio_field}' column. Columns: {sorted(cols)}"
                )
            # Cast audio column to raw bytes (skip torchcodec auto-decode).
            from datasets import Audio
            hf_dataset = hf_dataset.cast_column(
                self.config.audio_field, Audio(decode=False)
            )
            self.ds = hf_dataset
            self._audio_tokenizer = self._load_s3tokenizer()
            logger.warning(
                f"'{self.config.speech_tokens_field}' column missing — falling "
                f"back to on-the-fly s3tokenizer ({self.config.s3tokenizer_model} "
                f"on {self.config.s3tokenizer_device}). This is ~100x slower "
                f"than precomputed speech_tokens. For real training runs, "
                f"precompute the column upstream. Use num_workers=0 since the "
                f"tokenizer holds CUDA tensors."
            )

        # Precompute the per-turn template prefix / suffix tokens — they're
        # the same for every sample, so encode once.
        self._task_prefix_ids: List[int] = [
            self._special["task_podcast"],
            self._special["speaker_0"],
            self._special["text_start"],
        ]
        # After text: text_end + semantic_token_start
        self._text_to_speech_ids: List[int] = [
            self._special["text_end"],
            self._special["semantic_token_start"],
        ]
        # After speech tokens: semantic_token_end (== speech EOS)
        self._speech_suffix_ids: List[int] = [
            self._special["semantic_token_end"],
        ]

    def _load_s3tokenizer(self):
        """Lazy-load s3tokenizer on the configured device."""
        import s3tokenizer
        model = s3tokenizer.load_model(self.config.s3tokenizer_model)
        if self.config.s3tokenizer_device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        model = model.eval()
        return model

    # ---- HF Dataset interface ------------------------------------------ #

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        row = self.ds[idx]
        if self._has_speech_tokens:
            speech_tokens_raw = self._parse_speech_tokens(
                row[self.config.speech_tokens_field]
            )
        else:
            # Fallback: decode audio bytes → log-mel → s3tokenizer.quantize.
            audio_field = row[self.config.audio_field]
            speech_tokens_raw = self._tokenize_audio_bytes(audio_field["bytes"])
        return self.build_sample(
            text=row[self.config.text_field],
            speech_tokens_raw=speech_tokens_raw,
            lang=row[self.config.lang_field],
        )

    # ---- Core conversion (exposed for testing/inspection) -------------- #

    def build_sample(
        self,
        text: str,
        speech_tokens_raw: List[int],
        lang: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build one sample. Returns None if the sample should be filtered.

        `speech_tokens_raw` is a list of 0-based s3tokenizer ids (i.e. before
        the speech_token_offset is applied). The offset is applied here so
        callers can pass either the parsed precomputed column or fresh
        tokenizer output uniformly.

        Filter conditions (any one drops the sample):
          - empty / None speech_tokens (audio was too long for s3tokenizer,
            so no precomputed tokens exist)
          - fewer than `min_speech_tokens` (trivially-short clip)
          - more than `max_speech_tokens` (would inflate batch length / OOM)
          - full assembled sequence longer than `max_total_tokens`
        """
        n_speech = len(speech_tokens_raw)
        if n_speech < self.config.min_speech_tokens:
            return None
        if n_speech > self.config.max_speech_tokens:
            return None

        # Apply offset to map s3tokenizer ids → LLM-vocab ids.
        speech_tokens_llm = [t + self.config.speech_token_offset
                             for t in speech_tokens_raw]

        # Text → token ids. Prepend dialect tag for non-Mandarin/English.
        text_with_prefix = self._apply_dialect_prefix(text, lang)
        text_ids: List[int] = self.tokenizer.encode(
            text_with_prefix, add_special_tokens=False
        )

        # Assemble the full per-turn sequence (matches inference grammar).
        ids = (
            self._task_prefix_ids
            + text_ids
            + self._text_to_speech_ids
            + speech_tokens_llm
            + self._speech_suffix_ids
        )

        if len(ids) > self.config.max_total_tokens:
            # Length-filter rather than truncate — truncating the speech-token
            # stream would corrupt MTP training (the model would learn to end
            # mid-utterance).
            return None

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # speech_mask: 1 at positions that ARE speech tokens. MTP loss is
        # computed only at these positions, and only when the offset target
        # is also a speech token (handled in the training loop).
        speech_mask = torch.zeros_like(input_ids)
        speech_start = (
            len(self._task_prefix_ids) + len(text_ids) + len(self._text_to_speech_ids)
        )
        speech_end = speech_start + len(speech_tokens_llm)  # excludes EOS
        speech_mask[speech_start:speech_end] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "speech_mask": speech_mask,
            "length": torch.tensor(input_ids.shape[0], dtype=torch.long),
        }

    # ---- Helpers ------------------------------------------------------- #

    def _parse_speech_tokens(self, s) -> List[int]:
        """Parse "1 50 102 1205" → [1, 50, 102, 1205]. Robust to extra spaces.

        Also accepts a list/sequence of ints directly (some HF datasets store
        the column as `Sequence(Value("int32"))` instead of a string)."""
        if s is None:
            return []
        if isinstance(s, str):
            return [int(x) for x in s.split()] if s else []
        # Already a sequence of ints.
        return [int(x) for x in s]

    def _tokenize_audio_bytes(self, audio_bytes: bytes) -> List[int]:
        """Decode audio bytes → log-mel → s3tokenizer.quantize. Slow fallback.

        Expects WAV-format bytes (the standard for HF Audio columns when not
        decoded). Decoded via soundfile (a librosa dep, already installed).
        """
        import s3tokenizer
        import soundfile as sf
        import numpy as np

        # Decode bytes to float32 mono waveform.
        with io.BytesIO(audio_bytes) as buf:
            wav, src_sr = sf.read(buf, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)  # downmix to mono

        # Resample to s3tokenizer's expected 16 kHz if needed.
        if src_sr != self.config.s3tokenizer_sr:
            import torchaudio
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav_t = torchaudio.functional.resample(
                wav_t, orig_freq=src_sr, new_freq=self.config.s3tokenizer_sr
            )
            wav = wav_t.squeeze(0).numpy()

        # Match the inference-side normalization (utils/audio.py).
        from soulxpodcast.utils.audio import audio_volume_normalize
        audio = torch.from_numpy(wav).float()
        audio = audio_volume_normalize(audio)

        # Log-mel + quantize. s3tokenizer.padding expects [num_mels, T] inputs.
        log_mel = s3tokenizer.log_mel_spectrogram(audio)            # [num_mels, T]
        mels, mel_lens = s3tokenizer.padding([log_mel])             # [1, num_mels, T'], [1]
        if self.config.s3tokenizer_device == "cuda" and torch.cuda.is_available():
            mels = mels.cuda()
            mel_lens = mel_lens.cuda()
        with torch.no_grad():
            ids, lens = self._audio_tokenizer.quantize(mels, mel_lens)  # [1, T''], [1]
        return ids[0, : int(lens[0].item())].cpu().tolist()

    def _apply_dialect_prefix(self, text: str, lang: str) -> str:
        if self.config.skip_dialect_prefix:
            return text
        prefix = DIALECT_PREFIX.get(lang, "")
        if prefix and not text.startswith(prefix):
            return prefix + text
        return text

    @staticmethod
    def _resolve_special_tokens(tokenizer) -> Dict[str, int]:
        """Map symbolic names → token ids. Fails loudly if any are missing."""
        resolved = {}
        missing = []
        for key, sym in SPECIAL_TOKENS.items():
            ids = tokenizer.encode(sym, add_special_tokens=False)
            if len(ids) != 1:
                # Either the tokenizer split the special token (added_tokens
                # not loaded?) or it's genuinely missing.
                missing.append(f"{sym} (encoded to {ids})")
                continue
            resolved[key] = ids[0]
        if missing:
            raise ValueError(
                "Tokenizer is missing required special tokens — these must "
                "be in added_tokens.json: " + ", ".join(missing)
            )
        return resolved


@dataclass
class MtpCollator:
    """Pad to longest-in-batch. None-returning samples are filtered upstream."""
    pad_token_id: int = 0  # padding for input_ids; masked out by attention_mask

    def __call__(self, batch: Sequence[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Drop None samples (filtered by length / min_speech_tokens).
        batch = [b for b in batch if b is not None]
        if not batch:
            # Caller decides what to do with an empty batch (e.g. skip step).
            return {}

        max_len = max(int(b["length"].item()) for b in batch)
        B = len(batch)

        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        speech_mask = torch.zeros((B, max_len), dtype=torch.long)
        lengths = torch.zeros(B, dtype=torch.long)

        for i, b in enumerate(batch):
            L = int(b["length"].item())
            input_ids[i, :L] = b["input_ids"]
            attention_mask[i, :L] = b["attention_mask"]
            speech_mask[i, :L] = b["speech_mask"]
            lengths[i] = L

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "speech_mask": speech_mask,
            "lengths": lengths,
        }


def filtering_collate(collator: MtpCollator):
    """Wrap an MtpCollator so DataLoader doesn't choke on all-None batches.

    Returns a callable suitable for DataLoader(collate_fn=...). If every
    sample in a batch was filtered, returns an empty dict; train loop checks
    for `if not batch: continue`.
    """
    def _call(batch):
        return collator(batch)
    return _call
