"""Audio byte helpers for HTTP responses."""

from __future__ import annotations

import struct

import numpy as np
import torch


SAMPLE_RATE = 24000

# Target loudness for output audio: top-1% amplitude at this fraction of full scale.
# 0.35 ≈ -9 dBFS headroom — comfortable podcast listening level.
_OUTPUT_LOUDNESS_TARGET = 0.35


def _normalize_output(audio: np.ndarray) -> np.ndarray:
    """Normalize output PCM so loud speech peaks sit near _OUTPUT_LOUDNESS_TARGET.

    Uses the same percentile-based approach as audio_volume_normalize (training),
    but with a higher target so the output sounds at a consistent, audible level.
    Silences and very short clips are passed through unchanged to avoid
    amplifying noise between utterances.
    """
    significant = np.abs(audio)
    significant = significant[significant > 0.01]
    if significant.size < 10:
        return audio
    L = significant.size
    volume = np.mean(np.sort(significant)[int(0.99 * L):])  # top 1% amplitude
    if volume < 1e-4:
        return audio
    scale = np.clip(_OUTPUT_LOUDNESS_TARGET / volume, 0.1, 10.0)
    audio = audio * scale
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak
    return audio


def tensor_to_pcm16_bytes(wav: torch.Tensor, normalize: bool = True) -> bytes:
    """Convert a `[1, T]` or `[T]` float tensor to little-endian PCM16 bytes.

    Args:
        normalize: if True (default), apply per-chunk loudness normalization so
            output volume is consistent across speakers and turns.
    """
    audio = wav.detach().float().cpu().squeeze().numpy()
    if normalize:
        audio = _normalize_output(audio)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2", copy=False)
    return pcm.tobytes()


def wav_header(data_size: int | None, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Return a PCM16 WAV header.

    When `data_size` is unknown for chunked transfer, use the common
    placeholder length. For exact files, pass the final PCM byte count.
    """
    if data_size is None:
        data_size = 0xFFFFFFFF
        riff_size = 0xFFFFFFFF
    else:
        riff_size = 36 + data_size
    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def wav_bytes_from_pcm(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    return wav_header(len(pcm), sample_rate=sample_rate) + pcm


def normalize_turn_audio(
    chunks: list,
    target_p99: float = 0.35,
    max_gain: float = 10.0,
) -> list:
    """Return a new list of tensors with consistent loudness for the full turn.

    Computes a single gain from the concatenated turn audio — all chunks share
    the same factor so there are no volume jumps at chunk boundaries.

    Uses percentile-based targeting (same approach as _normalize_output):
    the top-1% amplitude of the turn is brought to target_p99 (≈ -9 dBFS).
    This caps loud spikes (including those from finalize=True lookahead tokens)
    while keeping relative dynamics intact.

    max_gain limits amplification for very quiet turns.
    """
    if not chunks:
        return chunks
    full = torch.cat(chunks, dim=-1).float()
    samples = np.abs(full.cpu().numpy().ravel())
    significant = samples[samples > 0.01]
    if significant.size < 10:
        return chunks
    L = significant.size
    volume = np.mean(np.sort(significant)[int(0.99 * L):])  # top-1% amplitude
    if volume < 1e-4:
        return chunks
    gain = float(np.clip(target_p99 / volume, 1.0 / max_gain, max_gain))
    peak = float(samples.max())
    if peak * gain > 1.0:
        gain = 1.0 / peak
    return [(c.float() * gain).clamp(-1.0, 1.0) for c in chunks]
