"""Audio byte helpers for HTTP responses."""

from __future__ import annotations

import struct

import numpy as np
import torch


SAMPLE_RATE = 24000


def tensor_to_pcm16_bytes(wav: torch.Tensor) -> bytes:
    """Convert a `[1, T]` or `[T]` float tensor to little-endian PCM16 bytes."""
    audio = wav.detach().float().cpu().squeeze().numpy()
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
