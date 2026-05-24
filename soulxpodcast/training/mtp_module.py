"""Sequential MTP (Multi-Token Prediction) heads for SoulX-Podcast.

Implements the DeepSeek-V3 / "Sequential MTP" pattern (PLAN.md Phase 2):
- K-1 lightweight residual-mixing layers stacked sequentially
- Each layer mixes the previous hidden state with the embedding of the
  previously-predicted token, restoring micro-causality between heads
- All layers share the base model's `lm_head` (no per-head vocab projection)

At inference, this enables drafting K speech tokens per trunk forward pass.
At training (Medusa-1 / Medusa-2 style), the trunk is frozen and only the
MTP layers are trained — CE on hard labels + KL against the trunk's own
soft logits (PLAN.md Phase 2 spec).

The trainable params are tiny (~200M for K=4 with hidden=2048) vs the 1.7B
frozen base, so this fits comfortably on a single 24-80 GB GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MtpConfig:
    """Sized to match Qwen3-1.7B by default (see soulxpodcast_config.json)."""
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 8       # GQA: same as Qwen3
    intermediate_size: int = 6144      # SwiGLU MLP
    num_mtp_layers: int = 3            # K-1 for K=4 total predictions
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bf16-safe: cast to fp32 for the norm, back to original dtype after.
        out_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(out_dtype)


class MTPLayer(nn.Module):
    """One MTP step: mix(prev_h, target_embed) → transformer block → h'.

    Architecturally a single Qwen3-style decoder layer prepended by a
    concat+project mixing block. Following DeepSeek-V3:
        h'_k = TransformerBlock( Proj_k( [RMSNorm(prev_h); RMSNorm(target_embed)] ) )
    """

    def __init__(self, config: MtpConfig, decoder_layer_cls, base_hf_config):
        super().__init__()
        self.config = config

        # Mixing: normalize both inputs separately, concat, project to hidden.
        self.norm_h = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_e = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # Transformer block — fresh-init instance of the trunk's decoder layer
        # class, built with the trunk's HF config (so attention impl, RoPE,
        # GQA, SwiGLU all match exactly). Random weights — trained from scratch.
        self.transformer = decoder_layer_cls(base_hf_config, layer_idx=0)

        self._init_weights()

    def _init_weights(self):
        # Light init — matches Qwen3's default scheme.
        nn.init.normal_(self.proj.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        prev_h: torch.Tensor,        # [B, T, H]
        target_embed: torch.Tensor,  # [B, T, H]
        position_ids: torch.Tensor,  # [B, T]
        position_embeddings: tuple,  # (cos, sin) from trunk's rotary_emb
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Mix prev_h and embed of the actually-predicted token (teacher forcing).
        mixed = torch.cat([self.norm_h(prev_h), self.norm_e(target_embed)], dim=-1)
        h = self.proj(mixed)
        # Qwen3DecoderLayer requires precomputed RoPE cos/sin tensors.
        out = self.transformer(
            hidden_states=h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out


class SequentialMTP(nn.Module):
    """Stack of K-1 MTP layers.

    Forward expects the trunk's hidden states + the LLM input ids and the
    trunk's `embed_tokens` (shared, frozen). Produces a list of length K-1
    of mixed hidden states, one per MTP head, each [B, T-k, H] where k is
    the head index (1-indexed).

    The caller projects each head's hidden through the shared `lm_head`
    to get logits, then computes the CE+KL loss against the appropriate
    offset targets.
    """

    def __init__(self, config: MtpConfig, decoder_layer_cls, base_hf_config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            MTPLayer(config, decoder_layer_cls, base_hf_config)
            for _ in range(config.num_mtp_layers)
        ])

    def forward(
        self,
        trunk_hidden: torch.Tensor,            # [B, T, H]  (no grad, frozen)
        input_ids: torch.LongTensor,           # [B, T]
        embed_tokens: nn.Module,               # base_model.embed_tokens (frozen)
        rotary_emb: nn.Module,                 # base_model.model.rotary_emb (frozen)
        position_ids: Optional[torch.LongTensor] = None,
        causal_mask_4d: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Returns K-1 hidden states. Layer k's output is [B, T-k, H]."""
        B, T = input_ids.shape
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        outputs: List[torch.Tensor] = []
        prev_h = trunk_hidden  # [B, T, H]

        for k_idx, layer in enumerate(self.layers, start=1):
            # k_idx-th MTP layer: mix prev_h[:, :T-k_idx] with embed(input_ids[:, k_idx:])
            # to predict tokens at positions [k_idx+1, T] (one step further out
            # than what the embedding already showed it).
            valid_len = T - k_idx
            if valid_len <= 0:
                break

            target_embed = embed_tokens(input_ids[:, k_idx:])  # [B, T-k_idx, H]
            pos_ids_k = position_ids[:, :valid_len]
            mask_k = None
            if causal_mask_4d is not None:
                # Slice the 4-D causal mask to the valid region.
                mask_k = causal_mask_4d[:, :, :valid_len, :valid_len]

            # Compute RoPE cos/sin for this MTP layer's positions.
            h_slice = prev_h[:, :valid_len]
            cos_sin = rotary_emb(h_slice, pos_ids_k)

            h_k = layer(h_slice, target_embed, pos_ids_k, cos_sin, mask_k)
            outputs.append(h_k)
            prev_h = h_k  # next MTP layer sees this

        return outputs


def build_causal_mask_4d(
    attention_mask_2d: torch.Tensor,   # [B, T] 1=keep 0=pad
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct a 4-D additive causal mask for Qwen3-style attention.

    Output shape: [B, 1, T, T] with 0 where attention is allowed and
    -inf where it's forbidden (pad columns + future positions).
    """
    B, T = attention_mask_2d.shape
    min_val = torch.finfo(dtype).min
    # Causal upper-triangular: -inf above the diagonal.
    causal = torch.full((T, T), min_val, dtype=dtype, device=attention_mask_2d.device)
    causal = torch.triu(causal, diagonal=1)
    # Pad mask: column j is pad → forbid attention to it.
    pad_cols = (attention_mask_2d == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
    pad_addend = torch.where(pad_cols,
        torch.tensor(min_val, dtype=dtype, device=attention_mask_2d.device),
        torch.tensor(0.0, dtype=dtype, device=attention_mask_2d.device))
    mask = causal.unsqueeze(0).unsqueeze(0) + pad_addend  # broadcast
    return mask
