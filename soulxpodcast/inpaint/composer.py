"""Phoneme embedding + composer module for pronunciation inpainting.

The composer takes a per-text-token slot buffer of phoneme ids and produces a
``d_model``-shaped embedding that replaces the LLM's text embedding at the
corresponding position. The LLM backbone stays frozen during training; only
this module's parameters receive gradients
(see ``.claude/skills/cosyvoice-inpaint/SKILL.md`` §2).

Slot layout, K=8 per text token::

    phone_token: LongTensor (B, K * L)
                  per-text-token block: [slot_0, slot_1, ..., slot_{K-1}]
                  id 0 means "no phoneme at this slot"
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from soulxpodcast.inpaint._vocab import TOTAL_VOCAB_SIZE


class PhonemeComposer(nn.Module):
    """Composes per-text-token phoneme slots into a single d_model embedding.

    Architecture: mean-pool over non-pad slots → 2-layer MLP::

      slot_emb[k] = phone_emb(ids[..., k])                # (B, L, K, d)
      pooled     = mean(slot_emb over non-pad slots)      # (B, L, d)
      composed   = Linear(d → d)(pooled) → GELU → Linear(d → d)

    Pad slots (id=0) are zero by ``padding_idx=0``; the denominator counts
    non-pad slots only so empty positions don't divide by zero.

    Forward returns ``(composed, mask)`` where ``composed`` has shape
    ``(B, L, d_model)`` (zero at positions with no phoneme slot) and
    ``mask`` is a bool ``(B, L)`` True wherever ``composed`` should
    replace the text embedding.
    """

    def __init__(self, d_model: int, slots_per_token: int = 8):
        super().__init__()
        self.d_model = d_model
        self.K = slots_per_token
        self.vocab_size = TOTAL_VOCAB_SIZE

        self.phone_emb = nn.Embedding(TOTAL_VOCAB_SIZE, d_model, padding_idx=0)
        self.composer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.phone_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.phone_emb.weight[0].zero_()  # keep pad row at exactly zero

    @torch.no_grad()
    def init_from_text_embed(self, text_embed: nn.Embedding) -> None:
        """Shift non-pad phoneme rows toward the average text-embedding row.

        Mirrors upstream CosyVoice-Inpaint's ``init_component_from_text_embed``:
        keeps composed embeddings on the LLM's input-embedding manifold so
        the frozen backbone sees a sane vector from step 1.
        """
        if text_embed.embedding_dim != self.d_model:
            raise ValueError(
                f"text_embed dim {text_embed.embedding_dim} != composer d_model {self.d_model}"
            )
        avg = text_embed.weight.mean(dim=0)
        self.phone_emb.weight.data[1:] += avg.to(self.phone_emb.weight.dtype).unsqueeze(0)
        self.phone_emb.weight.data[0].zero_()

    def forward(self, phone_token: Tensor) -> tuple[Tensor, Tensor]:
        """Compose per-text-token phoneme embeddings via mean-pool + linear.

        Args:
            phone_token: LongTensor of shape ``(B, K * L)``.

        Returns:
            composed: (B, L, d_model)
            mask:     (B, L) bool, True at positions with at least one
                      non-pad slot.
        """
        if phone_token.dim() != 2:
            raise ValueError(
                f"phone_token must be 2-D (B, K*L); got shape {tuple(phone_token.shape)}"
            )
        B, KL = phone_token.shape
        if KL % self.K != 0:
            raise ValueError(
                f"phone_token width {KL} is not a multiple of slots_per_token={self.K}"
            )
        L = KL // self.K

        ids = phone_token.view(B, L, self.K)               # (B, L, K)
        slot_emb = self.phone_emb(ids)                     # (B, L, K, d)

        # Mean-pool over the slot axis, ignoring pad slots (id=0).
        # ``padding_idx=0`` makes pad rows exactly zero, so the sum
        # naturally excludes them; we just need to divide by the
        # non-pad count per position.
        is_non_pad = (ids != 0).to(slot_emb.dtype)          # (B, L, K)
        n_slots = is_non_pad.sum(dim=-1, keepdim=True).clamp_min(1)  # (B, L, 1)
        pooled = slot_emb.sum(dim=2) / n_slots              # (B, L, d)

        composed = self.composer(pooled)                    # (B, L, d)

        # Position mask: True where this text token has at least one phoneme slot.
        position_mask = (ids != 0).any(dim=-1)              # (B, L)
        # Zero positions that had no phoneme so callers can trust
        # ``composed[~mask] == 0`` for safe inject downstream.
        composed = composed * position_mask.unsqueeze(-1).to(composed.dtype)
        return composed, position_mask


def apply_phoneme_inpaint(text_emb: Tensor, composed: Tensor, mask: Tensor) -> Tensor:
    """Replace text-embedding rows with composed phoneme embeddings.

    Args:
        text_emb: (B, L, D) — output of ``model.get_input_embeddings()(input_ids)``
        composed: (B, L, D) — output of ``PhonemeComposer.forward``
        mask:     (B, L)    — bool, True wherever ``composed`` should win

    The text embedding is preserved where ``mask`` is False; at True
    positions it is fully replaced (not blended) by ``composed``.
    """
    if text_emb.shape != composed.shape:
        raise ValueError(
            f"text_emb shape {tuple(text_emb.shape)} != composed shape {tuple(composed.shape)}"
        )
    if mask.shape != text_emb.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != (B, L) from text_emb {tuple(text_emb.shape)}"
        )
    keep = (~mask).unsqueeze(-1).to(text_emb.dtype)
    take = mask.unsqueeze(-1).to(text_emb.dtype)
    return text_emb * keep + composed * take
