"""Phoneme embedding + per-alphabet composer module for pronunciation inpainting.

The composer takes a per-text-token slot buffer of phoneme ids and produces a
``d_model``-shaped embedding that replaces the LLM's text embedding at the
corresponding position. The LLM backbone stays frozen during training; only
this module's parameters receive gradients.

Slot layout (storage K = 6 for all alphabets, with per-alphabet active slots)::

    phone_token: LongTensor (B, K * L)
                  per-text-token block: [slot_0, ..., slot_{K-1}]
                  id 0 means "no phoneme at this slot"

  jyutping (alphabet 0): K_active = 2  → slots [initial, final-with-tone]
  pinyin   (alphabet 1): K_active = 2  → slots [initial, final-with-tone]
  cmu      (alphabet 2): K_active = 6  → slots [phone_0, ..., phone_5]

Composition: concat the alphabet's active slots along the feature dim, then
``Linear(K_active * d → d) → GELU → Linear(d → d)``. Two heads share the
same ``phone_emb``: ``head_cn`` for K=2 Chinese (jyutping ∪ pinyin share
the head — their phone-vocab id ranges are disjoint so the same Linear
learns alphabet-specific patterns via embedding lookup), ``head_cmu`` for
K=6 English.

See ``docs/inpaint_v11_spec.md`` for the design rationale.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from soulxpodcast.inpaint._vocab import TOTAL_VOCAB_SIZE


# Alphabet id space — used by dataset and inference to tell the composer
# which head to route a row through. Keep in sync with
# `K_ACTIVE_PER_ALPHABET` below.
ALPHABET_JYUTPING: int = 0
ALPHABET_PINYIN: int = 1
ALPHABET_CMU: int = 2

# Mapping from the user-facing alphabet string (matches `LANG_TO_ALPHABET`
# in `training.inpaint_dataset`) to the integer id the composer expects.
ALPHABET_STR_TO_ID: dict[str, int] = {
    "jyutping": ALPHABET_JYUTPING,
    "pinyin": ALPHABET_PINYIN,
    "cmu": ALPHABET_CMU,
}

# How many slots each alphabet actually fills. The remaining (K_STORAGE -
# K_active) slots are always pad id 0. Storage K matches max active.
K_ACTIVE_PER_ALPHABET: dict[int, int] = {
    ALPHABET_JYUTPING: 2,
    ALPHABET_PINYIN: 2,
    ALPHABET_CMU: 6,
}

# Storage K — must be ≥ max(K_ACTIVE_PER_ALPHABET.values()).
K_STORAGE: int = 6


class PhonemeComposer(nn.Module):
    """Per-alphabet composer over a shared phoneme embedding table.

    Forward returns ``(composed, mask)`` where ``composed`` has shape
    ``(B, L, d_model)`` (zero at positions with no phoneme slot) and
    ``mask`` is a bool ``(B, L)`` True wherever ``composed`` should replace
    the text embedding.

    The two heads:
      * ``head_cn``  — ``Linear(2*d → d) → GELU → Linear(d → d)``.
        Handles jyutping (alphabet 0) AND pinyin (alphabet 1). Reads
        slots [0, 1] (initial, final-with-tone); slots [2..5] are pad.
      * ``head_cmu`` — ``Linear(6*d → d) → GELU → Linear(d → d)``.
        Reads all 6 slots. Tone information is encoded by the vocab's
        ``_on``/``_co`` consonant position tags, not a separate dim.
    """

    K_STORAGE = K_STORAGE
    K_ACTIVE_PER_ALPHABET = K_ACTIVE_PER_ALPHABET

    def __init__(self, d_model: int, slots_per_token: int = K_STORAGE):
        super().__init__()
        if slots_per_token != K_STORAGE:
            raise ValueError(
                f"slots_per_token must equal K_STORAGE={K_STORAGE} for v11; "
                f"got {slots_per_token}"
            )
        self.d_model = d_model
        self.K = K_STORAGE
        self.vocab_size = TOTAL_VOCAB_SIZE

        self.phone_emb = nn.Embedding(TOTAL_VOCAB_SIZE, d_model, padding_idx=0)
        self.head_cn = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.head_cmu = nn.Sequential(
            nn.Linear(6 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.phone_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.phone_emb.weight[0].zero_()  # keep pad row exactly zero

    @torch.no_grad()
    def init_from_text_embed(self, text_embed: nn.Embedding) -> None:
        """Shift non-pad phoneme rows toward the average text-embedding row.

        Keeps composed embeddings on the LLM's input-embedding manifold so
        the frozen backbone sees a sane vector from step 1.
        """
        if text_embed.embedding_dim != self.d_model:
            raise ValueError(
                f"text_embed dim {text_embed.embedding_dim} != composer d_model {self.d_model}"
            )
        avg = text_embed.weight.mean(dim=0)
        self.phone_emb.weight.data[1:] += avg.to(self.phone_emb.weight.dtype).unsqueeze(0)
        self.phone_emb.weight.data[0].zero_()

    def forward(
        self, phone_token: Tensor, alphabet_id: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compose per-text-token phoneme embeddings.

        Args:
            phone_token: LongTensor of shape ``(B, K_STORAGE * L)``.
            alphabet_id: LongTensor of shape ``(B,)``, one of
                ``{ALPHABET_JYUTPING, ALPHABET_PINYIN, ALPHABET_CMU}``.

        Returns:
            composed: (B, L, d_model). Zero at positions with no phoneme slot.
            mask:     (B, L) bool, True where the composer fired.
        """
        if phone_token.dim() != 2:
            raise ValueError(
                f"phone_token must be 2-D (B, K*L); got shape {tuple(phone_token.shape)}"
            )
        B, KL = phone_token.shape
        if KL % self.K != 0:
            raise ValueError(
                f"phone_token width {KL} is not a multiple of K_STORAGE={self.K}"
            )
        L = KL // self.K

        if alphabet_id.dim() != 1 or alphabet_id.shape[0] != B:
            raise ValueError(
                f"alphabet_id must be shape (B={B},); got {tuple(alphabet_id.shape)}"
            )

        ids = phone_token.view(B, L, self.K)           # (B, L, K, )
        slot_emb = self.phone_emb(ids)                 # (B, L, K, d)

        # Position mask: True where ANY slot has a non-pad id.
        position_mask = (ids != 0).any(dim=-1)         # (B, L)

        # Route per row. We compute both heads on each row (cheap relative to
        # the LLM forward) and then select via the alphabet routing mask. This
        # keeps the code branchless on the GPU and avoids irregular indexing.
        # For Chinese: slots [0, 1].
        # For CMU:     slots [0, 1, 2, 3, 4, 5].
        cn_in = slot_emb[:, :, :2, :].reshape(B, L, 2 * self.d_model)
        cmu_in = slot_emb.reshape(B, L, self.K * self.d_model)

        out_cn = self.head_cn(cn_in)
        out_cmu = self.head_cmu(cmu_in)

        # alphabet_id broadcasts to (B, 1, 1) for masking
        is_cmu = (alphabet_id == ALPHABET_CMU).view(B, 1, 1).to(out_cmu.dtype)
        composed = out_cmu * is_cmu + out_cn * (1.0 - is_cmu)

        # Zero positions that had no phoneme.
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
