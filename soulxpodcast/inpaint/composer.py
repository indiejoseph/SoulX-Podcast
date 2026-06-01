"""Phoneme embedding + composer module for pronunciation inpainting.

The composer takes a per-text-token slot buffer of phoneme ids and
produces a ``d_model``-shaped embedding that replaces the LLM's text
embedding at the corresponding position. The LLM backbone stays
frozen during training; only this module's parameters receive
gradients (see ``.claude/skills/cosyvoice-inpaint/SKILL.md`` §2).

Slot layout, identical to the upstream CosyVoice-Inpaint convention
but with ``K`` slots per text token instead of upstream's fixed 4
``[onset, nucleus, coda, tone]``:

    phone_token: LongTensor (B, K * L)
                  per-text-token block: [slot_0, slot_1, ..., slot_{K-1}]
                  id 0 means "no phoneme at this slot"

Composition is mean-pool over non-pad slots → 2-layer MLP, matching
the design notes in ``references/soulx-adaptation.md`` §4.

Auxiliary head: a 3-way linear classifier on the composed embedding
predicting the alphabet (cmu / jyutping / pinyin). Used only by the
trainer to push alphabet signal into the composed embedding so it
can't be recovered purely from neighbouring text context.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from soulxpodcast.inpaint._vocab import (
    CMU_BOUNDARY_ID,
    CMU_CODA_BASE,
    CMU_ONSET_BASE,
    CMU_VOWEL_BASE,
    JP_FINAL_BASE,
    JP_INITIAL_BASE,
    N_CMU_CODA,
    N_CMU_ONSET,
    N_CMU_VOWEL,
    N_JP_FINAL,
    N_JP_INITIAL,
    N_PY_FINAL,
    N_PY_INITIAL,
    PY_FINAL_BASE,
    PY_INITIAL_BASE,
    TOTAL_VOCAB_SIZE,
)

NUM_ALPHABETS: int = 3  # cmu / jyutping / pinyin

# Alphabet label ids used by the auxiliary classifier.
ALPHABET_LABEL: dict[str, int] = {"cmu": 0, "jyutping": 1, "pinyin": 2}
LABEL_TO_ALPHABET: dict[int, str] = {v: k for k, v in ALPHABET_LABEL.items()}


def _build_id_to_alphabet_label() -> Tensor:
    """Lookup: global phoneme id → alphabet label (or -100 for pad).

    Used by the trainer to derive alphabet supervision labels per
    inpainted position from the raw phoneme ids without having to
    plumb a separate label tensor.
    """
    table = torch.full((TOTAL_VOCAB_SIZE,), -100, dtype=torch.long)
    cmu = ALPHABET_LABEL["cmu"]
    table[CMU_VOWEL_BASE : CMU_VOWEL_BASE + N_CMU_VOWEL] = cmu
    table[CMU_ONSET_BASE : CMU_ONSET_BASE + N_CMU_ONSET] = cmu
    table[CMU_CODA_BASE : CMU_CODA_BASE + N_CMU_CODA] = cmu
    table[CMU_BOUNDARY_ID] = cmu
    table[JP_INITIAL_BASE : JP_INITIAL_BASE + N_JP_INITIAL] = ALPHABET_LABEL["jyutping"]
    table[JP_FINAL_BASE : JP_FINAL_BASE + N_JP_FINAL] = ALPHABET_LABEL["jyutping"]
    table[PY_INITIAL_BASE : PY_INITIAL_BASE + N_PY_INITIAL] = ALPHABET_LABEL["pinyin"]
    table[PY_FINAL_BASE : PY_FINAL_BASE + N_PY_FINAL] = ALPHABET_LABEL["pinyin"]
    # index 0 (pad) stays at -100, which is `ignore_index` for cross_entropy.
    return table


class PhonemeComposer(nn.Module):
    """Composes per-text-token phoneme slots into a single d_model embedding.

    Forward returns ``(composed, mask)`` where ``composed`` is shape
    ``(B, L, d_model)`` (zero at positions with no phoneme slot) and
    ``mask`` is a bool ``(B, L)`` that is True wherever the composed
    embedding should replace the text embedding.
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
        self.alphabet_head = nn.Linear(d_model, NUM_ALPHABETS)

        self.register_buffer(
            "id_to_alphabet_label", _build_id_to_alphabet_label(), persistent=False
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.phone_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.phone_emb.weight[0].zero_()  # keep pad row at exactly zero

    @torch.no_grad()
    def init_from_text_embed(self, text_embed: nn.Embedding) -> None:
        """Shift the non-pad phoneme rows toward the average text-embedding row.

        Mirrors the upstream ``init_component_from_text_embed`` heuristic
        (see CosyVoice-Inpaint ``pron_inpaint/modeling.py``): keeps the
        composed embeddings on the LLM's input-embedding manifold so the
        frozen transformer sees a sane vector from step 1.
        """
        if text_embed.embedding_dim != self.d_model:
            raise ValueError(
                f"text_embed dim {text_embed.embedding_dim} != composer d_model {self.d_model}"
            )
        avg = text_embed.weight.mean(dim=0)
        self.phone_emb.weight.data[1:] += avg.to(self.phone_emb.weight.dtype).unsqueeze(0)
        self.phone_emb.weight.data[0].zero_()

    def forward(self, phone_token: Tensor) -> tuple[Tensor, Tensor]:
        """Compose per-text-token phoneme embeddings.

        Args:
            phone_token: LongTensor of shape ``(B, K * L)``.

        Returns:
            composed: (B, L, d_model)
            mask:     (B, L) bool, True at positions with at least one
                      non-pad slot.
        """
        if phone_token.dim() != 2:
            raise ValueError(f"phone_token must be 2-D (B, K*L); got shape {tuple(phone_token.shape)}")
        B, KL = phone_token.shape
        if KL % self.K != 0:
            raise ValueError(
                f"phone_token width {KL} is not a multiple of slots_per_token={self.K}"
            )
        L = KL // self.K

        ids = phone_token.view(B, L, self.K)
        slot_emb = self.phone_emb(ids)  # (B, L, K, D)
        slot_mask = ids != 0  # (B, L, K)

        denom = slot_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(slot_emb.dtype)
        pooled = (slot_emb * slot_mask.unsqueeze(-1).to(slot_emb.dtype)).sum(dim=-2) / denom

        composed = self.composer(pooled)  # (B, L, D)
        position_mask = slot_mask.any(dim=-1)  # (B, L)
        # zero positions that had no phoneme so callers can trust composed[~mask] == 0
        composed = composed * position_mask.unsqueeze(-1).to(composed.dtype)
        return composed, position_mask

    def alphabet_labels(self, phone_token: Tensor) -> Tensor:
        """Per-text-token alphabet supervision derived from the first non-pad slot.

        Returns LongTensor ``(B, L)`` with values in ``{0, 1, 2, -100}``
        where -100 marks positions that have no phoneme (ignore in CE).
        """
        if phone_token.dim() != 2:
            raise ValueError("phone_token must be 2-D (B, K*L)")
        B, KL = phone_token.shape
        L = KL // self.K
        ids = phone_token.view(B, L, self.K)
        labels = self.id_to_alphabet_label[ids]  # (B, L, K), -100 for pad
        # take the first non-ignore label along the slot axis
        is_valid = labels != -100
        # argmax of bool gives index of first True (or 0 if all False)
        first_valid = is_valid.float().argmax(dim=-1, keepdim=True)  # (B, L, 1)
        gathered = labels.gather(-1, first_valid).squeeze(-1)  # (B, L)
        # rows with no valid slot retain -100
        has_any = is_valid.any(dim=-1)
        gathered = torch.where(has_any, gathered, gathered.new_full(gathered.shape, -100))
        return gathered


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
