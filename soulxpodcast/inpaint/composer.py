"""Phoneme embedding + composer module for pronunciation inpainting.

The composer takes a per-text-token slot buffer of phoneme ids and
produces a ``d_model``-shaped embedding that replaces the LLM's text
embedding at the corresponding position. The LLM backbone stays
frozen during training; only this module's parameters receive
gradients (see ``.claude/skills/cosyvoice-inpaint/SKILL.md`` §2).

Slot layout, K=8 per text token:

    phone_token: LongTensor (B, K * L)
                  per-text-token block: [slot_0, slot_1, ..., slot_{K-1}]
                  id 0 means "no phoneme at this slot"

Composition: **concat + linear** (matches the upstream CosyVoice-Inpaint
``concat_linear`` mode at
``third_party/CosyVoice-Inpaint/pron_inpaint/modeling.py::compose_phoneme``).
Each slot index ``i ∈ [0..K-1]`` always feeds the same dedicated weight
block of the first ``Linear(K*d → d)`` layer, so the MLP can learn
slot-specific feature transformations (e.g., slot 0 = initial-position,
slot 1 = final-position for Chinese; slot N = N-th ARPAbet position for
English). This preserves slot identity by construction.

We deliberately do NOT mean-pool the slots, because mean-pool is
commutative — it destroys the (initial, final) vs (final, initial)
distinction that the LLM needs to disambiguate phoneme content. With
abundant data (Cantonese, 261K rows) the phone_emb rows separate enough
that even mean-pool works; with sparse data (Mandarin, 11.7K rows) it
collapses to silent-token attractors. The upstream ``concat_linear``
approach is robust to data sparsity because each slot has dedicated
weights regardless of how many syllables in that role were seen.

Earlier versions (v1-v6) carried a 3-way auxiliary alphabet classifier
head on the composed embedding. Removed in v7 — the loss was always
~0 because phoneme ids live in disjoint per-alphabet ranges, so the
alphabet is trivially decodable from the input id alone. The head was
solving an already-solved problem and contributed no useful gradient.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from soulxpodcast.inpaint._vocab import TOTAL_VOCAB_SIZE


class PhonemeComposer(nn.Module):
    """Composes per-text-token phoneme slots into a single d_model embedding.

    Architecture (v10 — mean-pool + linear MLP):

      slot_emb[k] = phone_emb(ids[..., k])                # (B, L, K, d)
      pooled     = mean(slot_emb over non-pad slots)      # (B, L, d)
      composed   = Linear(d → d)(pooled) → GELU → Linear(d → d)

    v1-v3 used mean-pool, v4-v9 used concat+linear (each slot fed a
    dedicated weight block) on the theory that slot-order matters
    (initial vs final). After the v9 dataset fix (filter zh AISHELL-3
    silence-heavy rows) the original motivation for concat+linear
    dissolves — phoneme IDs live in disjoint per-alphabet vocab ranges,
    so slot role is already encoded in the ID. Mean-pool drops composer
    params 38.9M → 9.6M (-75%), which is healthier on the small filtered
    zh slice (5,625 rows).

    The mean ignores pad slots (id=0): pad rows are zero by
    ``padding_idx=0`` and the denominator counts non-pad slots only.

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
        # Mean-pool composer: a single Linear(d → d) projects the
        # pooled embedding. The per-slot role information is preserved
        # by the phoneme vocab layout — IDs in CMU/JP/PY ranges are
        # disjoint, so e.g. JP initial 'b' and JP final 'a1' embed to
        # different rows and their mean is unique. Order info ("which
        # slot was initial") is lost but redundant given the ID ranges.
        self.composer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # NOTE on the removed output LayerNorm (v6 → v8): early v6 added
        # an LN here, calibrated to text_embed.weight's per-channel
        # stats, to "anchor the composed embedding to the LLM manifold".
        # Empirically wrong: on the H100 v7 run the composer barely
        # beat the text-only baseline (gap_vs_text +0.05, full_vs_text
        # +0.15) and |grad| collapsed to ~0.07. The LN was renormalising
        # the composer's per-row mean/var on every forward, squashing
        # exactly the directions the composer needed to learn — letting
        # it only modify embedding *direction*, not magnitude. Removed
        # in v8. ``init_from_text_embed`` (next method) still seeds
        # phone_emb rows near the avg text embedding so the frozen LLM
        # sees a reasonable vector from step 0.
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


def filter_compatible_state_dict(
    composer: PhonemeComposer, state_dict: dict
) -> tuple[dict, list[str]]:
    """Drop keys whose shape doesn't match the current composer module.

    v10 changed the composer's first Linear from ``(K*d, d) → d`` to
    ``(d, d) → d`` (mean-pool replaces concat). Any pre-v10 checkpoint
    has ``composer.0.weight`` with shape ``(d, K*d)`` which won't load
    into v10's ``(d, d)`` shape. We drop such keys here and let
    ``load_state_dict(strict=False)`` use the random init for them.

    Returns (filtered_state_dict, dropped_keys).
    """
    own = composer.state_dict()
    filtered: dict = {}
    dropped: list[str] = []
    for k, v in state_dict.items():
        if k in own and own[k].shape != v.shape:
            dropped.append(f"{k} {tuple(v.shape)} → {tuple(own[k].shape)}")
            continue
        filtered[k] = v
    return filtered, dropped


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
