"""MTP speculative decoding for SoulX-Podcast (PLAN.md Phase 2, inference side).

Greedy Medusa-style speculative decoding using K-1 trained MTP heads.

Per step:
  1. Trunk forward → h_t (last hidden state of accumulated sequence)
  2. Primary head:  ŷ_{t+1} = argmax(lm_head(h_t))
  3. MTP heads draft K-1 more tokens sequentially:
       ŷ_{t+k+1} = argmax(lm_head( MTP_k(h'_{k-1}, embed(ŷ_{t+k})) ))
  4. Validate: run trunk on the drafted block and compare predictions.
  5. Accept the longest matching prefix; resample first mismatch from trunk.
  6. Commit accepted prefix + 1 bonus token; loop.

Each step commits between 1 (no acceptance) and K+1 (all drafts accepted +
bonus) tokens. Average acceptance length × per-step cost ratio = speedup.

This implementation does NOT yet manage KV cache (each step re-runs the
trunk on the full accumulated sequence) — it's correct for measuring
acceptance length, but not for measuring wall-time speedup. Cache management
is the obvious next optimization once acceptance behavior is validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SpecDecodeResult:
    """Output of one speculative-decoding generation."""
    tokens: torch.LongTensor               # [1, T_total] — full sequence including prompt
    generated_tokens: torch.LongTensor      # [1, T_new]   — only the newly generated portion
    accept_lengths: List[int]              # accepted drafts per step (1 = only primary accepted)
    n_steps: int                           # number of speculative steps
    n_committed: int                       # total tokens committed (excluding prompt)
    eos_hit: bool
    # Optional timing breakdown (seconds). Filled by cached impl; 0 in no-cache.
    t_trunk: float = 0.0                   # all trunk forwards combined (verify + extend)
    t_mtp: float = 0.0                     # all MTP layer forwards combined

    @property
    def mean_accept_length(self) -> float:
        return sum(self.accept_lengths) / max(len(self.accept_lengths), 1)

    @property
    def tokens_per_step(self) -> float:
        """Average tokens committed per trunk forward pair (= drafting + verify).

        Each step commits `accept_length + 1` tokens (the bonus token from
        the trunk's prediction at the divergence point). With K total heads,
        max per-step = K+1 (all K drafts accepted + 1 bonus).
        """
        return self.n_committed / max(self.n_steps, 1)


def _mtp_layer_forward_seq(
    layer: nn.Module,
    prev_h: torch.Tensor,        # [1, L, H]
    target_embed: torch.Tensor,  # [1, L, H]
    position_ids: torch.LongTensor,  # [1, L]
    rotary_emb: nn.Module,
    causal_mask_4d: torch.Tensor,    # [1, 1, L, L]
) -> torch.Tensor:
    """Full-sequence MTP layer forward. Returns [1, L, H].

    This matches the training-time forward pass: the transformer self-attends
    over all L positions with a causal mask. At inference we then take the
    last position as our prediction for the next draft.
    """
    mixed = torch.cat([layer.norm_h(prev_h), layer.norm_e(target_embed)], dim=-1)
    h = layer.proj(mixed)
    cos_sin = rotary_emb(h, position_ids)
    out = layer.transformer(
        hidden_states=h,
        attention_mask=causal_mask_4d,
        position_ids=position_ids,
        position_embeddings=cos_sin,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def _build_causal_mask(L: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Standard 4-D additive causal mask for Qwen3 attention. Shape [1, 1, L, L]."""
    min_val = torch.finfo(dtype).min
    mask = torch.full((L, L), min_val, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def mtp_speculative_decode(
    base,                              # AutoModelForCausalLM (frozen)
    mtp,                                # SequentialMTP
    input_ids: torch.LongTensor,        # [1, T_prompt]
    *,
    max_new_tokens: int = 500,
    eos_token_id: Optional[int] = None,
    K: Optional[int] = None,            # tokens per step = 1 primary + (K-1) MTP drafts
) -> SpecDecodeResult:
    """Greedy MTP speculative decoding. Returns full generated sequence + stats."""
    device = input_ids.device
    embed_tokens = base.model.embed_tokens
    lm_head = base.lm_head
    rotary_emb = base.model.rotary_emb

    n_mtp_layers = len(mtp.layers)
    if K is None:
        K = n_mtp_layers + 1  # 1 primary + n_mtp drafts

    # Number of MTP drafts to actually use (cap at available layers).
    n_drafts = min(K - 1, n_mtp_layers)

    full_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]
    accept_lengths: List[int] = []
    eos_hit = False
    n_steps = 0

    while full_ids.shape[1] - prompt_len < max_new_tokens:
        T = full_ids.shape[1]

        # ---- Step 1: trunk forward on the full accumulated sequence -----
        out = base.model(
            input_ids=full_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=False,
        )
        trunk_hidden_full = out.last_hidden_state  # [1, T, H]
        h_t = trunk_hidden_full[:, -1:]            # [1, 1, H]

        # ---- Step 2: primary head's pick ---------------------------------
        primary_logit = lm_head(h_t)               # [1, 1, V]
        d0 = primary_logit.argmax(dim=-1)          # [1, 1]
        drafts = [d0]

        # ---- Step 3: MTP heads draft additional tokens -------------------
        # Match the training distribution exactly: run each MTP layer over a
        # FULL sequence (positions 0..T-1) so the transformer block sees the
        # same kind of self-attention context it learned with. Take the last
        # position's output as our prediction.
        #
        # The target_embed for MTP layer k at position p in training was
        # embed(input_ids[p+k]). At inference we have input_ids = full_ids
        # of length T plus the drafted token(s). We construct target_embed
        # of length T per layer:
        #   layer 1: target_embed[p] = embed(full_ids[p+1]) for p<T-1,
        #            target_embed[T-1] = embed(d_0)
        #   layer 2: target_embed[p] = embed(full_ids[p+2]) for p<T-2,
        #            target_embed[T-2] = embed(d_0),
        #            target_embed[T-1] = embed(d_1)
        #   layer k: shift by k, with the LAST k positions filled by drafts.
        prev_h = trunk_hidden_full          # [1, T, H]
        pos_ids_full = torch.arange(T, device=device).unsqueeze(0)
        causal_mask_4d = _build_causal_mask(T, prev_h.dtype, device)

        for k in range(1, n_drafts + 1):
            # Construct target token ids of length T, shifted by k.
            # Drafts so far: drafts[0..k-1] (k tokens). Of those, the last
            # k that fall into the trailing slice get appended.
            shift_from_full = full_ids[:, k:]    # [1, T-k]
            draft_tail = torch.cat(drafts[:k], dim=1)  # [1, k]
            tgt_ids = torch.cat([shift_from_full, draft_tail], dim=1)  # [1, T]
            target_embed = embed_tokens(tgt_ids)  # [1, T, H]

            out_k = _mtp_layer_forward_seq(
                mtp.layers[k - 1], prev_h, target_embed,
                pos_ids_full, rotary_emb, causal_mask_4d,
            )  # [1, T, H]
            # Prediction is at the last position; lm_head it.
            d_k = lm_head(out_k[:, -1:]).argmax(dim=-1)  # [1, 1]
            drafts.append(d_k)
            prev_h = out_k                              # next MTP layer sees the FULL sequence output

        # drafts = [d_0, d_1, ..., d_{n_drafts}]  total K tokens
        draft_seq = torch.cat(drafts, dim=1)        # [1, K]

        # ---- Step 4: validate by running trunk on full_ids + draft_seq ---
        extended = torch.cat([full_ids, draft_seq], dim=1)  # [1, T+K]
        out2 = base.model(
            input_ids=extended, use_cache=False, return_dict=True,
            output_hidden_states=False,
        )
        # Hidden at position T+i-1 predicts token at position T+i.
        # We need predictions for positions T, T+1, ..., T+K-1 → use hidden
        # at positions T-1, T, ..., T+K-2.
        verify_hidden = out2.last_hidden_state[:, T - 1 : T - 1 + draft_seq.shape[1]]  # [1, K, H]
        verify_logits = lm_head(verify_hidden)       # [1, K, V]
        verify_ids = verify_logits.argmax(dim=-1)     # [1, K]

        # ---- Step 5: accept longest matching prefix ----------------------
        # d_0 always equals verify_ids[0,0] by construction (both = argmax(lm_head(h_t))).
        # For k=1..n_drafts: accept if d_k == verify_ids[0,k].
        n_accept = 1
        K_cur = draft_seq.shape[1]
        for k in range(1, K_cur):
            if int(draft_seq[0, k].item()) == int(verify_ids[0, k].item()):
                n_accept += 1
            else:
                break

        # ---- Step 6: commit accepted prefix + 1 bonus token --------------
        # Bonus = trunk's pick at the divergence point (or one past the end
        # if all accepted). Either way it's verify_ids[0, n_accept-1] when
        # n_accept == K_cur, or verify_ids[0, n_accept] when n_accept < K_cur.
        # Simpler formulation: trunk_logits[n_accept] is always the right
        # bonus (predicting position T + n_accept), but we only have hidden
        # up to T+K-1 so n_accept ≤ K_cur. When n_accept == K_cur, bonus = verify[K-1].
        accepted = draft_seq[:, :n_accept]
        if n_accept < K_cur:
            # Replace the first rejected draft with trunk's pick.
            replacement = verify_ids[:, n_accept : n_accept + 1]
            committed = torch.cat([accepted, replacement], dim=1)
        else:
            # All drafts accepted — the trunk's prediction after the last
            # draft is a free extra token.
            bonus = verify_ids[:, n_accept - 1 : n_accept]
            committed = torch.cat([accepted, bonus], dim=1)

        full_ids = torch.cat([full_ids, committed], dim=1)
        accept_lengths.append(n_accept)
        n_steps += 1

        if eos_token_id is not None and (committed == eos_token_id).any().item():
            eos_hit = True
            # Truncate at the EOS.
            committed_list = committed[0].tolist()
            eos_pos = committed_list.index(eos_token_id)
            keep_n = eos_pos + 1
            extra = committed.shape[1] - keep_n
            if extra > 0:
                full_ids = full_ids[:, :-extra]
            break

    generated_tokens = full_ids[:, prompt_len:]
    return SpecDecodeResult(
        tokens=full_ids,
        generated_tokens=generated_tokens,
        accept_lengths=accept_lengths,
        n_steps=n_steps,
        n_committed=int(generated_tokens.shape[1]),
        eos_hit=eos_hit,
        t_trunk=0.0,
        t_mtp=0.0,
    )


@torch.inference_mode()
def baseline_greedy_decode(
    base,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int = 500,
    eos_token_id: Optional[int] = None,
) -> Tuple[torch.LongTensor, int]:
    """Trunk-only greedy autoregressive baseline (one trunk forward per token,
    no KV cache — matches the unfair-but-comparable apples-to-apples setup of
    `mtp_speculative_decode`)."""
    full_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]
    lm_head = base.lm_head
    n_steps = 0
    while full_ids.shape[1] - prompt_len < max_new_tokens:
        out = base.model(input_ids=full_ids, use_cache=False, return_dict=True)
        next_logit = lm_head(out.last_hidden_state[:, -1:])
        next_id = next_logit.argmax(dim=-1)
        full_ids = torch.cat([full_ids, next_id], dim=1)
        n_steps += 1
        if eos_token_id is not None and int(next_id.item()) == eos_token_id:
            break
    return full_ids, n_steps


@torch.inference_mode()
def baseline_greedy_decode_cached(
    base,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int = 500,
    eos_token_id: Optional[int] = None,
) -> Tuple[torch.LongTensor, int]:
    """KV-cached greedy autoregressive baseline. This is the fair comparison
    against the KV-cached spec decoder — both use cache, so the only
    difference is the speculative multi-token-per-step pattern."""
    lm_head = base.lm_head
    full_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]

    # Initial prompt forward populates the cache.
    out = base.model(input_ids=full_ids, use_cache=True, return_dict=True)
    cache = out.past_key_values
    last_hidden = out.last_hidden_state[:, -1:]

    n_steps = 0
    while full_ids.shape[1] - prompt_len < max_new_tokens:
        next_id = lm_head(last_hidden).argmax(dim=-1)  # [1, 1]
        full_ids = torch.cat([full_ids, next_id], dim=1)
        n_steps += 1
        if eos_token_id is not None and int(next_id.item()) == eos_token_id:
            break
        # Extend cache by 1 token and refresh last_hidden.
        out = base.model(
            input_ids=next_id, past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        cache = out.past_key_values
        last_hidden = out.last_hidden_state[:, -1:]
    return full_ids, n_steps


@torch.inference_mode()
def mtp_speculative_decode_cached(
    base,                              # AutoModelForCausalLM (frozen)
    mtp,                                # SequentialMTP
    input_ids: torch.LongTensor,        # [1, T_prompt]
    *,
    max_new_tokens: int = 500,
    eos_token_id: Optional[int] = None,
    K: Optional[int] = None,
) -> SpecDecodeResult:
    """KV-cached MTP speculative decoding. The win over `mtp_speculative_decode`:

      - Prompt is forwarded through trunk once (not every spec step).
      - Validation only processes the K drafted tokens (not the full sequence).
      - One extra trunk forward per step (for the bonus/replacement token).

    Per-step trunk cost: O(K+1) tokens, independent of accumulated length.
    Compare to no-cache spec: O(2T+K) per step, growing with T.

    MTP layers are still run on the full sequence per step (using the
    incrementally-maintained `trunk_hidden_full`). MTP-side caching is a
    further optimization not implemented here — MTP is much cheaper than the
    trunk so the marginal saving is small.
    """
    device = input_ids.device
    embed_tokens = base.model.embed_tokens
    lm_head = base.lm_head
    rotary_emb = base.model.rotary_emb

    n_mtp_layers = len(mtp.layers)
    if K is None:
        K = n_mtp_layers + 1
    n_drafts = min(K - 1, n_mtp_layers)

    full_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]
    accept_lengths: List[int] = []
    eos_hit = False
    n_steps = 0

    # Initial trunk forward → cache + full hidden states.
    out_init = base.model(input_ids=full_ids, use_cache=True, return_dict=True)
    cache = out_init.past_key_values
    trunk_hidden_full = out_init.last_hidden_state  # [1, T_prompt, H]

    t_trunk = 0.0
    t_mtp = 0.0
    import time as _time

    while full_ids.shape[1] - prompt_len < max_new_tokens:
        T = full_ids.shape[1]
        h_t = trunk_hidden_full[:, -1:]  # [1, 1, H]

        # ---- Drafts: primary head + MTP heads (full sequence each layer) ----
        d_0 = lm_head(h_t).argmax(dim=-1)
        drafts = [d_0]

        prev_h = trunk_hidden_full
        pos_ids_full = torch.arange(T, device=device).unsqueeze(0)
        causal_mask_4d = _build_causal_mask(T, prev_h.dtype, device)
        torch.cuda.synchronize()
        _t_mtp_start = _time.perf_counter()
        for k in range(1, n_drafts + 1):
            shift_from_full = full_ids[:, k:]
            draft_tail = torch.cat(drafts[:k], dim=1)
            tgt_ids = torch.cat([shift_from_full, draft_tail], dim=1)
            target_embed = embed_tokens(tgt_ids)
            out_k = _mtp_layer_forward_seq(
                mtp.layers[k - 1], prev_h, target_embed,
                pos_ids_full, rotary_emb, causal_mask_4d,
            )
            d_k = lm_head(out_k[:, -1:]).argmax(dim=-1)
            drafts.append(d_k)
            prev_h = out_k
        torch.cuda.synchronize()
        t_mtp += _time.perf_counter() - _t_mtp_start

        draft_seq = torch.cat(drafts, dim=1)  # [1, K]
        K_cur = draft_seq.shape[1]

        # ---- Validation: forward drafts through trunk with cache ----
        torch.cuda.synchronize()
        _t_trunk_start = _time.perf_counter()
        verify_out = base.model(
            input_ids=draft_seq, past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        cache = verify_out.past_key_values  # extended by K
        verify_hidden = verify_out.last_hidden_state    # [1, K, H]
        verify_ids = lm_head(verify_hidden).argmax(dim=-1)  # [1, K]
        # verify_ids[i] predicts position T+i+1 (i.e. should match drafts[i+1]).

        # ---- Accept ----
        # drafts[0] (= primary head pick) always matches trunk by construction.
        n_accept = 1
        for i in range(1, K_cur):
            if int(draft_seq[0, i].item()) == int(verify_ids[0, i - 1].item()):
                n_accept += 1
            else:
                break

        # ---- Bonus / replacement ----
        if n_accept == K_cur:
            # All drafts accepted; bonus is trunk's pred at position T+K (one beyond).
            bonus_id = verify_ids[:, K_cur - 1 : K_cur]
        else:
            # First mismatch at draft index n_accept (1-based: n_accept tokens accepted).
            # Replace with trunk's pred for that position.
            bonus_id = verify_ids[:, n_accept - 1 : n_accept]

        # ---- Cache management ----
        # Cache has T + K positions. Keep T + n_accept (prompt + accepted drafts).
        # The position about to be filled (T + n_accept) currently holds the
        # rejected/extra draft's K/V — must crop before re-forwarding.
        if hasattr(cache, "crop"):
            cache.crop(T + n_accept)
        else:
            # Older transformers cache APIs use `.cache_position` or similar.
            # Fall back path: re-forward the accepted prefix from a fresh cache.
            # (Not implemented — modern transformers should have crop.)
            raise NotImplementedError("DynamicCache.crop() required for spec decoding")

        # Forward bonus to extend cache + get its hidden state.
        out_extend = base.model(
            input_ids=bonus_id, past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        cache = out_extend.past_key_values
        bonus_hidden = out_extend.last_hidden_state  # [1, 1, H]
        torch.cuda.synchronize()
        t_trunk += _time.perf_counter() - _t_trunk_start

        # Maintain trunk_hidden_full incrementally.
        accepted_hidden = verify_hidden[:, :n_accept]  # [1, n_accept, H]
        trunk_hidden_full = torch.cat(
            [trunk_hidden_full, accepted_hidden, bonus_hidden], dim=1
        )

        # Update full_ids.
        accepted = draft_seq[:, :n_accept]
        committed = torch.cat([accepted, bonus_id], dim=1)
        full_ids = torch.cat([full_ids, committed], dim=1)
        accept_lengths.append(n_accept)
        n_steps += 1

        if eos_token_id is not None and (committed == eos_token_id).any().item():
            eos_hit = True
            committed_list = committed[0].tolist()
            eos_pos = committed_list.index(eos_token_id)
            keep_n = eos_pos + 1
            extra = committed.shape[1] - keep_n
            if extra > 0:
                full_ids = full_ids[:, :-extra]
            break

    generated_tokens = full_ids[:, prompt_len:]
    return SpecDecodeResult(
        tokens=full_ids,
        generated_tokens=generated_tokens,
        accept_lengths=accept_lengths,
        n_steps=n_steps,
        n_committed=int(generated_tokens.shape[1]),
        eos_hit=eos_hit,
        t_trunk=t_trunk,
        t_mtp=t_mtp,
    )
