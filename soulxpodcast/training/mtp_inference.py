"""MTP speculative decoding for SoulX-Podcast (PLAN.md Phase 2, inference side).

Two decoding variants:

  1. `mtp_speculative_decode_cached` — GREEDY validation (argmax-match).
     Correct only when downstream uses greedy decoding. Simple, fast,
     useful for correctness benchmarks.

  2. `mtp_speculative_sample_cached` — Leviathan-Kalman speculative sampling.
     Preserves the trunk's sampling distribution for the normal sampling
     processors (temperature / top-K / top-P / repetition penalty). Required
     for production deployment where the existing SamplingParams are
     non-trivial — the SoulX-Podcast model is trained with sampling at
     inference and greedy output sounds robotic.

The Leviathan-Kalman acceptance rule:
    u ~ Uniform(0, 1)
    accept if u < min(1, p(x) / q(x))   where x is the drafted token,
                                          q = draft distribution,
                                          p = target (trunk) distribution
Strictly, on reject the replacement should come from `max(0, p - q)`
normalized, which gives output ~ p exactly. The cached sampled decoder below
uses a cheaper trunk-resample approximation on the rejection path; when q is
close to p, the bias is small, and the audio A/B harness is meant to catch
practical regressions.

NOTE on RAS: this implementation can apply SoulX-Podcast's Repetition-Aware
Sampling branch in both trunk and draft samplers. RAS is a stochastic reset
on top of the filtered distribution, so the exact distributional guarantee is
less clean than plain top-k/top-p sampling, but the branch is matched between
the sampled baseline and speculative path for practical timing and audio A/Bs.

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

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    return layer.final_norm(out)


def _build_causal_mask(L: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Standard 4-D additive causal mask for Qwen3 attention. Shape [1, 1, L, L]."""
    min_val = torch.finfo(dtype).min
    mask = torch.full((L, L), min_val, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _fork_dynamic_cache(cache):
    """Copy cache containers while sharing read-only prefix KV tensors.

    Transformers DynamicCache mutates layer containers by assigning concatenated
    key/value tensors. Copying the containers keeps the cached prefix tensors
    shared, while preventing per-request crop/append operations from modifying
    the prompt-cache entry.
    """
    forked = copy.copy(cache)
    if hasattr(cache, "layers"):
        forked.layers = [copy.copy(layer) for layer in cache.layers]
    return forked


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


# ============================================================================
# Sampling-aware speculative decoding (Leviathan & Kalman 2023)
# ============================================================================


def _apply_sampling_processors(
    logits: torch.Tensor,            # [B, V]
    context_ids: torch.LongTensor,    # [B, T] — for repetition penalty
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Apply HF-style sampling processors. Order matches transformers' default:
    repetition_penalty → top_k → top_p → temperature.

    Returns logits with -inf at filtered positions; subsequent softmax produces
    the proper sampling distribution.
    """
    out = logits.clone()

    # Repetition penalty: tokens appearing in context have their score divided
    # by penalty if positive, multiplied if negative (HF convention).
    if repetition_penalty != 1.0 and context_ids.numel() > 0:
        # context_ids is [B, T]; gather scores of those positions.
        # Use unique per-batch for efficiency.
        for b in range(out.size(0)):
            ids = context_ids[b].unique()
            scores = out[b, ids]
            scores = torch.where(scores > 0,
                                 scores / repetition_penalty,
                                 scores * repetition_penalty)
            out[b, ids] = scores

    # Temperature (apply BEFORE filtering so top-K/P operate on softened dist).
    if temperature != 1.0:
        out = out / max(temperature, 1e-6)

    # Top-K: keep only top K, mask rest.
    if top_k > 0:
        k = min(top_k, out.size(-1))
        kth = out.topk(k, dim=-1).values[..., -1, None]
        out = torch.where(out < kth, torch.full_like(out, -float("inf")), out)

    # Top-P (nucleus): keep smallest set with cumulative prob >= p; mask rest.
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Tokens to remove: those past the nucleus boundary. Shift right to
        # always keep the top-1.
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        # Map back to original ordering and mask.
        indices_to_remove = remove.scatter(-1, sorted_idx, remove)
        out = out.masked_fill(indices_to_remove, -float("inf"))

    return out


def _sample_with_ras(
    raw_logits: torch.Tensor,         # [1, V] — raw model output
    context_ids: torch.LongTensor,    # [1, T] — for rep_penalty + RAS window
    *,
    temperature: float, top_k: int, top_p: float, repetition_penalty: float,
    use_ras: bool, win_size: int, tau_r: float,
    generator: Optional[torch.Generator],
) -> tuple:
    """Apply processors → optionally RAS → sample. Returns (token, log_prob).

    `log_prob` is under the EFFECTIVE distribution used to sample (i.e. raw
    distribution if RAS fired, filtered otherwise). This is what the
    Leviathan-Kalman acceptance ratio needs as `q(d_k)`.
    """
    filtered = _apply_sampling_processors(
        raw_logits, context_ids,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    def _safe_sample(logits):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        if not torch.isfinite(probs).any() or probs.sum() == 0:
            tok = logits.argmax(dim=-1, keepdim=True)
        else:
            tok = torch.multinomial(probs, num_samples=1, generator=generator)
        return tok, log_probs.gather(-1, tok)

    if not use_ras:
        return _safe_sample(filtered)

    # RAS step 1: sample candidate from filtered, check repetition.
    candidate, _ = _safe_sample(filtered)
    window = context_ids[:, -win_size:] if context_ids.size(1) > win_size else context_ids
    rep_count = (window == candidate).sum().item() + 1

    if rep_count < win_size * tau_r:
        # Candidate is fine — commit it. Reuse its filtered log-prob.
        filtered_log_probs = F.log_softmax(filtered, dim=-1)
        return candidate, filtered_log_probs.gather(-1, candidate)

    # RAS fires: discard filtered, fall back to RAW logits (no processors).
    # This lets the model break out of repetition loops by escaping the
    # rep_penalty / top_k / top_p / temperature filter.
    return _safe_sample(raw_logits)


def _sample_residual(p_probs: torch.Tensor, q_probs: torch.Tensor,
                     generator: Optional[torch.Generator] = None) -> torch.LongTensor:
    """Sample from the residual distribution max(0, p - q) / normalizer.

    Used when a draft is rejected: the replacement comes from the residual,
    ensuring the overall output distribution remains p exactly.
    """
    residual = (p_probs - q_probs).clamp_min(0.0)
    total = residual.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    residual = residual / total
    return torch.multinomial(residual, num_samples=1, generator=generator)


@torch.inference_mode()
def mtp_speculative_sample_cached(
    base,                                # AutoModelForCausalLM (frozen)
    mtp,                                  # SequentialMTP
    input_ids: torch.LongTensor,          # [1, T_prompt]
    *,
    max_new_tokens: int = 500,
    min_new_tokens: int = 0,
    eos_token_id: Optional[int] = None,
    K: Optional[int] = None,              # 1 primary + (K-1) MTP drafts
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    # Repetition-Aware Sampling (VALL-E 2 style; matches production sampler).
    # When the sampled candidate appears too often in the last `win_size`
    # committed tokens, reset to raw model logits (no temp/top_k/top_p/rep_pen)
    # and resample. Helps break degenerate repetition loops, particularly on
    # languages where the base model is under-trained (e.g. HK Cantonese).
    use_ras: bool = False,
    ras_win_size: int = 25,
    ras_tau_r: float = 0.2,
    allow_eos_from_drafts: bool = False,
    seed: Optional[int] = None,
    streamer=None,
    prefix_cache=None,
    prefix_hidden: Optional[torch.Tensor] = None,
    prefix_len: int = 0,
) -> SpecDecodeResult:
    """KV-cached MTP speculative decoding with sampling-aware validation.

    The acceptance test is based on the trunk's sampling distribution under
    the given temperature/top_k/top_p/repetition_penalty — NOT just greedy.
    This is what's needed for SoulX-Podcast production deployment, where the
    existing SamplingParams (top_k=100, top_p=0.9, temperature=0.6, etc.)
    must be preserved to maintain audio quality.

    Algorithm per spec step:
      1. Trunk forward → primary distribution p_0 (with sampling processors)
      2. Sample d_0 ~ p_0  (always accepted; d_0 IS from the target distribution)
      3. MTP heads draft d_1..d_{K-1} from their distributions q_1..q_{K-1}
         (each q_k computed with the same sampling processors applied)
      4. Verify: trunk forward on drafts → true distributions p_1..p_{K-1}
         (with sampling processors using committed + drafts[0..k-1] for rep penalty)
      5. For k=1..K-1:
           u ~ Uniform(0, 1)
           if u < min(1, p_k(d_k) / q_k(d_k)): accept
           else: reject; sample bonus from max(0, p_k - q_k) normalized; break
      6. If all accepted: sample bonus ~ p_K (one position beyond)
      7. Commit accepted prefix + bonus; loop.

    See `mtp_speculative_decode_cached` for the greedy variant and shared
    KV-cache plumbing details.
    """
    device = input_ids.device
    embed_tokens = base.model.embed_tokens
    lm_head = base.lm_head
    rotary_emb = base.model.rotary_emb

    n_mtp_layers = len(mtp.layers)
    if K is None:
        K = n_mtp_layers + 1
    n_drafts = min(K - 1, n_mtp_layers)

    # RNG generator for reproducibility.
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    def _sample_from_logits(logits: torch.Tensor) -> torch.LongTensor:
        """Sample one token from already-processed logits. Returns [1, 1]."""
        probs = F.softmax(logits, dim=-1)
        if not torch.isfinite(probs).any() or probs.sum() == 0:
            return logits.argmax(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1, generator=gen)

    def _process(logits_1d: torch.Tensor,
                 context_ids: torch.LongTensor) -> torch.Tensor:
        """Apply sampling processors to a [1, V] logit row."""
        return _apply_sampling_processors(
            logits_1d, context_ids,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    def _ras_sample(raw_1d: torch.Tensor,
                    context_ids: torch.LongTensor,
                    *,
                    allow_eos: bool = True) -> tuple:
        """Sample with processors + RAS. Returns (token, log_prob_effective)."""
        raw_1d = _mask_eos_if_needed(raw_1d, context_ids, allow_eos=allow_eos)
        return _sample_with_ras(
            raw_1d, context_ids,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_ras=use_ras, win_size=ras_win_size, tau_r=ras_tau_r,
            generator=gen,
        )

    def _mask_eos_if_needed(logits: torch.Tensor,
                            context_ids: torch.LongTensor,
                            *,
                            allow_eos: bool = True) -> torch.Tensor:
        """Block EOS before min_new_tokens, and optionally for MTP drafts.

        HF `generate(min_new_tokens=N)` masks EOS while fewer than N tokens have
        already been generated. `allow_eos_from_drafts=False` additionally keeps
        termination in the trunk path: MTP heads draft speech tokens, while the
        trunk primary/replacement/bonus token decides when to stop.
        """
        if eos_token_id is None:
            return logits
        generated_so_far = context_ids.shape[1] - prompt_len
        if allow_eos and generated_so_far >= min_new_tokens:
            return logits
        masked = logits.clone()
        masked[..., eos_token_id] = -float("inf")
        return masked

    full_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]
    accept_lengths: List[int] = []
    eos_hit = False
    n_steps = 0
    if streamer is not None:
        # Match HF generate() streamer behavior: first put is the prompt and
        # SpeechTokenStreamer discards it before consuming generated tokens.
        streamer.put(input_ids.detach().cpu())

    # Initial trunk forward → cache + full hidden states. When a prompt-cache
    # entry provides prefix KV + hidden states, only prefill the per-request
    # target text tail here.
    if prefix_cache is not None:
        if prefix_hidden is None or prefix_len <= 0:
            raise ValueError("prefix_cache requires prefix_hidden and prefix_len")
        if prefix_len > full_ids.shape[1]:
            raise ValueError("prefix_len exceeds input length")
        cache = _fork_dynamic_cache(prefix_cache)
        target_ids = full_ids[:, prefix_len:]
        if target_ids.numel() > 0:
            out_init = base.model(
                input_ids=target_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out_init.past_key_values
            trunk_hidden_full = torch.cat(
                [prefix_hidden.to(out_init.last_hidden_state.device), out_init.last_hidden_state],
                dim=1,
            )
        else:
            trunk_hidden_full = prefix_hidden
    else:
        out_init = base.model(input_ids=full_ids, use_cache=True, return_dict=True)
        cache = out_init.past_key_values
        trunk_hidden_full = out_init.last_hidden_state  # [1, T_prompt, H]

    while full_ids.shape[1] - prompt_len < max_new_tokens:
        T = full_ids.shape[1]
        h_t = trunk_hidden_full[:, -1:]

        # --- d_0: sample from trunk's primary distribution -----------------
        # context for rep_penalty + RAS window = everything committed so far
        primary_raw = lm_head(h_t)[:, 0]                       # [1, V]
        d_0, _ = _ras_sample(primary_raw, full_ids)            # [1, 1]
        drafts = [d_0]

        # --- MTP draft loop: each layer produces q_k, sample d_k ----------
        # Track log q_k(d_k) under the EFFECTIVE draft distribution (raw if
        # RAS fired, filtered otherwise). Needed for the rejection ratio.
        q_log_probs_at_d: List[torch.Tensor] = []  # length K-1, each [1, 1]
        prev_h = trunk_hidden_full
        pos_ids_full = torch.arange(T, device=device).unsqueeze(0)
        causal_mask_4d = _build_causal_mask(T, prev_h.dtype, device)
        for k in range(1, n_drafts + 1):
            shift_from_full = full_ids[:, k:]
            draft_tail = torch.cat(drafts[:k], dim=1)
            tgt_ids = torch.cat([shift_from_full, draft_tail], dim=1)
            target_embed = embed_tokens(tgt_ids)
            out_k = _mtp_layer_forward_seq(
                mtp.layers[k - 1], prev_h, target_embed,
                pos_ids_full, rotary_emb, causal_mask_4d,
            )
            # Context for rep_penalty + RAS at this draft position =
            # committed + drafts[0..k-1]. (drafts[k] is what we're sampling.)
            ctx_k = torch.cat([full_ids] + drafts[:k], dim=1)
            qk_raw = lm_head(out_k[:, -1])                       # [1, V]
            d_k, qk_logp_at_d = _ras_sample(
                qk_raw, ctx_k, allow_eos=allow_eos_from_drafts
            )
            q_log_probs_at_d.append(qk_logp_at_d)               # [1, 1]
            drafts.append(d_k)
            prev_h = out_k

        draft_seq = torch.cat(drafts, dim=1)  # [1, K]
        K_cur = draft_seq.shape[1]

        # --- Validation: trunk forward on drafts with cache ---------------
        verify_out = base.model(
            input_ids=draft_seq, past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        cache = verify_out.past_key_values
        verify_hidden = verify_out.last_hidden_state             # [1, K, H]

        # --- Rejection sampling --------------------------------------------
        # drafts[0] is from trunk directly (p_0 == target), always accepted.
        n_accept = 1
        replacement = None  # set if a draft is rejected
        for k in range(1, K_cur):
            # p_k = trunk's distribution at position T+k-1 (predicts position T+k)
            # context for rep penalty + RAS = committed + drafts[0..k-1]
            ctx_k = torch.cat([full_ids] + drafts[:k], dim=1)
            pk_raw = lm_head(verify_hidden[:, k - 1])  # [1, V]

            # RAS-aware p_k(d_k): if d_k would trigger RAS at this position,
            # the effective sampling distribution is the raw one. Otherwise
            # use filtered. This makes the acceptance ratio comparable to q.
            pk_raw = _mask_eos_if_needed(pk_raw, ctx_k, allow_eos=True)
            if use_ras:
                window = ctx_k[:, -ras_win_size:] if ctx_k.size(1) > ras_win_size else ctx_k
                rep_count = (window == drafts[k]).sum().item() + 1
                if rep_count >= ras_win_size * ras_tau_r:
                    pk_log_eff = F.log_softmax(pk_raw, dim=-1)
                else:
                    pk_filtered = _process(pk_raw, ctx_k)
                    pk_log_eff = F.log_softmax(pk_filtered, dim=-1)
            else:
                pk_filtered = _process(pk_raw, ctx_k)
                pk_log_eff = F.log_softmax(pk_filtered, dim=-1)

            # pk_probs only needed at the drafted position (no need to materialize full softmax)
            p_at_d = pk_log_eff.gather(-1, drafts[k]).exp()       # [1, 1]
            q_at_d = q_log_probs_at_d[k - 1].exp()                # [1, 1]

            # u ~ U(0,1). Accept if u < min(1, p/q).
            u = torch.rand((1, 1), device=device, generator=gen)
            ratio = (p_at_d / q_at_d.clamp_min(1e-20)).clamp(max=1.0)
            if (u < ratio).item():
                n_accept += 1
                continue
            # Rejected: sample replacement.
            # NOTE: strict Leviathan-Kalman would sample from `max(0, p_k - q_k)`
            # normalized; we use p_k directly (slight bias toward trunk's mode
            # but cheaper — avoids re-materializing q_k probs). When q ≈ p
            # (what KD training achieves), the bias is small. We apply
            # processors + RAS here for consistency with production behavior.
            replacement, _ = _ras_sample(pk_raw, ctx_k)
            break

        # --- Determine bonus / replacement token --------------------------
        if replacement is not None:
            # First rejection at position n_accept; commit accepted prefix +
            # the resampled replacement.
            bonus_id = replacement
        elif n_accept == K_cur:
            # All drafts accepted: sample one bonus token from p_K (trunk's
            # prediction at position T+K-1, one beyond all drafts).
            # Use RAS here too — matches production sampling behavior at this
            # position when generating autoregressively.
            ctx_K = torch.cat([full_ids] + drafts, dim=1)
            pK_raw = lm_head(verify_hidden[:, K_cur - 1])
            bonus_id, _ = _ras_sample(pK_raw, ctx_K)
        else:
            # Shouldn't happen given the loop above always sets replacement
            # when n_accept < K_cur. Defensive fallback.
            ctx_n = torch.cat([full_ids] + drafts[:n_accept], dim=1)
            pn_raw = lm_head(verify_hidden[:, n_accept - 1])
            bonus_id, _ = _ras_sample(pn_raw, ctx_n)

        # --- Cache management ---------------------------------------------
        # Cache has T + K positions after verify. Crop to T + n_accept (keep
        # prompt + accepted drafts), then forward bonus to extend by 1.
        if hasattr(cache, "crop"):
            cache.crop(T + n_accept)
        else:
            raise NotImplementedError("DynamicCache.crop() required for spec decoding")

        out_extend = base.model(
            input_ids=bonus_id, past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        cache = out_extend.past_key_values
        bonus_hidden = out_extend.last_hidden_state              # [1, 1, H]

        # Maintain trunk_hidden_full incrementally.
        accepted_hidden = verify_hidden[:, :n_accept]
        trunk_hidden_full = torch.cat(
            [trunk_hidden_full, accepted_hidden, bonus_hidden], dim=1
        )

        accepted = draft_seq[:, :n_accept]
        committed = torch.cat([accepted, bonus_id], dim=1)
        full_ids = torch.cat([full_ids, committed], dim=1)
        accept_lengths.append(n_accept)
        n_steps += 1

        stream_committed = committed
        if eos_token_id is not None and (committed == eos_token_id).any().item():
            eos_hit = True
            committed_list = committed[0].tolist()
            eos_pos = committed_list.index(eos_token_id)
            keep_n = eos_pos + 1
            stream_committed = committed[:, :keep_n]
            extra = committed.shape[1] - keep_n
            if extra > 0:
                full_ids = full_ids[:, :-extra]

        if streamer is not None:
            streamer.put(stream_committed[0].detach().cpu())

        if eos_hit:
            break

    generated_tokens = full_ids[:, prompt_len:]
    if streamer is not None:
        streamer.end()
    return SpecDecodeResult(
        tokens=full_ids,
        generated_tokens=generated_tokens,
        accept_lengths=accept_lengths,
        n_steps=n_steps,
        n_committed=int(generated_tokens.shape[1]),
        eos_hit=eos_hit,
        t_trunk=0.0,  # not instrumented in this variant
        t_mtp=0.0,
    )


@torch.inference_mode()
def baseline_sample_decode_cached(
    base,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int = 500,
    min_new_tokens: int = 0,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    use_ras: bool = False,
    ras_win_size: int = 25,
    ras_tau_r: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[torch.LongTensor, int]:
    """KV-cached sampling baseline. Fair comparison target for
    `mtp_speculative_sample_cached` — both use cache + the same sampling
    processors, including optional RAS.
    """
    device = input_ids.device
    lm_head = base.lm_head
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    full_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]

    out = base.model(input_ids=full_ids, use_cache=True, return_dict=True)
    cache = out.past_key_values
    last_hidden = out.last_hidden_state[:, -1:]

    n_steps = 0
    while full_ids.shape[1] - prompt_len < max_new_tokens:
        raw = lm_head(last_hidden)[:, 0]  # [1, V]
        if (
            eos_token_id is not None
            and full_ids.shape[1] - prompt_len < min_new_tokens
        ):
            raw = raw.clone()
            raw[..., eos_token_id] = -float("inf")
        next_id, _ = _sample_with_ras(
            raw, full_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_ras=use_ras,
            win_size=ras_win_size,
            tau_r=ras_tau_r,
            generator=gen,
        )
        full_ids = torch.cat([full_ids, next_id], dim=1)
        n_steps += 1
        if eos_token_id is not None and int(next_id.item()) == eos_token_id:
            break
        out = base.model(
            input_ids=next_id, past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        cache = out.past_key_values
        last_hidden = out.last_hidden_state[:, -1:]
    return full_ids, n_steps
