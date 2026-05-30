from __future__ import annotations

import os
from collections.abc import Sequence

import torch


_SAMPLING_EPS = 1e-5
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in _FALSE_VALUES


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def multinomial_sample(
    probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.multinomial(
        probs,
        1,
        replacement=True,
        generator=generator,
    ).reshape(())


def nucleus_sample_one(
    weighted_scores: torch.Tensor,
    *,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> int:
    if top_k > 0 and int(top_k) < int(weighted_scores.numel()):
        top_scores, sorted_idx = torch.topk(
            weighted_scores,
            k=int(top_k),
            dim=0,
            largest=True,
            sorted=True,
        )
        sorted_prob = (top_scores - torch.logsumexp(top_scores, dim=0)).exp()
    else:
        probs = weighted_scores.softmax(dim=0)
        sorted_prob, sorted_idx = probs.sort(descending=True, stable=True)

    sorted_prob_f32 = sorted_prob.float()
    cum_prob = sorted_prob_f32.cumsum(dim=0)
    mask = (cum_prob - sorted_prob_f32) < top_p
    mask[0] = True

    kept_probs = torch.nan_to_num(
        sorted_prob[mask],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min_(0.0)
    kept_indices = sorted_idx[mask]
    kept_probs[0] = torch.where(
        kept_probs.sum() > 0,
        kept_probs[0],
        torch.ones_like(kept_probs[0]),
    )

    sample_pos = multinomial_sample(kept_probs, generator=generator)
    return int(kept_indices[sample_pos].item())


def ras_sample_one(
    candidate_scores: torch.Tensor,
    fallback_scores: torch.Tensor,
    decoded_tokens: Sequence[int],
    *,
    top_p: float,
    top_k: int,
    win_size: int,
    tau_r: float,
    generator: torch.Generator | None = None,
) -> int:
    top_id = nucleus_sample_one(
        candidate_scores,
        top_p=top_p,
        top_k=top_k,
        generator=generator,
    )

    if win_size > 0 and decoded_tokens:
        window_len = min(int(win_size), len(decoded_tokens))
        recent = decoded_tokens[-window_len:]
        rep_num = sum(1 for token in recent if int(token) == top_id) + 1
        if rep_num >= window_len * tau_r:
            fallback_probs = fallback_scores.softmax(dim=0)
            top_id = int(multinomial_sample(fallback_probs, generator=generator).item())
    return top_id


def _req_scalar(param: torch.Tensor | None, req_idx: int, default: float | int) -> float | int:
    if param is None or param.numel() == 0:
        return default
    index = min(req_idx, int(param.numel()) - 1)
    value = param.reshape(-1)[index].item()
    if isinstance(default, int):
        return int(value)
    return float(value)


def _soulx_ras_enabled(sampling_metadata) -> bool:
    if not _env_enabled("SOULX_VLLM_MODEL_RAS", default=True):
        return False
    if sampling_metadata.max_num_logprobs is not None:
        return False
    if getattr(sampling_metadata, "logprob_token_ids", None):
        return False
    if sampling_metadata.temperature is None:
        return False
    if sampling_metadata.all_greedy:
        return False
    if sampling_metadata.allowed_token_ids_mask is not None:
        return False
    if bool(sampling_metadata.bad_words_token_ids):
        return False
    if torch.any(sampling_metadata.frequency_penalties != 0):
        return False
    if torch.any(sampling_metadata.presence_penalties != 0):
        return False
    return True


def _soulx_qwen3_sample(self, logits: torch.Tensor, sampling_metadata):
    if logits is None or logits.numel() == 0:
        return None

    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.sampler import Sampler

    sampler = getattr(self, "_soulx_base_sampler", None)
    if sampler is None:
        sampler = Sampler()
        self._soulx_base_sampler = sampler

    if not _soulx_ras_enabled(sampling_metadata):
        return sampler(logits=logits, sampling_metadata=sampling_metadata)

    raw_logits = logits.to(torch.float32)
    processed_logits = sampler.apply_logits_processors(
        raw_logits.clone(),
        sampling_metadata,
        predict_bonus_token=False,
    )

    candidate_logits = processed_logits.clone()
    candidate_logits = sampler.apply_temperature(
        candidate_logits,
        sampling_metadata.temperature,
        sampling_metadata.all_random,
    )
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        candidate_logits = processor.apply(candidate_logits)

    win_size = _env_int("SOULX_VLLM_RAS_WIN_SIZE", 25)
    tau_r = _env_float("SOULX_VLLM_RAS_TAU_R", 0.2)

    sampled_ids: list[int] = []
    for req_idx in range(int(logits.shape[0])):
        row_processed = processed_logits[req_idx]
        temperature = float(_req_scalar(sampling_metadata.temperature, req_idx, 1.0))
        if temperature < _SAMPLING_EPS:
            sampled_ids.append(int(torch.argmax(row_processed).item()))
            continue

        top_p = float(_req_scalar(sampling_metadata.top_p, req_idx, 1.0))
        top_k = int(_req_scalar(sampling_metadata.top_k, req_idx, 0))
        generator = sampling_metadata.generators.get(req_idx)
        decoded_tokens = (
            sampling_metadata.output_token_ids[req_idx]
            if req_idx < len(sampling_metadata.output_token_ids)
            else []
        )
        sampled_ids.append(
            ras_sample_one(
                candidate_logits[req_idx],
                raw_logits[req_idx],
                decoded_tokens,
                top_p=top_p,
                top_k=top_k,
                win_size=win_size,
                tau_r=tau_r,
                generator=generator,
            )
        )

    sampled = torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32)
    return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)


def install_soulx_vllm_ras_sampler(
    *,
    win_size: int = 25,
    tau_r: float = 0.2,
) -> bool:
    os.environ["SOULX_VLLM_RAS_WIN_SIZE"] = str(int(win_size))
    os.environ["SOULX_VLLM_RAS_TAU_R"] = str(float(tau_r))

    from vllm.model_executor.models import qwen3

    model_cls = getattr(qwen3, "Qwen3ForCausalLM")
    if getattr(model_cls, "_soulx_ras_sampler_installed", False):
        return True

    # vLLM V1 checks this flag and lets the model return SamplerOutput itself.
    model_cls.prefer_model_sampler = True
    model_cls.sample = _soulx_qwen3_sample
    model_cls._soulx_ras_sampler_installed = True
    return True
