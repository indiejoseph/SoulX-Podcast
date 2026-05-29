import os

from typing import Any, Callable, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList
)
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
from transformers import StoppingCriteria


_COMPACT_PROCESSOR_NAMES = {
    "RepetitionPenaltyLogitsProcessor",
    "MinLengthLogitsProcessor",
    "MinNewTokensLengthLogitsProcessor",
    "TemperatureLogitsWarper",
    "TopKLogitsWarper",
    "TopPLogitsWarper",
}


def _build_allowed_speech_ids(
    *,
    device: torch.device,
    speech_token_offset: int,
    speech_vocab_size: int,
    speech_eos_token_id: Optional[int],
) -> torch.LongTensor:
    speech_ids = torch.arange(
        int(speech_token_offset),
        int(speech_token_offset) + int(speech_vocab_size),
        dtype=torch.long,
        device=device,
    )
    if speech_eos_token_id is None:
        return speech_ids
    eos = torch.tensor([int(speech_eos_token_id)], dtype=torch.long, device=device)
    if int(speech_token_offset) <= int(speech_eos_token_id) < int(speech_token_offset) + int(speech_vocab_size):
        return speech_ids
    return torch.cat([eos, speech_ids], dim=0)


def _build_full_to_compact_lookup(allowed_ids: torch.LongTensor) -> torch.LongTensor:
    lookup = torch.full(
        (int(allowed_ids.max().item()) + 1,),
        -1,
        dtype=torch.long,
        device=allowed_ids.device,
    )
    lookup[allowed_ids] = torch.arange(allowed_ids.numel(), dtype=torch.long, device=allowed_ids.device)
    return lookup


def _compact_processors_supported(logits_processor: LogitsProcessorList) -> bool:
    return all(proc.__class__.__name__ in _COMPACT_PROCESSOR_NAMES for proc in logits_processor)


def _extract_repetition_penalty(logits_processor: LogitsProcessorList) -> tuple[float, int]:
    for proc in logits_processor:
        if proc.__class__.__name__ == "RepetitionPenaltyLogitsProcessor":
            penalty = float(getattr(proc, "penalty", 1.0))
            prompt_ignore_length = int(getattr(proc, "prompt_ignore_length", 0))
            return penalty, prompt_ignore_length
    return 1.0, 0


def _apply_repetition_penalty_compact(
    scores: torch.Tensor,
    input_ids: torch.LongTensor,
    full_to_compact: torch.LongTensor,
    *,
    penalty: float,
    prompt_ignore_length: int,
) -> torch.Tensor:
    if penalty == 1.0 or input_ids.numel() == 0:
        return scores

    out = scores.clone()
    context_ids = input_ids[:, prompt_ignore_length:]
    for batch_idx in range(out.size(0)):
        ids = context_ids[batch_idx]
        ids = ids[ids < full_to_compact.numel()]
        if ids.numel() == 0:
            continue
        compact_ids = full_to_compact[ids]
        compact_ids = compact_ids[compact_ids >= 0].unique()
        if compact_ids.numel() == 0:
            continue
        token_scores = out[batch_idx, compact_ids]
        token_scores = torch.where(token_scores < 0, token_scores * penalty, token_scores / penalty)
        out[batch_idx, compact_ids] = token_scores
    return out


def _mask_compact_eos_before_min_length(
    scores: torch.Tensor,
    input_ids: torch.LongTensor,
    *,
    prompt_len: int,
    generation_config: GenerationConfig,
    eos_compact_idx: Optional[int],
) -> torch.Tensor:
    if eos_compact_idx is None:
        return scores

    cur_len = input_ids.shape[-1]
    mask_eos = False
    min_length = getattr(generation_config, "min_length", None)
    if min_length is not None and min_length > 0 and cur_len < min_length:
        mask_eos = True
    min_new_tokens = getattr(generation_config, "min_new_tokens", None)
    if min_new_tokens is not None and min_new_tokens > 0 and cur_len - prompt_len < min_new_tokens:
        mask_eos = True

    if not mask_eos:
        return scores
    out = scores.clone()
    out[..., eos_compact_idx] = -float("inf")
    return out


def _apply_compact_sampling_warpers(scores: torch.Tensor, generation_config: GenerationConfig) -> torch.Tensor:
    out = scores
    temperature = getattr(generation_config, "temperature", None)
    if temperature is not None and temperature != 1.0:
        out = out / max(float(temperature), 1e-6)

    top_k = getattr(generation_config, "top_k", None)
    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), out.size(-1))
        kth = out.topk(k, dim=-1).values[..., -1, None]
        out = torch.where(out < kth, torch.full_like(out, -float("inf")), out)

    top_p = getattr(generation_config, "top_p", None)
    if top_p is not None and 0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        indices_to_remove = remove.scatter(-1, sorted_idx, remove)
        out = out.masked_fill(indices_to_remove, -float("inf"))

    return out


def _sample_or_argmax(scores: torch.Tensor, *, do_sample: bool) -> torch.LongTensor:
    if not do_sample:
        return torch.argmax(scores, dim=-1)
    probs = F.softmax(scores, dim=-1)
    finite = torch.isfinite(probs)
    probs = torch.where(finite, probs, torch.zeros_like(probs))
    totals = probs.sum(dim=-1, keepdim=True)
    if not torch.all(totals > 0):
        return torch.argmax(scores, dim=-1)
    return torch.multinomial(probs / totals, num_samples=1).squeeze(1)


def _restricted_lm_head_logits(
    model,
    hidden_states: torch.Tensor,
    allowed_ids: torch.LongTensor,
) -> torch.Tensor:
    lm_head = getattr(model, "lm_head")
    weight = lm_head.weight.index_select(0, allowed_ids)
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        bias = bias.index_select(0, allowed_ids)
    logits = F.linear(hidden_states.to(dtype=weight.dtype), weight, bias)
    return logits.to(dtype=torch.float32)


def _backbone_forward_for_restricted_logits(
    model,
    model_inputs: dict[str, Any],
    *,
    output_attentions: bool,
    output_hidden_states: bool,
):
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(model, "lm_head"):
        return None
    allowed_keys = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "use_cache",
        "cache_position",
    }
    kwargs = {k: v for k, v in model_inputs.items() if k in allowed_keys and v is not None}
    kwargs.update(
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    return backbone(**kwargs)


def _ras_sample_hf_engine(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    use_ras=False,
    win_size=25,
    tau_r=0.2,
    restrict_speech_vocab: bool = False,
    speech_token_offset: Optional[int] = None,
    speech_vocab_size: int = 6561,
    speech_eos_token_id: Optional[int] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    compact_allowed_ids = None
    full_to_compact = None
    eos_compact_idx = None
    repetition_penalty, prompt_ignore_length = _extract_repetition_penalty(logits_processor)
    use_compact_sampling = (
        restrict_speech_vocab
        and speech_token_offset is not None
        and not return_dict_in_generate
        and _compact_processors_supported(logits_processor)
        and hasattr(self, "model")
        and hasattr(self, "lm_head")
    )
    if use_compact_sampling:
        compact_allowed_ids = _build_allowed_speech_ids(
            device=input_ids.device,
            speech_token_offset=speech_token_offset,
            speech_vocab_size=speech_vocab_size,
            speech_eos_token_id=speech_eos_token_id,
        )
        full_to_compact = _build_full_to_compact_lookup(compact_allowed_ids)
        if speech_eos_token_id is not None and int(speech_eos_token_id) < full_to_compact.numel():
            compact_idx = int(full_to_compact[int(speech_eos_token_id)].item())
            eos_compact_idx = compact_idx if compact_idx >= 0 else None

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    prompt_len = cur_len
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if use_compact_sampling:
            outputs = _backbone_forward_for_restricted_logits(
                self,
                model_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            if outputs is None:
                use_compact_sampling = False

        if not use_compact_sampling:
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)
        else:
            is_prefill = False

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        reused_ras_candidate = False
        if use_compact_sampling:
            compact_logits = _restricted_lm_head_logits(
                self,
                outputs.last_hidden_state[:, -1, :],
                compact_allowed_ids,
            )
            next_token_logits = compact_logits
            next_token_scores = _apply_repetition_penalty_compact(
                compact_logits,
                input_ids,
                full_to_compact,
                penalty=repetition_penalty,
                prompt_ignore_length=prompt_ignore_length,
            )
            next_token_scores = _mask_compact_eos_before_min_length(
                next_token_scores,
                input_ids,
                prompt_len=prompt_len,
                generation_config=generation_config,
                eos_compact_idx=eos_compact_idx,
            )
            next_token_scores = _apply_compact_sampling_warpers(next_token_scores, generation_config)

            # Repetition Aware Sampling in VALL-E 2.
            # Candidate sampled once; reused when no repetition reset needed.
            ras_candidate = None
            if use_ras:
                candidate_compact = _sample_or_argmax(next_token_scores, do_sample=do_sample)
                next_tokens_candidate = compact_allowed_ids[candidate_compact]
                rep_num = (input_ids[:, -win_size:] == next_tokens_candidate).sum().item() + 1
                if rep_num < win_size * tau_r:
                    ras_candidate = next_tokens_candidate
                else:
                    next_token_scores = compact_logits
        else:
            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Repetition Aware Sampling in VALL-E 2.
            # Candidate is sampled once; reused when no repetition reset is needed so
            # we avoid a second multinomial draw that would change the distribution.
            ras_candidate = None
            if use_ras:
                probs_candidate = nn.functional.softmax(next_token_scores, dim=-1)
                ras_candidate = torch.multinomial(probs_candidate, num_samples=1).squeeze(1)
                rep_num = (input_ids[:, -win_size:] == ras_candidate).sum().item() + 1
                if rep_num >= win_size * tau_r:
                    # Repetition detected — reset to raw logits and resample.
                    next_token_scores = next_token_logits
                    ras_candidate = None

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if ras_candidate is not None:
            # No repetition reset — reuse the candidate to avoid double-sampling.
            next_tokens = ras_candidate
        elif use_compact_sampling:
            next_tokens_compact = _sample_or_argmax(next_token_scores, do_sample=do_sample)
            next_tokens = compact_allowed_ids[next_tokens_compact]
        elif do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids
