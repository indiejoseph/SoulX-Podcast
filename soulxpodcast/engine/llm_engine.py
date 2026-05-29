from __future__ import annotations

import os
import types
import atexit
import inspect
import json
import queue
import threading
import uuid
from time import perf_counter
from functools import partial
from dataclasses import fields, asdict

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from transformers import EosTokenCriteria, RepetitionPenaltyLogitsProcessor

# SoulX-Podcast relies on the patched vLLM 0.10.1 V0 sampler for RAS fields by
# default. Keep this overridable so experimental newer vLLM runtimes can opt in
# to V1/speculative decoding with VLLM_USE_V1=1.
os.environ.setdefault("VLLM_USE_V1", "0")
try:    
    from vllm import EngineArgs, LLMEngine
    from vllm import SamplingParams as VllmSamplingParams
    from vllm.inputs import TokensPrompt as TokensPrompt
    SUPPORT_VLLM = True
except ImportError:
    SUPPORT_VLLM = False

from soulxpodcast.config import Config, SamplingParams
from soulxpodcast.models.modules.sampler import _ras_sample_hf_engine

# Fields accepted by stock vLLM 0.10.x SamplingParams.
_VLLM_BASE_FIELDS = frozenset({
    "temperature", "top_k", "top_p", "min_tokens", "max_tokens",
    "repetition_penalty", "stop_token_ids",
})
# Extra fields from the Soul-AILab RAS patch (vllm@v0.10.1.1-soulxpodcast).
_VLLM_RAS_FIELDS = frozenset({"use_ras", "win_size", "tau_r"})


def _load_json_value_or_path(value: str, *, name: str) -> dict:
    value = value.strip()
    if not value:
        return {}
    if value[0] in "[{":
        parsed = json.loads(value)
    else:
        with open(value, "r", encoding="utf-8") as f:
            parsed = json.load(f)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object or a path to a JSON object")
    return parsed


def _make_vllm_engine_args(engine_kwargs: dict, *, speculative_requested: bool) -> EngineArgs:
    """Build EngineArgs while producing a useful error for unsupported P-EAGLE."""
    try:
        signature = inspect.signature(EngineArgs)
    except (TypeError, ValueError):
        accepted = set()
    else:
        has_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        accepted = set() if has_var_kwargs else set(signature.parameters)

    if accepted:
        unsupported = sorted(k for k in engine_kwargs if k not in accepted)
        if unsupported:
            if speculative_requested and "speculative_config" in unsupported:
                raise RuntimeError(
                    "VLLM_SPECULATIVE_CONFIG was set, but the installed vLLM "
                    "EngineArgs does not support speculative_config. P-EAGLE "
                    "requires a newer vLLM/speculators runtime than the default "
                    "patched vLLM 0.10.1 image."
                )
            engine_kwargs = {k: v for k, v in engine_kwargs.items() if k not in unsupported}

    try:
        return EngineArgs(**engine_kwargs)
    except TypeError as exc:
        if speculative_requested and "speculative" in str(exc):
            raise RuntimeError(
                "VLLM_SPECULATIVE_CONFIG was set, but this vLLM build rejected "
                "the speculative decoding arguments. Use a vLLM/speculators "
                "runtime with P-EAGLE support."
            ) from exc
        raise

class HFLLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        config.eos = config.hf_config.eos_token_id # speech eos token;
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map=self.device)
        self.config = config
        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(
        self,
        prompt: list[int],
        sampling_param: SamplingParams,
        past_key_values=None,
        streamer=None,
    ) -> dict:

        stopping_criteria = StoppingCriteriaList([EosTokenCriteria(eos_token_id=self.config.hf_config.eos_token_id)])
        if sampling_param.use_ras:
            # HF's generate() drops `streamer` from the kwargs it forwards to a
            # custom_generate callable (it filters to keys unique to the custom
            # function, and `streamer` is shared with the built-in `_sample`).
            # Bind it into the partial so the custom sampler still receives it.
            handler_kwargs = dict(
                use_ras=sampling_param.use_ras,
                win_size=sampling_param.win_size,
                tau_r=sampling_param.tau_r,
            )
            if streamer is not None:
                handler_kwargs["streamer"] = streamer
            sample_hf_engine_handler = partial(_ras_sample_hf_engine, **handler_kwargs)
        else:
            sample_hf_engine_handler = None
        rep_pen_processor = RepetitionPenaltyLogitsProcessor(
            penalty=sampling_param.repetition_penalty,
            prompt_ignore_length=len(prompt)
        ) # exclude the input prompt, consistent with vLLM implementation;
        with torch.no_grad():
            input_len = len(prompt)
            generated_ids = self.model.generate(
                input_ids = torch.tensor([prompt], dtype=torch.int64).to(self.device),
                do_sample=True,
                top_k=sampling_param.top_k,
                top_p=sampling_param.top_p,
                min_new_tokens=sampling_param.min_tokens,
                max_new_tokens=sampling_param.max_tokens,
                temperature=sampling_param.temperature,
                stopping_criteria=stopping_criteria,
                past_key_values=past_key_values,
                custom_generate=sample_hf_engine_handler,
                use_cache=True,
                logits_processor=[rep_pen_processor],
                streamer=streamer,
            )
            generated_ids = generated_ids[:, input_len:].cpu().numpy().tolist()[0]
        output = {
            "text": self.tokenizer.decode(generated_ids),
            "token_ids": generated_ids,
        }
        return output

class VLLMEngine:

    def __init__(self, model, **kwargs):
        
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = config.hf_config.eos_token_id # speech eos token;
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if SUPPORT_VLLM:
            # Auto-detect AWQ / other quantization from the model's config.json.
            # vLLM 0.10.1 needs explicit `quantization=` kwarg even when the
            # config has the marker, so we read it ourselves and pass through.
            engine_kwargs = dict(
                model=model,
                enforce_eager=config.enforce_eager,
                dtype="bfloat16",
                max_model_len=config.max_model_len,
                gpu_memory_utilization=config.gpu_memory_utilization,
                tensor_parallel_size=config.tensor_parallel_size,
                enable_prefix_caching=True,
            )
            cfg_path = os.path.join(model, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as _f:
                    _cfg = json.load(_f)
                qcfg = _cfg.get("quantization_config")
                if qcfg and qcfg.get("quant_method"):
                    qmethod = qcfg["quant_method"]
                    # Prefer the faster Marlin kernel for AWQ on supported GPUs.
                    engine_kwargs["quantization"] = "awq_marlin" if qmethod == "awq" else qmethod
                    # AWQ packs in fp16, not bf16 — vLLM requires matching dtype.
                    engine_kwargs["dtype"] = "float16"
            speculative_requested = bool(config.vllm_speculative_config.strip())
            if speculative_requested:
                engine_kwargs["speculative_config"] = _load_json_value_or_path(
                    config.vllm_speculative_config,
                    name="VLLM_SPECULATIVE_CONFIG",
                )
            self.model = LLMEngine.from_engine_args(
                _make_vllm_engine_args(
                    engine_kwargs,
                    speculative_requested=speculative_requested,
                )
            )
        else:
            raise ImportError("Not Support VLLM now!!!")
        self.config = config
        self.pad_token_id = self.tokenizer.pad_token_id
        self._engine_lock = threading.Lock()
        self._engine_cv = threading.Condition(self._engine_lock)
        self._request_queues: dict[str, "queue.Queue"] = {}
        self._request_seen_tokens: dict[str, int] = {}
        self._shutdown = False
        self._engine_thread = threading.Thread(
            target=self._run_engine_loop,
            name="soulx-vllm-engine",
            daemon=True,
        )
        self._engine_thread.start()
        atexit.register(self.shutdown)

    def _make_sampling_params(self, sampling_param: SamplingParams) -> VllmSamplingParams:
        all_params = asdict(sampling_param)
        params = {k: v for k, v in all_params.items()
                  if k in _VLLM_BASE_FIELDS | _VLLM_RAS_FIELDS}
        params["stop_token_ids"] = [self.config.hf_config.eos_token_id]
        try:
            return VllmSamplingParams(**params)
        except TypeError:
            # Soul-AILab RAS patch not installed — fall back to base fields.
            params_base = {k: v for k, v in params.items() if k in _VLLM_BASE_FIELDS}
            return VllmSamplingParams(**params_base)

    def _run_engine_loop(self):
        while True:
            with self._engine_cv:
                while (
                    not self._shutdown
                    and (
                        not self._request_queues
                        or not self.model.has_unfinished_requests()
                    )
                ):
                    self._engine_cv.wait()
                if self._shutdown:
                    return

            try:
                with self._engine_lock:
                    request_outputs = self.model.step()
            except BaseException as exc:
                with self._engine_lock:
                    queues = list(self._request_queues.values())
                for output_queue in queues:
                    output_queue.put(("error", exc))
                continue

            for request_output in request_outputs:
                request_id = str(request_output.request_id)
                outputs = getattr(request_output, "outputs", None) or []
                if not outputs:
                    continue
                output = outputs[0]
                token_ids = list(output.token_ids)
                finish_reason = getattr(output, "finish_reason", None)
                finished = bool(getattr(request_output, "finished", False))

                with self._engine_lock:
                    output_queue = self._request_queues.get(request_id)
                    if output_queue is None:
                        continue
                    seen = self._request_seen_tokens.get(request_id, 0)
                    new_tokens = token_ids[seen:]
                    self._request_seen_tokens[request_id] = len(token_ids)

                output_queue.put(("tokens", new_tokens, finished, finish_reason, token_ids))

    def shutdown(self):
        with self._engine_cv:
            self._shutdown = True
            self._engine_cv.notify_all()
        if getattr(self, "_engine_thread", None) is not None and self._engine_thread.is_alive():
            self._engine_thread.join(timeout=1.0)

    def generate(
        self,
        prompt: list[int],
        sampling_param: SamplingParams,
        past_key_values=None,
        streamer=None,
    ) -> dict:
        request_id = f"vllm-{uuid.uuid4().hex}"
        output_queue: "queue.Queue" = queue.Queue()
        sampling_params = self._make_sampling_params(sampling_param)
        eos_id = self.config.hf_config.eos_token_id
        registered = False
        request_added = False

        generated_ids: list[int] = []
        finish_reason = None
        finished_request = False
        try:
            if streamer is not None:
                # Match HF generate(): first streamer.put() contains the prompt
                # and is intentionally skipped by SpeechTokenStreamer.
                streamer.put(torch.tensor(prompt, dtype=torch.long))

            with self._engine_cv:
                self._request_queues[request_id] = output_queue
                self._request_seen_tokens[request_id] = 0
                registered = True
                self.model.add_request(
                    request_id,
                    TokensPrompt(prompt_token_ids=prompt),
                    sampling_params,
                )
                request_added = True
                self._engine_cv.notify()

            while True:
                item = output_queue.get()
                kind = item[0]
                if kind == "error":
                    raise item[1]

                _, new_tokens, finished, finish_reason, token_ids = item
                for tok in new_tokens:
                    generated_ids.append(int(tok))
                    if streamer is not None:
                        streamer.put(torch.tensor([int(tok)], dtype=torch.long))

                if streamer is not None and getattr(streamer, "_cancelled", False):
                    # Consumer abandoned the stream (e.g. HTTP client disconnect).
                    # finished_request stays False so the finally block aborts the
                    # vLLM request and stops the engine from generating further.
                    break

                if finished:
                    # vLLM may omit special stop tokens from output token_ids.
                    # Downstream code expects HF-like output with EOS present
                    # unless generation stopped by max length.
                    if finish_reason != "length" and (not generated_ids or generated_ids[-1] != eos_id):
                        generated_ids.append(eos_id)
                    finished_request = True
                    break
        finally:
            if registered:
                with self._engine_cv:
                    if request_added and not finished_request:
                        try:
                            self.model.abort_request(request_id)
                        except BaseException:
                            pass
                    self._request_queues.pop(request_id, None)
                    self._request_seen_tokens.pop(request_id, None)
                    self._engine_cv.notify_all()
            if streamer is not None:
                streamer.end()

        output = {
            "text": self.tokenizer.decode(generated_ids),
            "token_ids": list(generated_ids),
        }
        return output
