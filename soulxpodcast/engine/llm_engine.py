from __future__ import annotations

import os
import types
import atexit
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

# SoulX-Podcast defaults to V0 for the patched vLLM 0.10.1 RAS sampler.
# Setting VLLM_USE_V1=1 in env switches to V1 (which requires reimplementing
# RAS as a LogitsProcessor plugin — RAS is silently disabled in V1 today).
# V1 is the prerequisite for speculative decoding (V0 has no spec_decode
# runtime in vLLM 0.10.1).
# Defensive: treat unset OR empty-string OR non-{0,1} values as default "0".
# Compose passes through `${VLLM_USE_V1:-}` which becomes an empty string when
# unset, and vLLM's envs.py does `int(os.getenv("VLLM_USE_V1", "1"))` which
# crashes on empty string.
_vuse_v1 = os.environ.get("VLLM_USE_V1", "").strip()
if _vuse_v1 not in ("0", "1"):
    _vuse_v1 = "0"
os.environ["VLLM_USE_V1"] = _vuse_v1
try:
    from vllm import EngineArgs
    if os.environ.get("VLLM_USE_V1") == "1":
        from vllm.v1.engine.llm_engine import LLMEngine
    else:
        from vllm import LLMEngine
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
            # Optional fp8 KV cache. On Ampere (no native fp8 hardware) this
            # still halves KV-read bandwidth at decode time at the cost of a
            # dequant kernel. Empirically a wash-to-small-win on single-request
            # TTS; expose as opt-in to allow benchmarking.
            #   VLLM_KV_CACHE_DTYPE=auto    (default — matches model dtype)
            #   VLLM_KV_CACHE_DTYPE=fp8_e5m2
            #   VLLM_KV_CACHE_DTYPE=fp8_e4m3
            _kv_dtype = os.environ.get("VLLM_KV_CACHE_DTYPE", "").strip()
            if _kv_dtype:
                engine_kwargs["kv_cache_dtype"] = _kv_dtype
            # Optional chunked prefill. vLLM V1 auto-enables this; V0 (our
            # engine, required for the Soul-AILab RAS patches) defaults to off.
            # For single-request multi-turn TTS, chunked prefill matters only
            # if turn N's prefill can overlap with turn N-1's decode tail — our
            # forward_longform_streaming runs turns sequentially so the overlap
            # window is small. Expose as opt-in so we can A/B benchmark.
            #   VLLM_ENABLE_CHUNKED_PREFILL=true|false
            #   VLLM_MAX_NUM_BATCHED_TOKENS=<int>  (per-step token budget)
            _chunked = os.environ.get("VLLM_ENABLE_CHUNKED_PREFILL", "").strip().lower()
            if _chunked in ("true", "1", "yes"):
                engine_kwargs["enable_chunked_prefill"] = True
            elif _chunked in ("false", "0", "no"):
                engine_kwargs["enable_chunked_prefill"] = False
            _mnbt = os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "").strip()
            if _mnbt:
                engine_kwargs["max_num_batched_tokens"] = int(_mnbt)
            # V1-specific: torch.compile / CUDA graph level. V1 default is
            # "level=3" (== -O3, today equivalent to -O2 per vLLM docs). For
            # small models at batch=1 this often costs more than it saves.
            # Allow override to bench cheaper compilation strategies.
            #   VLLM_COMPILATION_LEVEL=0|1|2|3
            _comp_level = os.environ.get("VLLM_COMPILATION_LEVEL", "").strip()
            if _comp_level:
                engine_kwargs["compilation_config"] = {"level": int(_comp_level)}
            # Optional attention backend override; vLLM auto-selects FLASH_ATTN
            # on Ampere with fp16/bf16 and supported head_dim. Override only if
            # you know the auto-pick is wrong for your hardware.
            #   VLLM_ATTENTION_BACKEND=FLASH_ATTN | FLASHINFER | XFORMERS | TRITON_ATTN_VLLM_V1
            # (also natively respected by vllm itself if set in env)
            import json as _json
            cfg_path = os.path.join(model, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as _f:
                    _cfg = _json.load(_f)
                qcfg = _cfg.get("quantization_config")
                if qcfg and qcfg.get("quant_method"):
                    qmethod = qcfg["quant_method"]
                    # Prefer the faster Marlin kernel for AWQ on supported GPUs.
                    engine_kwargs["quantization"] = "awq_marlin" if qmethod == "awq" else qmethod
                    # AWQ packs in fp16, not bf16 — vLLM requires matching dtype.
                    engine_kwargs["dtype"] = "float16"
            self.model = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
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
