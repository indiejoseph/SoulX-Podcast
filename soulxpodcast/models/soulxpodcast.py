import time
from datetime import datetime

from tqdm import tqdm
from itertools import chain
from copy import deepcopy

import numpy as np
import s3tokenizer
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from soulxpodcast.config import Config, SamplingParams, AutoPretrainedConfig
from soulxpodcast.engine.llm_engine import (
    HFLLMEngine, VLLMEngine
)
from soulxpodcast.models.modules.flow import CausalMaskedDiffWithXvec
from soulxpodcast.models.modules.hifigan import HiFTGenerator
from soulxpodcast.utils.streaming import SpeechTokenStreamer, run_llm_in_thread

def _remove_weight_norm_safe(module: torch.nn.Module) -> None:
    """Remove weight-norm from all submodules, handling both the legacy hook
    API (torch.nn.utils.weight_norm) and the new parametrize API
    (torch.nn.utils.parametrizations.weight_norm)."""
    for submodule in module.modules():
        # New parametrize API creates ParametrizedConv*/Linear modules.
        if hasattr(submodule, 'parametrizations') and 'weight' in submodule.parametrizations:
            try:
                torch.nn.utils.parametrize.remove_parametrizations(
                    submodule, 'weight', leave_parametrized=True
                )
            except Exception:
                pass
        else:
            # Legacy hook-based API.
            for k, hook in list(getattr(submodule, '_forward_pre_hooks', {}).items()):
                if type(hook).__name__ == 'WeightNorm':
                    try:
                        torch.nn.utils.remove_weight_norm(submodule)
                    except Exception:
                        pass
                    break


class SoulXPodcast(torch.nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = Config() if config is None else config

        self.audio_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").cuda().eval()
        if self.config.llm_engine == "hf":
            self.llm = HFLLMEngine(**self.config.__dict__)
        elif self.config.llm_engine == "vllm":
            self.llm = VLLMEngine(**self.config.__dict__)
        else:
            raise NotImplementedError

        self.use_tqdm = True

        self.flow = CausalMaskedDiffWithXvec()
        if self.config.hf_config.fp16_flow:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [INFO] - Casting flow to fp16")
            self.flow.half()
        self.flow.load_state_dict(torch.load(f"{self.config.model}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        self.flow.cuda().eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{self.config.model}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.cuda().eval()
        # Remove weight-norm parametrizations — safe at inference, eliminates
        # the per-conv weight recomputation overhead on every forward call.
        _remove_weight_norm_safe(self.hift)

    def compile_for_inference(self):
        """Apply torch.compile to the flow estimator.

        Only the estimator is compiled — it is a pure attention/FFN stack.
        HiFT is NOT compiled: stft/istft produce complex tensors Inductor cannot lower.

        suppress_errors=True: if Inductor fails on a new graph shape (e.g. the bool
        attention-mask buffer in the streaming path that triggers NaN bounds analysis),
        Dynamo falls back to eager for that subgraph rather than crashing the request.
        """
        tqdm.write("[INFO] Applying torch.compile to flow.decoder.estimator ...")
        torch._dynamo.config.suppress_errors = True
        self.flow.decoder.estimator = torch.compile(
            self.flow.decoder.estimator,
            mode="default",
            dynamic=True,
        )
        tqdm.write("[INFO] torch.compile applied. Run warmup_compiled() to pre-pay compilation cost.")

    def warmup_compiled(self):
        """Pre-pay torch.compile JIT cost with synthetic inputs matching the real call signature.

        In solve_euler, CFG doubles the batch (batch_size * 2 = 2 for a single request).
        t_in is a 1-D tensor of shape [batch_size*2], not a scalar.
        spks/cond are always present in normal inference.
        """
        tqdm.write("[INFO] Warming up compiled flow.decoder.estimator with synthetic input ...")
        n_feats = 80
        spk_emb_dim = 80   # output_size of CausalMaskedDiffWithXvec (post-affine projection)
        cfg_batch = 2      # batch_size * 2 (CFG unconditional + conditional)
        dt = torch.float16 if self.config.hf_config.fp16_flow else torch.float32
        with torch.inference_mode():
            # Cover both streaming=False (forward_longform) and streaming=True
            # (forward_longform_streaming, FLOW_STREAMING=true default), and both
            # short and long T so dynamic-shape guards cover the common range.
            for streaming in (False, True):
                for T in (50, 150):
                    dummy_x = torch.randn(cfg_batch, n_feats, T, device="cuda", dtype=dt)
                    dummy_mu = torch.randn(cfg_batch, n_feats, T, device="cuda", dtype=dt)
                    dummy_mask = torch.ones(cfg_batch, 1, T, device="cuda", dtype=dt)
                    dummy_t = torch.full((cfg_batch,), 0.5, device="cuda", dtype=dt)
                    dummy_spks = torch.randn(cfg_batch, spk_emb_dim, device="cuda", dtype=dt)
                    dummy_cond = torch.randn(cfg_batch, n_feats, T, device="cuda", dtype=dt)
                    _ = self.flow.decoder.estimator(
                        dummy_x, dummy_mask, dummy_mu, dummy_t,
                        dummy_spks, dummy_cond, streaming,
                    )
        tqdm.write("[INFO] Warmup complete.")

    @torch.inference_mode()
    def forward_longform(
        self, prompt_mels_for_llm,
        prompt_mels_lens_for_llm: torch.Tensor,
        prompt_text_tokens_for_llm: list[list[int]],
        text_tokens_for_llm: list[list[int]],
        prompt_mels_for_flow_ori, 
        spk_emb_for_flow: torch.Tensor,
        sampling_params: SamplingParams | list[SamplingParams],
        spk_ids: list[list[int]],
        use_dialect_prompt: bool = False,
        dialect_prompt_text_tokens_for_llm: list[list[int]] = None,
        dialect_prefix: list[list[int]] = None,
        **kwargs,  # for compatibility
    ):

        prompt_size, turn_size = len(prompt_mels_for_llm), len(text_tokens_for_llm)

        # Audio tokenization
        prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = self.audio_tokenizer.quantize(
            prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
        )

        # align speech token with speech feat as to reduce
        #    the noise ratio during the generation process.
        prompt_speech_tokens = []
        prompt_mels_for_flow, prompt_mels_lens_for_flow = [], []

        for prompt_index in range(prompt_size):
            prompt_speech_token_len = prompt_speech_tokens_lens_ori[prompt_index].item()
            prompt_speech_token = prompt_speech_tokens_ori[prompt_index, :prompt_speech_token_len]
            prompt_mel = prompt_mels_for_flow_ori[prompt_index]
            prompt_mel_len = prompt_mel.shape[0]
            if prompt_speech_token_len * 2 > prompt_mel_len:
                prompt_speech_token = prompt_speech_token[:int(prompt_mel_len/2)]
                prompt_mel_len = torch.tensor([prompt_mel_len]).cuda()
            else:
                prompt_mel = prompt_mel.detach().clone()[:prompt_speech_token_len * 2].cuda()
                prompt_mel_len = torch.tensor([prompt_speech_token_len * 2]).cuda()
            prompt_speech_tokens.append(prompt_speech_token)
            prompt_mels_for_flow.append(prompt_mel)
            prompt_mels_lens_for_flow.append(prompt_mel_len)

        # Prepare LLM inputs
        prompt_inputs = []
        history_inputs = []
        
        for i in range(prompt_size):
            speech_tokens_i = [token+self.config.hf_config.speech_token_offset for token in prompt_speech_tokens[i].tolist()]
            speech_tokens_i += [self.config.hf_config.eos_token_id]
            if use_dialect_prompt and len(dialect_prompt_text_tokens_for_llm[i])>0:
                dialect_prompt_input = prompt_text_tokens_for_llm[i] + speech_tokens_i + dialect_prompt_text_tokens_for_llm[i]
                if i>0:
                    dialect_prompt_input = dialect_prefix[0] + dialect_prompt_input
                prompt_input = self.llm.generate(dialect_prompt_input, sampling_params, past_key_values=None)['token_ids']
                prompt_inputs.append(dialect_prefix[i+1]+dialect_prompt_text_tokens_for_llm[i] + prompt_input)
                history_inputs.append(dialect_prefix[i+1]+dialect_prompt_text_tokens_for_llm[i] + prompt_input)
            else:
                prompt_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i )
                history_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i )

        generated_wavs, results_dict = [], {}
        per_turn_speech_tokens: list[list[int]] = []  # captured for diagnostics
        
        # LLM generation
        inputs = list(chain.from_iterable(prompt_inputs))
        cache_config = AutoPretrainedConfig().from_dataclass(self.llm.config.hf_config)
        past_key_values = DynamicCache(config=cache_config)
        valid_turn_size = prompt_size
        for i in range(turn_size):

            # # set ratio: reach the reset cache ratio;
            if valid_turn_size > self.config.max_turn_size or len(inputs)>self.config.turn_tokens_threshold:
                assert self.config.max_turn_size >= self.config.prompt_context + self.config.history_context, "Invalid Long history size setting, "
                prompt_text_bound = max(self.config.prompt_context, len(history_inputs)-self.config.history_text_context-self.config.history_context)
                inputs = list(chain.from_iterable(
                    history_inputs[:self.config.prompt_context]+ \
                    history_inputs[prompt_text_bound:-self.config.history_context]+ \
                    prompt_inputs[-self.config.history_context:]
                ))
                valid_turn_size = self.config.prompt_context + len(history_inputs) - prompt_text_bound
                past_key_values = DynamicCache(config=cache_config)
            valid_turn_size += 1
            
            inputs.extend(text_tokens_for_llm[i])
            start_time = time.time()
            llm_outputs = self.llm.generate(inputs, sampling_params, past_key_values=past_key_values)

            inputs.extend(llm_outputs['token_ids'])
            prompt_inputs.append(text_tokens_for_llm[i]+llm_outputs['token_ids'])
            history_inputs.append(text_tokens_for_llm[i][:-1]) # remove the <|audio_start|>
            
            # Prepare Flow inputs
            turn_spk = spk_ids[i]
            generated_speech_tokens = [token - self.config.hf_config.speech_token_offset for token in  llm_outputs['token_ids'][:-1]]  # ignore last eos
            per_turn_speech_tokens.append(generated_speech_tokens)
            prompt_speech_token = prompt_speech_tokens[turn_spk].tolist()
            flow_input = torch.tensor([prompt_speech_token + generated_speech_tokens])
            flow_inputs_len = torch.tensor([len(prompt_speech_token) + len(generated_speech_tokens)])

            # Flow generation and HiFi-GAN generation            
            start_idx = spk_ids[i]
            prompt_mels = prompt_mels_for_flow[start_idx][None]
            prompt_mels_lens = prompt_mels_lens_for_flow[start_idx][None]
            spk_emb = spk_emb_for_flow[start_idx:start_idx+1]

            # Flow generation
            with torch.amp.autocast("cuda", dtype=torch.float16 if self.config.hf_config.fp16_flow else torch.float32):
                generated_mels, generated_mels_lens = self.flow(
                    flow_input.cuda(), flow_inputs_len.cuda(),
                    prompt_mels, prompt_mels_lens, spk_emb.cuda(),
                    streaming=False, finalize=True
                )

            # HiFi-GAN generation
            mel = generated_mels[:, :, prompt_mels_lens[0].item():generated_mels_lens[0].item()]
            wav, _ = self.hift(speech_feat=mel)
            generated_wavs.append(wav)

        # Save the generated wav;
        results_dict['generated_wavs'] = generated_wavs
        results_dict['generated_speech_tokens'] = per_turn_speech_tokens
        return results_dict

    # ------------------------------------------------------------------ #
    # Streaming variant of forward_longform.
    # Yields audio chunks as the LLM generates speech tokens, per turn.
    # Same multi-speaker / dialect / cache-reset semantics as the batch path,
    # but uses the SpeechTokenStreamer + sliding-window flow+HiFT pattern
    # validated in bistream_test.py (PLAN.md Phase 0 task 2b).
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def forward_longform_streaming(
        self, prompt_mels_for_llm,
        prompt_mels_lens_for_llm: torch.Tensor,
        prompt_text_tokens_for_llm: list[list[int]],
        text_tokens_for_llm: list[list[int]],
        prompt_mels_for_flow_ori,
        spk_emb_for_flow: torch.Tensor,
        sampling_params: SamplingParams | list[SamplingParams],
        spk_ids: list[list[int]],
        use_dialect_prompt: bool = False,
        dialect_prompt_text_tokens_for_llm: list[list[int]] = None,
        dialect_prefix: list[list[int]] = None,
        chunk_size: int = 50,
        first_chunk_size: int | None = None,
        flow_streaming: bool = False,
        flow_steps: int = 15,
        **kwargs,
    ):
        """Generator yielding audio chunks across turns.

        Defaults keep full-context flow attention and 15 CFM steps, matching
        historical bi-stream behavior. For low-TTFA sweeps, opt into
        flow_streaming=True and lower flow_steps only after audio A/B checks.

        Yields dicts:
            {"turn": int, "speaker": int, "chunk": int, "audio": Tensor[1, T],
             "is_first_in_turn": bool, "is_last_in_turn": bool}
        """
        prompt_size, turn_size = len(prompt_mels_for_llm), len(text_tokens_for_llm)

        # ---- Setup: identical to forward_longform up to the main loop ----
        prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = self.audio_tokenizer.quantize(
            prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
        )
        prompt_speech_tokens = []
        prompt_mels_for_flow, prompt_mels_lens_for_flow = [], []
        for prompt_index in range(prompt_size):
            prompt_speech_token_len = prompt_speech_tokens_lens_ori[prompt_index].item()
            prompt_speech_token = prompt_speech_tokens_ori[prompt_index, :prompt_speech_token_len]
            prompt_mel = prompt_mels_for_flow_ori[prompt_index]
            prompt_mel_len = prompt_mel.shape[0]
            if prompt_speech_token_len * 2 > prompt_mel_len:
                prompt_speech_token = prompt_speech_token[:int(prompt_mel_len/2)]
                prompt_mel_len = torch.tensor([prompt_mel_len]).cuda()
            else:
                prompt_mel = prompt_mel.detach().clone()[:prompt_speech_token_len * 2].cuda()
                prompt_mel_len = torch.tensor([prompt_speech_token_len * 2]).cuda()
            prompt_speech_tokens.append(prompt_speech_token)
            prompt_mels_for_flow.append(prompt_mel)
            prompt_mels_lens_for_flow.append(prompt_mel_len)

        prompt_inputs = []
        history_inputs = []
        for i in range(prompt_size):
            speech_tokens_i = [t + self.config.hf_config.speech_token_offset
                               for t in prompt_speech_tokens[i].tolist()]
            speech_tokens_i += [self.config.hf_config.eos_token_id]
            if use_dialect_prompt and len(dialect_prompt_text_tokens_for_llm[i]) > 0:
                dialect_prompt_input = (prompt_text_tokens_for_llm[i] + speech_tokens_i
                                        + dialect_prompt_text_tokens_for_llm[i])
                if i > 0:
                    dialect_prompt_input = dialect_prefix[0] + dialect_prompt_input
                # Dialect prompt is one-shot setup; no streaming needed here.
                prompt_input = self.llm.generate(dialect_prompt_input, sampling_params,
                                                  past_key_values=None)['token_ids']
                prompt_inputs.append(dialect_prefix[i+1] + dialect_prompt_text_tokens_for_llm[i] + prompt_input)
                history_inputs.append(dialect_prefix[i+1] + dialect_prompt_text_tokens_for_llm[i] + prompt_input)
            else:
                prompt_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i)
                history_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i)

        # ---- Per-turn streaming loop ----
        inputs = list(chain.from_iterable(prompt_inputs))
        cache_config = AutoPretrainedConfig().from_dataclass(self.llm.config.hf_config)
        past_key_values = DynamicCache(config=cache_config)
        valid_turn_size = prompt_size

        for turn_i in range(turn_size):
            # Cache reset (mirrors forward_longform exactly).
            if (valid_turn_size > self.config.max_turn_size
                    or len(inputs) > self.config.turn_tokens_threshold):
                assert self.config.max_turn_size >= self.config.prompt_context + self.config.history_context, \
                    "Invalid Long history size setting, "
                prompt_text_bound = max(
                    self.config.prompt_context,
                    len(history_inputs) - self.config.history_text_context - self.config.history_context
                )
                inputs = list(chain.from_iterable(
                    history_inputs[:self.config.prompt_context]
                    + history_inputs[prompt_text_bound:-self.config.history_context]
                    + prompt_inputs[-self.config.history_context:]
                ))
                valid_turn_size = self.config.prompt_context + len(history_inputs) - prompt_text_bound
                past_key_values = DynamicCache(config=cache_config)
            valid_turn_size += 1

            inputs.extend(text_tokens_for_llm[turn_i])

            # ---- Stream LLM tokens; chunked flow+HiFT in the main thread ----
            # Use separate CUDA streams so LLM work in the worker thread and
            # flow+HiFT work in the main thread overlap on the GPU instead of
            # serializing on the default stream (PLAN.md Phase 0 task B3).
            turn_spk = spk_ids[turn_i]
            spk_prompt_speech_tokens = prompt_speech_tokens[turn_spk].tolist()
            spk_prompt_mel = prompt_mels_for_flow[turn_spk][None]
            spk_prompt_mel_len_t = prompt_mels_lens_for_flow[turn_spk]
            spk_emb = spk_emb_for_flow[turn_spk:turn_spk+1].cuda()

            llm_stream = torch.cuda.Stream()
            flow_stream = torch.cuda.Stream()

            streamer = SpeechTokenStreamer(
                eos_token_id=self.config.hf_config.eos_token_id
            )
            llm_thread = run_llm_in_thread(
                self.llm,
                list(inputs),  # snapshot — LLM thread mutates input via generate
                sampling_params,
                past_key_values=past_key_values,
                streamer=streamer,
                cuda_stream=llm_stream,
            )

            accumulated_speech_tokens = []
            prev_audio_len = 0
            chunk_idx = 0
            # Low first_chunk_size is only useful for turn 0 (user is waiting for
            # first audio). For later turns the previous turn's audio is already
            # playing, so a tiny first chunk just wastes a full Flow call (~0.23s)
            # to emit ~40 ms of audio.
            effective_first_chunk_size = first_chunk_size if turn_i == 0 else chunk_size
            try:
                for chunk, is_final_partial in streamer.iter_chunks(
                    chunk_size=chunk_size,
                    first_chunk_size=effective_first_chunk_size,
                    yield_final_flag=True,
                ):
                    cur_speech_tokens = [t - self.config.hf_config.speech_token_offset for t in chunk]
                    accumulated_speech_tokens.extend(cur_speech_tokens)
                    # For turns > 0, skip finalize=False on the final partial chunk.
                    # finalize=True (the flush below) handles those tokens in one call,
                    # saving one ~0.23s Flow call per turn without losing any audio.
                    if is_final_partial and turn_i > 0:
                        break
                    with torch.cuda.stream(flow_stream):
                        audio = self._stream_synth_chunk(
                            spk_prompt_speech_tokens, accumulated_speech_tokens,
                            spk_prompt_mel, spk_prompt_mel_len_t, spk_emb,
                            finalize=False,
                            streaming=flow_streaming,
                            flow_steps=flow_steps,
                        )
                    # `.detach().cpu()` syncs flow_stream — gives us the wav.
                    new_audio = audio[:, prev_audio_len:].detach().cpu()
                    prev_audio_len = audio.shape[-1]
                    yield {
                        "turn": turn_i,
                        "speaker": turn_spk,
                        "chunk": chunk_idx,
                        "audio": new_audio,
                        "is_first_in_turn": chunk_idx == 0,
                        "is_last_in_turn": False,
                    }
                    chunk_idx += 1

                # Final finalize=True flush to emit audio for trailing lookahead.
                with torch.cuda.stream(flow_stream):
                    audio = self._stream_synth_chunk(
                        spk_prompt_speech_tokens, accumulated_speech_tokens,
                        spk_prompt_mel, spk_prompt_mel_len_t, spk_emb,
                        finalize=True,
                        streaming=flow_streaming,
                        flow_steps=flow_steps,
                    )
                final_audio = audio[:, prev_audio_len:].detach().cpu()
                yield {
                    "turn": turn_i,
                    "speaker": turn_spk,
                    "chunk": chunk_idx,
                    "audio": final_audio,
                    "is_first_in_turn": chunk_idx == 0,
                    "is_last_in_turn": True,
                }
            finally:
                # Cancel the streamer so the LLM thread can detect abandonment
                # (e.g. client disconnect) and call abort_request before exiting.
                streamer.cancel()
                llm_thread.join(timeout=2.0)

            # Block until LLM is fully done before reading results for next turn.
            # join() on an already-finished thread returns immediately.
            llm_thread.join()
            if llm_thread.exc:
                raise llm_thread.exc
            llm_outputs = llm_thread.result

            inputs.extend(llm_outputs['token_ids'])
            prompt_inputs.append(text_tokens_for_llm[turn_i] + llm_outputs['token_ids'])
            history_inputs.append(text_tokens_for_llm[turn_i][:-1])

    def _stream_synth_chunk(self, prompt_speech_tokens, generated_speech_tokens,
                             prompt_mel, prompt_mel_len_t, spk_emb, finalize: bool,
                             streaming: bool, flow_steps: int):
        """Run flow+HiFT on (prompt_speech_tokens + generated_speech_tokens).
        Returns the full waveform; caller slices off already-emitted portion."""
        import os
        _time_stages = os.getenv("PROFILE_FLOW_STAGES")
        device = prompt_mel.device
        flow_input = torch.tensor(
            [prompt_speech_tokens + generated_speech_tokens],
            device=device,
        )
        flow_input_len = torch.tensor([flow_input.shape[1]], device=device)
        if _time_stages:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()
        with torch.amp.autocast("cuda",
                dtype=torch.float16 if self.config.hf_config.fp16_flow else torch.float32):
            mels, mels_lens = self.flow(
                flow_input, flow_input_len,
                prompt_mel, prompt_mel_len_t, spk_emb,
                streaming=streaming, finalize=finalize,
                n_timesteps=flow_steps,
            )
        if _time_stages:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
        mel = mels[:, :, prompt_mel_len_t[0].item(): mels_lens[0].item()]
        wav, _ = self.hift(speech_feat=mel)
        if _time_stages:
            torch.cuda.synchronize()
            _t2 = time.perf_counter()
            mel_frames = mel.shape[-1]
            tqdm.write(f"[PROFILE] flow={_t1-_t0:.3f}s  hift={_t2-_t1:.3f}s  mel_frames={mel_frames}")
        return wav
