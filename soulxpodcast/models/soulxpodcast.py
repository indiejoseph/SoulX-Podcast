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
        **kwargs,
    ):
        """Generator yielding audio chunks across turns.

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
            for chunk in streamer.iter_chunks(chunk_size=chunk_size):
                cur_speech_tokens = [t - self.config.hf_config.speech_token_offset for t in chunk]
                accumulated_speech_tokens.extend(cur_speech_tokens)
                with torch.cuda.stream(flow_stream):
                    audio = self._stream_synth_chunk(
                        spk_prompt_speech_tokens, accumulated_speech_tokens,
                        spk_prompt_mel, spk_prompt_mel_len_t, spk_emb,
                        finalize=False,
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

            # Block until LLM done so we can update cache state for next turn.
            llm_thread.join()
            if llm_thread.exc:
                raise llm_thread.exc
            llm_outputs = llm_thread.result

            inputs.extend(llm_outputs['token_ids'])
            prompt_inputs.append(text_tokens_for_llm[turn_i] + llm_outputs['token_ids'])
            history_inputs.append(text_tokens_for_llm[turn_i][:-1])

    def _stream_synth_chunk(self, prompt_speech_tokens, generated_speech_tokens,
                             prompt_mel, prompt_mel_len_t, spk_emb, finalize: bool):
        """Run flow+HiFT on (prompt_speech_tokens + generated_speech_tokens).
        Returns the full waveform; caller slices off already-emitted portion."""
        flow_input = torch.tensor([prompt_speech_tokens + generated_speech_tokens])
        flow_input_len = torch.tensor([flow_input.shape[1]])
        with torch.amp.autocast("cuda",
                dtype=torch.float16 if self.config.hf_config.fp16_flow else torch.float32):
            mels, mels_lens = self.flow(
                flow_input.cuda(), flow_input_len.cuda(),
                prompt_mel, prompt_mel_len_t, spk_emb,
                streaming=False, finalize=finalize,
            )
        mel = mels[:, :, prompt_mel_len_t[0].item(): mels_lens[0].item()]
        wav, _ = self.hift(speech_feat=mel)
        return wav