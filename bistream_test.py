"""True bi-streaming demo (PLAN.md Phase 0 task 2b).

LLM streams speech tokens in a worker thread. Main thread processes each
token chunk through flow + HiFT concurrently, yielding audio chunks as they
become available. Measures TTFA (time to first audio) — the real metric
streaming improves.

The flow's streaming= parameter only enables chunk-masked attention, not a
stateful cache. So per-chunk synthesis re-runs flow on (prompt + all tokens
so far) and slices out the new audio portion. This is O(N^2) on flow compute
but flow is only ~12% of total time so the redundancy is acceptable for now.

Usage:
    python bistream_test.py [chunk_size=50]
"""

import sys
import time
from pathlib import Path

import torch
import torchaudio

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.utils.streaming import SpeechTokenStreamer, run_llm_in_thread
from streaming_hook_test import build_first_turn_prompt


def synthesize_chunk(model, prepared, all_speech_tokens, finalize):
    """Run flow+HiFT on (prompt_speech_tokens + all_speech_tokens). Returns full waveform.

    Caller slices off the audio portion already emitted on prior calls.
    """
    prompt_mels = prepared["prompt_mels_for_llm"]
    prompt_mels_lens = prepared["prompt_mels_lens_for_llm"]
    prompt_spk_tokens, prompt_lens = model.audio_tokenizer.quantize(
        prompt_mels.cuda(), prompt_mels_lens.cuda()
    )
    spk0_prompt_tokens = prompt_spk_tokens[0, : prompt_lens[0].item()].tolist()

    flow_input = torch.tensor([spk0_prompt_tokens + all_speech_tokens])
    flow_input_len = torch.tensor([flow_input.shape[1]])

    prompt_mel = prepared["prompt_mels_for_flow_ori"][0]
    prompt_mel = prompt_mel[: prompt_lens[0].item() * 2].cuda()
    prompt_mel_len = torch.tensor([prompt_mel.shape[0]]).cuda()
    spk_emb = prepared["spk_emb_for_flow"][0:1].cuda()

    with torch.amp.autocast("cuda",
            dtype=torch.float16 if model.config.hf_config.fp16_flow else torch.float32):
        mels, mels_lens = model.flow(
            flow_input.cuda(), flow_input_len.cuda(),
            prompt_mel[None], prompt_mel_len, spk_emb,
            streaming=False, finalize=finalize,
        )
    # Drop the prompt-mel prefix (matches forward_longform).
    mel = mels[:, :, prompt_mel_len[0].item(): mels_lens[0].item()]
    wav, _ = model.hift(speech_feat=mel)
    return wav  # [1, T]


def main():
    model_path = "pretrained_models/SoulX-Podcast-1.7B-dialect"
    chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print(f"[init] loading model (hf engine, chunk_size={chunk_size})")
    t0 = time.perf_counter()
    model, dataset = initiate_model(seed=198964, model_path=model_path,
                                     llm_engine="hf", fp16_flow=True)
    print(f"[init] load: {time.perf_counter()-t0:.1f}s")

    data = {
        "speakers": {
            "S1": {
                "prompt_audio": Path("example/audios/female_mandarin.wav"),
                "prompt_text": "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
            }
        },
        "text": [
            ["S1", "Hello everyone, welcome to our show. Hey, I want to ask you "
                   "something today, do you feel that, um, driving an electric "
                   "car nowadays, the most annoying thing is what?"]
        ],
    }
    inputs = podcast_format_parser(data)
    prepared = process_single_input(
        dataset, inputs["text"], inputs["prompt_wav"], inputs["prompt_text"],
        inputs["use_dialect_prompt"], inputs["dialect_prompt_text"],
    )
    prompt_ids, eos_id, offset = build_first_turn_prompt(model, prepared)

    sp = prepared["sampling_params"]
    if isinstance(sp, list):
        sp = sp[0]

    out_dir = Path("outputs/bistream") / f"chunk{chunk_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    streamer = SpeechTokenStreamer(eos_token_id=eos_id)
    # Separate CUDA streams so LLM and flow+HiFT overlap on the GPU (B3).
    llm_stream = torch.cuda.Stream()
    flow_stream = torch.cuda.Stream()

    print(f"\n[bistream] starting LLM worker + main-thread synthesizer "
          f"(separate CUDA streams)")
    t_start = time.perf_counter()
    llm_thread = run_llm_in_thread(
        model.llm, prompt_ids, sp,
        past_key_values=None, streamer=streamer,
        cuda_stream=llm_stream,
    )

    accumulated_speech_tokens = []  # in s3tokenizer space (already offset-stripped)
    prev_audio_len = 0  # samples already emitted
    audio_chunks = []
    ttfa = None
    flow_chunk_times = []

    # Process each chunk as soon as it arrives — DO NOT peek ahead, that would
    # block on the LLM thread and serialize. Treat each intermediate chunk as
    # finalize=False (last 3 tokens act as lookahead context per the flow's
    # streaming convention). After the LLM thread finishes, do one final pass
    # with finalize=True to flush the trailing lookahead tokens.
    for chunk_idx, cur_chunk in enumerate(streamer.iter_chunks(chunk_size=chunk_size)):
        cur_speech_tokens = [t - offset for t in cur_chunk]
        accumulated_speech_tokens.extend(cur_speech_tokens)

        t_chunk_start = time.perf_counter()
        with torch.cuda.stream(flow_stream):
            wav_full = synthesize_chunk(model, prepared,
                                        accumulated_speech_tokens, finalize=False)
        new_audio = wav_full[:, prev_audio_len:].detach().cpu()
        flow_chunk_times.append(time.perf_counter() - t_chunk_start)

        prev_audio_len = wav_full.shape[-1]

        t_now = time.perf_counter() - t_start
        marker = "  ← TTFA" if ttfa is None else ""
        if ttfa is None:
            ttfa = t_now
        print(f"[bistream] chunk {chunk_idx}: {len(cur_chunk)} tok, "
              f"+{new_audio.shape[-1]/24000:.2f}s audio, "
              f"flow+HiFT={flow_chunk_times[-1]:.3f}s, t={t_now:.2f}s{marker}")

        torchaudio.save(str(out_dir / f"chunk_{chunk_idx:02d}.wav"),
                        new_audio, 24000)
        audio_chunks.append(new_audio)

    # Final flush: finalize=True on the complete token list to emit audio for
    # the trailing tokens the intermediate calls treated as lookahead context.
    t_final = time.perf_counter()
    with torch.cuda.stream(flow_stream):
        wav_full = synthesize_chunk(model, prepared,
                                    accumulated_speech_tokens, finalize=True)
    final_audio = wav_full[:, prev_audio_len:].detach().cpu()
    flow_chunk_times.append(time.perf_counter() - t_final)
    if final_audio.shape[-1] > 0:
        torchaudio.save(str(out_dir / f"chunk_{len(audio_chunks):02d}_final.wav"),
                        final_audio, 24000)
        audio_chunks.append(final_audio)
        print(f"[bistream] final flush: +{final_audio.shape[-1]/24000:.2f}s audio, "
              f"flow+HiFT={flow_chunk_times[-1]:.3f}s, "
              f"t={time.perf_counter()-t_start:.2f}s")

    llm_thread.join()
    if llm_thread.exc:
        raise llm_thread.exc

    # Concatenate per-chunk audio into one file (this is what would be played
    # back end-to-end after streaming).
    full_audio = torch.cat(audio_chunks, dim=-1)
    final_path = out_dir / "concatenated.wav"
    torchaudio.save(str(final_path), full_audio, 24000)

    t_end = time.perf_counter() - t_start
    audio_dur = full_audio.shape[-1] / 24000

    print(f"\n[bistream] DONE")
    print(f"  TTFA (time to first audio): {ttfa:.3f}s")
    print(f"  total wall time:            {t_end:.3f}s")
    print(f"  total audio:                {audio_dur:.2f}s @ 24kHz")
    print(f"  per-chunk flow+HiFT (avg):  {sum(flow_chunk_times)/len(flow_chunk_times):.3f}s")
    print(f"  per-chunk flow+HiFT range:  {min(flow_chunk_times):.3f}s – {max(flow_chunk_times):.3f}s")
    print(f"  RTF (wall/audio):           {t_end/audio_dur:.3f}")
    print(f"  output: {final_path}")


if __name__ == "__main__":
    main()
