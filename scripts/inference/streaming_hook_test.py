"""Smoke test for the LLM streaming hook (PLAN.md Phase 0 task 2a).

Wires SpeechTokenStreamer through HFLLMEngine.generate(), runs it in a worker
thread, and drains tokens in fixed-size chunks from the main thread. Logs
per-chunk timing so we can see the streaming cadence.

This validates the foundation needed for chunked flow+HiFT synthesis — once
this works, the next step (task 2b) feeds each token chunk into
self.flow(..., streaming=True) and HiFT.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import sys
import time
from pathlib import Path

import torch

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.utils.streaming import SpeechTokenStreamer, run_llm_in_thread


def build_first_turn_prompt(model, data_prepared):
    """Replicate the prompt construction from SoulXPodcast.forward_longform
    for just the first dialogue turn — enough to exercise the LLM streamer
    end-to-end without running the full flow/HiFT path."""
    import s3tokenizer
    from itertools import chain

    prompt_mels = data_prepared["prompt_mels_for_llm"]
    prompt_mels_lens = data_prepared["prompt_mels_lens_for_llm"]
    prompt_text_tokens = data_prepared["prompt_text_tokens_for_llm"]
    text_tokens = data_prepared["text_tokens_for_llm"]

    speech_token_offset = model.config.hf_config.speech_token_offset
    eos_id = model.config.hf_config.eos_token_id

    # Quantize the prompt audio into speech tokens (matches forward_longform)
    prompt_speech_tokens, prompt_lens = model.audio_tokenizer.quantize(
        prompt_mels.cuda(), prompt_mels_lens.cuda()
    )

    # Build prompt segment for speaker 0 only
    spk_tokens = prompt_speech_tokens[0, : prompt_lens[0].item()].tolist()
    spk_tokens = [t + speech_token_offset for t in spk_tokens] + [eos_id]
    prompt_segment = prompt_text_tokens[0] + spk_tokens

    # Append first dialogue turn's text tokens (these end with semantic_token_start)
    inputs = prompt_segment + text_tokens[0]
    return inputs, eos_id, speech_token_offset


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
    print(f"[init] prompt length: {len(prompt_ids)} tokens, eos_id={eos_id}, "
          f"speech_offset={offset}")

    sampling_params = prepared["sampling_params"]
    if isinstance(sampling_params, list):
        sampling_params = sampling_params[0]

    streamer = SpeechTokenStreamer(eos_token_id=eos_id)

    print(f"\n[stream] starting LLM in worker thread")
    t_start = time.perf_counter()
    llm_thread = run_llm_in_thread(
        model.llm, prompt_ids, sampling_params,
        past_key_values=None, streamer=streamer,
    )

    # Main thread: drain tokens in fixed-size chunks
    first_token_time = None
    total_tokens = 0
    chunks = []
    for chunk_idx, chunk in enumerate(streamer.iter_chunks(chunk_size=chunk_size)):
        now = time.perf_counter() - t_start
        if first_token_time is None:
            first_token_time = now
            print(f"[stream] chunk 0  ({len(chunk):>3} tok)  "
                  f"t={now:6.3f}s  (TTFC)")
        else:
            print(f"[stream] chunk {chunk_idx}  ({len(chunk):>3} tok)  "
                  f"t={now:6.3f}s  Δ={now - prev_t:6.3f}s")
        prev_t = now
        total_tokens += len(chunk)
        chunks.append(chunk)

    llm_thread.join()
    t_end = time.perf_counter() - t_start
    if llm_thread.exc:
        raise llm_thread.exc

    print(f"\n[stream] DONE")
    print(f"  total wall time:       {t_end:.3f}s")
    print(f"  time-to-first-chunk:   {first_token_time:.3f}s  (chunk_size={chunk_size})")
    print(f"  total speech tokens:   {total_tokens}  ({total_tokens / 25:.2f}s audio @ 25 Hz)")
    print(f"  avg generation rate:   {total_tokens / t_end:.1f} tok/s")

    # Sanity: tokens reported by streamer should match what generate() returned.
    final_ids = llm_thread.result["token_ids"]
    # generate() includes the EOS at the end; streamer dropped it.
    final_ids_no_eos = [t for t in final_ids if t != eos_id]
    streamed_flat = [t for c in chunks for t in c]
    assert streamed_flat == final_ids_no_eos, (
        f"streamed token mismatch: streamer={len(streamed_flat)} tokens, "
        f"generate()={len(final_ids_no_eos)} tokens"
    )
    print(f"[verify] streamed tokens match generate() output ✓")

    # ---- Synthesize wav from streamed tokens (one-shot flow+HiFT) --------
    # This proves the tokens are acoustically meaningful. We run flow+HiFT
    # in a single non-streaming call here — chunked synthesis is task 2b.
    import torchaudio

    print(f"\n[wav] synthesizing audio from streamed tokens")
    t_syn = time.perf_counter()
    generated_speech_tokens = [t - offset for t in streamed_flat]

    # Speaker-0 prompt speech tokens (same path forward_longform takes)
    prompt_mels = prepared["prompt_mels_for_llm"]
    prompt_mels_lens = prepared["prompt_mels_lens_for_llm"]
    prompt_spk_tokens, prompt_lens = model.audio_tokenizer.quantize(
        prompt_mels.cuda(), prompt_mels_lens.cuda()
    )
    spk0_tokens = prompt_spk_tokens[0, : prompt_lens[0].item()].tolist()

    flow_input = torch.tensor([spk0_tokens + generated_speech_tokens])
    flow_input_len = torch.tensor([flow_input.shape[1]])

    # Match prompt_mels_for_flow alignment from forward_longform (truncate
    # mel to 2x the prompt-speech-token length).
    prompt_mel = prepared["prompt_mels_for_flow_ori"][0]
    prompt_mel = prompt_mel[: prompt_lens[0].item() * 2].cuda()
    prompt_mel_len = torch.tensor([prompt_mel.shape[0]]).cuda()
    spk_emb = prepared["spk_emb_for_flow"][0:1].cuda()

    with torch.amp.autocast("cuda",
            dtype=torch.float16 if model.config.hf_config.fp16_flow else torch.float32):
        mels, mels_lens = model.flow(
            flow_input.cuda(), flow_input_len.cuda(),
            prompt_mel[None], prompt_mel_len, spk_emb,
            streaming=False, finalize=True,
        )

    # Drop the prompt-mel prefix from the output (mirrors forward_longform)
    mel = mels[:, :, prompt_mel_len[0].item(): mels_lens[0].item()]
    wav, _ = model.hift(speech_feat=mel)
    print(f"[wav] flow+HiFT: {time.perf_counter()-t_syn:.2f}s, "
          f"wav shape={tuple(wav.shape)}  ({wav.shape[-1]/24000:.2f}s @ 24 kHz)")

    out_path = Path("outputs/streaming") / f"streamed_chunk{chunk_size}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav.cpu(), 24000)
    print(f"[wav] saved -> {out_path}")


if __name__ == "__main__":
    main()
