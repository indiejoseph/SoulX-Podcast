"""MTP + bi-stream TTFA benchmark.

Runs verified MTP speculative sampling in a worker thread and streams only
committed speech tokens to the flow/vocoder chunker. Use a small first chunk
for TTFA, then larger chunks for better total wall time.

Usage:
    python scripts/mtp/mtp_bistream_test.py CKPT [chunk_size=100] [first_chunk_size=12]
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_ROOT))
_sys.path.insert(0, str(_ROOT / "scripts" / "inference"))

import argparse
from collections import Counter
import threading
import time
from pathlib import Path

import torch
import torchaudio

from bistream_test import prepare_synth_state, synthesize_chunk
from streaming_hook_test import build_first_turn_prompt
from soulxpodcast.training.mtp_inference import mtp_speculative_sample_cached
from soulxpodcast.training.mtp_module import MtpConfig, SequentialMTP
from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.utils.streaming import SpeechTokenStreamer


DEFAULT_BASE = "pretrained_models/SoulX-Podcast-1.7B-dialect"


def resolve_model_path(ckpt, cli_base: str | None) -> str:
    train_base = ckpt.get("train_config", {}).get("model_path")
    if cli_base:
        if train_base and cli_base != train_base:
            print(
                f"[warn] --base {cli_base!r} differs from checkpoint model_path {train_base!r}"
            )
        return cli_base
    if train_base:
        if not Path(train_base).is_dir():
            fallback = Path("runs/merged")
            if fallback.is_dir():
                print(
                    f"[warn] checkpoint model_path {train_base!r} does not exist; "
                    f"using local {str(fallback)!r}"
                )
                return str(fallback)
            print(
                f"[warn] checkpoint model_path {train_base!r} does not exist locally; "
                "pass --base to override"
            )
        return train_base
    fallback = Path("runs/merged")
    if fallback.is_dir():
        print("[warn] checkpoint has no train_config.model_path; using local 'runs/merged'")
        return str(fallback)
    print(f"[warn] checkpoint has no train_config.model_path; using {DEFAULT_BASE!r}")
    return DEFAULT_BASE


def load_mtp(ckpt, base):
    mtp_config = MtpConfig(**ckpt["mtp_config"])
    decoder_layer_cls = base.model.layers[0].__class__
    mtp = SequentialMTP(mtp_config, decoder_layer_cls, base.config)
    mtp.load_state_dict(ckpt["mtp_state"])
    return mtp.to(device="cuda").eval()


def build_demo_input():
    return {
        "speakers": {
            "S1": {
                "prompt_audio": Path("example/audios/female_mandarin.wav"),
                "prompt_text": (
                    "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当"
                    "去景德镇做陶瓷的白日梦想家。"
                ),
            }
        },
        "text": [
            [
                "S1",
                "Hello everyone, welcome to our show. Hey, I want to ask you "
                "something today, do you feel that, um, driving an electric "
                "car nowadays, the most annoying thing is what?",
            ]
        ],
    }


def run_mtp_in_thread(model, mtp, input_ids, sampling_params, eos_id, streamer, cuda_stream):
    class _MTPThread(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.result = None
            self.exc = None

        def run(self):
            try:
                with torch.cuda.stream(cuda_stream), torch.autocast("cuda", dtype=torch.bfloat16):
                    self.result = mtp_speculative_sample_cached(
                        model.llm.model,
                        mtp,
                        input_ids,
                        max_new_tokens=sampling_params.max_tokens,
                        min_new_tokens=sampling_params.min_tokens,
                        eos_token_id=eos_id,
                        temperature=sampling_params.temperature,
                        top_k=sampling_params.top_k,
                        top_p=sampling_params.top_p,
                        repetition_penalty=sampling_params.repetition_penalty,
                        use_ras=sampling_params.use_ras,
                        ras_win_size=sampling_params.win_size,
                        ras_tau_r=sampling_params.tau_r,
                        allow_eos_from_drafts=False,
                        seed=198964,
                        streamer=streamer,
                    )
            except BaseException as e:
                self.exc = e
                streamer.end()
                raise

    thread = _MTPThread()
    thread.start()
    return thread


@torch.inference_mode()
def warmup(model, mtp, input_ids, sampling_params, eos_id, synth_state):
    print("[warmup] running short MTP decode + first flow/HiFT call")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        spec = mtp_speculative_sample_cached(
            model.llm.model,
            mtp,
            input_ids,
            max_new_tokens=16,
            min_new_tokens=min(sampling_params.min_tokens, 8),
            eos_token_id=eos_id,
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
            repetition_penalty=sampling_params.repetition_penalty,
            use_ras=sampling_params.use_ras,
            ras_win_size=sampling_params.win_size,
            ras_tau_r=sampling_params.tau_r,
            allow_eos_from_drafts=False,
            seed=198964,
        )
    warm_tokens = spec.generated_tokens[0].tolist()
    if warm_tokens and warm_tokens[-1] == eos_id:
        warm_tokens = warm_tokens[:-1]
    offset = model.config.hf_config.speech_token_offset
    warm_speech = [t - offset for t in warm_tokens[:8]]
    if warm_speech:
        _ = synthesize_chunk(model, synth_state, warm_speech, finalize=False)
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("chunk_size", nargs="?", type=int, default=100)
    ap.add_argument("first_chunk_size", nargs="?", type=int, default=12)
    ap.add_argument("--base", default=None)
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--no_warmup", action="store_true",
                    help="Report cold TTFA including first CUDA/kernel overhead.")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_path = resolve_model_path(ckpt, args.base)
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path("outputs/mtp_bistream")
        / f"first{args.first_chunk_size}_chunk{args.chunk_size}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[init] model={model_path}  ckpt={args.ckpt}  "
        f"chunk_size={args.chunk_size}  first_chunk_size={args.first_chunk_size}"
    )
    model, dataset = initiate_model(
        seed=198964,
        model_path=model_path,
        llm_engine="hf",
        fp16_flow=True,
    )
    mtp = load_mtp(ckpt, model.llm.model)

    inputs = podcast_format_parser(build_demo_input())
    prepared = process_single_input(
        dataset,
        inputs["text"],
        inputs["prompt_wav"],
        inputs["prompt_text"],
        inputs["use_dialect_prompt"],
        inputs["dialect_prompt_text"],
    )
    prompt_ids, eos_id, offset = build_first_turn_prompt(model, prepared)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    synth_state = prepare_synth_state(model, prepared)
    sampling_params = prepared["sampling_params"]

    if not args.no_warmup:
        warmup(model, mtp, input_ids, sampling_params, eos_id, synth_state)

    streamer = SpeechTokenStreamer(eos_token_id=eos_id)
    mtp_stream = torch.cuda.Stream()
    flow_stream = torch.cuda.Stream()

    t_start = time.perf_counter()
    mtp_thread = run_mtp_in_thread(
        model, mtp, input_ids, sampling_params, eos_id, streamer, mtp_stream
    )

    accumulated_speech_tokens = []
    prev_audio_len = 0
    audio_chunks = []
    flow_chunk_times = []
    ttfa = None

    for chunk_idx, cur_chunk in enumerate(
        streamer.iter_chunks(
            chunk_size=args.chunk_size,
            first_chunk_size=args.first_chunk_size,
        )
    ):
        accumulated_speech_tokens.extend([t - offset for t in cur_chunk])
        t_flow = time.perf_counter()
        with torch.cuda.stream(flow_stream):
            wav_full = synthesize_chunk(
                model, synth_state, accumulated_speech_tokens, finalize=False
            )
        new_audio = wav_full[:, prev_audio_len:].detach().cpu()
        flow_chunk_times.append(time.perf_counter() - t_flow)
        prev_audio_len = wav_full.shape[-1]

        now = time.perf_counter() - t_start
        if ttfa is None:
            ttfa = now
        print(
            f"[mtp-bistream] chunk {chunk_idx}: {len(cur_chunk)} tok, "
            f"+{new_audio.shape[-1] / 24000:.2f}s audio, "
            f"flow+HiFT={flow_chunk_times[-1]:.3f}s, t={now:.2f}s"
            f"{'  <- TTFA' if chunk_idx == 0 else ''}"
        )
        if new_audio.shape[-1] > 0:
            torchaudio.save(str(out_dir / f"chunk_{chunk_idx:02d}.wav"), new_audio, 24000)
            audio_chunks.append(new_audio)

    t_flow = time.perf_counter()
    with torch.cuda.stream(flow_stream):
        wav_full = synthesize_chunk(
            model, synth_state, accumulated_speech_tokens, finalize=True
        )
    final_audio = wav_full[:, prev_audio_len:].detach().cpu()
    flow_chunk_times.append(time.perf_counter() - t_flow)
    if final_audio.shape[-1] > 0:
        torchaudio.save(str(out_dir / f"chunk_{len(audio_chunks):02d}_final.wav"), final_audio, 24000)
        audio_chunks.append(final_audio)

    mtp_thread.join()
    if mtp_thread.exc:
        raise mtp_thread.exc
    spec = mtp_thread.result

    full_audio = torch.cat(audio_chunks, dim=-1) if audio_chunks else torch.zeros(1, 0)
    final_path = out_dir / "concatenated.wav"
    torchaudio.save(str(final_path), full_audio, 24000)

    total = time.perf_counter() - t_start
    audio_dur = full_audio.shape[-1] / 24000
    hist = Counter(spec.accept_lengths)
    bucket = " ".join(f"{i}:{hist.get(i, 0)}" for i in range(1, len(mtp.layers) + 2))

    print("\n[mtp-bistream] DONE")
    print(f"  TTFA:              {ttfa:.3f}s")
    print(f"  total wall:        {total:.3f}s")
    print(f"  total audio:       {audio_dur:.2f}s")
    print(f"  RTF:               {total / max(audio_dur, 1e-6):.3f}")
    print(f"  mean accept:       {spec.mean_accept_length:.2f}")
    print(f"  accept histogram:  {bucket}")
    print(f"  flow avg/range:    {sum(flow_chunk_times)/len(flow_chunk_times):.3f}s / "
          f"{min(flow_chunk_times):.3f}-{max(flow_chunk_times):.3f}s")
    print(f"  output:            {final_path}")


if __name__ == "__main__":
    main()
