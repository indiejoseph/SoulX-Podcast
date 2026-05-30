"""End-to-end pipeline profiler — find the current bottleneck.

Runs `forward_longform_streaming` on a representative long dialogue with the
best-known defaults (FLOW_CHUNK_CACHE=1, FLOW_STEPS=4, flow_streaming=True)
and breaks total wall into:

  * Frontend  — `process_single_input` (text norm + tokenize + prompt audio
                + mel extract via s3tokenizer)
  * LLM       — speech token generation in the worker thread (start to last
                yielded token)
  * Flow      — sum of cache-aware flow chunk wall (encoder + CFM)
  * HiFT      — sum of vocoder wall
  * Other     — total wall minus the above (Python overhead, yield latency)

Because LLM and Flow+HiFT run on separate CUDA streams (B3 always on), the
LLM wall and Flow+HiFT wall **overlap**. The effective per-call bottleneck
is max(LLM share per token rate, Flow+HiFT per chunk). The total wall is
roughly `max(LLM_total, Flow+HiFT_total) + small overheads`.

Usage:
    python scripts/inference/profile_pipeline.py [engine=hf] [chunk=150]
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os
import sys
import time
from collections import defaultdict

import torch

os.environ.setdefault("FLOW_CHUNK_CACHE", "1")
os.environ.setdefault("PROFILE_FLOW_STAGES", "1")  # surfaces per-call flow vs HiFT timing

from soulxpodcast import config as _config

# Leave headroom for the flow cache K/V tensors that accumulate over the run.
# Default 0.9 collides with the U-Net cache on a 24GB card.
_config.Config.gpu_memory_utilization = 0.6  # type: ignore[assignment]

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input


LONG_DIALOGUE = {
    "speakers": {
        "S1": {"prompt_audio": "example/audios/female_mandarin.wav"},
        "S2": {"prompt_audio": "example/audios/male_mandarin.wav"},
    },
    "turns": [
        ("S1", "大家好，欢迎收听今天的播客节目。今天我们要聊的话题非常有意思，关于人工智能的最新发展，以及它如何改变我们日常生活的方方面面。"),
        ("S2", "对，这真的是一个非常热门的话题。我自己最近一直在研究大型语言模型，发现它们的进步速度真的超乎想象。"),
        ("S1", "那你能不能跟我们详细聊聊，这些模型到底是怎么工作的？为什么它们能产生如此惊人的效果？我们普通人也想了解一下背后的原理。"),
    ],
}


def _collect_event_timings(events_iter):
    """Drain the streaming generator and record per-event wall time."""
    timings = {"events": [], "audio_chunks": []}
    t0 = time.perf_counter()
    prev = t0
    for ev in events_iter:
        now = time.perf_counter()
        timings["events"].append({
            "turn": ev["turn"],
            "chunk": ev["chunk"],
            "is_last_in_turn": ev["is_last_in_turn"],
            "audio_samples": ev["audio"].numel(),
            "since_prev": now - prev,
            "since_start": now - t0,
        })
        timings["audio_chunks"].append(ev["audio"])
        prev = now
    timings["wall"] = time.perf_counter() - t0
    return timings


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "hf"
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    model_path = os.environ.get(
        "MODEL_PATH",
        "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/pretrained_models/SoulX-Podcast-1.7B-dialect",
    )
    print(f"[init] engine={engine} chunk_size={chunk_size}")
    print(f"[init] FLOW_CHUNK_CACHE={os.environ['FLOW_CHUNK_CACHE']} "
          f"PROFILE_FLOW_STAGES={os.environ['PROFILE_FLOW_STAGES']}")
    t0 = time.perf_counter()
    model, dataset = initiate_model(
        seed=198964, model_path=model_path, llm_engine=engine, fp16_flow=True
    )
    print(f"[init] model load: {time.perf_counter()-t0:.1f}s")

    # Frontend timing
    target_texts = [f"[{spk}]{txt}" for spk, txt in LONG_DIALOGUE["turns"]]
    prompt_wavs = [LONG_DIALOGUE["speakers"][s]["prompt_audio"]
                   for s in sorted(LONG_DIALOGUE["speakers"].keys())]
    prompt_texts = [
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        "我是一个资深的播客主持人，对科技和人工智能领域有深入研究。",
    ]

    t0 = time.perf_counter()
    prepared = process_single_input(
        dataset,
        target_text_list=target_texts,
        prompt_wav_list=prompt_wavs,
        prompt_text_list=prompt_texts,
        use_dialect_prompt=False,
        dialect_prompt_text_list=None,
    )
    frontend_wall = time.perf_counter() - t0
    print(f"[frontend] process_single_input: {frontend_wall*1000:.0f} ms")

    # Capture flow/HiFT per-call timings from PROFILE_FLOW_STAGES output.
    # tqdm.write goes to stderr; we hook stderr via a redirect.
    flow_calls = []
    hift_calls = []

    import io
    import contextlib

    class _Tee(io.StringIO):
        def write(self, s):
            super().write(s)
            sys.__stdout__.write(s)
            sys.__stdout__.flush()
            return len(s)

    tee = _Tee()

    print(f"[run] forward_longform_streaming (chunk_size={chunk_size})", flush=True)
    torch.cuda.synchronize()
    t_pipeline_start = time.perf_counter()
    # tqdm.write defaults to sys.stdout — redirect both stdout and stderr
    # so we catch the PROFILE lines wherever they go.
    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
        events_iter = model.forward_longform_streaming(
            **prepared,
            chunk_size=chunk_size,
            first_chunk_size=4,
            flow_streaming=True,
            flow_steps=4,
        )
        timings = _collect_event_timings(events_iter)
    torch.cuda.synchronize()
    t_pipeline_wall = time.perf_counter() - t_pipeline_start

    # Parse profile lines from tee
    for line in tee.getvalue().splitlines():
        if "[PROFILE] flow=" in line and "hift=" in line:
            # e.g. "[PROFILE] flow=0.260s  hift=0.049s  mel_frames=172"
            try:
                parts = line.split()
                flow_s = float(parts[1].split("=")[1].rstrip("s"))
                hift_s = float(parts[2].split("=")[1].rstrip("s"))
                flow_calls.append(flow_s)
                hift_calls.append(hift_s)
            except (IndexError, ValueError):
                pass
        elif "[PROFILE] cached_flow=" in line:
            # e.g. "[PROFILE] cached_flow=0.260s  mel_frames=172"
            try:
                flow_s = float(line.split("cached_flow=")[1].split("s")[0])
                flow_calls.append(flow_s)
            except (IndexError, ValueError):
                pass
        elif "[PROFILE] cached_hift=" in line:
            # e.g. "[PROFILE] cached_hift=0.049s  mel_frames=300"
            try:
                hift_s = float(line.split("cached_hift=")[1].split("s")[0])
                hift_calls.append(hift_s)
            except (IndexError, ValueError):
                pass

    flow_total = sum(flow_calls)
    hift_total = sum(hift_calls)
    n_flow = len(flow_calls)
    n_hift = len(hift_calls)

    # Audio total
    total_samples = sum(c.numel() for c in timings["audio_chunks"])
    audio_s = total_samples / 24000.0

    other = t_pipeline_wall - max(flow_total + hift_total, 0.0)  # rough: LLM overlaps with flow
    print()
    print(f"=== Pipeline breakdown (engine={engine}, chunk={chunk_size}, FLOW_CHUNK_CACHE=1) ===")
    print(f"  total wall      : {t_pipeline_wall:.2f}s")
    print(f"  audio produced  : {audio_s:.2f}s  ({total_samples} samples)")
    print(f"  end-to-end RTF  : {t_pipeline_wall / audio_s:.3f}")
    print(f"  frontend prep   : {frontend_wall*1000:.0f} ms  ({frontend_wall/t_pipeline_wall*100:.1f}%)")
    print(f"  Flow (cached)   : {flow_total:.2f}s  ({n_flow} calls × {flow_total/max(n_flow,1)*1000:.0f}ms avg)  ({flow_total/t_pipeline_wall*100:.1f}% of wall)")
    print(f"  HiFT            : {hift_total:.2f}s  ({n_hift} calls × {hift_total/max(n_hift,1)*1000:.0f}ms avg)  ({hift_total/t_pipeline_wall*100:.1f}% of wall)")
    print(f"  Flow + HiFT sum : {flow_total + hift_total:.2f}s  ({(flow_total+hift_total)/t_pipeline_wall*100:.1f}% of wall)")
    print()
    print(f"  LLM time (implied): {t_pipeline_wall - max(flow_total + hift_total, 0):.2f}s  "
          f"(wall − flow+HiFT; LLM overlaps with flow via B3 dual streams)")
    print()

    # Per-turn breakdown
    by_turn = defaultdict(lambda: {"chunks": 0, "audio_samples": 0, "wall": 0.0})
    for ev in timings["events"]:
        by_turn[ev["turn"]]["chunks"] += 1
        by_turn[ev["turn"]]["audio_samples"] += ev["audio_samples"]
        by_turn[ev["turn"]]["wall"] += ev["since_prev"]
    print(f"=== Per-turn ===")
    for turn, d in sorted(by_turn.items()):
        print(f"  turn {turn}: chunks={d['chunks']}  audio={d['audio_samples']/24000:.1f}s  wall={d['wall']:.2f}s")

    # First chunk latency (TTFA)
    first_audio_chunks = [ev for ev in timings["events"] if ev["audio_samples"] > 0]
    ttfa = first_audio_chunks[0]["since_start"] if first_audio_chunks else float("nan")
    print(f"\n[ttfa] first audio at: {ttfa:.3f}s")

    # Verdict
    print()
    print("=== Bottleneck verdict ===")
    llm_implied = t_pipeline_wall - max(flow_total + hift_total, 0)
    if llm_implied > flow_total + hift_total:
        delta = llm_implied - (flow_total + hift_total)
        print(f"  LLM dominates by ~{delta:.2f}s ({delta/t_pipeline_wall*100:.0f}% of wall is LLM-only).")
        print(f"  Further flow/HiFT cuts won't help — LLM is the new ceiling.")
    elif flow_total + hift_total > llm_implied * 1.1:
        delta = flow_total + hift_total - llm_implied
        print(f"  Flow+HiFT still dominates by ~{delta:.2f}s.")
        print(f"  More flow/HiFT optimization is worthwhile.")
    else:
        print(f"  LLM and Flow+HiFT are roughly balanced (~{llm_implied:.2f}s vs "
              f"~{flow_total+hift_total:.2f}s).")
        print(f"  Both must improve for a meaningful end-to-end win.")


if __name__ == "__main__":
    main()
