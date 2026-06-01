"""Concurrent-stream benchmark against /generate-stream.

Drives N concurrent HTTP clients at the production TTS service, each running a
fixed dialogue (default the long Cantonese set). Measures per-client TTFA,
end-to-end wall, audio duration, and computes RTF.

This is the production-relevant test for max-QPS-on-single-GPU: it measures
what each client actually sees when sharing the GPU with N-1 others. The
single-user RTF numbers we'd been benching previously do not predict this.

Usage:
    python scripts/inference/bench_concurrent.py [--n N] [--long] [--chunk N]

Run it twice — once with ``FLOW_HIFT_BATCHING`` env unset on the service,
once with it set — to A/B the post-LLM batcher.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

API = "http://localhost:8000"
KEY = os.getenv("API_KEY", "soulx-local-dev-2026-xK9mP")
HDR = {"Authorization": f"Bearer {KEY}"}

AUDIO = ["example/audios/female_mandarin.wav", "example/audios/male_mandarin.wav"]

SHORT_DIALOGUE = (
    "[S1]大家好，欢迎收听今天的节目。"
    "[S2]是的，今天我们要聊聊人工智能。"
    "[S1]这个话题确实很有趣。"
)
LONG_DIALOGUE = (
    "[S1]大家好，欢迎收听今天的节目，我是主持人小明。"
    "[S2]我是嘉宾小红，很高兴今天能来到这里和大家分享。"
    "[S1]今天我们要聊的话题是人工智能在语音合成领域的最新进展。"
    "[S2]是的，这个领域发展非常快，尤其是近几年大模型的出现，让语音合成的质量有了质的飞跃。"
    "[S1]您能给我们介绍一下目前最先进的语音合成技术吗？"
    "[S2]当然，目前最先进的系统通常结合了大型语言模型和扩散模型，能够生成非常自然流畅的语音。"
)


def run_one(client_idx, dialogue_text, chunk, first_chunk):
    """One streaming client. Returns (idx, ttfa, wall, audio_dur, rtf, err)."""
    data = {
        "prompt_texts": json.dumps([
            "喜欢攀岩、徒步、滑雪的语言爱好者。",
            "资深科技播客主持人。",
        ]),
        "dialogue_text": dialogue_text,
        # Distinct seed per client so we don't accidentally hit a vLLM prefix
        # cache hit that would invalidate the QPS measurement.
        "seed": 1988 + client_idx,
    }
    if chunk is not None:
        data["chunk_size"] = chunk
    if first_chunk is not None:
        data["first_chunk_size"] = first_chunk

    files = [("prompt_audio", open(f, "rb")) for f in AUDIO]
    t0 = time.perf_counter()
    ttfa = None
    total_bytes = 0
    err = None
    try:
        with requests.post(
            f"{API}/generate-stream", files=files, data=data,
            headers=HDR, stream=True, timeout=300,
        ) as resp:
            resp.raise_for_status()
            for chunk_b in resp.iter_content(chunk_size=None):
                if chunk_b:
                    if ttfa is None and total_bytes >= 44:
                        ttfa = time.perf_counter() - t0
                    total_bytes += len(chunk_b)
        wall = time.perf_counter() - t0
        dur = (total_bytes - 44) / (24000 * 2)
        rtf = wall / dur if dur > 0 else 0
        return client_idx, ttfa or 0.0, wall, dur, rtf, None
    except Exception as e:
        wall = time.perf_counter() - t0
        return client_idx, 0.0, wall, 0.0, 0.0, repr(e)
    finally:
        for _, f in files:
            f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="concurrent client count")
    ap.add_argument("--long", action="store_true", help="use long dialogue")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--first-chunk", type=int, default=None)
    ap.add_argument("--warmup", action="store_true",
                    help="run one warmup pass at N=1 before the real bench")
    args = ap.parse_args()

    dialogue = LONG_DIALOGUE if args.long else SHORT_DIALOGUE
    label = "long" if args.long else "short"

    if args.warmup:
        print(f"[warmup] one solo request to warm CUDA graphs ...")
        _, ttfa, wall, dur, rtf, err = run_one(0, dialogue, args.chunk, args.first_chunk)
        print(f"  warm: TTFA={ttfa:.2f}s wall={wall:.2f}s rtf={rtf:.3f}"
              + (f' ERR={err}' if err else ''))

    print(f"\n{'='*72}")
    print(f"Concurrent bench  N={args.n}  dialogue={label}  "
          f"chunk={args.chunk}  first_chunk={args.first_chunk}")
    print(f"{'='*72}")

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.n) as pool:
        futures = [
            pool.submit(run_one, i, dialogue, args.chunk, args.first_chunk)
            for i in range(args.n)
        ]
        results = [f.result() for f in as_completed(futures)]
    t_total = time.perf_counter() - t_start
    results.sort(key=lambda r: r[0])

    print(f"  {'cli':>3} {'TTFA':>7} {'wall':>7} {'audio':>7} {'RTF':>6}  err")
    ttfas, walls, rtfs, durs, errs = [], [], [], [], 0
    for idx, ttfa, wall, dur, rtf, err in results:
        marker = "ERR" if err else ""
        if err:
            errs += 1
        ttfas.append(ttfa); walls.append(wall); rtfs.append(rtf); durs.append(dur)
        print(f"  {idx:>3} {ttfa:>6.2f}s {wall:>6.2f}s {dur:>5.1f}s {rtf:>6.3f}  {marker}")

    n_ok = args.n - errs
    if n_ok == 0:
        print("\nAll clients errored. Check the service.")
        sys.exit(2)
    print(f"{'─'*72}")
    print(f"  N={args.n}  total wall {t_total:.2f}s  errors={errs}/{args.n}")
    print(f"  TTFA   p50={sorted(ttfas)[n_ok//2]:.2f}s  "
          f"max={max(ttfas):.2f}s  avg={sum(ttfas)/n_ok:.2f}s")
    print(f"  wall   p50={sorted(walls)[n_ok//2]:.2f}s  "
          f"max={max(walls):.2f}s  avg={sum(walls)/n_ok:.2f}s")
    print(f"  RTF    p50={sorted(rtfs)[n_ok//2]:.3f}  "
          f"max={max(rtfs):.3f}  avg={sum(rtfs)/n_ok:.3f}")
    audio_total = sum(durs)
    print(f"  audio_throughput = {audio_total/t_total:.2f}x realtime "
          f"(generated {audio_total:.1f}s in {t_total:.2f}s wall)")


if __name__ == "__main__":
    main()
