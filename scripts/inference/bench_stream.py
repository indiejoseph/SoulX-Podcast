"""Benchmark /generate-stream: TTFA, wall time, RTF.

Usage:
  python scripts/inference/bench_stream.py [N_RUNS] [--chunk N] [--first-chunk N] [--long]

  --chunk N        Override STREAM_CHUNK_SIZE per request (default: server config)
  --first-chunk N  Override STREAM_FIRST_CHUNK_SIZE per request (default: server config)
  --long           Use longer 6-turn dialogue (~18s audio) instead of short 3-turn
"""
import sys
import os
import time
import json
import argparse
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("n_runs", nargs="?", type=int, default=3)
parser.add_argument("--chunk", type=int, default=None)
parser.add_argument("--first-chunk", type=int, default=None)
parser.add_argument("--long", action="store_true")
args = parser.parse_args()

API  = "http://localhost:8000"
KEY  = os.getenv("API_KEY", "soulx-local-dev-2026-xK9mP")
HDR  = {"Authorization": f"Bearer {KEY}"}
N    = args.n_runs

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

DATA  = {
    "prompt_texts": json.dumps(["喜欢攀岩、徒步、滑雪的语言爱好者。", "资深科技播客主持人。"]),
    "dialogue_text": LONG_DIALOGUE if args.long else SHORT_DIALOGUE,
    "seed": 1988,
}
if args.chunk is not None:
    DATA["chunk_size"] = args.chunk
if args.first_chunk is not None:
    DATA["first_chunk_size"] = args.first_chunk

chunk_label  = str(args.chunk) if args.chunk else os.getenv("STREAM_CHUNK_SIZE", "srv")
fchunk_label = str(args.first_chunk) if args.first_chunk else os.getenv("STREAM_FIRST_CHUNK_SIZE", "srv")

print(f"\n{'='*60}")
print(f"Streaming benchmark  N={N}  dialogue={'long' if args.long else 'short'}")
print(f"Engine: {os.getenv('LLM_ENGINE','hf')}  MTP: {os.getenv('ENABLE_MTP','?')}  "
      f"first_chunk={fchunk_label}  chunk={chunk_label}")
print(f"{'='*60}")

rows = []
for i in range(N):
    files = [("prompt_audio", open(f, "rb")) for f in AUDIO]
    t0    = time.perf_counter()
    ttfa  = None
    total = 0
    with requests.post(f"{API}/generate-stream", files=files, data=DATA,
                       headers=HDR, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                if ttfa is None and total >= 44:   # past WAV header
                    ttfa = time.perf_counter() - t0
                total += len(chunk)
    wall  = time.perf_counter() - t0
    for _, f in files:
        f.close()
    dur   = (total - 44) / (24000 * 2)            # PCM16 mono 24 kHz
    rtf   = wall / dur if dur > 0 else 0
    rows.append((ttfa or 0.0, wall, dur, rtf))
    print(f"  run {i+1:2d}:  TTFA={ttfa:.2f}s  wall={wall:.2f}s  "
          f"audio={dur:.1f}s  RTF={rtf:.3f}")

ttfas = [r[0] for r in rows]
walls = [r[1] for r in rows]
rtfs  = [r[3] for r in rows]
print(f"{'─'*60}")
print(f"  avg:    TTFA={sum(ttfas)/N:.2f}s  wall={sum(walls)/N:.2f}s  "
      f"RTF={sum(rtfs)/N:.3f}")
