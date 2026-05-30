"""A/B test: /generate (sync) vs /generate-stream on the same dialogue + seed.

Goal: isolate whether the per-turn loudness variation we see on the streaming
endpoint comes from the streaming code path (per-chunk flow + cache + HiFT
re-runs) vs. the non-streaming one-shot path. If sync is consistent and
stream is not, streaming is the culprit; if both vary the same way, the
variation is intrinsic to model output and unrelated to streaming.

Same dialogue text, same prompts, same seed are sent to both endpoints. Both
WAVs are written, plus a per-segment peak/RMS table for each.

Usage:
    python scripts/inference/gen_stream_vs_sync.py [--lang yue|zh]
"""
import argparse
import io
import json
import os
import shutil
import sys

import numpy as np
import requests
import scipy.io.wavfile as wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

API = "http://localhost:8000"
KEY = os.getenv("API_KEY", "soulx-local-dev-2026-xK9mP")
HDR = {"Authorization": f"Bearer {KEY}"}

# Mirror the dialogues used by gen_cantonese.py and gen_mandarin_qc.py so
# we A/B the exact same text the user heard the loudness inconsistency on.
DIALOGUES = {
    "yue": (
        "[S1]大家好，歡迎收聽今日嘅節目，我係主持人小明。"
        "[S2]我係嘉賓小紅，好開心今日可以嚟呢度同大家分享。"
        "[S1]今日我哋要傾嘅話題係人工智能喺語音合成領域嘅最新進展。"
        "[S2]係囉，呢個領域發展好快，尤其係近幾年大模型嘅出現，令語音合成嘅質素有咗質嘅飛躍。"
        "[S1]你可唔可以同我哋介紹一下目前最先進嘅語音合成技術？"
        "[S2]當然，目前最先進嘅系統通常結合咗大型語言模型同擴散模型，能夠生成非常自然流暢嘅語音。"
    ),
    "zh": (
        "[S1]大家好，欢迎收听今天的播客节目，我是主持人小明。"
        "[S2]我是嘉宾小红，今天非常开心能来到这里和大家交流。"
        "[S1]今天我们要聊的话题是人工智能在语音合成领域的最新进展，你能先简单介绍一下吗？"
        "[S2]当然可以，目前最先进的语音合成系统通常会结合大型语言模型和扩散模型，能够生成非常自然流畅的人声。"
    ),
}

PROMPTS = {
    "yue": [
        ("example/audios/female_mandarin.wav",
         "<|Yue|>喜歡攀岩、行山、滑雪嘅語言愛好者。"),
        ("example/audios/male_mandarin.wav",
         "<|Yue|>資深科技播客主持人。"),
    ],
    "zh": [
        ("example/audios/female_mandarin.wav",
         "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。"),
        ("example/audios/male_mandarin.wav",
         "我是一个资深的科技播客主持人，喜欢和大家分享前沿技术。"),
    ],
}


def post_and_save(endpoint: str, data: dict, audio_paths, out_path: str) -> bytes:
    files = [("prompt_audio", open(p, "rb")) for p in audio_paths]
    buf = io.BytesIO()
    with requests.post(f"{API}{endpoint}", files=files, data=data,
                       headers=HDR, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                buf.write(chunk)
    buf.seek(0)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(buf, f)
    return buf.getvalue()


def analyze(path: str, label: str):
    sr, data = wavfile.read(path)
    samples = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
    duration = len(samples) / sr
    peak_overall = float(np.abs(samples).max())
    rms_overall = float(np.sqrt((samples**2).mean()))

    # Segment by silence (smoothed envelope)
    energy = np.abs(samples)
    win = sr // 10
    smoothed = np.convolve(energy, np.ones(win) / win, mode="same")
    speech = smoothed > 0.02
    diff = np.diff(speech.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if speech[0]:
        starts = np.insert(starts, 0, 0)
    if speech[-1]:
        ends = np.append(ends, len(samples) - 1)
    segs = [(s, e) for s, e in zip(starts, ends) if (e - s) / sr >= 0.4]

    print(f"\n=== {label}  ({path}) ===")
    print(f"  duration={duration:.1f}s  overall peak={peak_overall:.3f}  rms={rms_overall:.4f}  segments={len(segs)}")
    print(f"  {'seg':>4} {'start':>7} {'dur':>6} {'peak':>7} {'rms':>7} {'p99':>7}")
    rms_vals, p99_vals = [], []
    for i, (s, e) in enumerate(segs):
        seg = samples[s:e]
        rms = np.sqrt((seg**2).mean())
        p99 = np.percentile(np.abs(seg), 99)
        rms_vals.append(rms); p99_vals.append(p99)
        print(f"  {i:>4} {s/sr:>6.2f}s {(e-s)/sr:>5.2f}s "
              f"{np.abs(seg).max():>7.3f} {rms:>7.4f} {p99:>7.3f}")
    if len(rms_vals) > 1:
        r = np.array(rms_vals); p = np.array(p99_vals)
        rms_db = 20 * np.log10(r.max() / r.min())
        p99_db = 20 * np.log10(p.max() / p.min())
        print(f"  spread: rms σ/μ={r.std()/r.mean():.2f}  p99 σ/μ={p.std()/p.mean():.2f}  "
              f"max/min rms={rms_db:.1f}dB  p99={p99_db:.1f}dB")
    return {"rms": rms_vals, "p99": p99_vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["yue", "zh"], default="zh")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    os.makedirs("outputs/stream_vs_sync", exist_ok=True)

    prompts = PROMPTS[args.lang]
    audio_paths = [p for p, _ in prompts]
    prompt_texts = [t for _, t in prompts]

    common = {
        "prompt_texts": json.dumps(prompt_texts),
        "dialogue_text": DIALOGUES[args.lang],
        "seed": args.seed,
    }

    sync_out = f"outputs/stream_vs_sync/{args.lang}_sync.wav"
    print(f"[1/2] POST /generate (sync) → {sync_out}")
    post_and_save("/generate", common, audio_paths, sync_out)

    stream_out = f"outputs/stream_vs_sync/{args.lang}_stream.wav"
    print(f"[2/2] POST /generate-stream → {stream_out}")
    post_and_save("/generate-stream", {**common, "chunk_size": 150}, audio_paths, stream_out)

    a = analyze(sync_out, f"SYNC   ({args.lang})")
    b = analyze(stream_out, f"STREAM ({args.lang})")
    print()


if __name__ == "__main__":
    main()
