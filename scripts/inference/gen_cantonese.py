"""Generate a Cantonese (Yue) quality-check WAV via the streaming API.

Usage:
    python scripts/inference/gen_cantonese.py

Output: /home/joseph/projects/SoulX-Podcast/outputs/cantonese_quality_check.wav
"""
import sys, os, io, shutil, numpy as np, requests, json
import scipy.io.wavfile as wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

API = "http://localhost:8000"
KEY = os.getenv("API_KEY", "soulx-local-dev-2026-xK9mP")
HDR = {"Authorization": f"Bearer {KEY}"}

AUDIO = [
    "example/audios/female_mandarin.wav",
    "example/audios/male_mandarin.wav",
]

# Cantonese (Yue) dialogue
DIALOGUE = (
    "[S1]大家好，歡迎收聽今日嘅節目，我係主持人小明。"
    "[S2]我係嘉賓小紅，好開心今日可以嚟呢度同大家分享。"
    "[S1]今日我哋要傾嘅話題係人工智能喺語音合成領域嘅最新進展。"
    "[S2]係囉，呢個領域發展好快，尤其係近幾年大模型嘅出現，令語音合成嘅質素有咗質嘅飛躍。"
    "[S1]你可唔可以同我哋介紹一下目前最先進嘅語音合成技術？"
    "[S2]當然，目前最先進嘅系統通常結合咗大型語言模型同擴散模型，能夠生成非常自然流暢嘅語音。"
)

# <|Yue|> prefix in prompt_texts activates Cantonese dialect
DATA = {
    "prompt_texts": json.dumps([
        "<|Yue|>喜歡攀岩、行山、滑雪嘅語言愛好者。",
        "<|Yue|>資深科技播客主持人。",
    ]),
    "dialogue_text": DIALOGUE,
    "chunk_size": 150,
    "seed": 42,
}

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.makedirs("outputs", exist_ok=True)

print("Generating Cantonese audio ...")
buf = io.BytesIO()
files = [("prompt_audio", open(f, "rb")) for f in AUDIO]
with requests.post(f"{API}/generate-stream", files=files, data=DATA,
                   headers=HDR, stream=True, timeout=300) as resp:
    resp.raise_for_status()
    for chunk in resp.iter_content(chunk_size=None):
        if chunk:
            buf.write(chunk)

total_bytes = buf.tell()
print(f"Received {total_bytes} bytes")

buf.seek(0)
magic = buf.read(4)
buf.seek(0)

out_path = "outputs/cantonese_quality_check.wav"

if magic == b"RIFF":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, data = wavfile.read(buf)
    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float32) / 2**31
    else:
        samples = data.astype(np.float32)
    duration = len(samples) / sr
    print(f"WAV: {sr} Hz, {len(samples)} samples, {duration:.1f}s")
    nan_count = int(np.isnan(samples).sum())
    if nan_count:
        print(f"WARNING: {nan_count} NaN samples")
    else:
        print(f"Clean audio  range: [{samples.min():.4f}, {samples.max():.4f}]")
    buf.seek(0)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(buf, f)
else:
    import wave
    print(f"No RIFF header — treating as raw float32 PCM")
    buf.seek(0)
    samples = np.frombuffer(buf.read(), dtype=np.float32)
    duration = len(samples) / 24000
    with wave.open(out_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        pcm16 = (np.clip(np.nan_to_num(samples), -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())

print(f"Saved: {out_path}  ({duration:.1f}s)")
