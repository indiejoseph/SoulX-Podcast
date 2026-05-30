"""Mandarin loudness QC: 4-turn dialogue across both prompt voices.

Pairs with gen_cantonese.py — same /generate-stream path, no normalization.
Writes outputs/mandarin_quality_check.wav so we can audibly compare per-turn
and per-speaker volume to judge whether the original loudness inconsistency
still needs a fix.
"""
import io, json, os, shutil, sys
import numpy as np, requests
import scipy.io.wavfile as wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

API = "http://localhost:8000"
KEY = os.getenv("API_KEY", "soulx-local-dev-2026-xK9mP")
HDR = {"Authorization": f"Bearer {KEY}"}

AUDIO = [
    "example/audios/female_mandarin.wav",
    "example/audios/male_mandarin.wav",
]

DIALOGUE = (
    "[S1]大家好，欢迎收听今天的播客节目，我是主持人小明。"
    "[S2]我是嘉宾小红，今天非常开心能来到这里和大家交流。"
    "[S1]今天我们要聊的话题是人工智能在语音合成领域的最新进展，你能先简单介绍一下吗？"
    "[S2]当然可以，目前最先进的语音合成系统通常会结合大型语言模型和扩散模型，能够生成非常自然流畅的人声。"
)

DATA = {
    "prompt_texts": json.dumps([
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        "我是一个资深的科技播客主持人，喜欢和大家分享前沿技术。",
    ]),
    "dialogue_text": DIALOGUE,
    "chunk_size": 150,
    "seed": 42,
}

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.makedirs("outputs", exist_ok=True)

print("Generating Mandarin audio ...")
buf = io.BytesIO()
files = [("prompt_audio", open(f, "rb")) for f in AUDIO]
with requests.post(f"{API}/generate-stream", files=files, data=DATA,
                   headers=HDR, stream=True, timeout=300) as resp:
    resp.raise_for_status()
    for chunk in resp.iter_content(chunk_size=None):
        if chunk:
            buf.write(chunk)

print(f"Received {buf.tell()} bytes")
buf.seek(0)
magic = buf.read(4)
buf.seek(0)
out_path = "outputs/mandarin_quality_check.wav"

if magic == b"RIFF":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, data = wavfile.read(buf)
    samples = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
    duration = len(samples) / sr
    print(f"WAV: {sr} Hz, {len(samples)} samples, {duration:.1f}s")
    print(f"Clean audio  range: [{samples.min():.4f}, {samples.max():.4f}]  rms={np.sqrt((samples**2).mean()):.4f}")
    buf.seek(0)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(buf, f)
    print(f"Saved: {out_path}")
