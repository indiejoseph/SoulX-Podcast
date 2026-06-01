"""Bench English content to test whether the on-policy drafter actually works
on its training distribution (English read speech)."""
import os, sys, time, json, requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

API = "http://localhost:8000"
KEY = os.getenv("API_KEY", "soulx-local-dev-2026-xK9mP")
HDR = {"Authorization": f"Bearer {KEY}"}

AUDIO = ["example/audios/female_mandarin.wav", "example/audios/male_mandarin.wav"]

# English dialogue — matches the LibriSpeech-style training distribution roughly
DIALOGUE = (
    "[S1]Hello everyone, welcome to today's podcast about artificial intelligence."
    "[S2]Yes, today we're going to discuss recent advances in speech synthesis."
    "[S1]This is a fascinating topic with lots of new developments."
    "[S2]Indeed, the combination of large language models and diffusion has been transformative."
    "[S1]Can you tell us more about the current state-of-the-art systems?"
    "[S2]The most advanced systems combine autoregressive and flow-matching components."
)

DATA = {
    "prompt_texts": json.dumps(["A native English speaker.", "Another native English speaker."]),
    "dialogue_text": DIALOGUE,
    "seed": 1988,
}

for i in range(3):
    files = [("prompt_audio", open(f, "rb")) for f in AUDIO]
    try:
        t0 = time.perf_counter()
        ttfa = None
        total = 0
        with requests.post(f"{API}/generate-stream", files=files, data=DATA,
                           headers=HDR, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    if ttfa is None and total + len(chunk) > 44:
                        ttfa = time.perf_counter() - t0
                    total += len(chunk)
        wall = time.perf_counter() - t0
    finally:
        for _, f in files:
            f.close()
    dur = (total - 44) / (24000 * 2)
    rtf = wall / dur if dur > 0 else 0
    print(f"  run {i+1}: TTFA={ttfa:.2f}s wall={wall:.2f}s audio={dur:.1f}s RTF={rtf:.3f}")
