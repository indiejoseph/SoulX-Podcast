"""Bench Cantonese content — the drafter's training-distribution majority (72.8%)."""
import os, sys, time, json, requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

API = "http://localhost:8000"
KEY = os.getenv("API_KEY", "soulx-local-dev-2026-xK9mP")
HDR = {"Authorization": f"Bearer {KEY}"}

with open("example/podcast_script/script_yue.json") as f:
    spec = json.load(f)

AUDIO = [spec["speakers"]["S1"]["prompt_audio"], spec["speakers"]["S2"]["prompt_audio"]]
prompt_texts = [spec["speakers"]["S1"]["prompt_text"], spec["speakers"]["S2"]["prompt_text"]]
dialect_prompt_texts = [spec["speakers"]["S1"]["dialect_prompt"], spec["speakers"]["S2"]["dialect_prompt"]]

# Build dialogue text from script
dialogue = "".join(f"[{spk}]{text}" for spk, text in spec["text"])
print(f"Dialogue length: {len(dialogue)} chars, {len(spec['text'])} turns")

DATA = {
    "prompt_texts": json.dumps(prompt_texts),
    "dialect_prompt_texts": json.dumps(dialect_prompt_texts),
    "dialogue_text": dialogue,
    "seed": 1988,
}

for i in range(3):
    files = [("prompt_audio", open(f, "rb")) for f in AUDIO]
    t0 = time.perf_counter()
    ttfa = None
    total = 0
    with requests.post(f"{API}/generate-stream", files=files, data=DATA,
                       headers=HDR, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                if ttfa is None and total >= 44:
                    ttfa = time.perf_counter() - t0
                total += len(chunk)
    wall = time.perf_counter() - t0
    for _, f in files:
        f.close()
    dur = (total - 44) / (24000 * 2)
    rtf = wall / dur if dur > 0 else 0
    print(f"  run {i+1}: TTFA={ttfa:.2f}s wall={wall:.2f}s audio={dur:.1f}s RTF={rtf:.3f}")
