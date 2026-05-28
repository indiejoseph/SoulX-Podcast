"""Side-by-side comparison of HF vs vLLM bf16 speech-token output.

Runs the same first-turn prompt through both engines (same seed, same
sampling params) and reports:
  - per-turn token count (length divergence)
  - per-turn diff: where the two sequences first deviate
  - whether vLLM is producing more repetitive tokens

This isolates the source of the 18% audio-length disparity we saw in the
3-way benchmark.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import time
from collections import Counter
from pathlib import Path

import torch

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser


MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect-avg"
SEED = 198964


# Single-turn Cantonese — simpler to compare than the full 4-turn dialogue.
DATA = {
    "speakers": {
        "S1": {
            "prompt_audio": Path("example/audios/female_mandarin.wav"),
            "prompt_text": "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
            "dialect_prompt": "<|Yue|>真係冇讲错啊！攀山滑雪嘅语言专家几巴闭，都唔及我听日拖成副身家去景德镇玩泥巴，呢铺真系发哂白日梦咯！",
        },
    },
    "text": [
        ["S1", "<|Yue|>哈囉大家好啊，歡迎收聽我哋嘅節目。喂，我今日想問你樣嘢啊，你覺唔覺得，嗯，而家揸電動車，最煩，最煩嘅一樣嘢係咩啊？"],
    ],
}


def run_engine(engine: str, greedy: bool = False):
    model, dataset = initiate_model(SEED, MODEL_PATH, engine, fp16_flow=True)
    inputs = podcast_format_parser(DATA)
    prepared = process_single_input(
        dataset, inputs["text"], inputs["prompt_wav"],
        inputs["prompt_text"], inputs["use_dialect_prompt"],
        inputs["dialect_prompt_text"],
    )
    if greedy:
        # Override sampling to argmax: temp=0 + top_k=1 + no RAS + no rep_penalty.
        # Eliminates the RNG path so any token divergence comes from the model
        # forward (attention impl, dtype precision, RoPE) — NOT the sampler.
        sp = prepared["sampling_params"]
        # HF transformers rejects temperature=0 (use do_sample=False instead,
        # but the engine code path uses do_sample=True). Use a tiny positive
        # value — combined with top_k=1, output is deterministic argmax.
        sp.temperature = 1e-5
        sp.top_k = 1
        sp.top_p = 1.0
        sp.repetition_penalty = 1.0
        sp.use_ras = False
        # Cap generation. Sampled runs landed at ~300 tokens; greedy without
        # RAS can loop indefinitely on speech tokens — without this the flow
        # OOMs at ~3000 tokens trying to build its attention matrix.
        sp.max_tokens = 400
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = model.forward_longform(**prepared)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0

    # generated_speech_tokens is a list of 0-based s3tokenizer ids per turn
    # (after the LLM offset has been stripped). For the diff we want those.
    tok_lists = results.get("generated_speech_tokens", None)
    if tok_lists is None:
        # Older API: pull from the LLM-side ids; strip the speech_token_offset.
        offset = model.config.hf_config.speech_token_offset
        raw = results.get("llm_output_ids", [])
        tok_lists = [[int(t) - offset for t in seq] for seq in raw]

    return {
        "engine": engine,
        "infer_sec": t,
        "tokens_per_turn": [list(map(int, ts)) for ts in tok_lists],
    }


def diff_seqs(a, b):
    """Find first divergence index and longest common prefix length."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def token_stats(name, toks):
    n = len(toks)
    if n == 0:
        return f"  {name}: EMPTY"
    c = Counter(toks)
    top3 = c.most_common(3)
    repeat_rate = 1.0 - len(c) / n   # unique / total
    return (f"  {name}: n={n}  unique={len(c)}  repeat_rate={repeat_rate:.1%}  "
            f"top3={[(t, n) for t, n in top3]}")


def main():
    import sys
    greedy = "--greedy" in sys.argv
    mode = "GREEDY (argmax)" if greedy else "SAMPLING (production params)"
    print(f"[INFO] Mode: {mode}")
    print("[INFO] Running HF bf16 ...")
    hf = run_engine("hf", greedy=greedy)
    print(f"[INFO] HF done in {hf['infer_sec']:.2f}s")

    print("\n[INFO] Running vLLM bf16 ...")
    vl = run_engine("vllm", greedy=greedy)
    print(f"[INFO] vLLM done in {vl['infer_sec']:.2f}s")

    print("\n" + "=" * 70)
    print("PER-TURN TOKEN COMPARISON")
    print("=" * 70)
    n_turns = max(len(hf["tokens_per_turn"]), len(vl["tokens_per_turn"]))
    for i in range(n_turns):
        ht = hf["tokens_per_turn"][i] if i < len(hf["tokens_per_turn"]) else []
        vt = vl["tokens_per_turn"][i] if i < len(vl["tokens_per_turn"]) else []
        print(f"\n--- Turn {i} ---")
        print(token_stats("HF  ", ht))
        print(token_stats("vLLM", vt))
        if ht and vt:
            div = diff_seqs(ht, vt)
            print(f"  diff: first divergence at position {div} "
                  f"(common prefix len = {div})")
            print(f"  ratio: len(vLLM)/len(HF) = {len(vt)/len(ht):.3f}")
            # Show a glimpse of the divergence region.
            window = 5
            lo = max(0, div - window)
            hi_h = min(len(ht), div + window)
            hi_v = min(len(vt), div + window)
            print(f"  HF   [{lo}:{hi_h}] = {ht[lo:hi_h]}")
            print(f"  vLLM [{lo}:{hi_v}] = {vt[lo:hi_v]}")


if __name__ == "__main__":
    main()
