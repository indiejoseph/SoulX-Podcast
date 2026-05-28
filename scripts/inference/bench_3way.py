"""3-way inference benchmark: HF bf16 vs vLLM bf16 vs vLLM AWQ-INT4.

Runs the same 4-turn Cantonese dialogue through three configurations and
reports per-engine RTF + wall time. Saves wavs to outputs/bench_3way/<mode>/
for blind audio A/B.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import time
from pathlib import Path

import torch
import torchaudio


BASE_BF16 = "pretrained_models/SoulX-Podcast-1.7B-dialect-avg"
BASE_AWQ  = "pretrained_models/SoulX-Podcast-1.7B-dialect-avg-awq"

# Reused 4-turn Cantonese dialogue from inference_test.py.
S1_PROMPT_WAV = Path("example/audios/female_mandarin.wav")
S2_PROMPT_WAV = Path("example/audios/male_mandarin.wav")
DATA = {
    "speakers": {
        "S1": {
            "prompt_audio": S1_PROMPT_WAV,
            "prompt_text": "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
            "dialect_prompt": "<|Yue|>真係冇讲错啊！攀山滑雪嘅语言专家几巴闭，都唔及我听日拖成副身家去景德镇玩泥巴，呢铺真系发哂白日梦咯！",
        },
        "S2": {
            "prompt_audio": S2_PROMPT_WAV,
            "prompt_text": "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
            "dialect_prompt": "<|Yue|>咪搞错啊！陕北民谣响度唱咗几十年，黑神话边有咁大面啊？你估佢哋抄游戏咩！",
        },
    },
    "text": [
        ["S1", "<|Yue|>哈囉大家好啊，歡迎收聽我哋嘅節目。喂，我今日想問你樣嘢啊，你覺唔覺得，嗯，而家揸電動車，最煩，最煩嘅一樣嘢係咩啊？"],
        ["S2", "<|Yue|>梗係充電啦。大佬啊，搵個位都已經好煩，搵到個位仲要喺度等，你話快極都要半個鐘一個鐘，真係，有時諗起都覺得好冇癮。"],
        ["S1", "<|Yue|>係咪先。如果我而家同你講，充電可以快到同入油差唔多時間，你信唔信先？喂你平時喺油站入滿一缸油，要幾耐啊？五六分鐘？"],
        ["S2", "<|Yue|>差唔多啦，七八分鐘，點都走得啦。電車喎，可以做到咁快？你咪玩啦。"],
    ],
}

SEED = 198964


def run_one(mode: str, engine: str, model_path: str, out_dir: Path):
    from soulxpodcast.utils.infer_utils import process_single_input, initiate_model
    from soulxpodcast.utils.parser import podcast_format_parser

    print(f"\n{'='*70}")
    print(f"[mode={mode}]  engine={engine}  model={model_path}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    # AWQ uses fp16 flow (matching the fp16 LLM dtype is harmless; the flow is
    # already small). Keep the same for bf16 modes for consistency.
    model, dataset = initiate_model(SEED, model_path, engine, fp16_flow=True)
    t_load = time.perf_counter() - t0
    print(f"[mode={mode}]  load: {t_load:.2f}s")

    inputs = podcast_format_parser(DATA)
    prepared = process_single_input(
        dataset, inputs["text"], inputs["prompt_wav"],
        inputs["prompt_text"], inputs["use_dialect_prompt"],
        inputs["dialect_prompt_text"],
    )

    # Warmup turn dropped — for a multi-turn dialogue the per-turn dynamics
    # already amortize cold-start effects, and we want load-time visible in
    # the cold-start RTF (relevant for real deployment cost).
    print(f"[mode={mode}]  starting inference ({len(inputs['text'])} turns)")
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    results = model.forward_longform(**prepared)
    torch.cuda.synchronize()
    t_infer = time.perf_counter() - t0

    wavs = results["generated_wavs"]
    audio_sec = sum(w.shape[-1] / 24000.0 for w in wavs)
    rtf = t_infer / audio_sec
    print(f"[mode={mode}]  infer wall: {t_infer:.2f}s  audio: {audio_sec:.2f}s  RTF: {rtf:.3f}")

    out_sub = out_dir / mode
    out_sub.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(wavs):
        w_cpu = w.cpu()
        torchaudio.save(
            str(out_sub / f"turn_{i:02d}.wav"),
            w_cpu.unsqueeze(0) if w_cpu.dim() == 1 else w_cpu,
            24000,
        )
    print(f"[mode={mode}]  saved {len(wavs)} wavs -> {out_sub}/")

    return {
        "mode": mode, "engine": engine, "model_path": model_path,
        "load_sec": round(t_load, 3),
        "infer_sec": round(t_infer, 3),
        "audio_sec": round(audio_sec, 3),
        "rtf": round(rtf, 3),
        "turns": len(wavs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--modes", nargs="+",
        default=["hf_bf16", "vllm_bf16", "vllm_awq"],
        choices=["hf_bf16", "vllm_bf16", "vllm_awq"],
    )
    args = ap.parse_args()

    out_dir = Path("outputs/bench_3way")
    out_dir.mkdir(parents=True, exist_ok=True)

    plans = {
        "hf_bf16":   ("hf",   BASE_BF16),
        "vllm_bf16": ("vllm", BASE_BF16),
        "vllm_awq":  ("vllm", BASE_AWQ),
    }

    results = []
    for mode in args.modes:
        engine, model_path = plans[mode]
        try:
            r = run_one(mode, engine, model_path, out_dir)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"\n[mode={mode}]  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({"mode": mode, "error": f"{type(e).__name__}: {e}"})

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'mode':<12} {'load(s)':>8} {'infer(s)':>9} {'audio(s)':>9} {'RTF':>6}")
    for r in results:
        if "error" in r:
            print(f"{r['mode']:<12}  ERROR: {r['error']}")
        else:
            print(f"{r['mode']:<12} {r['load_sec']:>8.2f} {r['infer_sec']:>9.2f} "
                  f"{r['audio_sec']:>9.2f} {r['rtf']:>6.3f}")
    print(f"\nWavs: {out_dir}/<mode>/turn_*.wav  (A/B listen for quality)")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
