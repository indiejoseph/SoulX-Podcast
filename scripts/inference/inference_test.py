"""
Thorough inference test for SoulX-Podcast.

Runs the demo dialogue through both HF and vLLM engines, captures:
  - end-to-end wall time per turn
  - LLM token count + tokens/sec
  - generated audio (saved to outputs/)

This is the baseline characterization for the PLAN.md Phase 0.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import time
import json
import argparse
from pathlib import Path

import torch
import torchaudio


def run_engine(
    llm_engine: str,
    model_path: str,
    data: dict,
    seed: int = 198964,
    fp16_flow: bool = True,
    max_new_tokens: int | None = None,
    vllm_enforce_eager: bool = False,
    vllm_speculative_config: str = "",
):
    """Run a single dialogue through the model with the given LLM engine."""
    from soulxpodcast.utils.infer_utils import process_single_input, initiate_model
    from soulxpodcast.utils.parser import podcast_format_parser

    print(f"\n{'='*70}")
    print(f"[engine={llm_engine}]  loading model from {model_path}")
    print('='*70)

    t_load_start = time.perf_counter()
    model, dataset = initiate_model(
        seed,
        model_path,
        llm_engine,
        fp16_flow,
        enforce_eager=vllm_enforce_eager,
        vllm_speculative_config=vllm_speculative_config,
    )
    t_load = time.perf_counter() - t_load_start
    print(f"[engine={llm_engine}]  load time: {t_load:.2f}s")

    inputs = podcast_format_parser(data)
    prepared = process_single_input(
        dataset,
        inputs["text"],
        inputs["prompt_wav"],
        inputs["prompt_text"],
        inputs["use_dialect_prompt"],
        inputs["dialect_prompt_text"],
    )
    if max_new_tokens is not None:
        prepared["sampling_params"].max_tokens = max_new_tokens

    # Warmup not strictly necessary for end-to-end, but stabilizes cold-cache effects.
    print(f"[engine={llm_engine}]  starting inference ({len(inputs['text'])} turns)")
    t_infer_start = time.perf_counter()
    torch.cuda.synchronize()
    results = model.forward_longform(**prepared)
    torch.cuda.synchronize()
    t_infer = time.perf_counter() - t_infer_start
    print(f"[engine={llm_engine}]  inference wall time: {t_infer:.2f}s")

    wavs = results["generated_wavs"]
    total_audio_sec = sum(w.shape[-1] / 24000.0 for w in wavs)
    print(f"[engine={llm_engine}]  generated {len(wavs)} turn(s), total audio = {total_audio_sec:.2f}s")
    print(f"[engine={llm_engine}]  RTF (real-time factor) = {t_infer / total_audio_sec:.3f}  "
          f"(lower is faster; <1.0 = faster than realtime)")

    out_dir = Path("outputs") / f"engine_{llm_engine}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(wavs):
        wav_path = out_dir / f"turn_{i:02d}.wav"
        torchaudio.save(str(wav_path), w.cpu().unsqueeze(0) if w.dim() == 1 else w.cpu(), 24000)
        print(f"[engine={llm_engine}]  saved {wav_path}  ({w.shape[-1]/24000:.2f}s)")

    return {
        "engine": llm_engine,
        "load_time_sec": round(t_load, 3),
        "infer_time_sec": round(t_infer, 3),
        "audio_duration_sec": round(total_audio_sec, 3),
        "rtf": round(t_infer / total_audio_sec, 3),
        "num_turns": len(wavs),
        "max_new_tokens": max_new_tokens,
        "vllm_enforce_eager": vllm_enforce_eager,
        "vllm_speculative_config": vllm_speculative_config,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", choices=["hf", "vllm"], help="Engines to run. Defaults to hf vllm.")
    ap.add_argument("--engine", action="append", choices=["hf", "vllm"], help="Engine to run; can be repeated.")
    ap.add_argument("--model-path", default="pretrained_models/SoulX-Podcast-1.7B-dialect")
    ap.add_argument("--seed", type=int, default=198964)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--no-dialect-prompt", action="store_true")
    ap.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--vllm-speculative-config",
        default="",
        help="Experimental vLLM speculative_config JSON string or JSON file path, e.g. a trained P-EAGLE config.",
    )
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    model_path = args.model_path

    # Multi-turn Cantonese dialogue (demo.ipynb dialect path)
    S1_PROMPT_WAV = Path("example/audios/female_mandarin.wav")
    S2_PROMPT_WAV = Path("example/audios/male_mandarin.wav")
    data = {
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
    if args.no_dialect_prompt:
        for speaker in data["speakers"].values():
            speaker.pop("dialect_prompt", None)

    engines = args.engine or args.engines or ["hf", "vllm"]
    print(f"\nrunning engines: {engines}\n")

    results = []
    for eng in engines:
        try:
            r = run_engine(
                eng,
                model_path,
                data,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                vllm_enforce_eager=args.vllm_enforce_eager,
                vllm_speculative_config=args.vllm_speculative_config,
            )
            results.append(r)
        except Exception as e:
            import traceback
            print(f"\n[engine={eng}]  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({"engine": eng, "error": f"{type(e).__name__}: {e}"})

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(json.dumps(results, indent=2))
    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
