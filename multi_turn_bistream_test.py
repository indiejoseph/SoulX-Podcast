"""Multi-turn bi-streaming test (PLAN.md Phase 0 task D).

Exercises SoulXPodcast.forward_longform_streaming on the 4-turn Cantonese
dialect dialogue from inference_test.py. Each turn:
  - Streams LLM tokens via SpeechTokenStreamer in a worker thread
  - Synthesizes per-chunk audio with sliding-window flow+HiFT
  - Yields audio chunks tagged with turn / speaker / chunk index

Validates:
  - Speaker switching (S1 ↔ S2) with correct prompt mel / spk_emb per turn
  - KV cache continuity across turns
  - Per-turn TTFA (time from turn start to first audio chunk of that turn)
  - End-to-end concatenated audio playable
"""

import sys
import time
from pathlib import Path

import torch
import torchaudio

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser


def main():
    model_path = "pretrained_models/SoulX-Podcast-1.7B-dialect"
    chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print(f"[init] loading model (hf engine, chunk_size={chunk_size})")
    t0 = time.perf_counter()
    model, dataset = initiate_model(seed=198964, model_path=model_path,
                                     llm_engine="hf", fp16_flow=True)
    print(f"[init] load: {time.perf_counter()-t0:.1f}s")

    # Same 4-turn Cantonese dialect dialogue as inference_test.py
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
    inputs = podcast_format_parser(data)
    prepared = process_single_input(
        dataset, inputs["text"], inputs["prompt_wav"], inputs["prompt_text"],
        inputs["use_dialect_prompt"], inputs["dialect_prompt_text"],
    )

    out_dir = Path("outputs/bistream_multiturn") / f"chunk{chunk_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-turn state for TTFA measurement and per-turn wav assembly.
    turn_start_time = None
    turn_first_audio_time = None
    turn_audio_chunks = []
    turn_chunk_count = 0
    current_turn = -1

    all_audio = []
    per_turn_ttfa = []
    t_start = time.perf_counter()

    print(f"\n[bistream] starting multi-turn streaming")
    for event in model.forward_longform_streaming(chunk_size=chunk_size, **prepared):
        turn = event["turn"]
        speaker = event["speaker"]
        chunk_idx = event["chunk"]
        audio = event["audio"]
        is_first = event["is_first_in_turn"]
        is_last = event["is_last_in_turn"]
        now = time.perf_counter() - t_start

        # Track turn-level state
        if turn != current_turn:
            # Flush previous turn (if any)
            if current_turn >= 0 and turn_audio_chunks:
                turn_wav = torch.cat(turn_audio_chunks, dim=-1)
                tpath = out_dir / f"turn_{current_turn:02d}_full.wav"
                torchaudio.save(str(tpath), turn_wav, 24000)
                print(f"[bistream] turn {current_turn}: total "
                      f"{turn_wav.shape[-1]/24000:.2f}s audio "
                      f"({turn_chunk_count} chunks), saved {tpath}")
            current_turn = turn
            turn_audio_chunks = []
            turn_chunk_count = 0
            turn_start_time = now
            turn_first_audio_time = None
            print(f"\n[bistream] === TURN {turn} (speaker S{speaker+1}) ===")

        if is_first and audio.shape[-1] > 0:
            turn_first_audio_time = now
            per_turn_ttfa.append(now - turn_start_time)
            print(f"[bistream] turn {turn} chunk {chunk_idx}: "
                  f"+{audio.shape[-1]/24000:.2f}s audio, "
                  f"t={now:.2f}s  ← turn-TTFA={now - turn_start_time:.2f}s")
        else:
            marker = " (final flush)" if is_last else ""
            print(f"[bistream] turn {turn} chunk {chunk_idx}: "
                  f"+{audio.shape[-1]/24000:.2f}s audio, t={now:.2f}s{marker}")

        if audio.shape[-1] > 0:
            torchaudio.save(str(out_dir / f"turn_{turn:02d}_chunk_{chunk_idx:02d}.wav"),
                            audio, 24000)
            turn_audio_chunks.append(audio)
            all_audio.append(audio)
            turn_chunk_count += 1

    # Flush final turn
    if turn_audio_chunks:
        turn_wav = torch.cat(turn_audio_chunks, dim=-1)
        tpath = out_dir / f"turn_{current_turn:02d}_full.wav"
        torchaudio.save(str(tpath), turn_wav, 24000)
        print(f"[bistream] turn {current_turn}: total "
              f"{turn_wav.shape[-1]/24000:.2f}s audio "
              f"({turn_chunk_count} chunks), saved {tpath}")

    # Concatenated full dialogue
    full = torch.cat(all_audio, dim=-1)
    final_path = out_dir / "full_dialogue.wav"
    torchaudio.save(str(final_path), full, 24000)

    t_end = time.perf_counter() - t_start
    audio_dur = full.shape[-1] / 24000

    print(f"\n[bistream] DONE")
    print(f"  total wall time:           {t_end:.3f}s")
    print(f"  total audio:               {audio_dur:.2f}s @ 24kHz")
    print(f"  RTF (wall/audio):          {t_end/audio_dur:.3f}")
    for i, ttfa in enumerate(per_turn_ttfa):
        print(f"  turn {i} TTFA:               {ttfa:.3f}s")
    print(f"  global TTFA (turn 0):      {per_turn_ttfa[0]:.3f}s")
    print(f"  output: {final_path}")


if __name__ == "__main__":
    main()
