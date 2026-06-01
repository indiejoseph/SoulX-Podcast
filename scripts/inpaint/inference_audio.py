"""End-to-end audio synthesis with pronunciation inpaint.

Pipeline per test case:
  1. Process voice prompt → 80-dim mels (flow), 128-dim log-mels (s3tokenizer),
     CAMPPlus speaker embedding. (Uses SoulXPodcast's PodcastInferHandler.)
  2. Quantize voice prompt audio → prompt_speech_tokens. (model.audio_tokenizer)
  3. Generate continuation speech tokens via InpaintInferenceEngine
     (composer-injected Qwen3 LLM, SSML-aware).
  4. Concatenate prompt_speech_tokens + generated_speech_tokens and feed
     through model.flow → 80-dim generated mels.
  5. Run model.hift on the *non-prompt portion* of the mels → 24 kHz WAV.

We free SoulXPodcast's own LLM after extraction since InpaintInferenceEngine
ships its own — keeps GPU memory in check on the 3090.

Test cases focus on Cantonese (yue, 73% of training corpus) where the smoke
test already showed the composer producing localised, well-formed
divergence from baseline.

Outputs land at outputs/inpaint_audio/*.wav.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.inpaint.inference import InpaintInferenceEngine
from soulxpodcast.utils.infer_utils import initiate_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inference_audio")


# ------------------------------------------------------------------
# Test cases
# ------------------------------------------------------------------

# Voice prompts from the SoulX-Podcast repo demo set.
PROMPT_FEMALE = {
    "audio": "example/audios/female_mandarin.wav",
    "text": "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
}
PROMPT_MALE = {
    "audio": "example/audios/male_mandarin.wav",
    "text": "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家。",
}

# Each case generates two WAVs: baseline (no SSML) + inpaint (with SSML).
# Focus on yue — that's where the LLM smoke showed clean divergence at
# position 1 with both versions terminating cleanly.
CASES: list[dict] = [
    # ===== Cantonese (yue) — primary use case, 73% of training corpus =====
    {
        "name": "yue_keoi5_correct",
        "lang": "yue",
        "prompt": PROMPT_FEMALE,
        "plain":  "我同佢一齊去飲茶。",
        # Correct Jyutping override: should sound the SAME as baseline.
        "ssml":   '我同<phoneme alphabet="jyutping" ph="k eoi5">佢</phoneme>一齊去飲茶。',
    },
    {
        "name": "yue_keoi5_wrong_neoi5",
        "lang": "yue",
        "prompt": PROMPT_FEMALE,
        "plain":  "我同佢一齊去飲茶。",
        # WRONG initial: replaces 'keoi5' (he/she) with 'neoi5' (girl).
        "ssml":   '我同<phoneme alphabet="jyutping" ph="n eoi5">佢</phoneme>一齊去飲茶。',
    },
    {
        "name": "yue_keoi5_wrong_baai6",
        "lang": "yue",
        "prompt": PROMPT_FEMALE,
        "plain":  "我同佢一齊去飲茶。",
        # Maximally different override: 'baai6' is unrelated.
        "ssml":   '我同<phoneme alphabet="jyutping" ph="b aai6">佢</phoneme>一齊去飲茶。',
    },

    # ===== Mandarin (zh) — training format: pure CJK, NO punctuation =====
    # Target word: 银行 (yin2 hang2 = "bank"). The character 行 has two
    # readings — hang2 ("row/industry/bank") or xing2 ("walk/action") — so
    # this is a classic disambiguation case for phoneme override.
    {
        "name": "zh_hang2_correct",
        "lang": "zh",
        "prompt": PROMPT_FEMALE,
        "plain":  "我去银行办事",  # NO period — matches training
        "ssml":   '我去<phoneme alphabet="pinyin" ph="y in2 h ang2">银行</phoneme>办事',
    },
    {
        "name": "zh_hang2_wrong_xing2",
        "lang": "zh",
        "prompt": PROMPT_FEMALE,
        "plain":  "我去银行办事",
        # Wrong reading: forces 行 to xing2 — should produce different audio.
        "ssml":   '我去<phoneme alphabet="pinyin" ph="y in2 x ing2">银行</phoneme>办事',
    },

    # ===== English (en) — training format: ALL CAPS, NO punctuation =====
    # 200/200 sampled training rows are uppercase + no terminal punct.
    # The previous "Hello world today." test was an OOD format that the
    # model had never seen — that's why EOS failed, not the inpaint.
    {
        "name": "en_world_correct",
        "lang": "en",
        "prompt": PROMPT_MALE,
        "plain":  "HELLO WORLD TODAY",
        "ssml":   'HELLO <phoneme alphabet="cmu" ph="W ER L D">WORLD</phoneme> TODAY',
    },
    {
        "name": "en_world_wrong",
        "lang": "en",
        "prompt": PROMPT_MALE,
        "plain":  "HELLO WORLD TODAY",
        # Maximally different override.
        "ssml":   'HELLO <phoneme alphabet="cmu" ph="B AA T">WORLD</phoneme> TODAY',
    },
]


def trim_trailing_silence(
    wav: torch.Tensor,
    sample_rate: int = 24000,
    silence_db: float = -40.0,
    min_keep_sec: float = 0.1,
    pad_tail_sec: float = 0.1,
) -> torch.Tensor:
    """Crop trailing low-energy samples from a mono/stereo waveform.

    Used because the LLM can fail to emit ``<|semantic_token_end|>`` and
    keeps generating silence-like speech tokens until the max_new_tokens
    cap — the audio is fine, just has a long quiet tail.

    Strategy: find the last sample whose absolute amplitude exceeds the
    threshold (silence_db dBFS), then keep up to ``pad_tail_sec`` after
    that. ``min_keep_sec`` is a floor so we don't accidentally over-trim
    a near-silent intentional ending.
    """
    if wav.numel() == 0:
        return wav
    abs_wav = wav.abs()
    if abs_wav.dim() == 2:  # (C, T)
        env = abs_wav.amax(dim=0)
    else:
        env = abs_wav
    threshold = 10.0 ** (silence_db / 20.0)
    above = (env > threshold).nonzero(as_tuple=False)
    if above.numel() == 0:
        return wav[..., : int(min_keep_sec * sample_rate)]
    last_loud = int(above[-1].item())
    end = min(env.numel(), last_loud + int(pad_tail_sec * sample_rate))
    end = max(end, int(min_keep_sec * sample_rate))
    return wav[..., :end]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        default="/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged",
    )
    p.add_argument(
        "--composer_ckpt",
        default="outputs/step_0030000/composer.pt",
    )
    p.add_argument("--output_dir", default="outputs/inpaint_audio")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--silence_db", type=float, default=-40.0,
                   help="dBFS below which we treat samples as silence for trimming")
    p.add_argument("--no_trim", action="store_true",
                   help="Skip trailing-silence trim (keep raw HiFT output)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_sample", action="store_true",
                   help="Greedy decoding for deterministic A/B.")
    p.add_argument("--fp16_flow", action="store_true",
                   help="Run the flow in fp16 (faster on 3090).")
    return p.parse_args()


@torch.no_grad()
def synthesize_one(
    soulx_model,
    dataset,
    inpaint_engine: InpaintInferenceEngine,
    prompt_audio: str,
    prompt_text: str,
    ssml_or_text: str,
    lang: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    fp16_flow: bool,
) -> tuple[torch.Tensor, dict]:
    """Run the full pipeline on a single (prompt, text) pair. Returns (wav, debug)."""
    # ---- 1. Process voice prompt features (re-uses SoulX's dataloader) ----
    # We feed a dummy text just so process_data doesn't crash; the LLM step
    # gets bypassed in favour of our inpaint engine, so the actual text
    # tokens here are unused.
    dataitem = {
        "key": "inpaint_test",
        "prompt_text": [prompt_text],
        "prompt_wav": [prompt_audio],
        "text": ["dummy"],
        "spk": [0],
    }
    data = dataset.process_dataitem(dataitem)
    assert data is not None, "voice prompt processing failed"

    # ---- 2. Quantize prompt audio → s3tokenizer speech tokens ----
    log_mel = data["log_mel"][0].unsqueeze(0)  # (1, 128, T_audio)
    log_mel_len = torch.tensor([log_mel.shape[-1]], dtype=torch.int32)
    prompt_speech_tokens, prompt_speech_tokens_lens = soulx_model.audio_tokenizer.quantize(
        log_mel.cuda(), log_mel_len.cuda()
    )
    prompt_speech_token_list = prompt_speech_tokens[0, : prompt_speech_tokens_lens[0].item()].tolist()

    # Align prompt mels with prompt speech token count, same heuristic as
    # SoulXPodcast.forward_longform (each token = 2 mel frames at 50 Hz vs 25 Hz).
    prompt_mel = data["mel"][0]  # (T_mel, 80)
    prompt_mel_len = prompt_mel.shape[0]
    if len(prompt_speech_token_list) * 2 > prompt_mel_len:
        prompt_speech_token_list = prompt_speech_token_list[: prompt_mel_len // 2]
        prompt_mel_cu = prompt_mel.clone().cuda()
        prompt_mel_len_cu = torch.tensor([prompt_mel_len], device="cuda")
    else:
        prompt_mel_cu = prompt_mel[: len(prompt_speech_token_list) * 2].clone().cuda()
        prompt_mel_len_cu = torch.tensor(
            [len(prompt_speech_token_list) * 2], device="cuda"
        )
    spk_emb = torch.tensor(data["spk_emb"][0]).cuda().unsqueeze(0)

    # ---- 3. Generate continuation speech tokens via inpaint LLM ----
    t_llm = time.perf_counter()
    gen = inpaint_engine.generate_speech_tokens(
        ssml_or_text=ssml_or_text,
        lang=lang,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        disable_inpaint=("<phoneme" not in ssml_or_text),
    )
    dt_llm = time.perf_counter() - t_llm
    if not gen.eos_hit:
        log.warning(
            f"LLM didn't emit EOS within {max_new_tokens} tokens — audio may be truncated"
        )

    # ---- 4. Flow on (prompt + generated) ----
    full_tokens = prompt_speech_token_list + gen.speech_tokens
    flow_input = torch.tensor([full_tokens], dtype=torch.long, device="cuda")
    flow_input_lens = torch.tensor([len(full_tokens)], dtype=torch.long, device="cuda")
    t_flow = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.float16 if fp16_flow else torch.float32):
        generated_mels, generated_mels_lens = soulx_model.flow(
            flow_input,
            flow_input_lens,
            prompt_mel_cu.unsqueeze(0).cuda(),
            prompt_mel_len_cu,
            spk_emb,
            streaming=False,
            finalize=True,
        )
    dt_flow = time.perf_counter() - t_flow

    # ---- 5. HiFT on the non-prompt portion of the mels ----
    mel = generated_mels[:, :, prompt_mel_len_cu[0].item(): generated_mels_lens[0].item()]
    t_hift = time.perf_counter()
    wav, _ = soulx_model.hift(speech_feat=mel)
    dt_hift = time.perf_counter() - t_hift

    return wav.cpu(), {
        "n_prompt_tokens": len(prompt_speech_token_list),
        "n_generated_tokens": len(gen.speech_tokens),
        "eos_hit": gen.eos_hit,
        "n_phonemes_aligned": gen.n_phonemes_aligned,
        "phone_positions": sum(gen.phone_mask),
        "dt_llm": dt_llm,
        "dt_flow": dt_flow,
        "dt_hift": dt_hift,
        "wav_samples": wav.shape[-1],
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load SoulXPodcast for flow + HiFT + dataset ----
    log.info(f"loading SoulXPodcast model from {args.model_path}")
    t0 = time.perf_counter()
    soulx_model, dataset = initiate_model(
        seed=args.seed,
        model_path=args.model_path,
        llm_engine="hf",
        fp16_flow=args.fp16_flow,
    )
    log.info(f"SoulXPodcast loaded in {time.perf_counter() - t0:.1f}s")

    # Free SoulX's own LLM — InpaintInferenceEngine ships its own.
    try:
        del soulx_model.llm
        torch.cuda.empty_cache()
        log.info("freed SoulXPodcast.llm (using InpaintInferenceEngine LLM instead)")
    except AttributeError:
        pass

    # ---- Load Inpaint engine ----
    log.info(f"loading InpaintInferenceEngine from {args.composer_ckpt}")
    inpaint_engine = InpaintInferenceEngine(
        model_path=args.model_path,
        composer_ckpt_path=args.composer_ckpt,
    )

    # ---- Run each case (baseline + inpaint) ----
    summary: list[dict] = []
    for case in CASES:
        for variant, text in [("baseline", case["plain"]), ("inpaint", case["ssml"])]:
            name = f"{case['name']}_{variant}"
            log.info(f"--- {name} ---")
            log.info(f"  text: {text!r}")

            wav, debug = synthesize_one(
                soulx_model,
                dataset,
                inpaint_engine,
                prompt_audio=case["prompt"]["audio"],
                prompt_text=case["prompt"]["text"],
                ssml_or_text=text,
                lang=case["lang"],
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.no_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
                fp16_flow=args.fp16_flow,
            )
            raw_dur = debug["wav_samples"] / 24000.0
            if not args.no_trim:
                wav_2d = wav.squeeze(0) if wav.dim() == 3 else wav  # → (C, T)
                wav_2d = trim_trailing_silence(
                    wav_2d, sample_rate=24000, silence_db=args.silence_db
                )
                wav = wav_2d
            dur_sec = wav.shape[-1] / 24000.0
            debug["trimmed_dur_sec"] = dur_sec
            debug["raw_dur_sec"] = raw_dur
            out_path = out_dir / f"{name}.wav"
            torchaudio.save(str(out_path), wav.float(), 24000)
            log.info(
                f"  ✓ {out_path}  dur={dur_sec:.2f}s  "
                f"toks(p={debug['n_prompt_tokens']} g={debug['n_generated_tokens']})  "
                f"eos={debug['eos_hit']}  phone_pos={debug['phone_positions']}  "
                f"llm={debug['dt_llm']:.2f}s flow={debug['dt_flow']:.2f}s hift={debug['dt_hift']:.2f}s"
            )
            summary.append({"name": name, "case": case["name"], "variant": variant, **debug, "path": str(out_path)})

    log.info("=" * 64)
    log.info("Summary:")
    for s in summary:
        log.info(
            f"  {s['name']:42s}  gen={s['n_generated_tokens']:4d}  "
            f"eos={s['eos_hit']}  audio={s['wav_samples']/24000:.2f}s"
        )


if __name__ == "__main__":
    main()
