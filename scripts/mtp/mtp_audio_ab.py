"""MTP audio A/B test — generate paired wavs for ear comparison.

For each test prompt, generates two wavs with identical inputs but different
LLM decoding paths:

  - `*_baseline.wav`  : standard SoulXPodcast forward_longform (no MTP),
                        uses the production sampler (RAS + top_k + top_p).
  - `*_mtp_spec.wav`  : same prompt, but LLM generation goes through
                        mtp_speculative_sample_cached (Leviathan-Kalman
                        rejection sampling against trunk's distribution).

Listen to pairs side-by-side. The output distributions are *provably*
equivalent under proper speculative sampling, so any audible difference
indicates either (a) a bug in the spec decoder, (b) the approximation in our
residual-resample (we sample from p instead of max(0, p-q) on reject — see
mtp_inference.py notes), or (c) a difference between the production sampler
(RAS) and the spec sampler (no RAS). (a) and (b) are the things to watch
for; (c) is expected.

Usage:
    python mtp_audio_ab.py <path/to/mtp_final.pt> [output_dir]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torchaudio

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.training.mtp_inference import mtp_speculative_sample_cached
from soulxpodcast.training.mtp_module import MtpConfig, SequentialMTP


MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect"

# --- Test prompts (same voice clones as demo.ipynb) -------------------------

S1_PROMPT_WAV = "example/audios/female_mandarin.wav"
S2_PROMPT_WAV = "example/audios/male_mandarin.wav"
S1_PROMPT_TEXT = "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。"
S1_DIALECT_PROMPT = "<|Yue|>真係冇讲错啊！攀山滑雪嘅语言专家几巴闭，都唔及我听日拖成副身家去景德镇玩泥巴，呢铺真系发哂白日梦咯！"

TEST_CASES = [
    # (name, target_text, voice_clone_prompt_dict)
    (
        "english_short",
        "Hello everyone, welcome to our show.",
        {"prompt_audio": S1_PROMPT_WAV, "prompt_text": S1_PROMPT_TEXT},
    ),
    (
        "cantonese_short",
        "<|Yue|>哈囉大家好啊，歡迎收聽我哋嘅節目。",
        {
            "prompt_audio": S1_PROMPT_WAV,
            "prompt_text": S1_PROMPT_TEXT,
            "dialect_prompt": S1_DIALECT_PROMPT,
        },
    ),
    (
        "mandarin_medium",
        "今天天气真好，我们一起出去走走吧。听说附近新开了一家咖啡店，环境很不错。",
        {"prompt_audio": S1_PROMPT_WAV, "prompt_text": S1_PROMPT_TEXT},
    ),
]


def load_mtp(ckpt_path: str, base):
    """Reconstruct SequentialMTP from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mtp_config = MtpConfig(**ckpt["mtp_config"])
    decoder_layer_cls = base.model.layers[0].__class__
    mtp = SequentialMTP(mtp_config, decoder_layer_cls, base.config)
    mtp.load_state_dict(ckpt["mtp_state"])
    mtp = mtp.to(device="cuda", dtype=torch.bfloat16).eval()
    print(f"  loaded MTP: {ckpt['train_config'].get('num_mtp_layers', '?')} layers, "
          f"step={ckpt['step']}, "
          f"kl_top_k={ckpt['train_config'].get('kl_top_k', '?')}, "
          f"kl_temp={ckpt['train_config'].get('kl_temperature', '?')}, "
          f"ce_weight={ckpt['train_config'].get('ce_weight', '?')}")
    return mtp


def make_data(name, target_text, prompt_dict, dataset_handler):
    """Build the input data dict that forward_longform expects."""
    speakers = {"S1": {**prompt_dict, "prompt_audio": Path(prompt_dict["prompt_audio"])}}
    data = {
        "speakers": speakers,
        "text": [["S1", target_text]],
    }
    parsed = podcast_format_parser(data)
    prepared = process_single_input(
        dataset_handler,
        parsed["text"],
        parsed["prompt_wav"],
        parsed["prompt_text"],
        parsed["use_dialect_prompt"],
        parsed["dialect_prompt_text"],
    )
    return parsed, prepared


# --- Baseline path: standard forward_longform -------------------------------

@torch.inference_mode()
def generate_baseline(model, prepared):
    t0 = time.perf_counter()
    result = model.forward_longform(**prepared)
    t = time.perf_counter() - t0
    wav = result["generated_wavs"][0].cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav, t


# --- MTP path: spec decode + flow + HiFT ------------------------------------

@torch.inference_mode()
def generate_mtp(model, mtp, prepared, sampling_params):
    """Replicate forward_longform's first turn, but route LLM through MTP."""
    from itertools import chain

    cfg_off = model.config.hf_config.speech_token_offset
    eos_id = model.config.hf_config.eos_token_id

    # Same setup as forward_longform: quantize prompt audio.
    prompt_mels_for_llm = prepared["prompt_mels_for_llm"]
    prompt_mels_lens_for_llm = prepared["prompt_mels_lens_for_llm"]
    prompt_text_tokens = prepared["prompt_text_tokens_for_llm"]
    text_tokens = prepared["text_tokens_for_llm"]
    spk_ids = prepared["spk_ids"]
    prompt_mels_for_flow_ori = prepared["prompt_mels_for_flow_ori"]
    spk_emb_for_flow = prepared["spk_emb_for_flow"]

    prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = model.audio_tokenizer.quantize(
        prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
    )

    # Align speech tokens with mel (matches forward_longform exactly).
    prompt_speech_tokens, prompt_mels_for_flow, prompt_mels_lens_for_flow = [], [], []
    prompt_size = len(prompt_mels_for_llm)
    for prompt_index in range(prompt_size):
        plen = prompt_speech_tokens_lens_ori[prompt_index].item()
        pst = prompt_speech_tokens_ori[prompt_index, :plen]
        pmel = prompt_mels_for_flow_ori[prompt_index]
        pmel_len = pmel.shape[0]
        if plen * 2 > pmel_len:
            pst = pst[: int(pmel_len / 2)]
            pmel_len_t = torch.tensor([pmel_len]).cuda()
        else:
            pmel = pmel.detach().clone()[: plen * 2].cuda()
            pmel_len_t = torch.tensor([plen * 2]).cuda()
        prompt_speech_tokens.append(pst)
        prompt_mels_for_flow.append(pmel)
        prompt_mels_lens_for_flow.append(pmel_len_t)

    # Build LLM prompt for speaker 0 (single turn).
    speech_tokens_0 = [t + cfg_off for t in prompt_speech_tokens[0].tolist()] + [eos_id]
    prompt_input = prompt_text_tokens[0] + speech_tokens_0
    inputs = list(prompt_input) + list(text_tokens[0])
    input_ids = torch.tensor([inputs], dtype=torch.long, device="cuda")

    # --- MTP spec decode (sampling-aware, with RAS to match baseline) ---
    t0 = time.perf_counter()
    result = mtp_speculative_sample_cached(
        model.llm.model, mtp, input_ids,
        max_new_tokens=sampling_params.max_tokens,
        eos_token_id=eos_id,
        temperature=sampling_params.temperature,
        top_k=sampling_params.top_k,
        top_p=sampling_params.top_p,
        repetition_penalty=sampling_params.repetition_penalty,
        use_ras=sampling_params.use_ras,
        ras_win_size=sampling_params.win_size,
        ras_tau_r=sampling_params.tau_r,
        seed=42,
    )
    t_llm = time.perf_counter() - t0

    # Extract generated speech tokens (strip the trailing EOS).
    generated_ids = result.generated_tokens[0].tolist()
    if generated_ids and generated_ids[-1] == eos_id:
        generated_ids = generated_ids[:-1]
    generated_speech_tokens = [t - cfg_off for t in generated_ids]

    # --- Flow + HiFT (identical to forward_longform's per-turn synth) ---
    turn_spk = spk_ids[0]
    pst_list = prompt_speech_tokens[turn_spk].tolist()
    flow_input = torch.tensor([pst_list + generated_speech_tokens])
    flow_input_len = torch.tensor([flow_input.shape[1]])
    prompt_mel = prompt_mels_for_flow[turn_spk][None]
    prompt_mel_len = prompt_mels_lens_for_flow[turn_spk]
    spk_emb = spk_emb_for_flow[turn_spk: turn_spk + 1].cuda()

    t0 = time.perf_counter()
    with torch.amp.autocast(
        "cuda",
        dtype=torch.float16 if model.config.hf_config.fp16_flow else torch.float32,
    ):
        mels, mels_lens = model.flow(
            flow_input.cuda(), flow_input_len.cuda(),
            prompt_mel, prompt_mel_len, spk_emb,
            streaming=False, finalize=True,
        )
    mel = mels[:, :, prompt_mel_len[0].item(): mels_lens[0].item()]
    wav, _ = model.hift(speech_feat=mel)
    t_synth = time.perf_counter() - t0

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    return wav.cpu(), t_llm, t_synth, result


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <mtp_checkpoint.pt> [output_dir]")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/mtp_audio_ab")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] loading SoulXPodcast model")
    model, dataset_handler = initiate_model(
        seed=42, model_path=MODEL_PATH, llm_engine="hf", fp16_flow=True,
    )

    print(f"[init] loading MTP checkpoint: {ckpt_path}")
    mtp = load_mtp(ckpt_path, model.llm.model)

    # Production sampling params (matches what process_single_input attaches).
    from soulxpodcast.config import SamplingParams
    sampling_params = SamplingParams()
    print(f"[init] sampling: temp={sampling_params.temperature}, "
          f"top_k={sampling_params.top_k}, top_p={sampling_params.top_p}, "
          f"rep_pen={sampling_params.repetition_penalty}")

    summary_lines = []

    for case_name, target_text, prompt_dict in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"CASE: {case_name}")
        print(f"  target: {target_text[:80]}{'...' if len(target_text) > 80 else ''}")
        print(f"{'=' * 60}")

        parsed, prepared = make_data(case_name, target_text, prompt_dict, dataset_handler)

        # --- Baseline ---
        print("  [baseline] running forward_longform (no MTP)...")
        try:
            wav_base, t_base = generate_baseline(model, prepared)
            audio_sec_base = wav_base.shape[-1] / 24000
            base_path = out_dir / f"{case_name}_baseline.wav"
            torchaudio.save(str(base_path), wav_base, 24000)
            print(f"  [baseline] {t_base:.2f}s wall, "
                  f"{audio_sec_base:.2f}s audio, RTF={t_base/audio_sec_base:.3f}")
            print(f"  [baseline] saved → {base_path}")
        except Exception as e:
            print(f"  [baseline] FAILED: {type(e).__name__}: {e}")
            continue

        # --- MTP ---
        print("  [mtp_spec] running spec decode (sampling)...")
        try:
            wav_mtp, t_llm_mtp, t_synth_mtp, spec_result = generate_mtp(
                model, mtp, prepared, sampling_params,
            )
            audio_sec_mtp = wav_mtp.shape[-1] / 24000
            t_total_mtp = t_llm_mtp + t_synth_mtp
            mtp_path = out_dir / f"{case_name}_mtp_spec.wav"
            torchaudio.save(str(mtp_path), wav_mtp, 24000)
            print(f"  [mtp_spec] LLM={t_llm_mtp:.2f}s ({spec_result.n_steps} steps, "
                  f"mean accept={spec_result.mean_accept_length:.2f}, "
                  f"tok/step={spec_result.tokens_per_step:.2f})  "
                  f"synth={t_synth_mtp:.2f}s  "
                  f"total={t_total_mtp:.2f}s  "
                  f"RTF={t_total_mtp/audio_sec_mtp:.3f}")
            print(f"  [mtp_spec] saved → {mtp_path}")

            # Speedup summary (LLM stage only — flow+HiFT is identical).
            speedup = t_base / t_total_mtp
            from collections import Counter
            hist = Counter(spec_result.accept_lengths)
            n_drafts = len(mtp.layers) + 1
            bucket = " ".join(f"{i}:{hist.get(i, 0)}" for i in range(1, n_drafts + 1))
            print(f"  >>> overall speedup (LLM+synth): {speedup:.2f}x")
            print(f"      accept-len histogram (1..{n_drafts}): {bucket}")
            summary_lines.append(
                f"{case_name:>20s}  base={t_base:5.2f}s  mtp={t_total_mtp:5.2f}s  "
                f"speedup={speedup:.2f}x  mean_accept={spec_result.mean_accept_length:.2f}"
            )
        except Exception as e:
            import traceback
            print(f"  [mtp_spec] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    for line in summary_lines:
        print(line)
    print(f"\nWavs saved to {out_dir}/")
    print(f"Listen to pairs (baseline vs mtp_spec) — they should sound essentially identical.")
    print(f"Audible artifacts = bug; equivalent quality = ship-ready.")


if __name__ == "__main__":
    main()
