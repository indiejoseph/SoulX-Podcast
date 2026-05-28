"""MTP audio A/B test — generate paired wavs for ear comparison.

For each test prompt, generates two wavs with identical inputs but different
LLM decoding paths:

  - `*_baseline.wav`  : standard SoulXPodcast forward_longform (no MTP),
                        uses the production sampler (RAS + top_k + top_p).
  - `*_mtp_spec.wav`  : same prompt, but LLM generation goes through
                        mtp_speculative_sample_cached (Leviathan-Kalman
                        rejection sampling against trunk's distribution).

Listen to pairs side-by-side. The baseline and MTP path both use production
RAS. Audible differences are worth investigating as either a spec-decoder bug
or the known rejection-path approximation where we resample from p instead of
max(0, p-q) on reject.

Usage:
    python mtp_audio_ab.py <path/to/mtp_final.pt> [output_dir]
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
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


def resolve_model_path(ckpt, cli_base: str | None) -> str:
    train_base = ckpt.get("train_config", {}).get("model_path")
    if cli_base:
        if train_base and cli_base != train_base:
            print(
                f"[warn] --base {cli_base!r} differs from checkpoint train_config model_path {train_base!r}"
            )
        return cli_base
    if train_base:
        if not Path(train_base).is_dir():
            fallback = Path("runs/merged")
            if fallback.is_dir():
                print(
                    f"[warn] checkpoint model_path {train_base!r} does not exist; "
                    f"using local {str(fallback)!r}"
                )
                return str(fallback)
            print(
                f"[warn] checkpoint model_path {train_base!r} does not exist locally; "
                "pass --base to override"
            )
        return train_base
    fallback = Path("runs/merged")
    if fallback.is_dir():
        print("[warn] checkpoint has no train_config.model_path; using local 'runs/merged'")
        return str(fallback)
    print(
        f"[warn] checkpoint has no train_config.model_path; falling back to {MODEL_PATH!r}"
    )
    return MODEL_PATH


def load_mtp(ckpt, base):
    """Reconstruct SequentialMTP from a saved checkpoint."""
    train_cfg = ckpt.get("train_config", {})
    mtp_config = MtpConfig(**ckpt["mtp_config"])
    decoder_layer_cls = base.model.layers[0].__class__
    mtp = SequentialMTP(mtp_config, decoder_layer_cls, base.config)
    mtp.load_state_dict(ckpt["mtp_state"])
    mtp = mtp.to(device="cuda").eval()
    print(
        f"  loaded MTP: {train_cfg.get('num_mtp_layers', '?')} layers, "
        f"step={ckpt.get('step', '?')}, "
        f"model_path={train_cfg.get('model_path', '?')}, "
        f"kl_top_k={train_cfg.get('kl_top_k', '?')}, "
        f"kl_temp={train_cfg.get('kl_temperature', '?')}, "
        f"ce_weight={train_cfg.get('ce_weight', '?')}"
    )
    return mtp


def make_data(name, target_text, prompt_dict, dataset_handler, *, disable_dialect_prompt=False):
    """Build the input data dict that forward_longform expects."""
    prompt_payload = dict(prompt_dict)
    if disable_dialect_prompt:
        prompt_payload.pop("dialect_prompt", None)
    speakers = {
        "S1": {**prompt_payload, "prompt_audio": Path(prompt_payload["prompt_audio"])}
    }
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
    # forward_longform strips trailing EOS before appending to per_turn_speech_tokens,
    # so EOS-hit is inferred by: did we stop before max_tokens-1?
    n_tokens = len(result["generated_speech_tokens"][0])
    max_tok = (
        prepared["sampling_params"].max_tokens
        if "sampling_params" in prepared
        else 3000
    )
    eos_hit = n_tokens < max_tok - 1
    return wav, t, n_tokens, eos_hit


# --- MTP path: spec decode + flow + HiFT ------------------------------------


@torch.inference_mode()
def generate_mtp(model, mtp, prepared, sampling_params):
    """Replicate forward_longform's first turn, but route LLM through MTP."""
    from itertools import chain

    t_total_start = time.perf_counter()
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
    use_dialect_prompt = prepared.get("use_dialect_prompt", False)
    dialect_prompt_text_tokens = prepared.get("dialect_prompt_text_tokens_for_llm")
    dialect_prefix = prepared.get("dialect_prefix")

    prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = (
        model.audio_tokenizer.quantize(
            prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
        )
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

    # Build prompt_inputs exactly like forward_longform. For dialect prompts,
    # this includes the extra trunk LLM call that synthesizes the dialect prompt
    # continuation; skipping it made Cantonese A/B timings apples-to-oranges.
    prompt_inputs = []
    t_prompt_llm = 0.0
    for prompt_index in range(prompt_size):
        speech_tokens_i = [
            token + cfg_off for token in prompt_speech_tokens[prompt_index].tolist()
        ] + [eos_id]
        if (
            use_dialect_prompt
            and dialect_prompt_text_tokens is not None
            and len(dialect_prompt_text_tokens[prompt_index]) > 0
        ):
            dialect_prompt_input = (
                prompt_text_tokens[prompt_index]
                + speech_tokens_i
                + dialect_prompt_text_tokens[prompt_index]
            )
            if prompt_index > 0:
                dialect_prompt_input = dialect_prefix[0] + dialect_prompt_input
            t0 = time.perf_counter()
            dialect_output = model.llm.generate(
                dialect_prompt_input, sampling_params, past_key_values=None
            )["token_ids"]
            t_prompt_llm += time.perf_counter() - t0
            prompt_inputs.append(
                dialect_prefix[prompt_index + 1]
                + dialect_prompt_text_tokens[prompt_index]
                + dialect_output
            )
        else:
            prompt_inputs.append(prompt_text_tokens[prompt_index] + speech_tokens_i)

    inputs = list(chain.from_iterable(prompt_inputs)) + list(text_tokens[0])
    input_ids = torch.tensor([inputs], dtype=torch.long, device="cuda")

    # --- MTP spec decode (sampling-aware, with RAS to match baseline) ---
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = mtp_speculative_sample_cached(
            model.llm.model,
            mtp,
            input_ids,
            max_new_tokens=sampling_params.max_tokens,
            min_new_tokens=sampling_params.min_tokens,
            eos_token_id=eos_id,
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
            repetition_penalty=sampling_params.repetition_penalty,
            use_ras=sampling_params.use_ras,
            ras_win_size=sampling_params.win_size,
            ras_tau_r=sampling_params.tau_r,
            allow_eos_from_drafts=False,
            seed=42,
        )
    t_llm = time.perf_counter() - t0

    # Extract generated speech tokens (strip the trailing EOS).
    generated_ids = result.generated_tokens[0].tolist()
    eos_hit_mtp = bool(generated_ids and generated_ids[-1] == eos_id)
    if eos_hit_mtp:
        generated_ids = generated_ids[:-1]
    generated_speech_tokens = [t - cfg_off for t in generated_ids]
    n_tokens_mtp = len(generated_speech_tokens)

    # --- Flow + HiFT (identical to forward_longform's per-turn synth) ---
    turn_spk = spk_ids[0]
    pst_list = prompt_speech_tokens[turn_spk].tolist()
    flow_input = torch.tensor([pst_list + generated_speech_tokens])
    flow_input_len = torch.tensor([flow_input.shape[1]])
    prompt_mel = prompt_mels_for_flow[turn_spk][None]
    prompt_mel_len = prompt_mels_lens_for_flow[turn_spk]
    spk_emb = spk_emb_for_flow[turn_spk : turn_spk + 1].cuda()

    t0 = time.perf_counter()
    with torch.amp.autocast(
        "cuda",
        dtype=torch.float16 if model.config.hf_config.fp16_flow else torch.float32,
    ):
        mels, mels_lens = model.flow(
            flow_input.cuda(),
            flow_input_len.cuda(),
            prompt_mel,
            prompt_mel_len,
            spk_emb,
            streaming=False,
            finalize=True,
        )
    mel = mels[:, :, prompt_mel_len[0].item() : mels_lens[0].item()]
    wav, _ = model.hift(speech_feat=mel)
    t_synth = time.perf_counter() - t0

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    t_total = time.perf_counter() - t_total_start
    return wav.cpu(), t_total, t_prompt_llm, t_llm, t_synth, result, n_tokens_mtp, eos_hit_mtp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("output_dir", nargs="?", default="outputs/mtp_audio_ab")
    ap.add_argument(
        "--base",
        default=None,
        help="SoulXPodcast trunk path. Defaults to checkpoint train_config.model_path.",
    )
    ap.add_argument(
        "--no_warmup",
        action="store_true",
        help="Skip one untimed warmup pass before measurements.",
    )
    ap.add_argument(
        "--disable_dialect_prompt",
        action="store_true",
        help="Remove dialect_prompt from both baseline and MTP paths for fair production timing without dialect warmup.",
    )
    args = ap.parse_args()

    ckpt_path = args.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_path = resolve_model_path(ckpt, args.base)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] loading SoulXPodcast model: {model_path}")
    model, dataset_handler = initiate_model(
        seed=42,
        model_path=model_path,
        llm_engine="hf",
        fp16_flow=True,
    )

    print(f"[init] loading MTP checkpoint: {ckpt_path}")
    mtp = load_mtp(ckpt, model.llm.model)

    # Production sampling params (matches what process_single_input attaches).
    from soulxpodcast.config import SamplingParams

    sampling_params = SamplingParams()
    print(
        f"[init] sampling: temp={sampling_params.temperature}, "
        f"top_k={sampling_params.top_k}, top_p={sampling_params.top_p}, "
        f"rep_pen={sampling_params.repetition_penalty}"
    )

    summary_lines = []

    if not args.no_warmup:
        case_name, target_text, prompt_dict = TEST_CASES[0]
        print(f"[init] warmup on {case_name}")
        _, warm_prepared = make_data(
            case_name,
            target_text,
            prompt_dict,
            dataset_handler,
            disable_dialect_prompt=args.disable_dialect_prompt,
        )
        try:
            _ = generate_baseline(model, warm_prepared)
            _ = generate_mtp(model, mtp, warm_prepared, sampling_params)
        except Exception as e:
            print(f"[warn] warmup failed: {type(e).__name__}: {e}")

    for case_name, target_text, prompt_dict in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"CASE: {case_name}")
        print(f"  target: {target_text[:80]}{'...' if len(target_text) > 80 else ''}")
        print(f"{'=' * 60}")

        parsed, prepared = make_data(
            case_name,
            target_text,
            prompt_dict,
            dataset_handler,
            disable_dialect_prompt=args.disable_dialect_prompt,
        )

        # --- Baseline ---
        print("  [baseline] running forward_longform (no MTP)...")
        try:
            wav_base, t_base, n_tok_base, eos_base = generate_baseline(model, prepared)
            audio_sec_base = wav_base.shape[-1] / 24000
            base_path = out_dir / f"{case_name}_baseline.wav"
            torchaudio.save(str(base_path), wav_base, 24000)
            print(
                f"  [baseline] {t_base:.2f}s wall, "
                f"{audio_sec_base:.2f}s audio, {n_tok_base} speech tokens, "
                f"eos_hit={eos_base}, RTF={t_base/audio_sec_base:.3f}"
            )
            print(f"  [baseline] saved → {base_path}")
        except Exception as e:
            print(f"  [baseline] FAILED: {type(e).__name__}: {e}")
            continue

        # --- MTP ---
        print("  [mtp_spec] running spec decode (sampling)...")
        try:
            (
                wav_mtp,
                t_total_mtp,
                t_prompt_llm_mtp,
                t_target_llm_mtp,
                t_synth_mtp,
                spec_result,
                n_tok_mtp,
                eos_mtp,
            ) = (
                generate_mtp(
                    model,
                    mtp,
                    prepared,
                    sampling_params,
                )
            )
            audio_sec_mtp = wav_mtp.shape[-1] / 24000
            mtp_path = out_dir / f"{case_name}_mtp_spec.wav"
            torchaudio.save(str(mtp_path), wav_mtp, 24000)
            print(
                f"  [mtp_spec] prompt_llm={t_prompt_llm_mtp:.2f}s  "
                f"target_llm={t_target_llm_mtp:.2f}s ({spec_result.n_steps} steps, "
                f"mean accept={spec_result.mean_accept_length:.2f}, "
                f"tok/step={spec_result.tokens_per_step:.2f}, "
                f"tokens={n_tok_mtp}, eos={eos_mtp})  "
                f"synth={t_synth_mtp:.2f}s  "
                f"total={t_total_mtp:.2f}s  "
                f"audio={audio_sec_mtp:.2f}s  RTF={t_total_mtp/audio_sec_mtp:.3f}"
            )
            print(f"  [mtp_spec] saved → {mtp_path}")

            # Per-case A/B report
            tok_ratio = n_tok_mtp / max(n_tok_base, 1)
            audio_ratio = audio_sec_mtp / max(audio_sec_base, 0.01)
            warn = ""
            if tok_ratio < 0.8 or audio_ratio < 0.8:
                warn = " ⚠ MTP shorter — possible early EOS / truncation"
            elif tok_ratio > 1.25 or audio_ratio > 1.25:
                warn = " ⚠ MTP longer than baseline (>25%)"
            print(
                f"  [report] tok_ratio={tok_ratio:.2f}  audio_ratio={audio_ratio:.2f}{warn}"
            )

            # Speedup summary over the same scope as baseline forward_longform:
            # prompt quantization + optional dialect prompt + target decode + flow/HiFT.
            speedup = t_base / t_total_mtp
            from collections import Counter

            hist = Counter(spec_result.accept_lengths)
            n_drafts = len(mtp.layers) + 1
            bucket = " ".join(f"{i}:{hist.get(i, 0)}" for i in range(1, n_drafts + 1))
            print(f"  >>> overall speedup (LLM+synth): {speedup:.2f}x")
            print(f"      accept-len histogram (1..{n_drafts}): {bucket}")
            summary_lines.append(
                f"{case_name:>20s}  base={t_base:5.2f}s  mtp={t_total_mtp:5.2f}s  "
                f"speedup={speedup:.2f}x  mean_accept={spec_result.mean_accept_length:.2f}  "
                f"base_audio={audio_sec_base:.2f}s  mtp_audio={audio_sec_mtp:.2f}s  "
                f"tokens={n_tok_mtp}  eos={eos_mtp}"
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
    print(
        f"Listen to pairs (baseline vs mtp_spec) — they should sound essentially identical."
    )
    print(f"Audible artifacts = bug; equivalent quality = ship-ready.")


if __name__ == "__main__":
    main()
