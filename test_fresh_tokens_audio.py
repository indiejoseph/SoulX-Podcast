"""Re-extract speech tokens for 3 samples using the FIXED script logic,
then synthesize audio via flow + vocoder. If these sound right, the
extraction script is correct and the dataset just needs re-extraction.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torchaudio
import whisper
import onnxruntime
import soundfile as sf
import s3tokenizer
from datasets import load_from_disk, Audio
from torchaudio.compliance import kaldi

from soulxpodcast.utils.audio import audio_volume_normalize, mel_spectrogram
from soulxpodcast.utils.infer_utils import initiate_model

BASE = "pretrained_models/SoulX-Podcast-1.7B-dialect"
DATASET = "tmp/dataset_small_with_tokens"
PROMPT_WAV = "example/audios/female_mandarin.wav"
OUT = Path("outputs/dataset_quality_fresh")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000

SAMPLES = {"en": 37, "zh": 0, "yue": 1}


def setup_ort():
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(
        "/notebooks/projects/CosyVoice/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx",
        sess_options=opts, providers=["CPUExecutionProvider"],
    )


def extract_fixed(audio_array, src_sr, ort):
    """Mirror of tmp/extract_speech_token.py:single_job logic."""
    audio = torch.from_numpy(audio_array).unsqueeze(0).float()
    if src_sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, src_sr, TARGET_SR)
    feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    return ort.run(None, {
        ort.get_inputs()[0].name: feat.detach().cpu().numpy(),
        ort.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
    })[0].flatten().tolist()


def load_spk_model():
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(
        f"{BASE}/campplus.onnx", sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def load_prompt(model, spk_model, prompt_wav_path):
    audio_16k = s3tokenizer.load_audio(prompt_wav_path, sr=16000)
    audio_16k = audio_volume_normalize(audio_16k)
    log_mel = s3tokenizer.log_mel_spectrogram(audio_16k)
    mels_for_llm, mels_lens_for_llm = s3tokenizer.padding([log_mel])
    prompt_speech_tokens, prompt_speech_tokens_lens = model.audio_tokenizer.quantize(
        mels_for_llm.cuda(), mels_lens_for_llm.cuda()
    )
    spk_feat = kaldi.fbank(audio_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
    spk_emb = spk_model.run(
        None, {spk_model.get_inputs()[0].name: spk_feat.unsqueeze(0).cpu().numpy()},
    )[0].flatten().tolist()
    spk_emb = torch.tensor([spk_emb], dtype=torch.float32)

    wav_24k, sr = torchaudio.load(prompt_wav_path, backend="soundfile")
    wav_24k = wav_24k[0]
    wav_24k = audio_volume_normalize(wav_24k).unsqueeze(0)
    if sr != 24000:
        wav_24k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)(wav_24k)
    flow_prompt_mel = mel_spectrogram(wav_24k).transpose(1, 2).squeeze(0)
    if flow_prompt_mel.shape[0] % 2 != 0:
        flow_prompt_mel = flow_prompt_mel[:-1]
    mel_len = flow_prompt_mel.shape[0]

    plen = int(prompt_speech_tokens_lens[0].item())
    pst = prompt_speech_tokens[0, :plen]
    if plen * 2 > mel_len:
        pst = pst[: mel_len // 2]
        mel_len_aligned = mel_len
    else:
        flow_prompt_mel = flow_prompt_mel[: plen * 2]
        mel_len_aligned = plen * 2

    return {
        "prompt_speech_tokens": pst.cuda(),
        "prompt_mel": flow_prompt_mel.cuda().unsqueeze(0),
        "prompt_mel_lens": torch.tensor([mel_len_aligned]).cuda(),
        "spk_emb": spk_emb.cuda(),
    }


@torch.inference_mode()
def synth(model, prompt, speech_tokens):
    prompt_st = prompt["prompt_speech_tokens"]
    ds_st = torch.tensor(speech_tokens, dtype=prompt_st.dtype, device=prompt_st.device)
    flow_input = torch.cat([prompt_st, ds_st]).unsqueeze(0)
    flow_input_len = torch.tensor([flow_input.shape[1]], device=flow_input.device)
    use_fp16 = model.config.hf_config.fp16_flow
    with torch.amp.autocast("cuda", dtype=torch.float16 if use_fp16 else torch.float32):
        gen_mels, gen_mels_lens = model.flow(
            flow_input, flow_input_len,
            prompt["prompt_mel"], prompt["prompt_mel_lens"], prompt["spk_emb"],
            streaming=False, finalize=True,
        )
    cutoff = int(prompt["prompt_mel_lens"][0].item())
    end = int(gen_mels_lens[0].item())
    mel = gen_mels[:, :, cutoff:end]
    wav, _ = model.hift(speech_feat=mel)
    return wav


def main():
    print("[load] SoulX pipeline")
    sp_model, _ = initiate_model(seed=42, model_path=BASE, llm_engine="hf", fp16_flow=True)
    print("[load] speaker ONNX + s3tokenizer ONNX (fixed script's tokenizer)")
    spk_model = load_spk_model()
    ort = setup_ort()
    prompt = load_prompt(sp_model, spk_model, PROMPT_WAV)

    ds = load_from_disk(DATASET).cast_column("audio", Audio(decode=False))

    for lang, idx in SAMPLES.items():
        s = ds[idx]
        ab = s["audio"]
        with io.BytesIO(ab["bytes"]) as buf:
            wav, sr = sf.read(buf, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        # Fresh extraction (CORRECT script logic)
        fresh_tokens = extract_fixed(wav, sr, ort)
        n_fresh = len(fresh_tokens)
        expected = int((len(wav) / sr) * 25)

        print(f"\n=== {lang} idx={idx} ===")
        print(f"  TEXT: {s['text']!r}")
        print(f"  audio: sr={sr} Hz, dur={len(wav)/sr:.3f}s")
        print(f"  FRESH tokens (fixed script): {n_fresh}  (expected {expected})")

        out_wav = synth(sp_model, prompt, fresh_tokens).cpu()
        if out_wav.dim() == 1:
            out_wav = out_wav.unsqueeze(0)
        out_dur = out_wav.shape[-1] / 24000
        path = OUT / f"{lang}_sample{idx}_FRESH.wav"
        torchaudio.save(str(path), out_wav, 24000)
        print(f"  synth audio: {out_dur:.2f}s -> {path}")


if __name__ == "__main__":
    main()
