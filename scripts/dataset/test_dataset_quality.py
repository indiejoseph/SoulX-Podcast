"""Sanity check the dataset's speech_tokens by decoding them straight to audio
through the SoulX flow + vocoder — bypassing the LLM entirely.

Picks one sample per language (en/zh/yue) from the small dataset, uses
example/audios/female_mandarin.wav as the voice clone reference, concatenates
prompt speech tokens + dataset speech tokens, runs flow+hift, saves wav.

If the wavs match the printed transcription audibly, the dataset tokens are
clean. If they sound garbled, the dataset (or its extraction script) has a bug.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from pathlib import Path

import torch
import torchaudio
from datasets import load_from_disk

import s3tokenizer
import onnxruntime
from torchaudio.compliance import kaldi
from soulxpodcast.utils.audio import audio_volume_normalize, mel_spectrogram
from soulxpodcast.utils.infer_utils import initiate_model

BASE = "pretrained_models/SoulX-Podcast-1.7B-dialect"
DATASET = "tmp/dataset_small_with_tokens"
PROMPT_WAV = "example/audios/female_mandarin.wav"
OUT = Path("outputs/dataset_quality_check")
OUT.mkdir(parents=True, exist_ok=True)

# (idx, lang)
SAMPLES = {
    "en": 37,
    "zh": 0,
    "yue": 1,
}


def load_spk_model():
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(
        f"{BASE}/campplus.onnx", sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def load_prompt(model, spk_model, prompt_wav_path):
    """Build the flow conditioning inputs from a voice prompt wav."""
    # 1) s3tokenizer mels for speech token quantization
    audio_16k = s3tokenizer.load_audio(prompt_wav_path, sr=16000)
    audio_16k = audio_volume_normalize(audio_16k)
    log_mel = s3tokenizer.log_mel_spectrogram(audio_16k)             # [num_mels=128, T]
    mels_for_llm, mels_lens_for_llm = s3tokenizer.padding([log_mel]) # [1, 128, T']
    prompt_speech_tokens, prompt_speech_tokens_lens = model.audio_tokenizer.quantize(
        mels_for_llm.cuda(), mels_lens_for_llm.cuda()
    )

    # 2) speaker embedding via campplus.onnx
    spk_feat = kaldi.fbank(audio_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
    spk_emb = spk_model.run(
        None,
        {spk_model.get_inputs()[0].name: spk_feat.unsqueeze(0).cpu().numpy()},
    )[0].flatten().tolist()
    spk_emb = torch.tensor([spk_emb], dtype=torch.float32)

    # 3) 24kHz mel features for flow conditioning
    wav_24k, sr = torchaudio.load(prompt_wav_path, backend="soundfile")
    wav_24k = wav_24k[0]
    wav_24k = audio_volume_normalize(wav_24k).unsqueeze(0)
    if sr != 24000:
        wav_24k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)(wav_24k)
    flow_prompt_mel = mel_spectrogram(wav_24k).transpose(1, 2).squeeze(0)  # [T, 80]
    if flow_prompt_mel.shape[0] % 2 != 0:
        flow_prompt_mel = flow_prompt_mel[:-1]
    mel_len = flow_prompt_mel.shape[0]

    # Trim speech tokens to align with mel (matches forward_longform pattern)
    plen = int(prompt_speech_tokens_lens[0].item())
    pst = prompt_speech_tokens[0, :plen]
    if plen * 2 > mel_len:
        pst = pst[: mel_len // 2]
        mel_len_aligned = mel_len
    else:
        flow_prompt_mel = flow_prompt_mel[: plen * 2]
        mel_len_aligned = plen * 2

    return {
        "prompt_speech_tokens": pst.cuda(),                          # raw 0-6560 ids, on cuda
        "prompt_mel": flow_prompt_mel.cuda().unsqueeze(0),            # [1, T, 80]
        "prompt_mel_lens": torch.tensor([mel_len_aligned]).cuda(),
        "spk_emb": spk_emb.cuda(),
    }


@torch.inference_mode()
def synth_from_tokens(model, prompt, dataset_speech_tokens):
    """Run flow + hift on (prompt speech tokens || dataset speech tokens)."""
    prompt_st = prompt["prompt_speech_tokens"]
    ds_st = torch.tensor(dataset_speech_tokens, dtype=prompt_st.dtype, device=prompt_st.device)

    flow_input = torch.cat([prompt_st, ds_st]).unsqueeze(0)
    flow_input_len = torch.tensor([flow_input.shape[1]], device=flow_input.device)

    use_fp16 = model.config.hf_config.fp16_flow
    with torch.amp.autocast("cuda", dtype=torch.float16 if use_fp16 else torch.float32):
        generated_mels, generated_mels_lens = model.flow(
            flow_input, flow_input_len,
            prompt["prompt_mel"], prompt["prompt_mel_lens"], prompt["spk_emb"],
            streaming=False, finalize=True,
        )

    # Cut off the prompt portion of the generated mels
    cutoff = int(prompt["prompt_mel_lens"][0].item())
    end = int(generated_mels_lens[0].item())
    mel = generated_mels[:, :, cutoff:end]
    wav, _ = model.hift(speech_feat=mel)
    return wav


def main():
    print("[load] SoulX pipeline")
    sp_model, _dh = initiate_model(
        seed=42, model_path=BASE, llm_engine="hf", fp16_flow=True,
    )

    print(f"[load] speaker embedding ONNX")
    spk_model = load_spk_model()

    print(f"[prompt] preparing voice prompt: {PROMPT_WAV}")
    prompt = load_prompt(sp_model, spk_model, PROMPT_WAV)
    print(f"  prompt_speech_tokens: {prompt['prompt_speech_tokens'].shape}")
    print(f"  prompt_mel:           {prompt['prompt_mel'].shape}")

    print(f"\n[ds] loading {DATASET}")
    ds = load_from_disk(DATASET).remove_columns(["audio"])

    for lang, idx in SAMPLES.items():
        s = ds[idx]
        st_raw = [int(x) for x in s["speech_tokens"].split()]
        n = len(st_raw)
        expected_dur = n / 25.0

        print(f"\n=== {lang} sample idx={idx} ===")
        print(f"  TRANSCRIPTION: {s['text']!r}")
        print(f"  n_speech_tokens: {n}  (expected dur: {expected_dur:.2f}s)")
        print(f"  token range: [{min(st_raw)}, {max(st_raw)}]")

        wav = synth_from_tokens(sp_model, prompt, st_raw)
        wav = wav.cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        actual_dur = wav.shape[-1] / 24000
        path = OUT / f"{lang}_sample{idx}.wav"
        torchaudio.save(str(path), wav, 24000)
        print(f"  generated audio: {actual_dur:.2f}s → {path}")


if __name__ == "__main__":
    main()
