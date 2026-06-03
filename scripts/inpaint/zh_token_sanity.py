"""Reconstruct zh audio from the dataset's speech_tokens directly through flow + HiFT.

Skips the LLM entirely. If the resulting audio is intelligible Mandarin,
the speech tokens are valid and the issue is purely composer-level. If
the audio is garbled, the s3tokenizer is producing wrong codes for the
zh source audio (e.g., sample rate mismatch).

For comparison we also reconstruct a yue and an en row.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.utils.infer_utils import initiate_model
from scripts.inpaint.inference_audio import PROMPT_FEMALE, PROMPT_MALE

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("zh_token_sanity")


def pick_one_row_per_lang(jsonl_path: str) -> dict:
    """Get one zh, one yue, one en row from the dataset."""
    picks = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            lang = r["lang"]
            if lang in {"zh", "yue", "en"} and lang not in picks:
                # Pick a medium-length row (not too short, not too long)
                if 50 <= len(r["speech_tokens"]) <= 100:
                    picks[lang] = r
            if len(picks) == 3:
                break
    return picks


@torch.no_grad()
def synth_from_tokens(soulx_model, dataset, speech_tokens: list[int],
                     prompt_audio: str, prompt_text: str, name: str) -> dict:
    """Feed (prompt_tokens + speech_tokens) through flow + HiFT directly."""
    dataitem = {
        "key": name, "prompt_text": [prompt_text],
        "prompt_wav": [prompt_audio], "text": ["dummy"], "spk": [0],
    }
    data = dataset.process_dataitem(dataitem)
    import s3tokenizer

    # Tokenize the prompt audio's mel
    log_mel = data["log_mel"][0].unsqueeze(0)
    log_mel_len = torch.tensor([log_mel.shape[-1]], dtype=torch.int32)
    pt, pt_lens = soulx_model.audio_tokenizer.quantize(
        log_mel.cuda(), log_mel_len.cuda(),
    )
    prompt_speech = pt[0, : pt_lens[0].item()].tolist()

    prompt_mel = data["mel"][0]
    if len(prompt_speech) * 2 > prompt_mel.shape[0]:
        prompt_speech = prompt_speech[: prompt_mel.shape[0] // 2]
        pmel = prompt_mel.clone().cuda()
        pmel_len = torch.tensor([prompt_mel.shape[0]], device="cuda")
    else:
        pmel = prompt_mel[: len(prompt_speech) * 2].clone().cuda()
        pmel_len = torch.tensor([len(prompt_speech) * 2], device="cuda")
    spk_emb = torch.tensor(data["spk_emb"][0]).cuda().unsqueeze(0)

    full_tokens = prompt_speech + list(speech_tokens)
    flow_in = torch.tensor([full_tokens], dtype=torch.long, device="cuda")
    flow_len = torch.tensor([len(full_tokens)], dtype=torch.long, device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.float16):
        gen_mels, gen_mels_lens = soulx_model.flow(
            flow_in, flow_len, pmel.unsqueeze(0), pmel_len, spk_emb,
            streaming=False, finalize=True,
        )
    mel = gen_mels[:, :, pmel_len[0].item(): gen_mels_lens[0].item()]
    wav, _ = soulx_model.hift(speech_feat=mel)
    return {
        "wav": wav.cpu(),
        "dur_sec": wav.shape[-1] / 24000.0,
        "n_prompt": len(prompt_speech),
        "n_target": len(speech_tokens),
    }


def main():
    out_dir = Path("outputs/inpaint_zh_token_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading SoulX (flow + HiFT + audio_tokenizer)...")
    soulx_model, dataset = initiate_model(
        seed=42,
        model_path="/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged",
        llm_engine="hf", fp16_flow=True,
    )

    log.info("picking rows...")
    picks = pick_one_row_per_lang("tmp/dataset.jsonl")
    for lang, row in picks.items():
        log.info(f"  {lang}: text={row['text']!r}  n_tokens={len(row['speech_tokens'])}")

    # Use the Mandarin voice prompt for all reconstructions (consistent voice)
    prompt = PROMPT_FEMALE

    for lang, row in picks.items():
        log.info(f"\n=== reconstructing {lang} row id={row['id']!r} ===")
        log.info(f"  text: {row['text']!r}")
        log.info(f"  first 20 speech tokens: {row['speech_tokens'][:20]}")
        log.info(f"  last 20  speech tokens: {row['speech_tokens'][-20:]}")
        out = synth_from_tokens(
            soulx_model, dataset, row["speech_tokens"],
            prompt["audio"], prompt["text"], f"{lang}_{row['id']}",
        )
        out_path = out_dir / f"{lang}_{row['id']}_from_tokens.wav"
        torchaudio.save(str(out_path), out["wav"].float(), 24000)
        log.info(f"  → {out_path}  dur={out['dur_sec']:.2f}s")

        # Also: trim the trailing-silence tokens (4299/4218/6486) and synthesize
        # to see if the audio improves
        SILENCE_IDS = {4299, 4218, 6486}
        tokens = list(row["speech_tokens"])
        n_trimmed = 0
        while tokens and tokens[-1] in SILENCE_IDS:
            tokens.pop()
            n_trimmed += 1
        if n_trimmed > 0 and lang == "zh":
            log.info(f"  trimming {n_trimmed} trailing silence tokens → "
                     f"{len(tokens)} remaining")
            out_t = synth_from_tokens(
                soulx_model, dataset, tokens,
                prompt["audio"], prompt["text"], f"{lang}_{row['id']}_trimmed",
            )
            out_path_t = out_dir / f"{lang}_{row['id']}_trimmed.wav"
            torchaudio.save(str(out_path_t), out_t["wav"].float(), 24000)
            log.info(f"  → {out_path_t}  dur={out_t['dur_sec']:.2f}s  (after silence-trim)")


if __name__ == "__main__":
    main()
