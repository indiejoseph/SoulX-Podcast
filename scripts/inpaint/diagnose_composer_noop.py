"""Diagnose composer collapse — is the composer a no-op on a phoneme override?

The behavioral gate showed corr↔wrong edit distance = 0.0 on
``yue_dei6_byte_fallback`` and ``zh_yinhang_merged`` — the LLM emits
*identical* speech tokens whether given the correct or a deliberately wrong
phoneme override. Loss looked great; behaviour says no steering. This script
classifies the failure into one of three modes by inspecting the composed
phoneme embedding the composer injects at the pad position(s), BEFORE it
reaches the frozen trunk:

  (a) NEAR-ZERO   — ‖composed‖ ≪ the composer's typical injection norm.
                    The head produces no signal for this combo.
  (b) COLLAPSED   — cos(composed_correct, composed_wrong) ≈ 1.0 at the pad.
                    Different phonemes map to (nearly) the same vector, so the
                    trunk can't tell them apart. Loss-no-op failure.
  (c) TRUNK-IGNORED — composed_correct and composed_wrong are distinct and
                    large, yet the LLM output is identical. The injection is
                    fine but the frozen trunk doesn't act on it at that position
                    (hardest case — architecture, not training/loss).

Contrast fixtures that PASS M1 (zhongguo 0.583, keoi5 0.333) are run too, so
we can read the failing combos against a working baseline on the same axes.

Usage:
    python scripts/inpaint/diagnose_composer_noop.py \
        --composer_ckpt outputs/inpaint_final/step_0030000/composer.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.inpaint.composer import ALPHABET_STR_TO_ID, apply_phoneme_inpaint
from soulxpodcast.inpaint.inference import InpaintInferenceEngine
from soulxpodcast.inpaint.ssml import parse_ssml
from soulxpodcast.training.inpaint_dataset import DIALECT_PREFIX, LANG_TO_ALPHABET

from fixtures_adversarial import (
    YUE_KEOI5_BYTE_FALLBACK,
    YUE_DEI6_BYTE_FALLBACK,
    ZH_YINHANG_MERGED,
    ZH_ZHONGGUO_MERGED,
    ZH_WO_SINGLETON_BASELINE,
)

MODEL_PATH = "/home/joseph/projects/notebooks/notebooks/projects/SoulX-Podcast/runs/merged"

# (fixture, gate-status) — failing combos first, then contrast cases that the
# behavioral gate passed on M1 (the disagree assert).
PROBES = [
    (YUE_DEI6_BYTE_FALLBACK, "FAIL M1 (corr==wrong, edit=0.0)"),
    (ZH_YINHANG_MERGED, "FAIL M1 (corr==wrong, edit=0.0)"),
    (YUE_KEOI5_BYTE_FALLBACK, "pass M1 / FAIL D1 (edit=0.333)"),
    (ZH_ZHONGGUO_MERGED, "pass M1 (edit=0.583)"),
    (ZH_WO_SINGLETON_BASELINE, "pass M1 (singleton baseline)"),
]


@torch.no_grad()
def composed_at_pads(engine: InpaintInferenceEngine, ssml: str, lang: str):
    """Mirror the embed-building half of ``generate_speech_tokens`` and return
    the composed-embedding rows at the masked (pad) positions, plus the
    text-embedding norm for scale reference.

    Returns (composed_pad_rows: (n_pad, d), text_emb_norm_median: float,
             n_pad: int).
    """
    surface_text, spans = parse_ssml(ssml)
    prefix = DIALECT_PREFIX.get(lang, "")
    (_surface, text_ids, _offs, phone_token, phone_mask, _n) = (
        engine._build_text_and_phone_tokens(
            surface_text, spans, prefix, lang, disable_inpaint=False
        )
    )

    task_prefix = [
        engine.special["task_podcast"],
        engine.special["speaker_0"],
        engine.special["text_start"],
    ]
    bridge = [engine.special["text_end"], engine.special["semantic_token_start"]]
    input_ids_list = task_prefix + text_ids + bridge
    T = len(input_ids_list)
    text_token_global_start = len(task_prefix)
    K = engine.K

    full_phone_token = torch.zeros(K * T, dtype=torch.long)
    full_phone_mask = torch.zeros(T, dtype=torch.bool)
    for local_ti in range(len(text_ids)):
        gti = text_token_global_start + local_ti
        full_phone_token[gti * K : (gti + 1) * K] = phone_token[
            local_ti * K : (local_ti + 1) * K
        ]
        full_phone_mask[gti] = phone_mask[local_ti]

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=engine.device)
    full_phone_token = full_phone_token.unsqueeze(0).to(engine.device)
    alphabet = LANG_TO_ALPHABET[lang]
    alphabet_id = torch.tensor(
        [ALPHABET_STR_TO_ID[alphabet]], dtype=torch.long, device=engine.device
    )

    text_emb = engine.model.get_input_embeddings()(input_ids).to(engine.dtype)
    composed, mask = engine.composer(full_phone_token, alphabet_id)

    pad_idx = mask[0].nonzero(as_tuple=True)[0]
    composed_pad = composed[0, pad_idx].float()          # (n_pad, d)
    text_norm_med = text_emb[0].float().norm(dim=-1).median().item()
    return composed_pad, text_norm_med, int(pad_idx.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composer_ckpt", required=True)
    ap.add_argument("--model_path", default=MODEL_PATH)
    args = ap.parse_args()

    engine = InpaintInferenceEngine(
        model_path=args.model_path, composer_ckpt_path=args.composer_ckpt
    )

    print("\n" + "=" * 84)
    print("COMPOSER NO-OP DIAGNOSTIC — composed-embedding inspection at pad positions")
    print(f"text-emb median row norm is the scale reference for 'near-zero'.")
    print("=" * 84)

    rows = []
    for fx, status in PROBES:
        c_corr, tnorm, npad_c = composed_at_pads(engine, fx.ssml_correct, fx.lang)
        c_wrng, _, npad_w = composed_at_pads(engine, fx.ssml_wrong, fx.lang)

        n = min(c_corr.shape[0], c_wrng.shape[0])
        c_corr, c_wrng = c_corr[:n], c_wrng[:n]

        norm_corr = c_corr.norm(dim=-1).mean().item()
        norm_wrng = c_wrng.norm(dim=-1).mean().item()
        # cosine per pad row, then mean — 1.0 == collapsed onto same vector.
        cos = F.cosine_similarity(c_corr, c_wrng, dim=-1).mean().item()
        # L2 distance relative to the correct-row norm — how far apart are they?
        rel_l2 = ((c_corr - c_wrng).norm(dim=-1) / (c_corr.norm(dim=-1) + 1e-6)).mean().item()

        # Classify.
        ratio_to_text = norm_corr / (tnorm + 1e-6)
        if ratio_to_text < 0.1:
            verdict = "(a) NEAR-ZERO — head emits no signal"
        elif cos > 0.98:
            verdict = "(b) COLLAPSED — corr≈wrong, trunk can't distinguish"
        else:
            verdict = "(c) DISTINCT — injection differs; trunk-side issue"

        print(f"\n[{fx.name}]  {status}")
        print(f"    n_pad={npad_c}  text_emb_norm≈{tnorm:.3f}")
        print(f"    ‖composed_correct‖ = {norm_corr:8.3f}   "
              f"(×{ratio_to_text:.1f} vs text_emb)")
        print(f"    ‖composed_wrong‖   = {norm_wrng:8.3f}")
        print(f"    cos(correct,wrong) = {cos:8.4f}   "
              f"rel_L2 = {rel_l2:.4f}")
        print(f"    => {verdict}")
        rows.append((fx.name, status, norm_corr, cos, rel_l2, verdict))

    print("\n" + "=" * 84)
    print("SUMMARY")
    print("=" * 84)
    print(f"{'fixture':28s} {'‖cor‖':>8s} {'cos':>8s} {'relL2':>7s}  verdict")
    for name, _st, ncor, cos, rl2, verdict in rows:
        print(f"{name:28s} {ncor:8.2f} {cos:8.4f} {rl2:7.4f}  {verdict.split('—')[0].strip()}")

    # --- cross-fixture direction collapse check ------------------------- #
    # If composed embeddings point the SAME way across different phonemes AND
    # alphabets, the head learned a phoneme-agnostic "marker" direction and the
    # phoneme signal is only a small perturbation on top. Compare the first pad
    # row of each fixture's CORRECT override against every other.
    print("\n" + "=" * 84)
    print("CROSS-FIXTURE DIRECTION COLLAPSE  (cos of composed_correct[pad0] across fixtures)")
    print("high off-diagonal cos => a global phoneme-agnostic marker direction")
    print("=" * 84)
    firsts = []
    names = []
    for fx, _st in PROBES:
        c_corr, _, _ = composed_at_pads(engine, fx.ssml_correct, fx.lang)
        firsts.append(c_corr[0])
        names.append(fx.name.split("_")[1])  # short label
    M = torch.stack(firsts)                  # (F, d)
    cosmat = F.cosine_similarity(M.unsqueeze(1), M.unsqueeze(0), dim=-1)
    hdr = "          " + " ".join(f"{n[:7]:>8s}" for n in names)
    print(hdr)
    for i, n in enumerate(names):
        row = " ".join(f"{cosmat[i, j].item():8.3f}" for j in range(len(names)))
        print(f"{n[:9]:9s} {row}")
    off = cosmat[~torch.eye(len(names), dtype=torch.bool)]
    print(f"\noff-diagonal cos: mean={off.mean():.3f}  min={off.min():.3f}  max={off.max():.3f}")
    print("(different phonemes, mix of jyutping+pinyin — high mean = collapsed onto a marker)\n")


if __name__ == "__main__":
    main()
