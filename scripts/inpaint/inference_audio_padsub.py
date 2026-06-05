"""Audio synthesis with the upstream-style pad-token-substitution alignment.

The default `_build_text_and_phone_tokens` in InpaintInferenceEngine writes the
composer slot block over EVERY BPE that covers a span's char range. For chars
that byte-fallback into >1 BPE (e.g. 佢 → 2 BPEs) the composer fires twice and
the LLM emits the override syllable twice ("baai6 baai6"). For multi-syllable
Chinese words that merge into 1 BPE (e.g. 银行) all 4 phone ids land in one
slot block and mean-pool destroys order → wrong syllable.

CosyVoice-Inpaint upstream avoids both: it strips `[ph]` markup and inserts ONE
`<pad>` token per syllable into the LLM input. The composer fires only at pad
positions and each pad carries exactly one syllable's worth of slots.

This script monkey-patches the engine to do the same and re-runs the v10 A/B.

Outputs at outputs/inpaint_audio_v10_padsub/.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.inpaint import inference as _inf
from soulxpodcast.inpaint.composer import apply_phoneme_inpaint  # noqa: F401  (sanity)
from soulxpodcast.inpaint.ssml import parse_ssml
from soulxpodcast.inpaint.tokenizer import encode_arpabet_per_syllable
from soulxpodcast.training.inpaint_dataset import (
    DIALECT_PREFIX, LANG_TO_ALPHABET, normalise_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inference_audio_padsub")


def _build_text_and_phone_tokens_padsub(
    self,
    surface_text: str,
    spans,
    prefix: str,
    lang: str,
    disable_inpaint: bool,
):
    """Pad-token-substitution alignment (upstream CosyVoice-Inpaint style).

    For each <phoneme> span:
      - tokenize the text segment BEFORE the span normally,
      - insert exactly N pad-token ids (N = syllable count of the override),
      - write each syllable's phone ids into ONE pad token's slot block.
    Non-span text is tokenized as-is.

    For Chinese (jyutping/pinyin) N = len(ph_tokens) // 2 (initial-final pairs).
    For English (cmu) N = #syllables from encode_arpabet_per_syllable.
    """
    pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has no pad/eos id to substitute with")

    surface_text = normalise_text(surface_text, lang)
    K = self.K

    if disable_inpaint or not spans:
        text_ids = self.tokenizer(prefix + surface_text, add_special_tokens=False)["input_ids"]
        T = len(text_ids)
        return (prefix + surface_text, text_ids, [],
                torch.zeros(K * T, dtype=torch.long),
                torch.zeros(T, dtype=torch.bool), 0)

    spans_sorted = sorted(spans, key=lambda s: s.char_start)

    text_ids: list[int] = []
    # (span, list of token indices in text_ids occupied by its pad tokens)
    pad_positions_per_span: list[tuple[object, list[int]]] = []

    if prefix:
        text_ids.extend(self.tokenizer(prefix, add_special_tokens=False)["input_ids"])

    cursor = 0
    for span in spans_sorted:
        # 1. text segment before this span
        seg = surface_text[cursor:span.char_start]
        if seg:
            text_ids.extend(self.tokenizer(seg, add_special_tokens=False)["input_ids"])

        # 2. choose syllable count for this span
        ph_tokens = list(span.ph_tokens)
        if span.alphabet == "cmu":
            try:
                syll_groups = encode_arpabet_per_syllable(ph_tokens)
            except ValueError as e:
                log.warning(f"skipping unparseable cmu span {span}: {e}")
                cursor = span.char_end
                continue
            n_pads = max(1, len(syll_groups))
        else:
            # jyutping / pinyin → tokens alternate (initial, final, initial, final, ...)
            n_pads = max(1, len(ph_tokens) // 2)

        start_idx = len(text_ids)
        text_ids.extend([pad_id] * n_pads)
        pad_positions_per_span.append((span, list(range(start_idx, start_idx + n_pads))))
        cursor = span.char_end

    # 3. trailing text segment after the last span
    if cursor < len(surface_text):
        seg = surface_text[cursor:]
        text_ids.extend(self.tokenizer(seg, add_special_tokens=False)["input_ids"])

    # 4. write phoneme ids into each pad's slot block (one syllable per pad)
    T = len(text_ids)
    phone_token = torch.zeros(K * T, dtype=torch.long)
    phone_mask = torch.zeros(T, dtype=torch.bool)
    n_aligned = 0

    for span, pad_positions in pad_positions_per_span:
        ph_tokens = list(span.ph_tokens)
        try:
            flat_ids = self.phone_tok.encode_span(span.alphabet, ph_tokens)
        except ValueError as e:
            log.warning(f"skipping unparseable span {span}: {e}")
            continue

        if span.alphabet == "cmu":
            syll_groups = encode_arpabet_per_syllable(ph_tokens)
        else:
            # pair off (initial, final) — same length as len(flat_ids) // 2
            syll_groups = [flat_ids[2 * i : 2 * (i + 1)] for i in range(len(flat_ids) // 2)]

        n = min(len(pad_positions), len(syll_groups))
        for k in range(n):
            pos = pad_positions[k]
            group = syll_groups[k]
            if len(group) > K:
                group = list(group)[:K]
            phone_token[pos * K : pos * K + len(group)] = torch.tensor(group, dtype=torch.long)
            phone_mask[pos] = True
        if n > 0:
            n_aligned += 1

    # offsets list (unused downstream for inference) — return empty
    return (prefix + surface_text, text_ids, [], phone_token, phone_mask, n_aligned)


def install_padsub():
    """Monkey-patch the engine method."""
    _inf.InpaintInferenceEngine._build_text_and_phone_tokens = (
        _build_text_and_phone_tokens_padsub
    )
    log.info("installed pad-substitution alignment in InpaintInferenceEngine")


def main():
    install_padsub()

    # Reuse the same A/B harness as the original inference_audio.py.
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location(
        "_inference_audio",
        str(Path(__file__).parent / "inference_audio.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["_inference_audio"] = mod
    # Override default --output_dir + --composer_ckpt via argv if user didn't pass them.
    if "--composer_ckpt" not in _sys.argv:
        _sys.argv += [
            "--composer_ckpt",
            "outputs/inpaint_h100_v10_filtered_meanpool/step_0030000/composer.pt",
        ]
    if "--output_dir" not in _sys.argv:
        _sys.argv += ["--output_dir", "outputs/inpaint_audio_v10_padsub"]
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
