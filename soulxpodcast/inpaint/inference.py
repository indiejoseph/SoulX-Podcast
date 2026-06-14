"""Inference engine for pronunciation-inpaint.

Loads a trained :class:`PhonemeComposer` checkpoint + the SoulX Qwen3
backbone and exposes a single-utterance ``generate_speech_tokens``
method that accepts an SSML string (or plain text) and returns the
LLM's generated speech-token sequence.

This is the LLM-only path — flow + HiFT vocoder synthesis is *not*
included here. For audio, take the returned ``speech_tokens`` and feed
them through ``SoulXPodcast``'s existing flow/vocoder pipeline with a
voice-prompt of your choice.

Composer checkpoint format (from :mod:`soulxpodcast.training.train_inpaint`):

    {
      "step":      int,
      "config":    {"d_model": int, "slots_per_token": int,
                    "vocab_size": int},
      "composer":  state_dict,
      "optimizer": ...,
      "scheduler": ...,
    }

Only ``config`` + ``composer`` are required for inference.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soulxpodcast.inpaint.composer import (
    PhonemeComposer, apply_phoneme_inpaint,
    ALPHABET_STR_TO_ID, K_STORAGE,
)
from soulxpodcast.inpaint.ssml import PhonemeSpan, parse_ssml
from soulxpodcast.inpaint.tokenizer import PhonemeTokenizer
from soulxpodcast.training.inpaint_dataset import (
    DIALECT_PREFIX,
    LANG_TO_ALPHABET,
    NON_PHONEME_TOKENS,
    SPECIAL_TOKENS,
    _block,
    _resolve_special_tokens,
    _substitute_units,
    normalise_text,
)

log = logging.getLogger("inpaint.inference")


# Same default offset as in soulxpodcast_config.json. Override from
# the model's config file when constructing the engine if needed.
DEFAULT_SPEECH_TOKEN_OFFSET = 153595


@dataclass
class GenerateOutput:
    """One generate() call's result for a single SSML/text input."""

    speech_tokens: list[int]            # 0-based s3tokenizer ids
    raw_generated_ids: list[int]        # LLM-vocab ids (offset NOT subtracted)
    surface_text: str                   # SSML parsed → plain text (with dialect prefix)
    phone_mask: list[bool]              # True at BPE positions that got composer-inject
    phone_token_nnz: int                # count of non-pad slot ids written
    n_phonemes_aligned: int             # count of unit spans that aligned successfully
    spans: list[PhonemeSpan]            # parsed SSML spans (empty for plain text)
    prompt_len: int                     # number of tokens in the LLM input prompt
    eos_hit: bool                       # did the model emit <|semantic_token_end|>?


class InpaintInferenceEngine:
    """Single-GPU inference wrapper around a frozen Qwen3 + trained composer."""

    def __init__(
        self,
        model_path: str | Path,
        composer_ckpt_path: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        speech_token_offset: int = DEFAULT_SPEECH_TOKEN_OFFSET,
        speech_token_vocab: int = 6561,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.speech_token_offset = speech_token_offset
        self.speech_token_vocab = speech_token_vocab

        log.info(f"loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.special = _resolve_special_tokens(self.tokenizer)

        log.info(f"loading model from {model_path}  dtype={dtype}  attn={attn_implementation}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device,
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        log.info(f"loading composer checkpoint: {composer_ckpt_path}")
        ckpt = torch.load(composer_ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        log.info(
            f"composer config: d_model={cfg['d_model']} K={cfg['slots_per_token']} "
            f"vocab={cfg['vocab_size']}"
        )
        if cfg["d_model"] != self.model.config.hidden_size:
            raise ValueError(
                f"composer d_model ({cfg['d_model']}) != model hidden_size "
                f"({self.model.config.hidden_size}) — backbone mismatch?"
            )
        self.composer = PhonemeComposer(
            d_model=cfg["d_model"], slots_per_token=cfg["slots_per_token"]
        ).to(self.device, dtype=dtype)
        # Strict load — composer arch must match the checkpoint exactly.
        self.composer.load_state_dict(ckpt["composer"], strict=True)
        self.composer.eval()
        self.K = cfg["slots_per_token"]

        self.phone_tok = PhonemeTokenizer()

    # ----- public API ------------------------------------------------- #

    @torch.no_grad()
    def generate_speech_tokens(
        self,
        ssml_or_text: str,
        lang: str = "yue",
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
        disable_inpaint: bool = False,
    ) -> GenerateOutput:
        """Run a single utterance through the LLM, returning generated speech tokens.

        Args:
            ssml_or_text:   either a plain string OR an SSML string containing
                            ``<phoneme>`` tags. Plain text → composer never
                            fires; SSML → composer injects at annotated spans.
            lang:           one of ``en``, ``yue``, ``zh``. Controls the
                            dialect prefix prepended to text.
            disable_inpaint: if True, ignore SSML spans and route entirely
                            through the text path. Useful for A/B comparisons.

        Returns: :class:`GenerateOutput` with the generated speech_tokens
        (0-based s3tokenizer ids) plus diagnostics.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Parse SSML (or treat as plain text).
        if "<phoneme" in ssml_or_text or "<speak" in ssml_or_text:
            surface_text, spans = parse_ssml(ssml_or_text)
        else:
            surface_text, spans = ssml_or_text, []

        # 2. Build the BPE-aligned phone_token from SSML spans.
        prefix = DIALECT_PREFIX.get(lang, "")
        surface_with_prefix, text_ids, offsets, phone_token, phone_mask, n_aligned = (
            self._build_text_and_phone_tokens(
                surface_text, spans, prefix, lang, disable_inpaint=disable_inpaint
            )
        )

        # 3. Assemble the SoulX template (same shape as training):
        #    <|task_podcast|> <|SPEAKER_0|> <|text_start|>
        #    {prefix+text}
        #    <|text_end|> <|semantic_token_start|>
        # Then let the LLM continue to <|semantic_token_end|>.
        task_prefix = [
            self.special["task_podcast"],
            self.special["speaker_0"],
            self.special["text_start"],
        ]
        bridge = [self.special["text_end"], self.special["semantic_token_start"]]
        input_ids_list = task_prefix + text_ids + bridge

        T = len(input_ids_list)
        text_token_global_start = len(task_prefix)

        full_phone_token = torch.zeros(self.K * T, dtype=torch.long)
        full_phone_mask = torch.zeros(T, dtype=torch.bool)
        for local_ti in range(len(text_ids)):
            gti = text_token_global_start + local_ti
            full_phone_token[gti * self.K : (gti + 1) * self.K] = (
                phone_token[local_ti * self.K : (local_ti + 1) * self.K]
            )
            full_phone_mask[gti] = phone_mask[local_ti]

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        full_phone_token = full_phone_token.unsqueeze(0).to(self.device)
        full_phone_mask = full_phone_mask.unsqueeze(0).to(self.device)
        alphabet = LANG_TO_ALPHABET[lang]
        alphabet_id = torch.tensor(
            [ALPHABET_STR_TO_ID[alphabet]], dtype=torch.long, device=self.device,
        )

        # 4. Compute inputs_embeds with composer inject at masked positions.
        text_emb = self.model.get_input_embeddings()(input_ids).to(self.dtype)
        composed, mask_check = self.composer(full_phone_token, alphabet_id)
        # Sanity: mask derived from slots must match what we built.
        if not torch.equal(mask_check, full_phone_mask):
            log.warning(
                "composer-derived phone_mask diverges from dataset-derived mask; "
                "using composer's view."
            )
            full_phone_mask = mask_check
        # NOTE: composer outputs at norm ~300 vs text_emb ~1.5 — huge by
        # naive comparison, but the LLM was trained ON those magnitudes
        # (the composer is paired with the LLM during training). DO NOT
        # renormalise at inference; that creates a train/infer mismatch
        # and breaks the cases that worked.
        inputs_embeds = apply_phoneme_inpaint(text_emb, composed, full_phone_mask)

        # 5. Generate. EOS = <|semantic_token_end|>; we cap with max_new_tokens.
        eos_id = self.special["semantic_token_end"]
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_id,
            pad_token_id=self.tokenizer.pad_token_id or 0,
        )
        out = self.model.generate(**gen_kwargs)
        # When generating from inputs_embeds, HF returns ONLY the generated
        # token ids (no prompt prefix). Confirm shape.
        gen_ids = out[0].tolist()

        # 6. Split EOS off; convert LLM-vocab speech ids → 0-based s3tokenizer ids.
        eos_hit = bool(gen_ids and gen_ids[-1] == eos_id)
        if eos_hit:
            gen_ids = gen_ids[:-1]
        speech_tokens = [
            int(t - self.speech_token_offset)
            for t in gen_ids
            if self.speech_token_offset
            <= t
            < self.speech_token_offset + self.speech_token_vocab
        ]

        return GenerateOutput(
            speech_tokens=speech_tokens,
            raw_generated_ids=gen_ids,
            surface_text=surface_with_prefix,
            phone_mask=full_phone_mask[0].cpu().tolist(),
            phone_token_nnz=int((full_phone_token != 0).sum()),
            n_phonemes_aligned=n_aligned,
            spans=spans,
            prompt_len=T,
            eos_hit=eos_hit,
        )

    # ----- internals -------------------------------------------------- #

    def _build_text_and_phone_tokens(
        self,
        surface_text: str,
        spans: list[PhonemeSpan],
        prefix: str,
        lang: str,
        disable_inpaint: bool,
    ) -> tuple[str, list[int], list[tuple[int, int]], torch.Tensor, torch.Tensor, int]:
        """Grapheme-substitution alignment matching the training dataset's
        ``_align_grapheme_subst_*`` byte-for-byte (asserted by
        ``scripts/inpaint/test_alignment_parity.py``).

        Strategy:
          1. Tokenize natural (unmarked) text + dialect prefix once.
          2. For each SSML span, build a substitution unit
             ``(char_start, char_end, blocks)`` with one slot block per
             syllable (one per CJK char for zh/yue; one per syllable for en).
          3. Hand the units to the shared :func:`_substitute_units`, which
             REMOVES each unit's covering grapheme BPE(s) and splices the
             phoneme pad(s) in their place. ``phone_mask`` is True at the
             pads, False elsewhere. The composed embedding is therefore the
             only signal at the annotated position — the LM cannot fall back
             on the (now-deleted) grapheme.

        Returns surface_with_prefix, new text_ids (graphemes substituted),
        unused offsets (``[]`` placeholder — caller doesn't read them),
        phone_token, phone_mask, n_aligned.
        """
        from soulxpodcast.inpaint.tokenizer import encode_arpabet_per_syllable

        surface_text = normalise_text(surface_text, lang)
        surface_with_prefix = prefix + surface_text
        prefix_len = len(prefix)

        enc = self.tokenizer(
            surface_with_prefix, add_special_tokens=False, return_offsets_mapping=True
        )
        base_text_ids: list[int] = enc["input_ids"]
        base_offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]

        K = self.K
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id for padsub")

        if disable_inpaint or not spans:
            phone_token = torch.zeros(K * len(base_text_ids), dtype=torch.long)
            phone_mask = torch.zeros(len(base_text_ids), dtype=torch.bool)
            return surface_with_prefix, base_text_ids, [], phone_token, phone_mask, 0

        # Build substitution units (char_start, char_end, blocks) and
        # splice via the SHARED _substitute_units so the inference input_ids
        # match the training pipeline byte-for-byte (guarded by
        # test_alignment_parity.py). Each unit's covering grapheme BPE(s) are
        # removed and replaced by its phoneme pad(s).
        units: list[tuple[int, int, list[list[int]]]] = []
        n_aligned = 0
        for span in spans:
            ph_tokens = list(span.ph_tokens)
            if span.alphabet == "cmu":
                try:
                    syll_groups = encode_arpabet_per_syllable(ph_tokens)
                except ValueError as exc:
                    log.warning(f"skipping unparseable cmu span {span}: {exc}")
                    continue
                if not syll_groups:
                    continue
                blocks = [list(g) for g in syll_groups]
            else:
                # Chinese: one (initial, final-with-tone) pair per syllable.
                # ph_tokens alternates (initial, final, initial, final, ...).
                if len(ph_tokens) % 2 != 0:
                    log.warning(
                        f"chinese span ph_tokens length {len(ph_tokens)} is odd; "
                        f"expected (initial, final) pairs: {span}"
                    )
                    continue
                # SSML contract change (v11): one syllable per <phoneme> span.
                # Multi-syllable single-char wraps like
                # `<phoneme ph="y in2 h ang2">银行</phoneme>` are rejected;
                # the user must split into per-char spans.
                if len(ph_tokens) // 2 != 1 and span.char_end - span.char_start == 1:
                    raise ValueError(
                        f"chinese <phoneme> span wraps a single char but carries "
                        f"{len(ph_tokens) // 2} syllables. Split into one span "
                        f"per character. Got: ph={ph_tokens!r} "
                        f"char_range=[{span.char_start},{span.char_end})"
                    )
                try:
                    flat_ids = self.phone_tok.encode_span(span.alphabet, ph_tokens)
                except ValueError as exc:
                    log.warning(f"skipping unparseable span {span}: {exc}")
                    continue
                # Split flat_ids into (initial, final) pairs — one syllable each,
                # in reading order (matches the training group's per-char order).
                blocks = [list(flat_ids[2 * i : 2 * (i + 1)])
                          for i in range(len(flat_ids) // 2)]
            units.append(
                (span.char_start + prefix_len, span.char_end + prefix_len, blocks)
            )
            n_aligned += 1

        def _warn_fallback(unit):
            cs, ce, _ = unit
            log.warning(
                "inpaint span [%d,%d) %r partially covers a merged BPE; "
                "grapheme kept + pads inserted (v11-style) instead of "
                "substituted. Annotate the WHOLE merged token (e.g. both chars "
                "of 银行) for the trained substitution behaviour.",
                cs, ce, surface_with_prefix[cs:ce],
            )

        # Whitespace positions are free to delete with a word's leading-space
        # BPE (English); zh/yue have spaces stripped so this set is empty.
        # MUST match the training aligner's free_chars for parity.
        free_chars = {i for i, ch in enumerate(surface_with_prefix) if ch.isspace()}

        new_text_ids, new_slots, new_mask = _substitute_units(
            base_text_ids, base_offsets, units, pad_id, K,
            on_fallback=_warn_fallback, free_chars=free_chars,
        )
        # Pack slots into the flat phone_token shape the composer expects.
        phone_token = torch.zeros(K * len(new_text_ids), dtype=torch.long)
        for i, block in enumerate(new_slots):
            phone_token[i * K : (i + 1) * K] = torch.tensor(block, dtype=torch.long)
        phone_mask = torch.tensor(new_mask, dtype=torch.bool)

        return surface_with_prefix, new_text_ids, [], phone_token, phone_mask, n_aligned
