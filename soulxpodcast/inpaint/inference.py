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
    PhonemeComposer, apply_phoneme_inpaint, filter_compatible_state_dict,
)
from soulxpodcast.inpaint.ssml import PhonemeSpan, parse_ssml
from soulxpodcast.inpaint.tokenizer import PhonemeTokenizer
from soulxpodcast.training.inpaint_dataset import (
    DIALECT_PREFIX,
    LANG_TO_ALPHABET,
    NON_PHONEME_TOKENS,
    SPECIAL_TOKENS,
    _align_chinese,
    _align_english,
    _is_cjk,
    _resolve_special_tokens,
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
        # Non-strict load — older checkpoints (pre-v7) carry removed
        # alphabet_head / id_to_alphabet_label keys, and pre-v10
        # checkpoints have a (d, K*d) first-Linear shape that doesn't
        # fit v10's (d, d). Filter shape-mismatched keys before load.
        sd, dropped_shape = filter_compatible_state_dict(
            self.composer, ckpt["composer"]
        )
        missing, unexpected = self.composer.load_state_dict(sd, strict=False)
        for name, keys in (("dropped (shape mismatch)", dropped_shape),
                           ("missing", missing),
                           ("unexpected", unexpected)):
            if keys:
                log.info(f"  {name} composer keys: "
                         f"{len(keys)} — {keys[:6]}{'...' if len(keys) > 6 else ''}")
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

        # 4. Compute inputs_embeds with composer inject at masked positions.
        text_emb = self.model.get_input_embeddings()(input_ids).to(self.dtype)
        composed, mask_check = self.composer(full_phone_token)
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
        """Re-uses the SAME alignment helpers as
        :class:`soulxpodcast.training.inpaint_dataset.InpaintDataset` so train/
        infer can never drift apart.
        """
        # Normalise (strip spaces for zh/yue).
        surface_text = normalise_text(surface_text, lang)
        surface_with_prefix = prefix + surface_text

        enc = self.tokenizer(
            surface_with_prefix, add_special_tokens=False, return_offsets_mapping=True
        )
        text_ids = enc["input_ids"]
        offsets = [tuple(o) for o in enc["offset_mapping"]]

        K = self.K
        T_text = len(text_ids)
        phone_token = torch.zeros(K * T_text, dtype=torch.long)
        phone_mask = torch.zeros(T_text, dtype=torch.bool)
        n_aligned = 0

        if disable_inpaint or not spans:
            return surface_with_prefix, text_ids, offsets, phone_token, phone_mask, 0

        # Build the per-text-token alignment using the SAME helpers as the
        # dataset, but restrict to span-covered char ranges so non-annotated
        # words stay text-only (matches the inference-time sparse-annotation
        # distribution we trained for).
        alphabet = LANG_TO_ALPHABET[lang]
        prefix_len = len(prefix)
        # Reduce offsets to text-portion-relative (skip dialect prefix tokens).
        text_part_offsets: list[tuple[int, int]] = []
        text_part_token_index: list[int] = []
        for ti, (cs, ce) in enumerate(offsets):
            if ce <= prefix_len:
                continue
            adj = (max(0, cs - prefix_len), max(0, ce - prefix_len))
            text_part_offsets.append(adj)
            text_part_token_index.append(ti)

        # For inference we want only the SSML-annotated spans, not the entire
        # text. So we project each span's per-language phonemes onto a fresh
        # alignment buffer.
        per_token_text, kept = self._align_spans_to_tokens(
            surface_text, text_part_offsets, spans, alphabet, K,
        )
        n_aligned = kept

        for local_ti, global_ti in enumerate(text_part_token_index):
            block = per_token_text[local_ti]
            phone_token[global_ti * K : global_ti * K + K] = torch.tensor(
                block, dtype=torch.long
            )
            if any(b != 0 for b in block):
                phone_mask[global_ti] = True

        return surface_with_prefix, text_ids, offsets, phone_token, phone_mask, n_aligned

    def _align_spans_to_tokens(
        self,
        text: str,
        text_offsets: list[tuple[int, int]],
        spans: list[PhonemeSpan],
        alphabet: str,
        K: int,
    ) -> tuple[list[list[int]], int]:
        """Per-span placement, matching the dataset adapter's alignment rule.

        - Chinese (jyutping / pinyin) spans: broadcast the span's (initial,
          final) ids across every BPE token covering its char range. Matches
          ``inpaint_dataset._align_chinese``.
        - English (cmu) spans: SYLLABLE-DISTRIBUTE — for multi-BPE multi-
          syllable words, syllable groups are spread across BPEs in order.
          Matches ``inpaint_dataset._align_english`` which the composer
          was trained on.
        """
        from soulxpodcast.inpaint.tokenizer import encode_arpabet_per_syllable

        T_text = len(text_offsets)
        per_token = [[0] * K for _ in range(T_text)]
        cursor = [0] * T_text

        # char → list[bpe token index] (covers byte-fallback rare CJK chars).
        max_char = max((ce for _, ce in text_offsets), default=0)
        char_to_tokens: list[list[int]] = [[] for _ in range(max_char)]
        for ti, (cs, ce) in enumerate(text_offsets):
            for ci in range(cs, ce):
                char_to_tokens[ci].append(ti)

        kept = 0
        for span in spans:
            cs, ce = span.char_start, span.char_end
            ph_tokens = list(span.ph_tokens)

            # Collect BPE tokens overlapping the span's char range.
            token_set: list[int] = []
            seen: set[int] = set()
            for ci in range(cs, ce):
                if ci >= len(char_to_tokens):
                    continue
                for t in char_to_tokens[ci]:
                    if t not in seen:
                        seen.add(t)
                        token_set.append(t)
            if not token_set:
                continue

            if span.alphabet == "cmu":
                # Per-syllable distribution across the span's BPE tokens.
                try:
                    syll_groups = encode_arpabet_per_syllable(ph_tokens)
                except ValueError as exc:
                    log.warning(f"skipping unparseable cmu span {span}: {exc}")
                    continue
                if not syll_groups:
                    continue
                n_bpes = len(token_set)
                n_sylls = len(syll_groups)
                if n_bpes <= 1 or n_sylls <= 1:
                    flat = [i for grp in syll_groups for i in grp]
                    if self._write_unit(per_token, cursor, token_set, flat, K):
                        kept += 1
                else:
                    base = n_sylls // n_bpes
                    extra = n_sylls % n_bpes
                    wrote_any = False
                    idx = 0
                    for bi, bpe_token in enumerate(token_set):
                        take = base + (1 if bi < extra else 0)
                        if take == 0:
                            continue
                        group = syll_groups[idx : idx + take]
                        idx += take
                        flat = [i for grp in group for i in grp]
                        if self._write_unit(per_token, cursor, [bpe_token], flat, K):
                            wrote_any = True
                    if wrote_any:
                        kept += 1
            else:
                # Chinese: broadcast (initial, final) across all BPE tokens
                # covering the span — matches _align_chinese training behavior.
                try:
                    ids = self.phone_tok.encode_span(span.alphabet, ph_tokens)
                except ValueError as exc:
                    log.warning(f"skipping unparseable span {span}: {exc}")
                    continue
                if self._write_unit(per_token, cursor, token_set, ids, K):
                    kept += 1

        return per_token, kept

    @staticmethod
    def _write_unit(
        per_token: list[list[int]],
        cursor: list[int],
        token_indices: list[int],
        ids: list[int],
        K: int,
    ) -> bool:
        """Same broadcast writer used in inpaint_dataset._write_unit_into_tokens."""
        if not token_indices:
            return False
        if any(cursor[t] != cursor[token_indices[0]] for t in token_indices):
            target = token_indices[0]
            free = K - cursor[target]
            n = min(free, len(ids))
            per_token[target][cursor[target] : cursor[target] + n] = ids[:n]
            cursor[target] += n
            return n > 0
        cur = cursor[token_indices[0]]
        free = K - cur
        if free <= 0:
            return False
        n = min(free, len(ids))
        head = ids[:n]
        for t in token_indices:
            per_token[t][cur : cur + n] = head
            cursor[t] += n
        return True
