"""Pronunciation inpainting for SoulX-Podcast.

Public API:
    from soulxpodcast.inpaint import (
        PhonemeTokenizer,
        PhonemeComposer,
        parse_ssml,
        PhonemeSpan,
        apply_phoneme_inpaint,
    )

See .claude/skills/cosyvoice-inpaint/ for the design.
"""

from soulxpodcast.inpaint.composer import PhonemeComposer, apply_phoneme_inpaint
from soulxpodcast.inpaint.ssml import PhonemeSpan, parse_ssml
from soulxpodcast.inpaint.tokenizer import PhonemeTokenizer

__all__ = [
    "PhonemeComposer",
    "PhonemeSpan",
    "PhonemeTokenizer",
    "apply_phoneme_inpaint",
    "parse_ssml",
]
