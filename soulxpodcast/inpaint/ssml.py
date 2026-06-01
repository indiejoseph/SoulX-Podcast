"""SSML frontend for pronunciation inpainting.

Accepts the SSML subset documented at
https://docs.cloud.google.com/text-to-speech/docs/ssml#phoneme:

    <speak>
      Hello <phoneme alphabet="cmu" ph="W ER L D">world</phoneme>!
      上 <phoneme alphabet="jyutping" ph="t ong4">堂</phoneme>
      请 <phoneme alphabet="pinyin" ph="zh u3 yi4">注意</phoneme>
    </speak>

The ``<speak>`` root is optional. Multiple ``<phoneme>`` spans per
document are supported. Whitespace between tags is preserved as-is.
``alphabet="ipa"`` is rejected: an IPA→native mapper is out of scope
for v1 (see ``.claude/skills/cosyvoice-inpaint/SKILL.md`` §9).

Parse output::

    plain_text, spans = parse_ssml(ssml_string)

    plain_text  -- the surface text with <phoneme> tags collapsed to their
                   wrapped content (e.g. ``world`` survives in the text)
    spans       -- list[PhonemeSpan]; char offsets refer to plain_text
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal
from xml.etree import ElementTree as ET

Alphabet = Literal["cmu", "jyutping", "pinyin"]
_VALID_ALPHABETS: Final[frozenset[str]] = frozenset({"cmu", "jyutping", "pinyin"})


@dataclass(frozen=True, slots=True)
class PhonemeSpan:
    """A single ``<phoneme>`` annotation extracted from SSML."""

    char_start: int
    char_end: int  # exclusive
    alphabet: str
    ph_tokens: tuple[str, ...]

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


class SSMLParseError(ValueError):
    """Raised on malformed SSML or unsupported alphabet."""


_SPEAK_RE = re.compile(r"^\s*<speak\b[^>]*>(.*)</speak>\s*$", re.DOTALL)


def _strip_speak(ssml: str) -> str:
    """Strip an outer ``<speak>``…``</speak>`` wrapper if present."""
    m = _SPEAK_RE.match(ssml)
    return m.group(1) if m else ssml


def parse_ssml(ssml: str) -> tuple[str, list[PhonemeSpan]]:
    """Parse an SSML string with ``<phoneme>`` annotations.

    Returns a tuple of (plain_text, spans). Plain text contains the
    surface wording (i.e. ``<phoneme>`` tags collapsed to their inner
    content). Span char offsets are relative to plain_text and are
    half-open ``[start, end)``.
    """
    body = _strip_speak(ssml)
    # Wrap in a synthetic root so ElementTree accepts text + multiple
    # children without an enclosing element. Use namespace-free parse.
    try:
        root = ET.fromstring(f"<__root__>{body}</__root__>")
    except ET.ParseError as exc:
        raise SSMLParseError(f"malformed SSML: {exc}") from exc

    plain_parts: list[str] = []
    spans: list[PhonemeSpan] = []
    cursor = 0

    if root.text:
        plain_parts.append(root.text)
        cursor += len(root.text)

    for child in root:
        if child.tag != "phoneme":
            raise SSMLParseError(
                f"unsupported tag <{child.tag}> — only <phoneme> is recognised"
            )
        alphabet = child.attrib.get("alphabet")
        ph = child.attrib.get("ph")
        if alphabet is None:
            raise SSMLParseError("<phoneme> requires an `alphabet` attribute")
        if alphabet not in _VALID_ALPHABETS:
            raise SSMLParseError(
                f"unsupported alphabet={alphabet!r}; v1 supports {sorted(_VALID_ALPHABETS)}"
            )
        if not ph or not ph.strip():
            raise SSMLParseError("<phoneme> requires a non-empty `ph` attribute")

        surface = child.text or ""
        if not surface:
            raise SSMLParseError(
                "<phoneme> must wrap surface text — the model needs a text-token "
                "position to inject the phoneme embedding into"
            )

        start = cursor
        plain_parts.append(surface)
        cursor += len(surface)
        end = cursor

        spans.append(
            PhonemeSpan(
                char_start=start,
                char_end=end,
                alphabet=alphabet,
                ph_tokens=tuple(ph.split()),
            )
        )

        if child.tail:
            plain_parts.append(child.tail)
            cursor += len(child.tail)

    return "".join(plain_parts), spans
