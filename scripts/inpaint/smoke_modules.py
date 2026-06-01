"""End-to-end smoke test for `soulxpodcast.inpaint`.

Exercises the SSML parser, the PhonemeTokenizer (split-form encoding +
whole-syllable splitting for dataset conversion), the PhonemeComposer
forward + auxiliary-label derivation, and the inject helper.

Run with::

    python scripts/inpaint/smoke_modules.py

Exits non-zero on assertion failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.inpaint import (  # noqa: E402
    PhonemeComposer,
    PhonemeTokenizer,
    apply_phoneme_inpaint,
    parse_ssml,
)
from soulxpodcast.inpaint._vocab import TOTAL_VOCAB_SIZE  # noqa: E402


def test_tokenizer() -> None:
    tok = PhonemeTokenizer()
    # forward encode — CMU stream is syllabified internally so L_on vs L_co distinguish
    hello_ids = tok.encode_span("cmu", ["HH", "AH", "L", "OW"])
    world_ids = tok.encode_span("cmu", ["W", "ER", "L", "D"])
    # "hello": maximal-onset → syll 0 (HH onset, AH nucleus), syll 1 (L onset, OW nucleus)
    # "world": only one vowel ER → W onset, ER nucleus, L coda, D coda
    # So `L` in hello ≠ `L` in world (different ids).
    l_in_hello = hello_ids[2]
    l_in_world = world_ids[2]
    assert l_in_hello != l_in_world, (
        f"position tag failed — L_on and L_co should differ; got {l_in_hello} vs {l_in_world}"
    )
    # Boundary marker round-trips
    assert tok.encode_span("cmu", ["HH", "IY", "|", "W", "ER", "L", "D"])[2] != 0
    # Chinese encodings — pairs of (initial, final+tone)
    assert len(tok.encode_span("jyutping", ["j", "yut6", "p", "ing4"])) == 4
    assert len(tok.encode_span("pinyin", ["p", "in1", "y", "in1"])) == 4
    # zero-initial syllables
    assert len(tok.encode_span("jyutping", ["", "aa3"])) == 2
    assert len(tok.encode_span("pinyin", ["", "a1"])) == 2
    # whole-syllable split helper
    assert tok.split_whole_syllable("jyutping", "nei1") == ("n", "ei1")
    assert tok.split_whole_syllable("jyutping", "aa3") == ("", "aa3")
    assert tok.split_whole_syllable("jyutping", "gwong2") == ("gw", "ong2")
    assert tok.split_whole_syllable("pinyin", "zhuo1") == ("zh", "uo1")
    whole = tok.encode_whole_syllable_sequence("jyutping", ["nei1", "hou2", "aa3"])
    split = tok.encode_span("jyutping", ["n", "ei1", "h", "ou2", "", "aa3"])
    assert whole == split
    # error cases
    for fn in [
        lambda: tok.encode_span("cmu", ["XYZ"]),
        lambda: tok.encode_span("jyutping", ["j", "yut6", "p"]),
        lambda: tok.encode_span("jyutping", ["xx", "aa3"]),
        lambda: tok.encode_span("jyutping", ["j", "xxxxx"]),
        lambda: tok.encode_span("ipa", ["a"]),
    ]:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    # Syllabifier shape spot-check
    sylls = PhonemeTokenizer.syllabify_arpabet(["HH", "AH", "L", "OW"])
    assert len(sylls) == 2
    assert ("L", "onset") in sylls[1]
    sylls2 = PhonemeTokenizer.syllabify_arpabet(["W", "ER", "L", "D"])
    assert len(sylls2) == 1
    assert ("L", "coda") in sylls2[0] and ("D", "coda") in sylls2[0]
    print("tokenizer ✓ (L_on != L_co confirmed)")


def test_ssml() -> None:
    text, spans = parse_ssml(
        '<speak>Hello <phoneme alphabet="cmu" ph="HH AH L OW">hello</phoneme>!'
        '上<phoneme alphabet="jyutping" ph="t ong4">堂</phoneme>。</speak>'
    )
    assert text == "Hello hello!上堂。"
    assert len(spans) == 2
    assert spans[0].alphabet == "cmu" and spans[0].ph_tokens == ("HH", "AH", "L", "OW")
    assert text[spans[0].char_start : spans[0].char_end] == "hello"
    assert spans[1].alphabet == "jyutping" and text[spans[1].char_start : spans[1].char_end] == "堂"
    # no <speak> wrapper still works
    text2, spans2 = parse_ssml(
        'Hello <phoneme alphabet="cmu" ph="HH AH L OW">hello</phoneme>!'
    )
    assert text2 == "Hello hello!" and len(spans2) == 1
    print("ssml ✓")


def test_composer() -> None:
    torch.manual_seed(0)
    tok = PhonemeTokenizer()
    d_model, K, L, B = 32, 8, 5, 2
    composer = PhonemeComposer(d_model=d_model, slots_per_token=K)

    phone = torch.zeros(B, K * L, dtype=torch.long)
    ids_cmu = tok.encode_span("cmu", ["HH", "AH", "L", "OW"])  # "hello"
    phone[0, 0 * K : 0 * K + len(ids_cmu)] = torch.tensor(ids_cmu)
    ids_py = tok.encode_span("pinyin", ["p", "in1"])
    phone[1, 2 * K : 2 * K + len(ids_py)] = torch.tensor(ids_py)

    composed, mask = composer(phone)
    assert composed.shape == (B, L, d_model)
    assert mask.shape == (B, L)
    assert composed[~mask].abs().sum().item() == 0  # zero outside the mask

    fake_text_emb = torch.randn(B, L, d_model)
    merged = apply_phoneme_inpaint(fake_text_emb, composed, mask)
    assert (merged[~mask] == fake_text_emb[~mask]).all()
    assert (merged[mask] == composed[mask]).all()

    labels = composer.alphabet_labels(phone)
    assert labels[0, 0].item() == 0  # cmu
    assert labels[1, 2].item() == 2  # pinyin
    assert (labels[0, 1:] == -100).all()
    assert (labels[1, [0, 1, 3, 4]] == -100).all()

    fake_text_embedder = torch.nn.Embedding(1000, d_model)
    torch.nn.init.normal_(fake_text_embedder.weight, mean=0.5, std=0.01)
    composer.init_from_text_embed(fake_text_embedder)
    assert composer.phone_emb.weight[0].abs().sum().item() == 0

    n_train = sum(p.numel() for p in composer.parameters() if p.requires_grad)
    print(f"composer ✓  ({n_train:,} trainable params, vocab={TOTAL_VOCAB_SIZE}, d_model={d_model})")


if __name__ == "__main__":
    test_tokenizer()
    test_ssml()
    test_composer()
    print("ALL CHECKS PASSED ✓")
