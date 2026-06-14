"""Dataset format + composer-output magnitude audit.

Two checks bundled together because both came up while debugging why
multi-language inpaint inference produced silent / non-terminating
output:

1. ``--check format`` — per-language formatting in ``tmp/dataset.jsonl``:
   uppercase ratio, terminal-punctuation rate, sample texts. This caught
   that English training rows are 100% UPPERCASE without trailing
   punctuation and Mandarin rows have no punctuation. Inference test
   strings that don't match this distribution put the LLM out-of-domain
   and cause EOS failures.

2. ``--check composer`` — per-alphabet phone_emb row-norm stats from a
   composer checkpoint. Underrepresented alphabets (Pinyin at 3% of the
   training corpus here) end up with smaller delta-from-init magnitudes
   — a quantitative signal for "this alphabet won't perform well at
   inference".

Run examples::

    python scripts/inpaint/audit_format.py --check format
    python scripts/inpaint/audit_format.py --check composer \\
        --composer_ckpt outputs/inpaint_final/step_0030000/composer.pt
    python scripts/inpaint/audit_format.py --check both \\
        --composer_ckpt outputs/inpaint_final/step_0030000/composer.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _bucket_per_lang(jsonl_path: Path, max_per_lang: int) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"en": [], "zh": [], "yue": []}
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            lang = r.get("lang")
            if lang not in buckets or len(buckets[lang]) >= max_per_lang:
                if all(len(v) >= max_per_lang for v in buckets.values()):
                    break
                continue
            buckets[lang].append(r)
            if all(len(v) >= max_per_lang for v in buckets.values()):
                break
    return buckets


def audit_format(jsonl_path: Path, sample_n: int) -> None:
    print(f"\n=== format audit (sample {sample_n}/lang from {jsonl_path}) ===")
    buckets = _bucket_per_lang(jsonl_path, sample_n)

    for lang, rows in buckets.items():
        if not rows:
            print(f"  no rows for {lang}")
            continue
        has_lower = sum(any(c.islower() for c in r["text"]) for r in rows)
        has_upper = sum(any(c.isupper() for c in r["text"]) for r in rows)
        has_digit = sum(any(c.isdigit() for c in r["text"]) for r in rows)
        has_end_punct = sum(
            r["text"].rstrip().endswith((".", "。", "？", "?", "!", "！", "，", ","))
            for r in rows
        )
        print(f"\n  --- {lang} ({len(rows)} rows) ---")
        print(f"    has_lower_letter : {has_lower}/{len(rows)}")
        print(f"    has_upper_letter : {has_upper}/{len(rows)}")
        print(f"    has_digit        : {has_digit}/{len(rows)}")
        print(f"    ends_with_punct  : {has_end_punct}/{len(rows)}")
        print(f"    examples:")
        for r in rows[:3]:
            print(f"      {r['text'][:90]!r}")

    print()
    print("  → Inference text must match these patterns. Common failure modes:")
    print("    en: mixed-case + trailing period → LLM out-of-domain, EOS fails")
    print("    zh: trailing period            → same as en")
    print("    yue: text uses spaces, but the dataset adapter strips them")


def audit_composer(ckpt_path: Path) -> None:
    print(f"\n=== composer phone_emb audit ({ckpt_path}) ===")

    import torch

    from soulxpodcast.inpaint._vocab import (  # noqa: E402
        CMU_BOUNDARY_ID,
        CMU_CODA_BASE,
        CMU_ONSET_BASE,
        CMU_VOWEL_BASE,
        JP_FINAL_BASE,
        JP_INITIAL_BASE,
        N_CMU_CODA,
        N_CMU_ONSET,
        N_CMU_VOWEL,
        N_JP_FINAL,
        N_JP_INITIAL,
        N_PY_FINAL,
        N_PY_INITIAL,
        PY_FINAL_BASE,
        PY_INITIAL_BASE,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    phone_emb = ckpt["composer"]["phone_emb.weight"].float()
    print(f"  phone_emb shape: {tuple(phone_emb.shape)}")

    # Robust init-centre estimate: median of non-pad rows.
    init_centre = phone_emb[1:].median(dim=0).values
    delta_norms = (phone_emb - init_centre).norm(dim=-1)
    delta_norms[0] = 0.0
    row_norms = phone_emb.norm(dim=-1)

    def stats(name: str, lo: int, hi: int) -> None:
        rn = row_norms[lo:hi]
        dn = delta_norms[lo:hi]
        print(
            f"    {name:30s} ids[{lo:4d}..{hi - 1:4d}] n={hi - lo:4d}  "
            f"row_norm={rn.mean():.3f}±{rn.std():.3f}  "
            f"delta_from_init={dn.mean():.3f}±{dn.std():.3f}"
        )

    print()
    print("  per-alphabet phone_emb statistics:")
    stats("CMU vowels (nucleus)",   CMU_VOWEL_BASE,  CMU_VOWEL_BASE + N_CMU_VOWEL)
    stats("CMU consonants _on",     CMU_ONSET_BASE,  CMU_ONSET_BASE + N_CMU_ONSET)
    stats("CMU consonants _co",     CMU_CODA_BASE,   CMU_CODA_BASE + N_CMU_CODA)
    stats("CMU boundary |",         CMU_BOUNDARY_ID, CMU_BOUNDARY_ID + 1)
    stats("Jyutping initials",      JP_INITIAL_BASE, JP_INITIAL_BASE + N_JP_INITIAL)
    stats("Jyutping finals",        JP_FINAL_BASE,   JP_FINAL_BASE + N_JP_FINAL)
    stats("Pinyin initials",        PY_INITIAL_BASE, PY_INITIAL_BASE + N_PY_INITIAL)
    stats("Pinyin finals",          PY_FINAL_BASE,   PY_FINAL_BASE + N_PY_FINAL)

    print()
    print("  → Small delta_from_init = alphabet under-trained.")
    print("    At inference, that alphabet may push the LLM toward silence / no-EOS.")
    print("    Fix: rebalance training corpus (more zh / en rows) OR add per-alphabet")
    print("    inverse-frequency loss weighting.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", choices=["format", "composer", "both"], default="both")
    p.add_argument("--jsonl_path", default="tmp/dataset.jsonl")
    p.add_argument("--composer_ckpt", default="outputs/inpaint_final/step_0030000/composer.pt")
    p.add_argument("--sample_n", type=int, default=200,
                   help="rows to sample per language for the format check")
    args = p.parse_args()

    if args.check in {"format", "both"}:
        audit_format(Path(args.jsonl_path), args.sample_n)
    if args.check in {"composer", "both"}:
        audit_composer(Path(args.composer_ckpt))


if __name__ == "__main__":
    main()
