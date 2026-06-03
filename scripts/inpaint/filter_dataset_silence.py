"""Filter dataset.jsonl to drop silence-heavy rows (mainly zh AISHELL-3).

The v4-v8 composer collapse on Mandarin was traced to AISHELL-3 studio
recordings having ~20% silence-coding s3 tokens (vs yue's ~2%). v9
takes the upstream CosyVoice-Inpaint approach of plain CE on a cleaner
dataset rather than patching the loss around bad data.

This filter applies per-row:

  1. (optional) boundary strip — drop leading and trailing runs of
     ``SILENCE_TOKEN_IDS[lang]`` from speech_tokens. The audit at
     scripts/inpaint/audit_punct_silence_alignment.py showed zh
     position-0 silence rate is 68% and position-9 (end) is 83%;
     boundary tokens are virtually all silence, so this removal is
     safe and reflects what well-trimmed conversational audio
     naturally produces.
  2. silence-fraction filter — drop the row if the post-strip
     speech_tokens still contain more than ``--max_silence_frac``
     of silence-coding ids.

Default ``--max_silence_frac=0.05`` is chosen to land zh near yue's
natural silence profile (~2.25%). Use ``--max_silence_frac=0.10`` to
keep more rows at the cost of slightly higher silence pollution.

Output: filtered jsonl + a printed histogram + per-lang kept/dropped
counts. The filtered jsonl can be fed directly to ``train_inpaint``
with all silence-handling flags OFF (matching upstream CE behaviour).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.training.inpaint_dataset import SILENCE_TOKEN_IDS

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("filter")


def trim_boundary(tokens: list[int], silence_set: frozenset[int]) -> list[int]:
    """Strip leading and trailing runs of silence-coded tokens."""
    n = len(tokens)
    i = 0
    while i < n and tokens[i] in silence_set:
        i += 1
    j = n
    while j > i and tokens[j - 1] in silence_set:
        j -= 1
    return tokens[i:j]


def silence_fraction(tokens: list[int], silence_set: frozenset[int]) -> float:
    if not tokens:
        return 0.0
    n_silence = sum(1 for t in tokens if t in silence_set)
    return n_silence / len(tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tmp/dataset.jsonl")
    p.add_argument("--output", default="tmp/dataset_filtered.jsonl")
    p.add_argument(
        "--apply_to",
        default="zh",
        help="comma-separated languages to filter. Other langs always pass through. "
             "Default 'zh' — only the AISHELL-3 problem is being addressed.",
    )
    p.add_argument(
        "--boundary_strip",
        action="store_true",
        default=True,
        help="strip leading/trailing silence runs before computing the "
             "silence fraction. Default ON. Use --no_boundary_strip to disable.",
    )
    p.add_argument("--no_boundary_strip", dest="boundary_strip", action="store_false")
    p.add_argument(
        "--max_silence_frac",
        type=float,
        default=0.05,
        help="drop rows whose post-strip silence fraction exceeds this. "
             "Default 0.05 (matches yue's natural ~2-3%% rate with headroom).",
    )
    p.add_argument(
        "--min_speech_tokens",
        type=int,
        default=8,
        help="drop rows shorter than this after the boundary strip. "
             "Matches InpaintDatasetConfig.min_speech_tokens.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="don't write output, just report the distribution.",
    )
    args = p.parse_args()

    apply_langs = {x.strip() for x in args.apply_to.split(",") if x.strip()}
    log.info(f"filtering applied to langs: {apply_langs}")
    log.info(f"boundary_strip: {args.boundary_strip}")
    log.info(f"max_silence_frac (post-strip): {args.max_silence_frac:.1%}")

    in_path = Path(args.input)
    if not in_path.exists():
        log.error(f"input not found: {in_path}")
        sys.exit(1)

    # Histograms: silence fraction in 10% buckets per language, pre and post strip.
    n_seen = Counter()
    n_kept = Counter()
    n_dropped_short = Counter()
    n_dropped_silence = Counter()
    n_dropped_other = Counter()
    silence_pre_buckets: dict[str, list[int]] = {}   # lang → 11 buckets [0-10%, 10-20%, ..., 100%]
    silence_post_buckets: dict[str, list[int]] = {}

    out_path = Path(args.output)
    out_f = None if args.dry_run else out_path.open("w")

    with in_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            lang = r.get("lang")
            tokens = r.get("speech_tokens", [])
            if not lang or not tokens:
                n_dropped_other[lang or "unknown"] += 1
                continue
            n_seen[lang] += 1
            silence_set = SILENCE_TOKEN_IDS.get(lang, frozenset())

            # Pre-strip stats (for distribution reporting).
            pre_frac = silence_fraction(tokens, silence_set)
            silence_pre_buckets.setdefault(lang, [0] * 11)
            silence_pre_buckets[lang][min(int(pre_frac * 10), 10)] += 1

            if lang in apply_langs:
                tokens_filtered = (
                    trim_boundary(tokens, silence_set)
                    if args.boundary_strip else tokens
                )
                post_frac = silence_fraction(tokens_filtered, silence_set)
                silence_post_buckets.setdefault(lang, [0] * 11)
                silence_post_buckets[lang][min(int(post_frac * 10), 10)] += 1

                if len(tokens_filtered) < args.min_speech_tokens:
                    n_dropped_short[lang] += 1
                    continue
                if post_frac > args.max_silence_frac:
                    n_dropped_silence[lang] += 1
                    continue
                # Write through with the boundary-stripped tokens
                # so the trainer doesn't have to repeat the work.
                r_out = dict(r)
                r_out["speech_tokens"] = tokens_filtered
                if out_f:
                    out_f.write(json.dumps(r_out, ensure_ascii=False) + "\n")
            else:
                # Pass-through for non-filtered languages
                if out_f:
                    out_f.write(line if line.endswith("\n") else line + "\n")
                silence_post_buckets.setdefault(lang, [0] * 11)
                silence_post_buckets[lang][min(int(pre_frac * 10), 10)] += 1

            n_kept[lang] += 1

    if out_f:
        out_f.close()

    log.info(f"\n=== INPUT ROWS PER LANGUAGE ===")
    for lang in sorted(n_seen.keys()):
        log.info(f"  {lang:>5s}: {n_seen[lang]:>8,d}")

    log.info(f"\n=== KEPT / DROPPED ===")
    for lang in sorted(n_seen.keys()):
        kept = n_kept[lang]
        dropped_s = n_dropped_silence[lang]
        dropped_sh = n_dropped_short[lang]
        seen = n_seen[lang]
        log.info(f"  {lang:>5s}: kept {kept:>7,d}/{seen:>7,d} "
                 f"({kept/max(1,seen):.1%})  "
                 f"dropped[silence>{args.max_silence_frac:.0%}]={dropped_s:>6,d}  "
                 f"dropped[too_short]={dropped_sh:>5,d}")

    log.info(f"\n=== silence fraction histogram (pre-strip) ===")
    log.info(f"  {'lang':>5s}  " +
             "  ".join(f"[{i*10}-{(i+1)*10}%]" for i in range(11)))
    for lang, buckets in silence_pre_buckets.items():
        log.info(f"  {lang:>5s}  " + "  ".join(f"{b:>8,d}" for b in buckets))

    if args.boundary_strip:
        log.info(f"\n=== silence fraction histogram (post-strip, on filtered langs) ===")
        log.info(f"  {'lang':>5s}  " +
                 "  ".join(f"[{i*10}-{(i+1)*10}%]" for i in range(11)))
        for lang, buckets in silence_post_buckets.items():
            log.info(f"  {lang:>5s}  " + "  ".join(f"{b:>8,d}" for b in buckets))

    log.info(f"\n=== OUTPUT ===")
    if args.dry_run:
        log.info("  (dry run — no output written)")
    else:
        log.info(f"  wrote {sum(n_kept.values()):,} rows → {out_path}")


if __name__ == "__main__":
    main()
