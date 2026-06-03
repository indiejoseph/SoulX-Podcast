"""Re-audit zh silence-token set against tmp/dataset.jsonl.

Goal: surface s3 ids that behave like silence (heavy boundary presence
or heavy long-run usage) but are missing from
`SILENCE_TOKEN_IDS["zh"]`. Two of the overfit's still-collapsed rows
landed on s3 ids 3887 / 4523 — neither is in the current set, so the
filter let them through.

Outputs three rankings:
  * BOUNDARY  — token's share at the first/last 10 positions divided
                by its global unigram share. >=2× = boundary-enriched.
  * RUN       — max sustained-run length per row (P95 across rows).
                Silence-like ids show 10–30 token uninterrupted runs.
  * FREQUENCY — raw global frequency in zh; useful sanity check.

Then prints **proposed additions**: ids meeting BOTH "boundary share
≥0.1%" AND "boundary enrichment ≥2×" thresholds that are NOT already
in `SILENCE_TOKEN_IDS["zh"]`. Conservative — only adds tokens the data
clearly flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soulxpodcast.training.inpaint_dataset import SILENCE_TOKEN_IDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="tmp/dataset.jsonl")
    p.add_argument("--lang", default="zh")
    p.add_argument("--boundary_window", type=int, default=10,
                   help="how many positions count as 'boundary'")
    p.add_argument("--top_k", type=int, default=40,
                   help="rank length per ranking")
    p.add_argument("--min_boundary_share", type=float, default=0.001,
                   help="min frac of all boundary positions for inclusion")
    p.add_argument("--min_enrichment", type=float, default=2.0,
                   help="boundary_share / global_share ratio threshold")
    args = p.parse_args()

    # Global unigram + boundary unigram + max-run distribution
    global_count: Counter[int] = Counter()
    boundary_count: Counter[int] = Counter()
    max_runs: dict[int, list[int]] = defaultdict(list)
    n_rows = 0
    n_total_tokens = 0
    n_total_boundary = 0

    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r["lang"] != args.lang:
                continue
            tokens = r["speech_tokens"]
            if not tokens:
                continue
            n_rows += 1
            n_total_tokens += len(tokens)
            for t in tokens:
                global_count[t] += 1

            head = tokens[: args.boundary_window]
            tail = tokens[-args.boundary_window:]
            n_total_boundary += len(head) + len(tail)
            for t in head:
                boundary_count[t] += 1
            for t in tail:
                boundary_count[t] += 1

            # Max-run length per token in this row
            row_max_run: dict[int, int] = {}
            cur_t, cur_n = tokens[0], 1
            for t in tokens[1:]:
                if t == cur_t:
                    cur_n += 1
                else:
                    row_max_run[cur_t] = max(row_max_run.get(cur_t, 0), cur_n)
                    cur_t, cur_n = t, 1
            row_max_run[cur_t] = max(row_max_run.get(cur_t, 0), cur_n)
            for t, n in row_max_run.items():
                max_runs[t].append(n)

    print(f"[{args.lang}] {n_rows:,} rows, {n_total_tokens:,} tokens, "
          f"{n_total_boundary:,} boundary positions "
          f"(window={args.boundary_window})")
    current = SILENCE_TOKEN_IDS.get(args.lang, frozenset())
    print(f"current SILENCE_TOKEN_IDS[{args.lang}]: {len(current)} ids")

    # Build the table
    rows = []
    for t, b in boundary_count.items():
        g = global_count[t]
        b_share = b / n_total_boundary
        g_share = g / n_total_tokens
        enrich = (b_share / g_share) if g_share > 0 else 0.0
        runs = max_runs.get(t, [])
        runs_sorted = sorted(runs, reverse=True)
        p95_run = runs_sorted[len(runs_sorted) // 20] if runs_sorted else 0
        rows.append(dict(
            token=t, count=g, b_share=b_share, g_share=g_share,
            enrichment=enrich, p95_run=p95_run, n_rows_with_run=len(runs),
        ))

    # === Ranking 1: boundary share ===
    print(f"\n=== top {args.top_k} by BOUNDARY SHARE ===")
    print(f"{'tok':>6s}  {'in_set':>6s}  {'count':>10s}  {'b_share':>8s}  "
          f"{'g_share':>8s}  {'enrich':>7s}  {'p95_run':>8s}")
    for r in sorted(rows, key=lambda r: r["b_share"], reverse=True)[: args.top_k]:
        marker = " IN " if r["token"] in current else "  - "
        print(f"  {r['token']:>4d}  {marker:>6s}  {r['count']:>10,d}  "
              f"{r['b_share']:>7.2%}  {r['g_share']:>7.2%}  "
              f"{r['enrichment']:>6.1f}x  {r['p95_run']:>8d}")

    # === Ranking 2: long-run (silence-like sustained) ===
    print(f"\n=== top {args.top_k} by P95 RUN LENGTH ===")
    print(f"{'tok':>6s}  {'in_set':>6s}  {'p95_run':>8s}  {'rows':>8s}  "
          f"{'count':>10s}  {'g_share':>8s}")
    for r in sorted(rows, key=lambda r: r["p95_run"], reverse=True)[: args.top_k]:
        marker = " IN " if r["token"] in current else "  - "
        print(f"  {r['token']:>4d}  {marker:>6s}  {r['p95_run']:>8d}  "
              f"{r['n_rows_with_run']:>8,d}  {r['count']:>10,d}  "
              f"{r['g_share']:>7.2%}")

    # === Proposed additions ===
    proposed = []
    for r in rows:
        if r["token"] in current:
            continue
        if r["b_share"] < args.min_boundary_share:
            continue
        if r["enrichment"] < args.min_enrichment:
            continue
        proposed.append(r)
    proposed.sort(key=lambda r: r["b_share"], reverse=True)

    # Also check the two ids the overfit flagged
    overfit_flagged = [3887, 4523]
    print(f"\n=== overfit-flagged ids ===")
    for tid in overfit_flagged:
        r = next((r for r in rows if r["token"] == tid), None)
        if r is None:
            print(f"  {tid}: not present in {args.lang} corpus")
        else:
            marker = "IN" if tid in current else "MISSING"
            in_proposed = "→ PROPOSED" if any(p["token"] == tid for p in proposed) else ""
            print(f"  {tid}: {marker}  b_share={r['b_share']:.2%} "
                  f"enrich={r['enrichment']:.1f}x  p95_run={r['p95_run']} "
                  f"count={r['count']:,d}  {in_proposed}")

    print(f"\n=== PROPOSED ADDITIONS "
          f"(boundary≥{args.min_boundary_share:.1%}, enrich≥{args.min_enrichment}x) ===")
    print(f"{len(proposed)} new ids")
    cur_list = sorted(current)
    add_list = sorted(p["token"] for p in proposed)
    union = sorted(set(cur_list) | set(add_list))
    print(f"\nUNION ({len(union)} ids): {union}")
    print(f"\nADDED ({len(add_list)}):  {add_list}")


if __name__ == "__main__":
    main()
