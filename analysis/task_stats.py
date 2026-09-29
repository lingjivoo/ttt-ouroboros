"""Paired analysis of task-stream probe results.

The probe set is re-asked at every checkpoint, so start-vs-end is a paired
comparison, not two independent proportions. This reports the McNemar exact
test on the problems that flipped, which is what makes a null here meaningful.

  python analysis/task_stats.py parity/task2_algebra_*.json
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations


def mcnemar_exact(b, c):
    """Two-sided exact p for b gains vs c losses under H0: flips are 50/50."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--all-pairs", action="store_true",
                    help="compare every checkpoint pair, not just first vs last")
    args = ap.parse_args()

    for f in args.files:
        d = json.load(open(f))
        pr = d["probe"]
        if not all("marks" in p for p in pr):
            print(f"{f}: no per-problem marks (run predates paired logging)")
            continue
        n = len(pr[0]["marks"])
        print(f"=== {d['policy']}  ({d['model']}, n={n} probe problems)")
        print("    acc: " + "  ".join(f"{p['step']}:{p['acc']:.3f}" for p in pr))
        pairs = combinations(range(len(pr)), 2) if args.all_pairs else [(0, len(pr) - 1)]
        for i, j in pairs:
            a, b_ = pr[i], pr[j]
            gains = sum(1 for x, y in zip(a["marks"], b_["marks"]) if not x and y)
            losses = sum(1 for x, y in zip(a["marks"], b_["marks"]) if x and not y)
            p = mcnemar_exact(gains, losses)
            print(f"    step {a['step']:>3} -> {b_['step']:<3}  "
                  f"{b_['acc']-a['acc']:+.3f}  gained {gains:>3}  lost {losses:>3}  "
                  f"discordant {gains+losses:>3}  McNemar p={p:.4f}"
                  f"{'  *' if p < 0.05 else ''}")
        print()


if __name__ == "__main__":
    main()
