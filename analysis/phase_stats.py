"""Stable vs slowly-unstable cells, judged by terminal slope at 128K.

A terminal NLL cannot tell a stable cell from one that is drifting up slowly --
that was the reviewer's objection to the 64K phase diagram. Over 128 chunks the
late-window slope can: a stable cell's probe curve is flat at the end, an
unstable one is still climbing. Each cell is reported against the w=0 baseline
run at the SAME lambda, so decay is not confounded with write strength.

This is the criterion promised in REVIEW_RESPONSE.md (P3/P4): "terminal slope of
d_t over the final third, relative to the matched control's own drift". The runs
have 15 probes, so the default --tail 5 IS that final third, and fitting the
paired difference is what makes it relative to the control's own drift.

  python analysis/phase_stats.py --dir parity
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re


def probes(d):
    """[(chunk, nll)] from the single run stored in a horizon json."""
    (run,) = [v for v in d.values() if isinstance(v, dict) and "probes" in v]
    return [(int(c), float(v)) for c, v in run["probes"]]


def slope(pts):
    """Least-squares change per chunk, with the standard error of that slope.

    Probe NLL bounces ~0.1 between adjacent chunks purely from text difficulty,
    which swamps a real drift if you fit the raw curve. Callers fit the paired
    difference against the same-lambda w=0 run instead, where that noise cancels,
    and the residual SE says whether the remaining slope is meaningful.
    """
    n = len(pts)
    mx = sum(c for c, _ in pts) / n
    my = sum(v for _, v in pts) / n
    den = sum((c - mx) ** 2 for c, _ in pts)
    if den == 0 or n < 3:
        return 0.0, float("inf")
    b = sum((c - mx) * (v - my) for c, v in pts) / den
    a = my - b * mx
    rss = sum((v - (a + b * c)) ** 2 for c, v in pts)
    return b, (rss / (n - 2) / den) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="parity")
    ap.add_argument("--tail", type=int, default=5, help="probes in the late window")
    ap.add_argument("--dep", type=float, default=0.10,
                    help="NLL above the paired w=0 run that counts as departed")
    args = ap.parse_args()

    base = {}
    for f in glob.glob(os.path.join(args.dir, "ph2_w0_lam*.json")):
        lam = re.search(r"lam([\d.]+)\.json", f).group(1)
        base[lam] = probes(json.load(open(f)))

    print(f"{'lam':>5} {'w':>5} {'NLL_0':>7} {'NLL_end':>8} {'delta':>7} "
          f"{'vs w=0':>7} {'tail slope':>11} {'+/- SE':>9} {'t':>6} {'verdict':>9}")
    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "ph2_lam*_w*.json"))):
        m = re.search(r"ph2_lam([\d.]+)_w([\d.]+)\.json", f)
        if not m:
            continue
        rows.append((m.group(1), m.group(2), probes(json.load(open(f)))))
    for lam, p in sorted(base.items()):
        rows.append((lam, "0", p))

    for lam, w, pts in sorted(rows, key=lambda r: (-float(r[0]), float(r[1]))):
        d = pts[-1][1] - pts[0][1]
        b = base.get(lam)
        if b is None:
            print(f"{lam:>5} {w:>5} {pts[0][1]:>7.3f} {pts[-1][1]:>8.3f} {d:>+7.3f} "
                  f"{'--':>7} {'--':>11} {'--':>9} {'--':>6} {'no base':>9}")
            continue
        bm = dict(b)
        diff = [(c, v - bm[c]) for c, v in pts if c in bm]
        rel = diff[-1][1] - diff[0][1]
        s, se = slope(diff[-args.tail:])
        t = s / se if se else 0.0
        # Two ways to be in the unstable phase, and the slope alone catches only
        # one: a cell that already blew up saturates, so its tail is flat. Departure
        # from the write-free run is the primary call; a still-climbing tail is what
        # catches instability too young to have accumulated -- the case a 64K
        # terminal value cannot see.
        departed = rel > args.dep
        verdict = ("UNSTABLE" if departed else
                   "incipient" if (t > 2 and s > 1e-3) else "stable")
        print(f"{lam:>5} {w:>5} {pts[0][1]:>7.3f} {pts[-1][1]:>8.3f} {d:>+7.3f} "
              f"{rel:>+7.3f} {s:>+11.5f} {se:>9.5f} {t:>+6.1f} {verdict:>9}")

    print(f"\ntail slope fits the PAIRED difference (cell minus same-lambda w=0 run) "
          f"over the last {args.tail} probes, so text-difficulty noise cancels; "
          f"t = slope / SE.")


if __name__ == "__main__":
    main()
