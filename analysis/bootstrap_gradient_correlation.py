#!/usr/bin/env python3
"""Cluster bootstrap for the recovered direction-prediction experiment.

Primary analysis defaults to confirmation/closed.  The predictor is the exact
gradient-direction cosine stored as `cosine_q_delta`; outcomes are realized NLL
changes under the original candidate update.  Books and seeds are resampled as
clusters while all measured stream positions remain attached to their cluster.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        r[order[i:j]] = (i + j - 1) / 2
        i = j
    return r


def corr(x, y, kind):
    if kind == "spearman":
        x, y = ranks(x), ranks(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def flatten(files):
    rows = []
    for p in files:
        d = json.loads(Path(p).read_text())
        m = d["manifest"]
        for block in d["records"]:
            pos = block["generated_position"]
            for r in block["rows"]:
                z = r["scores"]
                rows.append({
                    "seed": m["seed"], "book": r["book_index"], "position": pos,
                    "cosine": r["cosine_q_delta"],
                    "q_damage": z["original"]["q"] - z["zero"]["q"],
                    "near_damage": z["original"]["r_near"] - z["zero"]["r_near"],
                    "far_damage": z["original"]["r_far"] - z["zero"]["r_far"],
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--split", default="confirmation")
    ap.add_argument("--history", default="closed")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(Path(a.input).glob(f"{a.split}_{a.history}_s*.json"))
    rows = flatten(files)
    seeds = sorted({r["seed"] for r in rows})
    books = sorted({r["book"] for r in rows})
    rng = np.random.default_rng(a.seed)
    metrics = ("q_damage", "near_damage", "far_damage")
    point = {}
    boot = {m: {k: [] for k in ("pearson", "spearman")} for m in metrics}
    x = np.array([r["cosine"] for r in rows])
    for m in metrics:
        y = np.array([r[m] for r in rows])
        point[m] = {k: corr(x, y, k) for k in ("pearson", "spearman")}
    for _ in range(a.draws):
        ss = rng.choice(seeds, len(seeds), replace=True)
        bs = rng.choice(books, len(books), replace=True)
        sample = []
        # Duplicate selected clusters explicitly; each occurrence is a bootstrap unit.
        for si, s in enumerate(ss):
            for bi, b in enumerate(bs):
                sample.extend(r for r in rows if r["seed"] == s and r["book"] == b)
        xx = np.array([r["cosine"] for r in sample])
        for m in metrics:
            yy = np.array([r[m] for r in sample])
            for k in ("pearson", "spearman"):
                boot[m][k].append(corr(xx, yy, k))
    out = {"schema":"gradient-correlation-cluster-bootstrap-v1",
           "files":[str(p) for p in files], "split":a.split,"history":a.history,
           "n_records":len(rows),"n_seeds":len(seeds),"n_books":len(books),
           "positions":sorted({r["position"] for r in rows}),"draws":a.draws,
           "bootstrap_seed":a.seed,"cluster_units":["seed","book"],"results":{}}
    for m in metrics:
        out["results"][m] = {}
        for k in ("pearson", "spearman"):
            v = np.asarray(boot[m][k])
            v = v[np.isfinite(v)]
            out["results"][m][k] = {
                "estimate": point[m][k],
                "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
            }
    Path(a.out).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
