#!/usr/bin/env python3
"""Paired seed/goal bootstrap for the formal WebShop JSON files."""
import argparse
import json
from pathlib import Path

import numpy as np


def load_final(path):
    d = json.loads(Path(path).read_text())
    ev = d["eval"][-1]
    rewards = np.asarray(ev["rewards"], dtype=float)
    wins = np.asarray(ev.get("wins", rewards >= 1.0), dtype=float)
    goal_ids = ev.get("goal_ids")
    return rewards, wins, goal_ids


def ci(x):
    return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="directory containing five seed JSONs")
    ap.add_argument(
        "--control",
        required=True,
        help="Writes Off JSON; its paired goal vector is shared",
    )
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(Path(a.arm).glob(a.pattern))
    if len(files) != 5:
        raise SystemExit(f"expected five seed files, found {len(files)}")
    control_r, control_w, control_ids = load_final(a.control)
    arm = [load_final(p) for p in files]
    if any(len(r) != len(control_r) for r, _, _ in arm):
        raise SystemExit("goal-vector length mismatch")
    for _, _, ids in arm:
        if control_ids is not None and ids is not None and ids != control_ids:
            raise SystemExit("goal identity/order mismatch")
    dr = np.stack([r - control_r for r, _, _ in arm])
    dw = np.stack([w - control_w for _, w, _ in arm])
    rng = np.random.default_rng(a.seed)
    nr = np.empty(a.draws)
    nw = np.empty(a.draws)
    S, G = dr.shape
    for b in range(a.draws):
        si = rng.integers(0, S, S)
        gi = rng.integers(0, G, G)
        nr[b] = dr[si][:, gi].mean()
        nw[b] = dw[si][:, gi].mean()
    out = {
        "schema": "webshop-paired-hierarchical-bootstrap-v1",
        "arm_files": [str(p) for p in files],
        "control_file": a.control,
        "independent_seed_count": S,
        "paired_goal_count": G,
        "goal_ids_verified": control_ids is not None and all(
            ids == control_ids for _, _, ids in arm
        ),
        "draws": a.draws,
        "bootstrap_seed": a.seed,
        "reward_difference": {"estimate": float(dr.mean()), "ci95": ci(nr)},
        "win_rate_difference": {"estimate": float(dw.mean()), "ci95": ci(nw)},
        "per_seed_reward_difference": dr.mean(1).tolist(),
        "per_seed_win_rate_difference": dw.mean(1).tolist(),
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
