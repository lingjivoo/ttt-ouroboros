"""Paired analysis of the ALFWorld write-policy arms.

Every arm is scored on the same held-out games in the same order at every
checkpoint, and all arms start from the identical frozen model, so two
comparisons are paired and both matter:

  within-arm   eval@k vs eval@0  — what this policy's writes did to it
  between-arm  eval@k vs none@k  — what it did relative to not writing at all

Success rates alone hide this: a 0.375 -> 0.400 move is one episode, and only the
flips say whether that is a real gain or two unrelated episodes trading places.

Read the two columns separately, because they answer different questions and the
frozen arm settles which null applies. Its discordance measures what this task
flips from numerics alone, and in ALFWorld that is 0 -- argmax over a candidate
set absorbs the rounding that free-form generation amplifies (the MATH probe, by
contrast, flips 4 of 200). With a zero-noise control, ANY discordance in a writing
arm is a real behavioural change, so the discordant count is the evidence that
writes did something. The p-value asks the narrower question of whether they
helped or hurt, and against a zero-noise control it is conservative: McNemar's
null is that flips are symmetric, while the control says the true null is no flips
at all.

  python analysis/agent_stats.py parity/agent_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def flips(a, b):
    g = sum(1 for x, y in zip(a, b) if not x and y)
    l = sum(1 for x, y in zip(a, b) if x and not y)
    return g, l


def line(label, a, b):
    g, l = flips(a, b)
    p = mcnemar_exact(g, l)
    d = (sum(b) - sum(a)) / len(a)
    return (f"  {label:<30} {d:+.3f}  gained {g:>2}  lost {l:>2}  "
            f"discordant {g+l:>2}  p={p:.4f}{'  *' if p < 0.05 else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=None)
    args = ap.parse_args()

    arms = {}
    for f in (args.files or sorted(glob.glob("parity/agent_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "eval" in d and d["eval"] and "marks" in d["eval"][0]:
            arms[d["policy"]] = d

    if not arms:
        print("no arm results with per-episode marks yet")
        return

    n = len(next(iter(arms.values()))["eval"][0]["marks"])
    print(f"{len(arms)} arms, {n} held-out episodes per checkpoint\n")
    for pol, d in arms.items():
        ev = d["eval"]
        print(f"=== {pol}   " + "  ".join(f"@{e['step']}:{e['success']:.3f}" for e in ev))
        base = ev[0]["marks"]
        for e in ev[1:]:
            print(line(f"vs own eval@0 (@{e['step']})", base, e["marks"]))
        print()

    if "none" in arms and len(arms) > 1:
        print("=== against the frozen arm at the same checkpoint")
        ctrl = {e["step"]: e["marks"] for e in arms["none"]["eval"]}
        for pol, d in arms.items():
            if pol == "none":
                continue
            for e in d["eval"]:
                if e["step"] in ctrl and e["step"] > 0:
                    print(line(f"{pol} vs none @{e['step']}", ctrl[e["step"]], e["marks"]))

    # The stream is three times the eval block and costs nothing extra: every arm
    # walks the same games in the same order, so episode i is a matched pair across
    # arms. It measures the writing model as it is actually being written to, which
    # is the quantity the experiment is about; the eval blocks answer the narrower
    # question of what survives with writes switched off.
    if "none" in arms and len(arms) > 1:
        base = [int(e["success"]) for e in arms["none"]["episodes"]]
        if base:
            print("\n=== stream, paired per episode (same games in the same order)")
            for pol, d in arms.items():
                if pol == "none":
                    continue
                cur = [int(e["success"]) for e in d["episodes"]][:len(base)]
                if cur:
                    # n is per comparison: an arm still running has fewer episodes
                    # than the control, and labelling its result with the control's
                    # length reports a partial run as if it were complete.
                    print(line(f"{pol} vs none (n={len(cur)})", base[:len(cur)], cur))


if __name__ == "__main__":
    main()
