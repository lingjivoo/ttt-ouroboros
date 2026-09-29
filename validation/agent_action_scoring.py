"""KV-cache-shared action scoring must match a naive full-prefill rescore.

agent_ttt.AgentTTT.act prefills the prompt once and reuses its cache for every
admissible action; this checks that shortcut against scoring each action with a
full forward pass, on both the per-action scores and the resulting choice.

  python tests/test_agent_act.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.agent_ttt import DEV, AgentTTT  # noqa: E402

ACTIONS = ["go to cabinet 1", "go to countertop 1", "take potato 1 from countertop 1",
           "open fridge 1", "clean potato 1 with sinkbasin 1", "put potato 1 in/on garbagecan 1",
           "examine cabinet 21", "go to sinkbasin 1", "close microwave 1", "heat egg 1 with microwave 1"]

PROMPT = ("You are an agent in a household environment.\n\nTask: put a clean potato in "
          "garbagecan.\n\nYou are in the middle of a room. You see a cabinet 1, a countertop 1, "
          "a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1.\n\nNext action:")


@torch.no_grad()
def naive_scores(eng, prompt, actions):
    """Length-normalised logprob per action, one full forward each."""
    out = []
    for a in actions:
        full, n = eng._pair(prompt, a)
        lg = eng.m(full).logits[0, n - 1:-1].float()
        lp = torch.log_softmax(lg, -1).gather(-1, full[0, n:, None])[:, 0]
        out.append(float(lp.mean()))
    return out


def cached_scores(eng, prompt, actions):
    """The shared-prefix batched path act() actually uses."""
    return eng.scores(prompt, actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    # In float32 the two paths are the same arithmetic and agree to ~1e-5. In
    # bfloat16 they do not: cached keys/values go through different kernel shapes
    # and reduction orders, and per-token logprobs carry ~1e-2 of rounding either
    # way, so the loose bound is the honest one there. Correctness is established
    # in float32; the bfloat16 run only checks the choice survives that noise.
    ap.add_argument("--tol", type=float, default=None)
    args = ap.parse_args()
    tol = args.tol if args.tol is not None else (1e-4 if args.dtype == "float32" else 6e-2)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa").to(DEV).eval()
    eng = AgentTTT(model, tok, [], lr=0.0)

    naive = naive_scores(eng, PROMPT, ACTIONS)
    cached = cached_scores(eng, PROMPT, ACTIONS)
    worst = max(abs(a - b) for a, b in zip(naive, cached))
    for a, x, y in zip(ACTIONS, naive, cached):
        print(f"  {x:+.5f}  {y:+.5f}  |diff| {abs(x-y):.2e}  {a}")
    pick_n = ACTIONS[max(range(len(naive)), key=lambda i: naive[i])]
    pick_c = eng.act(PROMPT, ACTIONS)
    print(f"\n[{args.dtype}] max |diff| {worst:.3e} (tol {tol:.1e})   "
          f"naive pick: {pick_n!r}   act() pick: {pick_c!r}")

    assert pick_n == pick_c, "cached scoring changed the chosen action"
    assert worst < tol, f"score mismatch {worst:.3e} >= {tol:.1e}"
    print(f"agent act parity passed ({args.dtype})")


if __name__ == "__main__":
    main()
