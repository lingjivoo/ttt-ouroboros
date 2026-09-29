"""Feasibility canary for ScienceWorld as a second agent environment.

TWC was abandoned after sixty episodes at zero wins, and the cost of that was
building a full adapter before measuring whether a frozen 4B agent could score
at all. So this measures first and builds nothing: it runs the existing scoring
engine over ScienceWorld's raw action list on a handful of variations of the
easiest-looking tasks, and reports the two numbers that decide the question --
the frozen score fraction, and seconds per episode.

ScienceWorld exposes ~115 valid actions per step against ALFWorld's 15-40, and
the harness scores every candidate with a forward pass, so the per-step cost is
the thing most likely to make this impractical even if the baseline is usable.
"""
import argparse
import json
import time

import numpy as np
import torch

from agent_ttt import AgentTTT, build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="",
                help="comma-separated; empty screens every task the "
                     "installation exposes")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--max-steps", type=int, default=50)
ap.add_argument("--max-actions", type=int, default=60,
                help="cap on candidates scored per step; ScienceWorld offers "
                     "~115 and the engine scores each with a forward pass")
ap.add_argument("--model", default="Qwen/Qwen3-4B")
ap.add_argument("--out", required=True)
args = ap.parse_args()

from scienceworld import ScienceWorldEnv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

dev = "cuda"
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
L = getattr(model.config, "num_hidden_layers", None) or \
    model.config.text_config.num_hidden_layers
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
    target_modules=["down_proj"], layers_to_transform=list(range((3 * L) // 4, L))))
fast = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
eng = AgentTTT(model, tok, fast, 0.0)   # frozen
eng.reset()

env = ScienceWorldEnv("", envStepLimit=args.max_steps)
tasks = [t for t in args.tasks.split(",") if t] or sorted(env.get_task_names())
print(f"screening {len(tasks)} tasks x {args.n} variations\n", flush=True)
res = {}
for task in tasks:
    scores, times, steps_used, wins, raw = [], [], [], [], []
    for v in range(args.n):
        env.load(task, v, "easy")
        obs, info = env.reset()
        goal = env.get_task_description()
        hist, t0 = [], time.time()
        score = 0
        for step in range(args.max_steps):
            adm = info.get("valid", [])[:args.max_actions]
            if not adm:
                break
            prompt = build_prompt(goal, hist, obs, adm)
            action = eng.act(prompt, adm)
            obs, r, done, info = env.step(action)
            score = max(score, info.get("score", 0))
            hist.append((action, obs))
            if done:
                break
        dt = time.time() - t0
        scores.append(max(0, score) / 100.0)
        # Binary completion, not just the graded score. This is the number that
        # decides whether an outcome-verified write policy has anything to
        # write: on find-non-living-thing it is zero in every episode, so that
        # arm wrote nothing and coincided with the frozen arm by construction.
        # A task is only usable as a write-authority testbed if it is well
        # inside (0, 1).
        wins.append(1 if score >= 100 else 0)
        raw.append(max(0, score))
        times.append(dt)
        steps_used.append(step + 1)
        print(f"  {task} v{v}: score={score} steps={step+1} {dt:.0f}s", flush=True)
    # Distinct score levels, the criterion the earlier screen was missing.
    # inclined-plane-friction-named-surfaces was chosen for its non-zero win
    # rate and turned out to score only +100 or -100, so its "graded" metric was
    # binary with a 12% base rate -- weaker than the ALFWorld signal it was
    # meant to improve on, and far too coarse to separate two arms that write 15
    # times each. A usable task needs BOTH a win rate inside (0, 1) AND more
    # than two score levels.
    levels = sorted(set(raw))
    res[task] = {"win_rate": float(np.mean(wins)),
                 "score_frac": float(np.mean(scores)),
                 "score_std": float(np.std(scores)),
                 "score_levels": len(levels),
                 "levels": levels[:12],
                 "usable": bool(0.0 < np.mean(wins) < 1.0 and len(levels) > 2),
                 "sec_per_episode": float(np.mean(times)),
                 "mean_steps": float(np.mean(steps_used))}
    r = res[task]
    print(f"== {task}: WIN {r['win_rate']:.2f}  "
          f"score-frac {r['score_frac']:.2f}+-{r['score_std']:.2f}  "
          f"levels {r['score_levels']} {r['levels']}  "
          f"{r['sec_per_episode']:.0f}s/ep  {r['mean_steps']:.0f} steps  "
          f"{'USABLE' if r['usable'] else '--'}", flush=True)

json.dump(res, open(args.out, "w"), indent=1)
ok = [t for t, r in res.items() if r["usable"]]
print(f"\n{len(ok)} of {len(res)} tasks have a win rate in (0,1) AND >2 score "
      f"levels:", flush=True)
for t in sorted(ok, key=lambda t: -res[t]["score_levels"]):
    r = res[t]
    print(f"   {t:46s} win {r['win_rate']:.2f}  levels {r['score_levels']:2d}  "
          f"{r['sec_per_episode']:.0f}s/ep", flush=True)
print("saved", args.out, flush=True)
