"""The four write policies on ScienceWorld, reusing the ALFWorld harness.

Only the environment differs: AgentTTT, run_episode, the visited-state mask, the
write policies and the eval protocol are imported unchanged from agent_ttt, so a
difference between the two environments cannot come from the machinery.

ScienceWorld supplies its own train/test variation split (150/75), so the eval
set shares no variation with the training stream by construction -- ALFWorld
needed valid_seen/valid_unseen for the same guarantee.

Success is graded 0-100. We record both the binary completion rate and the mean
score fraction; the probe put the frozen agent at 0.50 score-fraction, and a
graded signal has more power per episode than ALFWorld's binary one.
"""

import argparse
import json
import os

import numpy as np
import torch

from agent_ttt import AgentTTT, run_episode
from sw_env import SWBatchEnv


def evaluate(eng, env, n, seed):
    # Return the per-episode vector, not just its mean: binary completion is
    # zero throughout this environment, so the graded score is the only signal,
    # and a mean alone cannot be paired against another arm.
    env.seed(seed)
    marks, fracs = [], []
    for _ in range(n):
        ok, _, _ = run_episode(eng, env, max_steps=env.env.envStepLimit)
        marks.append(int(ok))
        fracs.append(env.last_score / 100.0)
    return marks, fracs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    choices=["none", "uniform", "verified", "random", "scored"])
    # On the original task (find-non-living-thing) binary completion was zero in
    # every episode, so `verified` wrote nothing and coincided with `none` by
    # construction; that is why this experiment moved to
    # inclined-plane-friction-named-surfaces, where completion is non-zero and
    # `verified` writes on 15 of 150 episodes. `scored` remains the graded
    # alternative -- write when the episode clears an absolute score bar. The bar is calibrated on the FROZEN arm's
    # stream scores, which are already on disk, so it never peeks at the arm's
    # own trajectory. The scores are coarse ({-100, 58, 67, 75}): 75 admits
    # 21/150 and 67 admits 76/150 on the frozen stream.
    ap.add_argument("--score-thr", type=float, default=75.0)
    ap.add_argument("--task", default="find-non-living-thing")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--eval-episodes", type=int, default=75)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--max-actions", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--write-frac", type=float, default=0.25,
                    help="rate for `random`. Prefer --write-k: a rate has to be "
                         "guessed before the run and the guess was wrong -- "
                         "0.38 was taken from an expected frozen win rate while "
                         "`verified` actually wrote 15/150, so the arm meant to "
                         "be dose-matched wrote three times as often.")
    ap.add_argument("--write-k", type=int, default=-1,
                    help="for `random`: write on exactly this many episodes, "
                         "drawn without replacement. Matches a measured count "
                         "instead of an assumed rate.")
    ap.add_argument("--write-seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--stream-seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import random as _r

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
        target_modules=["down_proj"],
        layers_to_transform=list(range((3 * L) // 4, L))))
    fast = [p for n, p in model.named_parameters()
            if "lora_" in n and p.requires_grad]
    lr = 0.0 if args.policy == "none" else args.lr
    eng = AgentTTT(model, tok, fast, lr)
    eng.reset()

    stream = SWBatchEnv(task=args.task, split="train",
                        max_episode_steps=args.max_steps,
                        max_actions=args.max_actions)
    if args.stream_seed is not None:
        stream.seed(args.stream_seed)
    ev = SWBatchEnv(task=args.task, split="test",
                    max_episode_steps=args.max_steps,
                    max_actions=args.max_actions)

    res = {"policy": args.policy, "task": args.task, "model": args.model,
           "episodes": [], "eval": []}
    ck = os.path.splitext(args.out)[0] + "_fast.pt"

    def save():
        torch.save({"fast": [p.detach().cpu() for p in fast],
                    "opt": eng.opt.state_dict() if eng.opt is not None else None,
                    "res": res, "policy": args.policy, "lr": lr}, ck)

    start = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cpu", weights_only=False)
        if st.get("res"):
            res = st["res"]
            start = len(res["episodes"])
            with torch.no_grad():
                for p, q in zip(fast, st["fast"]):
                    p.copy_(q.to(dev))
            if st.get("opt") and eng.opt is not None:
                eng.opt.load_state_dict(st["opt"])
            for _ in range(start):
                stream.reset()
            print(f"resuming {args.policy} at episode {start}", flush=True)

    if start == 0:
        m, fr = evaluate(eng, ev, args.eval_episodes, args.eval_seed)
        res["eval"].append({"step": 0, "success": float(np.mean(m)),
                            "score_frac": float(np.mean(fr)),
                            "fracs": fr, "marks": m})
        print(f"  eval@0: success {np.mean(m):.3f} "
              f"score-frac {np.mean(fr):.3f}", flush=True)
        save()

    wr = _r.Random(args.write_seed)
    kset = set()
    if args.policy == "random" and args.write_k >= 0:
        kset = set(_r.Random(args.write_seed).sample(range(args.episodes),
                                                     args.write_k))
        print(f"random arm writes on exactly {len(kset)} episodes", flush=True)

    # Distance of the adapter from its initial value. Without it, "the writes
    # changed nothing measurable" and "the writes never landed" produce the same
    # evidence: the evaluation only sees the discrete actions, so a small weight
    # change that flips no argmax is invisible. That ambiguity is what made the
    # first run of this experiment uninterpretable.
    init = [p.detach().clone() for p in fast]

    def adapter_drift():
        with torch.no_grad():
            return float(sum(((p - q) ** 2).sum() for p, q in zip(fast, init)) ** 0.5)

    for ep in range(start, args.episodes):
        ok, samples, _ = run_episode(eng, stream, max_steps=args.max_steps)
        wrote = False
        if args.policy == "uniform":
            eng.write(samples); wrote = True
        elif args.policy == "verified" and ok:
            eng.write(samples); wrote = True
        elif args.policy == "random" and (
                ep in kset if kset else wr.random() < args.write_frac):
            eng.write(samples); wrote = True
        elif args.policy == "scored" and stream.last_score >= args.score_thr:
            eng.write(samples); wrote = True
        res["episodes"].append({"ep": ep, "success": bool(ok),
                                "score": stream.last_score, "written": wrote,
                                "drift": adapter_drift()})
        if (ep + 1) % args.ckpt_every == 0:
            save()
            print(f"  [{ep+1}/{args.episodes}] wrote so far "
                  f"{sum(1 for e in res['episodes'] if e['written'])}", flush=True)

    m, fr = evaluate(eng, ev, args.eval_episodes, args.eval_seed)
    res["eval"].append({"step": args.episodes, "success": float(np.mean(m)),
                        "score_frac": float(np.mean(fr)),
                        "fracs": fr, "marks": m})
    print(f"  eval@{args.episodes}: success {np.mean(m):.3f} "
          f"score-frac {np.mean(fr):.3f}", flush=True)
    save()
    json.dump(res, open(args.out, "w"), indent=1)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
