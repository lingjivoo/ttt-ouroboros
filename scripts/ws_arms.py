"""WebShop causal policies used in the paper.

``none``, ``uniform``, ``fixed`` and ``settlement`` map to Writes Off,
Closed Loop, Fixed Generation and task-reward Settlement, respectively.
Every arm records per-goal rewards, exact success and adapter drift.

    python scripts/ws_arms.py --policy settlement --episodes 900 --out ws_settlement.json
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch


def evaluate(eng, env, n, seed):
    """Per-goal reward vector, not just its mean: arms are paired by goal, and a
    mean alone cannot be paired."""
    env.seed(seed)
    rewards, bought, goal_ids = [], [], []
    for _ in range(n):
        obs, info = env.reset()
        goal_ids.append(env.last_goal_id)
        obs, acts = obs[0], info["admissible_commands"][0]
        goal, hist, r, done = env.env.instruction_text, [], 0.0, False
        for _ in range(env.max_steps):
            if not acts:
                break
            a = eng.act(build_prompt(goal, hist, obs, acts), acts)
            obs, r, done, info = env.step([a])
            obs, r, done, acts = obs[0], r[0], done[0], info["admissible_commands"][0]
            hist.append((a, obs))
            if done:
                break
        rewards.append(float(r))
        bought.append(1 if done else 0)
    return rewards, bought, goal_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy", required=True, choices=["none", "uniform", "fixed", "settlement"]
    )
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--episodes", type=int, default=900)
    ap.add_argument("--eval-episodes", type=int, default=150)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--max-actions", type=int, default=60)
    ap.add_argument(
        "--score-bs",
        type=int,
        default=8,
        help="candidates per scoring forward. WebShop pages are "
        "long enough that expanding the prompt cache to 60 "
        "copies overflows an 80 GiB card. Scores DO shift with "
        "the width -- up to 5.2e-02, measured -- but the argmax "
        "did not move on any step tested, so the action taken "
        "is unchanged. Keep this identical across arms: a "
        "comparison across two widths mixes in that shift.",
    )
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--settle-every", type=int, default=25)
    ap.add_argument("--settle-eval-episodes", type=int, default=10)
    ap.add_argument("--settle-seed", type=int, default=4321)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--stream-seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, "scripts")
    global build_prompt
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from webshop_agent import AgentTTT, build_prompt, run_episode  # noqa: F811
    from ws_env import WSEnv

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(dev)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    L = (
        getattr(model.config, "num_hidden_layers", None)
        or model.config.text_config.num_hidden_layers
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules=["down_proj"],
            layers_to_transform=list(range((3 * L) // 4, L)),
        ),
    )
    fast = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    lr = 0.0 if args.policy == "none" else args.lr
    eng = AgentTTT(model, tok, fast, lr, score_bs=args.score_bs)
    eng.reset()

    def state():
        return {
            "fast": [p.detach().clone() for p in fast],
            "opt": copy.deepcopy(eng.opt.state_dict()) if eng.opt is not None else None,
        }

    def restore(st):
        with torch.no_grad():
            for p, q in zip(fast, st["fast"]):
                p.copy_(q.to(dev))
        if eng.opt is not None and st["opt"] is not None:
            eng.opt.load_state_dict(copy.deepcopy(st["opt"]))

    stream = WSEnv(
        split="train", max_steps=args.max_steps, max_actions=args.max_actions
    )
    if args.stream_seed is not None:
        stream.seed(args.stream_seed)
    ev = WSEnv(split="test", max_steps=args.max_steps, max_actions=args.max_actions)

    res = {
        "status": "running",
        "policy": args.policy,
        "model": args.model,
        "episodes": [],
        "eval": [],
        "settlement": [],
        "stream_manifest": stream.manifest(),
        "evaluation_manifest": ev.manifest(),
    }
    ck = os.path.splitext(args.out)[0] + "_fast.pt"

    init = [p.detach().clone() for p in fast]
    committed = state()

    def drift():
        with torch.no_grad():
            return float(sum(((p - q) ** 2).sum() for p, q in zip(fast, init)) ** 0.5)

    def save():
        torch.save(
            {
                "fast": [p.detach().cpu() for p in fast],
                "opt": eng.opt.state_dict() if eng.opt is not None else None,
                "committed": {
                    "fast": [p.detach().cpu() for p in committed["fast"]],
                    "opt": committed["opt"],
                },
                "res": res,
                "policy": args.policy,
                "lr": lr,
            },
            ck,
        )

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
            if st.get("committed"):
                committed = st["committed"]
                committed["fast"] = [p.to(dev) for p in committed["fast"]]
            for _ in range(start):
                stream.reset()
            print(f"resuming {args.policy} at episode {start}", flush=True)

    if start == 0:
        rw, bt, goal_ids = evaluate(eng, ev, args.eval_episodes, args.eval_seed)
        res["eval"].append(
            {
                "step": 0,
                "reward": float(np.mean(rw)),
                "buy_rate": float(np.mean(bt)),
                "win_rate": float(np.mean([r >= 1.0 for r in rw])),
                "rewards": rw,
                "bought": bt,
                "goal_ids": goal_ids,
            }
        )
        print(f"  eval@0: reward {np.mean(rw):.3f} buy {np.mean(bt):.2f}", flush=True)

    for ep in range(start, args.episodes):
        if args.policy == "settlement":
            candidate = state()
            restore(committed)
            ok, samples, _ = run_episode(eng, stream, max_steps=args.max_steps)
            restore(candidate)
        else:
            eng.actor_frozen = args.policy == "fixed"
            ok, samples, _ = run_episode(eng, stream, max_steps=args.max_steps)
            eng.actor_frozen = False
        r = stream.last_score
        # `ok` from run_episode is the environment's done flag, which here means
        # a purchase happened, not that the right thing was bought.
        success = r >= 1.0
        wrote, n_w_steps = False, 0
        admit = args.policy in ("uniform", "fixed", "settlement")
        if admit:
            eng.write(samples)
            wrote, n_w_steps = True, len(samples)
        if args.policy == "settlement" and (ep + 1) % args.settle_every == 0:
            candidate = state()
            cand_rw, _, cand_goal_ids = evaluate(
                eng, ev, args.settle_eval_episodes, args.settle_seed
            )
            restore(committed)
            base_rw, _, base_goal_ids = evaluate(
                eng, ev, args.settle_eval_episodes, args.settle_seed
            )
            if cand_goal_ids != base_goal_ids:
                raise RuntimeError("Settlement candidate/base validation goals differ")
            keep = float(np.mean(cand_rw)) > float(np.mean(base_rw))
            if keep:
                restore(candidate)
                committed = state()
            res["settlement"].append(
                {
                    "step": ep + 1,
                    "candidate_reward": float(np.mean(cand_rw)),
                    "current_reward": float(np.mean(base_rw)),
                    "accepted": keep,
                    "goal_ids": cand_goal_ids,
                }
            )
        res["episodes"].append(
            {
                "ep": ep,
                "reward": float(r),
                "bought": bool(ok),
                "success": bool(success),
                "written": wrote,
                "n_steps": len(samples),
                "n_written": n_w_steps,
                "drift": drift(),
                "goal_id": stream.last_goal_id,
            }
        )
        if (ep + 1) % args.ckpt_every == 0:
            save()
            n_w = sum(1 for e in res["episodes"] if e["written"])
            print(
                f"  [{ep + 1}/{args.episodes}] wrote {n_w}  drift {drift():.5f}",
                flush=True,
            )

    rw, bt, goal_ids = evaluate(eng, ev, args.eval_episodes, args.eval_seed)
    res["eval"].append(
        {
            "step": args.episodes,
            "reward": float(np.mean(rw)),
            "buy_rate": float(np.mean(bt)),
            "win_rate": float(np.mean([r >= 1.0 for r in rw])),
            "rewards": rw,
            "bought": bt,
            "goal_ids": goal_ids,
        }
    )
    res["writes"] = sum(1 for e in res["episodes"] if e["written"])
    res["retained"] = (
        sum(int(x["accepted"]) for x in res["settlement"])
        if args.policy == "settlement"
        else res["writes"]
    )
    res["written_steps"] = sum(e["n_written"] for e in res["episodes"])
    res["final_drift"] = drift()
    res["status"] = "passed"
    save()
    json.dump(res, open(args.out, "w"), indent=1)
    a, b = res["eval"][0], res["eval"][-1]
    print(f"\n{args.policy}: {res['writes']} writes, drift {res['final_drift']:.5f}")
    print(
        f"  reward {a['reward']:.3f} -> {b['reward']:.3f}   "
        f"buy {a['buy_rate']:.3f} -> {b['buy_rate']:.3f}"
    )
    print("saved", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
