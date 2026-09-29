"""Write policies on WebShop, reusing the ALFWorld/ScienceWorld harness.

Only the environment differs: AgentTTT, run_episode, the write policies and the
eval protocol are shared, so a difference between environments cannot come from
the machinery.

WebShop was chosen after two ScienceWorld tasks each failed one of the two
properties this experiment needs. find-non-living-thing scores in {58, 67, 75}
but never reaches a binary win, so an outcome-verified arm writes nothing and
coincides with the frozen arm by construction. inclined-plane-friction-named-
surfaces wins on 10% of episodes but scores only +-100, leaving a binary metric
at a 12% base rate over 75 variations of one task. Measured on 40 episodes,
WebShop gives a 0.30 completion rate, seven distinct reward levels, and 2,418
held-out goals that are independent shopping instructions rather than variations
of one task.

Three things this file does that the ScienceWorld version did not, each because
its absence made that experiment uninterpretable:

  --write-k matches the control's dose to a MEASURED count. Setting a rate in
  advance put the random arm at 47 writes against verified's 15 -- three times
  the dose of the arm it was controlling for.

  Adapter drift is recorded every episode. Without it, "the writes were too
  small to change any decision" and "the writes never landed" produce identical
  evidence, because the evaluation only sees discrete actions.

  Both success definitions are recorded. A purchase (reward > 0 and done) and a
  full match (reward == 1) are different events, and which one an
  outcome-verified policy keys on changes how much it writes.

    python scripts/ws_arms.py --policy verified --out ws_verified.json
"""
import argparse
import json
import os
import random as _r
import sys
import time

import numpy as np
import torch


def evaluate(eng, env, n, seed):
    """Per-goal reward vector, not just its mean: arms are paired by goal, and a
    mean alone cannot be paired."""
    env.seed(seed)
    rewards, bought = [], []
    for _ in range(n):
        obs, info = env.reset()
        obs, acts = obs[0], info['admissible_commands'][0]
        goal, hist, r, done = env.env.instruction_text, [], 0.0, False
        for _ in range(env.max_steps):
            if not acts:
                break
            a = eng.act(build_prompt(goal, hist, obs, acts), acts)
            obs, r, done, info = env.step([a])
            obs, r, done, acts = obs[0], r[0], done[0], info['admissible_commands'][0]
            hist.append((a, obs))
            if done:
                break
        rewards.append(float(r))
        bought.append(1 if done else 0)
    return rewards, bought


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    choices=["none", "uniform", "verified", "random", "scored"])
    ap.add_argument("--success", default="full", choices=["full", "bought"],
                    help="what `verified` counts as an outcome. `full` is "
                         "reward 1.0, the unambiguous success; `bought` is any "
                         "completed purchase, which is completion rather than "
                         "success and writes about three times as often.")
    ap.add_argument("--score-thr", type=float, default=0.5,
                    help="bar for `scored`, the graded alternative to a binary "
                         "outcome")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--eval-episodes", type=int, default=150)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--max-actions", type=int, default=60)
    ap.add_argument("--score-bs", type=int, default=8,
                    help="candidates per scoring forward. WebShop pages are "
                         "long enough that expanding the prompt cache to 60 "
                         "copies overflows an 80 GiB card. Scores DO shift with "
                         "the width -- up to 5.2e-02, measured -- but the argmax "
                         "did not move on any step tested, so the action taken "
                         "is unchanged. Keep this identical across arms: a "
                         "comparison across two widths mixes in that shift.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--write-k", type=int, default=-1,
                    help="for `random`: write on exactly this many episodes, "
                         "drawn without replacement. Matches a measured count "
                         "rather than a rate assumed before the run.")
    ap.add_argument("--write-frac", type=float, default=0.25)
    ap.add_argument("--write-steps", default="all",
                    choices=["all", "tail", "head"],
                    help="which steps of an admitted episode to write. The "
                         "paper's claim is that a trajectory-level success bit "
                         "can authorise steps that do not deserve it; this "
                         "splits the trajectory to test it. WebShop episodes "
                         "run search -> click product -> options -> Buy Now, "
                         "and the decomposition says the gain is a buy-format "
                         "dividend while the damage is in the choice. So `tail` "
                         "(the last --tail-k steps) should carry most of the "
                         "gain at a fraction of the writes, and `head` "
                         "(everything before them) should carry the damage. "
                         "Episode dose is held fixed across the three; only "
                         "which steps within an admitted episode are written "
                         "changes, and both counts are recorded."),
    ap.add_argument("--tail-k", type=int, default=2)
    ap.add_argument("--write-seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--stream-seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, "scripts")
    global build_prompt
    from agent_ttt import AgentTTT, run_episode, build_prompt  # noqa: F811
    from ws_env import WSEnv
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
        target_modules=["down_proj"],
        layers_to_transform=list(range((3 * L) // 4, L))))
    fast = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    lr = 0.0 if args.policy == "none" else args.lr
    eng = AgentTTT(model, tok, fast, lr, score_bs=args.score_bs)
    eng.reset()

    stream = WSEnv(split="train", max_steps=args.max_steps,
                   max_actions=args.max_actions)
    if args.stream_seed is not None:
        stream.seed(args.stream_seed)
    ev = WSEnv(split="test", max_steps=args.max_steps,
               max_actions=args.max_actions)

    res = {"policy": args.policy, "success": args.success, "model": args.model,
           "episodes": [], "eval": []}
    ck = os.path.splitext(args.out)[0] + "_fast.pt"

    init = [p.detach().clone() for p in fast]

    def drift():
        with torch.no_grad():
            return float(sum(((p - q) ** 2).sum()
                             for p, q in zip(fast, init)) ** 0.5)

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
        rw, bt = evaluate(eng, ev, args.eval_episodes, args.eval_seed)
        res["eval"].append({"step": 0, "reward": float(np.mean(rw)),
                            "buy_rate": float(np.mean(bt)),
                            "win_rate": float(np.mean([r >= 1.0 for r in rw])),
                            "rewards": rw, "bought": bt})
        print(f"  eval@0: reward {np.mean(rw):.3f} buy {np.mean(bt):.2f}",
              flush=True)

    kset = set()
    if args.policy == "random" and args.write_k >= 0:
        kset = set(_r.Random(args.write_seed).sample(range(args.episodes),
                                                     args.write_k))
        print(f"random arm writes on exactly {len(kset)} episodes", flush=True)
    wr = _r.Random(args.write_seed)

    for ep in range(start, args.episodes):
        ok, samples, _ = run_episode(eng, stream, max_steps=args.max_steps)
        r = stream.last_score
        # `ok` from run_episode is the environment's done flag, which here means
        # a purchase happened, not that the right thing was bought.
        success = (r >= 1.0) if args.success == "full" else bool(ok)
        def sel(sam):
            """Which steps of an admitted episode actually get written."""
            if args.write_steps == "tail":
                return sam[-args.tail_k:]
            if args.write_steps == "head":
                # everything the tail arm does not write; an episode shorter
                # than the split writes nothing rather than overlapping it
                return sam[:-args.tail_k] if len(sam) > args.tail_k else []
            return sam

        wrote, n_w_steps = False, 0
        admit = (args.policy == "uniform"
                 or (args.policy == "verified" and success)
                 or (args.policy == "random"
                     and (ep in kset if kset else wr.random() < args.write_frac))
                 or (args.policy == "scored" and r >= args.score_thr))
        if admit:
            sub = sel(samples)
            if sub:
                eng.write(sub)
                wrote, n_w_steps = True, len(sub)
        res["episodes"].append({"ep": ep, "reward": float(r),
                                "bought": bool(ok), "success": bool(success),
                                "written": wrote, "n_steps": len(samples),
                                "n_written": n_w_steps, "drift": drift()})
        if (ep + 1) % args.ckpt_every == 0:
            save()
            n_w = sum(1 for e in res["episodes"] if e["written"])
            print(f"  [{ep+1}/{args.episodes}] wrote {n_w}  drift {drift():.5f}",
                  flush=True)

    rw, bt = evaluate(eng, ev, args.eval_episodes, args.eval_seed)
    res["eval"].append({"step": args.episodes, "reward": float(np.mean(rw)),
                        "buy_rate": float(np.mean(bt)),
                        "win_rate": float(np.mean([r >= 1.0 for r in rw])),
                        "rewards": rw, "bought": bt})
    res["writes"] = sum(1 for e in res["episodes"] if e["written"])
    res["write_steps"] = args.write_steps
    res["tail_k"] = args.tail_k
    res["written_steps"] = sum(e["n_written"] for e in res["episodes"])
    res["final_drift"] = drift()
    save()
    json.dump(res, open(args.out, "w"), indent=1)
    a, b = res["eval"][0], res["eval"][-1]
    print(f"\n{args.policy}: {res['writes']} writes, drift {res['final_drift']:.5f}")
    print(f"  reward {a['reward']:.3f} -> {b['reward']:.3f}   "
          f"buy {a['buy_rate']:.3f} -> {b['buy_rate']:.3f}")
    print("saved", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
