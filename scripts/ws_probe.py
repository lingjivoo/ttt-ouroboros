"""Feasibility canary for WebShop, before anything is built on top of it.

Measures the three numbers that decide whether this environment can host a
write-authority experiment, and nothing else:

  frozen success rate   strictly inside (0, 1). At zero an outcome-verified
                        policy has nothing to write and coincides with the
                        frozen arm by construction; at one it writes on
                        everything and coincides with uniform.
  reward resolution     how many distinct reward values appear. The previous
                        environment was chosen for a "graded" score that turned
                        out to take two values, leaving a binary metric with a
                        12% base rate.
  seconds per episode   the engine scores every candidate action with a forward
                        pass, which is what made ScienceWorld's ~115 actions per
                        step expensive.

Two environments were adopted before being measured this way and each failed on
a property that a few minutes of probing would have shown. This runs the frozen
agent and reports; it builds no adapter and trains nothing.

    python scripts/ws_probe.py --n 40 --out ws_probe.json
"""
import argparse
import json
import sys
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
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
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, "scripts")
    from agent_ttt import AgentTTT, build_prompt
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
    eng = AgentTTT(model, tok, fast, 0.0, score_bs=args.score_bs)   # frozen: lr 0, write() is a no-op
    eng.reset()

    env = WSEnv(split=args.split, max_steps=args.max_steps,
                max_actions=args.max_actions)
    env.seed(1234)
    print(f"{len(env.goals)} goals in the {args.split} split", flush=True)

    rewards, times, steps_used, wins, bought = [], [], [], [], []
    for i in range(args.n):
        obs, info = env.reset()
        obs, acts = obs[0], info['admissible_commands'][0]
        goal = env.env.instruction_text
        hist, t0, r = [], time.time(), 0.0
        done = False
        for step in range(args.max_steps):
            if not acts:
                break
            action = eng.act(build_prompt(goal, hist, obs, acts), acts)
            obs, r, done, info = env.step([action])
            obs, r, done, acts = obs[0], r[0], done[0], info['admissible_commands'][0]
            hist.append((action, obs))
            if done:
                break
        dt = time.time() - t0
        rewards.append(float(r))
        # A purchase is what ends an episode. Separating "bought something" from
        # "bought the right thing" matters: an agent that never buys scores zero
        # for a different reason than one that buys badly, and only the second
        # gives an outcome-verified policy anything to gate on.
        bought.append(1 if done else 0)
        wins.append(1 if r >= 1.0 else 0)
        times.append(dt)
        steps_used.append(step + 1)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{args.n}] reward {np.mean(rewards):.3f}  "
                  f"bought {np.mean(bought):.2f}  win {np.mean(wins):.2f}  "
                  f"{np.mean(times):.0f}s/ep", flush=True)

    levels = sorted(set(round(x, 4) for x in rewards))
    res = {"split": args.split, "n": args.n,
           "mean_reward": float(np.mean(rewards)),
           "reward_levels": len(levels), "levels": levels[:15],
           "buy_rate": float(np.mean(bought)),
           "win_rate": float(np.mean(wins)),
           "sec_per_episode": float(np.mean(times)),
           "mean_steps": float(np.mean(steps_used)),
           "rewards": rewards}
    # Usable needs BOTH: something for a verified policy to write on, and enough
    # resolution to see a difference once it does.
    res["usable"] = bool(0.0 < np.mean(bought) < 1.0 and len(levels) > 2)
    json.dump(res, open(args.out, "w"), indent=1)

    print(f"\nmean reward      {res['mean_reward']:.3f}")
    print(f"reward levels    {res['reward_levels']}  {res['levels'][:8]}")
    print(f"buy rate         {res['buy_rate']:.3f}")
    print(f"full-match rate  {res['win_rate']:.3f}")
    print(f"{res['sec_per_episode']:.1f}s per episode, {res['mean_steps']:.1f} steps")
    print("USABLE" if res["usable"] else
          "NOT USABLE: needs a buy rate inside (0,1) and more than two reward levels")
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
