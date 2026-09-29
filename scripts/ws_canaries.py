"""Correctness checks for the WebShop wrapper, before any experiment uses it.

Every environment adopted in this project so far turned out to have a defect
that only a run would surface: a non-idempotent seed that silently misaligned
per-episode pairing across two evaluations in one process, a "graded" score that
was binary, a task whose binary completion was zero so the outcome-verified arm
wrote nothing. These are the properties an experiment on this wrapper would
assume without stating, checked directly.

  1  train and test goals are disjoint
  2  seed() is idempotent -- calling it twice gives the same order, so two
     evaluations in one process score the same goals in the same order
  3  the episode order is a function of the seed alone
  4  no state leaks between episodes: the same goal replayed gives the same
     first observation and the same action set
  5  search queries carry no template words ("instruction", "i", "need")
  6  rewards stay in [0, 1] and only a bought episode can score above zero
  7  the pyserini stub is never constructed
  8  a long run does not crash, leak goals, or drift in cost

Checks 1-7 need no model and run in seconds. Check 8 takes --n episodes with a
frozen agent.

    python scripts/ws_canaries.py --n 60 --out ws_canaries.json
"""
import argparse
import json
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="episodes for the load test")
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
    ap.add_argument("--no-model", action="store_true",
                    help="run only the checks that need no forward pass")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sys.path.insert(0, "scripts")
    from ws_env import WSEnv

    fails, checks = [], 0

    def check(name, cond, detail=""):
        nonlocal checks
        checks += 1
        print(f"[{'  ok  ' if cond else ' FAIL '}] {name}" + (f"  {detail}" if detail else ""),
              flush=True)
        if not cond:
            fails.append(name)

    tr = WSEnv(split="train", max_steps=args.max_steps, max_actions=args.max_actions)
    te = WSEnv(split="test", max_steps=args.max_steps, max_actions=args.max_actions)

    # 1. the split has to be a partition, or the held-out set is not held out
    overlap = set(tr.goals) & set(te.goals)
    check("train and test goals are disjoint", not overlap,
          f"{len(tr.goals)} train, {len(te.goals)} test, {len(overlap)} shared")

    # 2/3. seeding is idempotent and determines the order
    te.seed(7)
    o1 = list(te.order)
    te.seed(7)
    o2 = list(te.order)
    check("seed() is idempotent", o1 == o2,
          "" if o1 == o2 else "a second seed() permuted an already-permuted order")
    te.seed(8)
    o3 = list(te.order)
    check("a different seed gives a different order", o1 != o3)
    te.seed(7)
    check("re-seeding restores the first order", list(te.order) == o1)

    # 4. episodes must not leak into one another
    te.seed(7)
    first_obs, first_acts = te.reset()
    first_goal = te.env.instruction_text
    for _ in range(3):
        te.reset()
    te.seed(7)
    again_obs, again_acts = te.reset()
    check("replaying a goal reproduces its first observation",
          again_obs == first_obs and te.env.instruction_text == first_goal)
    check("replaying a goal reproduces its action set",
          again_acts == first_acts,
          f"{len(first_acts)} vs {len(again_acts)}")

    # 5. the query must describe the product, not the prompt template
    bad_words = {"instruction", "i", "need", "am", "looking", "the", "for"}
    offenders, n_q = [], 0
    te.seed(11)
    for _ in range(40):
        te.reset()
        for q in te._queries():
            n_q += 1
            hit = [w for w in q.split() if w in bad_words]
            if hit:
                offenders.append((q, hit))
    check("search queries carry no template words", not offenders,
          f"{len(offenders)} of {n_q} queries" +
          (f", e.g. {offenders[0]}" if offenders else ""))
    check("every goal yields at least one query", n_q >= 40,
          f"{n_q} queries over 40 goals")

    # 7. the stub must never be constructed; if it were, search would be dead
    import pyserini.search.lucene as _l
    try:
        _l.LuceneSearcher()
        stub_ok = False
    except RuntimeError:
        stub_ok = True
    except Exception:
        stub_ok = False
    check("the pyserini stub raises rather than pretending to search", stub_ok)

    res = {"train_goals": len(tr.goals), "test_goals": len(te.goals)}

    if not args.no_model:
        import torch
        from agent_ttt import AgentTTT, build_prompt
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
        fast = [p for n, p in model.named_parameters()
                if "lora_" in n and p.requires_grad]
        eng = AgentTTT(model, tok, fast, 0.0, score_bs=args.score_bs)
        eng.reset()

        te.seed(1234)
        rewards, bought, times, seen = [], [], [], []
        for i in range(args.n):
            obs, info = te.reset()
            obs, acts = obs[0], info['admissible_commands'][0]
            seen.append(te.env.instruction_text)
            goal, hist, t0, r, done = te.env.instruction_text, [], time.time(), 0.0, False
            for step in range(args.max_steps):
                if not acts:
                    break
                a = eng.act(build_prompt(goal, hist, obs, acts), acts)
                obs, r, done, info = te.step([a])
                obs, r, done, acts = obs[0], r[0], done[0], info['admissible_commands'][0]
                hist.append((a, obs))
                if done:
                    break
            rewards.append(float(r))
            bought.append(1 if done else 0)
            times.append(time.time() - t0)
            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{args.n}] reward {np.mean(rewards):.3f} "
                      f"buy {np.mean(bought):.2f} {np.mean(times):.0f}s/ep",
                      flush=True)

        # The check that was missing. Everything above tests properties of the
        # wrapper; none of them touches run_episode, which is what the training
        # loop calls. The wrapper first returned (obs, actions) while
        # run_episode reads info["admissible_commands"][0], so evaluation --
        # which has its own loop -- ran for thirty minutes and training then
        # died on its first episode. Four times, before anyone looked.
        from agent_ttt import run_episode as _run_ep
        try:
            ok, samples, _tr = _run_ep(eng, te, max_steps=4)
            drove = isinstance(samples, list)
        except Exception as e:
            drove = False
            print(f"    run_episode raised {e.__class__.__name__}: {e}", flush=True)
        check("run_episode can drive this environment", drove,
              "the training loop calls this; the evaluation loop does not")

        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        check("peak GPU memory leaves headroom", peak < 0.85 * total,
              f"{peak:.1f} GiB of {total:.1f} GiB")

        check("rewards stay within [0, 1]",
              all(0.0 <= x <= 1.0 for x in rewards),
              f"min {min(rewards):.3f} max {max(rewards):.3f}")
        # A reward without a purchase would mean the score is not tied to the
        # action that ends the episode, and an outcome-gated policy would be
        # gating on something else entirely.
        stray = [i for i, (r, b) in enumerate(zip(rewards, bought)) if r > 0 and not b]
        check("only a completed purchase scores above zero", not stray,
              f"{len(stray)} episodes scored without buying")
        check("no goal was served twice in one pass", len(set(seen)) == len(seen),
              f"{len(set(seen))} distinct of {len(seen)}")
        first, last = np.mean(times[:len(times)//3]), np.mean(times[-len(times)//3:])
        check("per-episode cost does not drift", last < 2.0 * first,
              f"first third {first:.0f}s, last third {last:.0f}s")
        res.update(mean_reward=float(np.mean(rewards)),
                   buy_rate=float(np.mean(bought)),
                   levels=sorted(set(round(x, 4) for x in rewards)),
                   sec_per_episode=float(np.mean(times)), n=args.n)
        print(f"\n  {args.n} episodes: reward {np.mean(rewards):.3f}, "
              f"buy {np.mean(bought):.2f}, {np.mean(times):.1f}s/ep, "
              f"{len(set(round(x,4) for x in rewards))} reward levels")

    res["checks"] = checks
    res["failed"] = fails
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)

    print()
    if fails:
        print(f"{len(fails)} of {checks} FAILED: {', '.join(fails)}")
        print("These are properties an experiment would assume silently.")
    else:
        print(f"all {checks} checks passed")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(main())
