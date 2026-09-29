"""Decode-time TTT in an agent loop: does write authority decide whether
self-improvement happens or self-destruction does?

ALFWorld gives what a math stream cannot: the same environment family visited
again and again, so there IS transferable structure to adapt to (this is the
regime our same-domain carry probe says adaptation should help in), and a free
verifier (the environment reports task success), so write authority can be
granted by evidence rather than guessed from content.

Loop: for each episode the agent acts with its current adapted weights; when the
episode ends the write policy decides whether to train on the trajectory.

  none      no adaptation                       (frozen reference)
  uniform   train on every trajectory           (standard self-training)
  verified  train only on SUCCESSFUL episodes   (write authority from the env)
  failneg   train on successes, unlearn failures (signed writes; exploratory)
  random    train on a random --write-frac of trajectories, ignoring outcome

Metric: success rate over episodes, and a held-out eval block run periodically
with writes disabled.

  python scripts/agent_ttt.py --model Qwen/Qwen3-8B --policy verified --out a.json
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"

SYS = ("You are an agent in a household environment. At each step you receive an "
       "observation and choose ONE admissible action, copied exactly.\n")

# Commands that consume a step without changing the world. The step budget is
# what makes these episodes hard, so they are never worth offering.
NOOPS = ("look", "inventory", "help")


def format_demos(demos, obs_chars=160):
    """Worked examples to condition on, in the same shape as the live episode."""
    if not demos:
        return ""
    blocks = []
    for d in demos:
        lines = [f"Task: {d['task']}"]
        for s in d["steps"]:
            lines.append(f"> {s['action']}")
            lines.append(s["obs"][:obs_chars].strip())
        blocks.append("\n".join(lines))
    return "Examples of completed tasks:\n\n" + "\n\n".join(blocks) + "\n\n---\n\n"


def build_prompt(task, history, obs, admissible, demos=""):
    h = "".join(f"> {a}\n{o}\n" for a, o in history[-8:])
    acts = "\n".join(f"- {a}" for a in admissible[:40])
    return (f"{SYS}\n{demos}Task: {task}\n\n{h}{obs}\n\nAdmissible actions:\n{acts}\n\n"
            f"Next action:")


class AgentTTT:
    def __init__(self, model, tok, fast, lr, steps=2, score_bs=64):
        # score_bs caps how many candidates share one forward. 64 is above any
        # action set ALFWorld or ScienceWorld produces, so those runs are
        # unchanged; WebShop needs a smaller value because its pages are long
        # and 60 copies of that prompt's KV cache overflow an 80 GiB card.
        self.m, self.tok, self.fast, self.lr, self.steps = model, tok, fast, lr, steps
        self.score_bs = max(1, score_bs)
        self.init = [p.detach().clone() for p in fast]
        self.opt = torch.optim.Adam(fast, lr=lr) if lr > 0 else None

    def reset(self):
        with torch.no_grad():
            for p, i0 in zip(self.fast, self.init):
                p.copy_(i0)
        self.opt = torch.optim.Adam(self.fast, lr=self.lr) if self.lr > 0 else None

    def _act_ids(self, action):
        return self.tok(" " + action, add_special_tokens=False,
                        return_tensors="pt").to(DEV)["input_ids"]

    def _pair(self, prompt, action):
        """(prompt+action ids, prompt length) — the boundary act() and write() share."""
        pids = self.tok(prompt, return_tensors="pt").to(DEV)["input_ids"]
        return torch.cat([pids, self._act_ids(action)], 1), pids.shape[1]

    @torch.no_grad()
    def scores(self, prompt, admissible):
        """Length-normalised logprob of each candidate action after the prompt.

        The prompt is prefilled once and all candidates are then scored in ONE
        batched forward, sharing that prefix. Scoring them one at a time is what
        the arithmetic suggests, but at ~5 action tokens each the GPU sits idle
        between kernel launches and a step costs ~4s; the batched form does the
        same arithmetic in a single launch sequence.
        """
        pids = self.tok(prompt, return_tensors="pt").to(DEV)["input_ids"]
        out = self.m(pids, use_cache=True)
        cache = out.past_key_values
        lp_first = torch.log_softmax(out.logits[0, -1].float(), -1)

        ids = [self._act_ids(a)[0] for a in admissible]
        first = torch.stack([lp_first[i[0]] for i in ids])
        lens = torch.tensor([len(i) for i in ids], device=DEV)
        N, L = len(ids), int(lens.max())
        if L == 1:
            return (first / lens).tolist()

        # right-padded; causal attention means real tokens never see the padding,
        # and the padded positions' logits are masked out of the sum below
        pad = torch.zeros(N, L, dtype=torch.long, device=DEV)
        for r, i in enumerate(ids):
            pad[r, :len(i)] = i

        # Score in slices. Expanding the prompt's KV cache to N copies is what
        # makes one launch possible, and it is also what runs out of memory:
        # WebShop's observations are far longer than ALFWorld's, and 60
        # candidates over a long page needed 2.8 GiB more than an 80 GiB card
        # had. Candidates are scored independently, so slicing changes only how
        # many copies exist at once.
        #
        # This is NOT numerically equivalent, and the difference is not small.
        # Measured on WebShop pages of 1200-1700 tokens with 12 candidates,
        # scores move by up to 5.2e-02 between a width of 12 and a width of 8 --
        # forty times the streaming-vs-standard discrepancy elsewhere in this
        # project. Batch width selects kernels; the arithmetic each candidate
        # receives is not the same arithmetic.
        #
        # It is usable because the agent takes an argmax and the gaps between
        # candidates are larger than that: the chosen action was identical on
        # 6/6 steps measured. So the justification is "smaller than the decision
        # margin", not "equivalent" -- which carries a hard constraint:
        # score_bs MUST be identical across arms, or a comparison between them
        # straddles two numerics.
        rest = torch.empty(N, device=DEV)
        base = [(l.keys, l.values) for l in cache.layers]
        for s0 in range(0, N, self.score_bs):
            s1 = min(s0 + self.score_bs, N)
            n = s1 - s0
            for layer, (k, v) in zip(cache.layers, base):
                layer.keys = k.expand(n, -1, -1, -1).contiguous()
                layer.values = v.expand(n, -1, -1, -1).contiguous()
            chunk = pad[s0:s1]
            lg = torch.log_softmax(
                self.m(chunk[:, :-1], past_key_values=cache).logits.float(), -1)
            tok_lp = lg.gather(-1, chunk[:, 1:, None])[:, :, 0]
            pos = torch.arange(1, L, device=DEV)[None, :]
            rest[s0:s1] = (tok_lp * (pos < lens[s0:s1, None])).sum(1)
            # Drop this slice's copies before making the next set. Leaving the
            # expanded tensors attached to cache.layers keeps one full extra
            # copy alive for the whole loop, which on a long page is most of
            # what the slicing was meant to avoid.
            del lg, tok_lp
            for layer, (k, v) in zip(cache.layers, base):
                layer.keys, layer.values = k, v
        return ((first + rest) / lens).tolist()

    def _act_adapted(self, prompt, admissible):
        """Pick an action using the currently active adapter state."""
        adm = admissible[:40]
        sc = self.scores(prompt, adm)
        return adm[max(range(len(adm)), key=lambda i: sc[i])]

    def act(self, prompt, admissible):
        """Act with the learner, or with frozen W0 when ``actor_frozen`` is set.

        PEFT's disable_adapter context removes the LoRA contribution while
        leaving the learner's adapter tensors untouched. The subsequent write
        therefore updates the learner from a trajectory generated by W0.
        """
        if getattr(self, "actor_frozen", False):
            with self.m.disable_adapter():
                return self._act_adapted(prompt, admissible)
        return self._act_adapted(prompt, admissible)

    def write(self, samples, sign=1.0):
        """SFT on (prompt, action) pairs from a trajectory. sign=-1 unlearns."""
        if self.lr <= 0 or not samples:
            return
        for _ in range(self.steps):
            tot = 0.0
            # Average over the pairs actually used. Dividing by
            # min(len(samples), 16) counted the ones skipped for having no
            # action tokens, which silently scaled the update down by the
            # skipped fraction -- an effective learning rate that varied with
            # the episode rather than a fixed one.
            used = 0
            for prompt, action in samples[:16]:
                full, n = self._pair(prompt, action)
                if full.shape[1] <= n:
                    continue
                lg = self.m(full).logits[0, n - 1:-1]
                loss = F.cross_entropy(lg.float(), full[0, n:])
                tot = tot + sign * loss
                used += 1
            if used:
                tot = tot / used
            if not torch.is_tensor(tot):
                return
            g = torch.autograd.grad(tot, self.fast)
            gn = torch.sqrt(sum(x.float().pow(2).sum() for x in g))
            sc = min(1.0, 1.0 / (float(gn) + 1e-8))
            for p, gr in zip(self.fast, g):
                p.grad = gr * sc
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)


def run_episode(eng, env, max_steps=30, demos="", no_repeat=True):
    """Returns (success, training samples, trace).

    The trace is what makes a 0%-success run diagnosable: a policy that never
    succeeds looks identical in the summary whether the agent is wandering, is
    repeating one action, or is picking sensibly and just running out of steps.
    """
    obs, info = env.reset()
    obs = obs[0] if isinstance(obs, (list, tuple)) else obs
    task = re.sub(r".*Your task is to:", "", obs, flags=re.S).strip()
    history, samples, done, success = [], [], False, False
    trace = {"task": task, "actions": []}
    # (observation, action) pairs already spent this episode. A greedy policy in a
    # deterministic environment that returns to a state it has seen must repeat the
    # action it chose there, so `take X from Y` / `move X to Y` becomes a permanent
    # 2-cycle; the traces showed one episode spending 47 of 50 steps in exactly
    # that. Refusing a pair twice makes the cycle impossible rather than unlikely.
    tried = set()
    for _ in range(max_steps):
        adm = info["admissible_commands"][0]
        adm = [a for a in adm if a not in NOOPS] or adm
        if no_repeat:
            # fall back to the full set if masking would leave nothing
            adm = [a for a in adm if (obs, a) not in tried] or adm
        prompt = build_prompt(task, history, obs, adm, demos)
        action = eng.act(prompt, adm)
        tried.add((obs, action))
        samples.append((prompt, action))
        obs, reward, done, info = env.step([action])
        obs = obs[0] if isinstance(obs, (list, tuple)) else obs
        done = done[0] if isinstance(done, (list, tuple)) else done
        trace["actions"].append({"a": action, "n_adm": len(adm), "obs": obs[:120]})
        history.append((action, obs))
        if done:
            success = bool(info.get("won", [False])[0]) if isinstance(
                info.get("won"), (list, tuple)) else bool(info.get("won", False))
            break
    trace["success"] = bool(success)
    return success, samples, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--policy", required=True,
                    choices=["none", "uniform", "verified", "failneg", "random"])
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--eval-episodes", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--keep-traces", type=int, default=6,
                    help="eval episodes to keep a full action trace for")
    ap.add_argument("--ckpt-every", type=int, default=25,
                    help="save fast weights every N episodes regardless of the "
                         "eval schedule; must stay well under the preemption "
                         "interval or a run loses everything between eval blocks")
    ap.add_argument("--write-frac", type=float, default=0.25,
                    help="policy=random: fraction of trajectories to write, set to "
                         "the rate the verified arm actually achieved so the two "
                         "differ only in WHICH trajectories are chosen")
    ap.add_argument("--write-seed", type=int, default=0,
                    help="policy=random: fixes which trajectories are chosen")
    ap.add_argument("--stream-seed", type=int, default=None,
                    help="re-seed the TRAINING stream for an independent replicate; "
                         "arms being compared must use the same value")
    ap.add_argument("--eval-seed", type=int, default=1234,
                    help="fixes which held-out games each eval block replays")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="ignore an existing checkpoint and start the arm over")
    ap.add_argument("--allow-repeat", action="store_true",
                    help="do not mask (observation, action) pairs already spent; "
                         "reproduces the looping baseline")
    ap.add_argument("--demos", default=None,
                    help="oracle demonstrations from alfworld_demos.py, used as "
                         "prompt context only — never written to the fast weights")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    from alfworld.agents.environment import get_environment
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    cfg_path = os.path.join(os.path.dirname(__file__), "alfworld_config.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    EnvCls = get_environment(cfg["env"]["type"])
    # TextworldBatchGymEnv.__init__ seeds itself with a fixed 1234, so by default every
    # policy walks the SAME stream games in the same order and the writes are the only
    # difference between arms. Verified by running two processes and diffing the task
    # sequence. --stream-seed re-seeds it to get an INDEPENDENT training stream: the
    # eval block is paired across 100 games, but a single stream order makes the
    # training run itself n=1, which supports "this adapter changed" and not "this
    # policy usually degrades agents". Arms to be compared must share the value.
    env = EnvCls(cfg, train_eval="train").init_env(batch_size=1)
    if args.stream_seed is not None:
        env.seed(args.stream_seed)
    # Eval draws from valid_seen, which shares no games with the 3553 train ones,
    # and is re-seeded before every block so the SAME games come up in the SAME
    # order. reset() otherwise pulls from a shuffled infinite cycle, and with
    # ALFWorld's spread of task lengths a fresh draw each block would measure
    # which games turned up rather than what the writes did.
    eval_env = EnvCls(cfg, train_eval="eval_in_distribution").init_env(batch_size=1)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    L = getattr(model.config, "num_hidden_layers", None) or \
        model.config.text_config.num_hidden_layers
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["down_proj"], layers_to_transform=list(range((3 * L) // 4, L))))
    fast = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    lr = 0.0 if args.policy == "none" else args.lr
    eng = AgentTTT(model, tok, fast, lr)
    eng.reset()
    print(f"{sum(p.numel() for p in fast)/1e6:.1f}M LoRA params, policy={args.policy}",
          flush=True)

    demos = format_demos(json.load(open(args.demos))) if args.demos else ""
    if demos:
        print(f"demo context: {len(tok(demos)['input_ids'])} tokens", flush=True)

    res = {"policy": args.policy, "model": args.model, "episodes": [], "eval": [],
           "traces": [], "demos": args.demos}

    # freecycle preempts these runs about half way through, and an arm restarted
    # from scratch costs ~3h. Checkpoint at every eval block instead: fast weights,
    # the Adam moments that go with them (resuming with those reset would put a
    # discontinuity in the middle of the very accumulation being measured), and the
    # results so far.
    ckpt_path = os.path.splitext(args.out)[0] + "_fast.pt"

    def save_ckpt():
        torch.save({"fast": [p.detach().cpu() for p in fast],
                    "opt": eng.opt.state_dict() if eng.opt is not None else None,
                    "res": res, "policy": args.policy, "episodes": args.episodes,
                    "lr": lr}, ckpt_path)

    start_ep = 0
    if args.resume and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if st.get("res"):
            res = st["res"]
            with torch.no_grad():
                for p, q in zip(fast, st["fast"]):
                    p.copy_(q.to(DEV))
            if st.get("opt") is not None and eng.opt is not None:
                eng.opt.load_state_dict(st["opt"])
            start_ep = len(res["episodes"])
            print(f"resuming {args.policy} at episode {start_ep} "
                  f"({len(res['eval'])} eval blocks done)", flush=True)
            # the stream env hands out games in a fixed order, so fast-forward it to
            # the same position rather than re-running the episodes
            for _ in range(start_ep):
                env.reset()

    def eval_block(step):
        eval_env.seed(args.eval_seed)
        marks = []
        for _ in range(args.eval_episodes):
            ok, _, tr = run_episode(eng, eval_env, args.max_steps, demos, not args.allow_repeat)
            marks.append(int(ok))
            if len(res["traces"]) < args.keep_traces:
                res["traces"].append({"step": step, **tr})
        r = sum(marks) / len(marks)
        # per-episode marks, not just the mean: the same games recur at every
        # checkpoint, so blocks can be compared pairwise on the episodes that flip
        res["eval"].append({"step": step, "success": r, "marks": marks})
        print(f"  eval@{step}: success {r:.3f}  {marks}", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
        save_ckpt()

    if start_ep == 0:
        eval_block(0)
    write_rng = np.random.default_rng(args.write_seed)
    for _ in range(start_ep):          # keep the draw aligned when resuming
        write_rng.random()
    n_written = sum(1 for e in res["episodes"] if e.get("written"))
    for ep in range(start_ep, args.episodes):
        ok, samples, _tr = run_episode(eng, env, args.max_steps, demos, not args.allow_repeat)
        wrote = False
        if args.policy == "uniform":
            eng.write(samples); wrote = True
        elif args.policy == "verified" and ok:
            eng.write(samples); wrote = True
        elif args.policy == "failneg":
            eng.write(samples, sign=1.0 if ok else -0.3); wrote = True
        elif args.policy == "random" and write_rng.random() < args.write_frac:
            # Dose-matched control for `verified`: same number of writes, chosen
            # without reference to the outcome. Comparing verified against uniform
            # confounds how much is written with which trajectories are chosen;
            # this arm holds the amount fixed so the comparison isolates whether
            # success actually identifies what is worth keeping.
            eng.write(samples); wrote = True
        n_written += wrote
        # recorded per episode so a resumed run recovers the count, and so the
        # analysis can relate churn to how much was actually written
        res["episodes"].append({"ep": ep, "success": bool(ok), "steps": len(samples),
                                "written": wrote})
        # Checkpoint on its own schedule. Saving only inside eval blocks ties
        # fault tolerance to how often we evaluate, and raising --eval-every to
        # trade trajectory detail for endpoint power silently raised the
        # checkpoint interval to 200 episodes -- about 2.5 hours, longer than the
        # gap between preemptions, so three arms lost everything.
        if args.ckpt_every > 0 and (ep + 1) % args.ckpt_every == 0 \
                and (ep + 1) % args.eval_every != 0:
            save_ckpt()
            print(f"[{ep+1}/{args.episodes}] checkpoint", flush=True)
        if (ep + 1) % args.eval_every == 0:
            recent = np.mean([e["success"] for e in res["episodes"][-args.eval_every:]])
            print(f"[{ep+1}/{args.episodes}] stream-recent {recent:.3f} "
                  f"written {n_written}", flush=True)
            eval_block(ep + 1)
    json.dump(res, open(args.out, "w"), indent=1)
    save_ckpt()
    print(f"final eval {res['eval'][-1]['success']:.3f} "
          f"(start {res['eval'][0]['success']:.3f})", flush=True)


if __name__ == "__main__":
    main()
