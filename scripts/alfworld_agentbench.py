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
import copy
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"

SYS = ("You are an agent in a household environment. At each step you receive an "
       "observation and choose ONE admissible action, copied exactly.\n")

AGENTBENCH_INSTRUCTION = (
    "Interact with a household to solve a task. At the beginning you receive "
    "the environment and goal. Every turn you receive AVAILABLE ACTIONS. "
    "Reply as `THOUGHT: ...\\n ACTION: ...` or `ACTION: ...`. The action must "
    "be chosen from AVAILABLE ACTIONS. Think when necessary and act directly "
    "when the next step is clear. If the environment says Nothing happened, "
    "the previous action was invalid and you should try another option."
)

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


def task_kind(task):
    task = task.lower()
    if "two " in task:
        return "puttwo"
    if "clean" in task:
        return "clean"
    if "heat" in task or "hot " in task:
        return "heat"
    if "cool" in task:
        return "cool"
    if "look at" in task or "examine" in task:
        return "examine"
    return "put"


def build_agentbench_prompt(task, history, obs, admissible, demos, initial_obs):
    """Reproduce AgentBench v0.2's alternating chat-message injection."""
    messages = [
        {"role": "user", "content": AGENTBENCH_INSTRUCTION},
        {"role": "assistant", "content":
         "OK. I'll follow your instructions and try my best to solve the task."},
    ]
    role = "user"
    for item in demos[task_kind(task)]:
        messages.append({"role": role, "content": item})
        role = "assistant" if role == "user" else "user"

    first = "Here is your task. " + initial_obs
    if not history:
        first += "\n AVAILABLE ACTIONS: " + "\n".join(admissible)
    messages.append({"role": "user", "content": first})
    for idx, (action, observation) in enumerate(history):
        messages.append({"role": "assistant", "content": "ACTION: " + action})
        content = observation
        if idx == len(history) - 1:
            content += "\n AVAILABLE ACTIONS: " + "\n".join(admissible)
        messages.append({"role": "user", "content": content})
    return messages


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
        if isinstance(prompt, list):
            encoded = self.tok.apply_chat_template(
                prompt, add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=False,
            ).to(DEV)
            pids = encoded["input_ids"]
            aids = self.tok("ACTION: " + action, add_special_tokens=False,
                            return_tensors="pt").to(DEV)["input_ids"]
            return torch.cat([pids, aids], 1), pids.shape[1]
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

        # Hybrid attention models such as Qwen3.8 mix ordinary KV-cache
        # layers with recurrent linear-attention layers.  The latter do not
        # expose ``keys``/``values``, so the cache-expansion optimization
        # below cannot clone the prefix state across candidate actions.  Fall
        # back to scoring complete prompt/action sequences in small batches.
        # This computes the same conditional action log-probability without
        # depending on private cache internals.
        if not all(hasattr(layer, "keys") and hasattr(layer, "values")
                   for layer in cache.layers):
            plen = pids.shape[1]
            scores = []
            pad_id = self.tok.pad_token_id
            if pad_id is None:
                pad_id = self.tok.eos_token_id
            for s0 in range(0, len(ids), self.score_bs):
                chunk_ids = ids[s0:s0 + self.score_bs]
                seqs = [torch.cat([pids[0], act]) for act in chunk_ids]
                max_len = max(len(seq) for seq in seqs)
                batch = torch.full((len(seqs), max_len), pad_id,
                                   dtype=torch.long, device=DEV)
                mask = torch.zeros_like(batch)
                for row, seq in enumerate(seqs):
                    batch[row, :len(seq)] = seq
                    mask[row, :len(seq)] = 1
                logits = self.m(batch, attention_mask=mask, use_cache=False).logits.float()
                for row, act in enumerate(chunk_ids):
                    n = len(act)
                    pred = logits[row, plen - 1:plen + n - 1]
                    lp = torch.log_softmax(pred, -1)
                    scores.append(float(lp.gather(-1, act[:, None]).sum() / n))
                del logits, batch, mask
            return scores

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
        if getattr(self, "agentbench", False):
            messages = prompt
            encoded = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=False,
            ).to(DEV)
            ids = encoded["input_ids"]
            out = self.m.generate(ids, max_new_tokens=192, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
            text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            if getattr(self, "parser_mode", "official") == "exact":
                match = re.search(r"ACTION:(.*)", text)
                if not match:
                    return text.strip().lower().split("\n")[0]
                return match.group(1).strip().lower().split("\n")[0]
            agentbench_root = os.environ.get("AGENTBENCH_ROOT")
            if not agentbench_root:
                raise RuntimeError(
                    "official parser requested but AGENTBENCH_ROOT is unset; "
                    "point it at an AgentBench v0.2 checkout"
                )
            if agentbench_root not in sys.path:
                sys.path.insert(0, agentbench_root)
            from src.server.tasks.alfworld.utils import process_action
            return process_action(text, admissible)
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
            accum = [torch.zeros_like(p) for p in self.fast]
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
                lg = self.m(full, use_cache=False).logits[0, n - 1:-1]
                loss = F.cross_entropy(lg.float(), full[0, n:])
                one = torch.autograd.grad(sign * loss, self.fast)
                for dst, src in zip(accum, one):
                    dst.add_(src)
                used += 1
                del full, lg, loss, one
            if not used:
                return
            g = [x / used for x in accum]
            gn = torch.sqrt(sum(x.float().pow(2).sum() for x in g))
            sc = min(1.0, 1.0 / (float(gn) + 1e-8))
            for p, gr in zip(self.fast, g):
                p.grad = gr * sc
            update_scale = getattr(self, "update_scale", 1.0)
            before = ([p.detach().clone() for p in self.fast]
                      if update_scale != 1.0 else None)
            self.opt.step()
            if before is not None:
                # Scale the realized Adam parameter displacement. Scaling the
                # gradient before Adam is nearly cancelled by its second-moment
                # normalization and does not implement a dose sweep.
                with torch.no_grad():
                    for p, old in zip(self.fast, before):
                        p.copy_(old + update_scale * (p - old))
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
    initial_obs = obs
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
        prompt = (build_agentbench_prompt(task, history, obs, adm, demos, initial_obs)
                  if getattr(eng, "agentbench", False)
                  else build_prompt(task, history, obs, adm, demos))
        action = eng.act(prompt, adm)
        if not action:
            trace["actions"].append({"a": None, "n_adm": len(adm),
                                     "obs": "invalid AgentBench action"})
            break
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
                    choices=["writes_off", "closed", "fixed_generation", "settlement",
                             "verified", "failure_only"])
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--eval-episodes", type=int, default=15)
    ap.add_argument("--eval-seen-episodes", type=int, default=0)
    ap.add_argument("--eval-unseen-episodes", type=int, default=0)
    ap.add_argument("--online-prequential", action="store_true",
                    help="run one continuous seen->unseen stream, updating after each task")
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
    ap.add_argument("--settle-every", type=int, default=25)
    ap.add_argument("--validation-episodes", type=int, default=20)
    ap.add_argument("--validation-seed", type=int, default=8765)
    ap.add_argument("--admit-margin", type=float, default=0.0)
    ap.add_argument("--update-scale", type=float, default=1.0,
                    help="multiply the already clipped LoRA update")
    ap.add_argument("--parser-mode", choices=["official", "exact"], default="official")
    ap.add_argument("--task-types", default=None,
                    help="comma-separated ALFWorld task ids, e.g. 2,3,4,5,6")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="ignore an existing checkpoint and start the arm over")
    ap.add_argument("--allow-repeat", action="store_true",
                    help="do not mask (observation, action) pairs already spent; "
                         "reproduces the looping baseline")
    ap.add_argument("--demos", default=None,
                    help="oracle demonstrations from alfworld_demos.py, used as "
                         "prompt context only — never written to the fast weights")
    ap.add_argument("--agentbench-prompts", default=None,
                    help="AgentBench v0.2 task-specific plan-first prompt JSON")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    from alfworld.agents.environment import get_environment
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    cfg_path = os.path.join(os.path.dirname(__file__), "alfworld_config.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    if args.task_types:
        cfg["env"]["task_types"] = [int(x) for x in args.task_types.split(",")]
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
    eval_seen = EnvCls(cfg, train_eval="eval_in_distribution").init_env(batch_size=1)
    eval_unseen = EnvCls(cfg, train_eval="eval_out_of_distribution").init_env(batch_size=1)
    validation_env = EnvCls(cfg, train_eval="train").init_env(batch_size=1)

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
    # A 27B backward pass over a multi-turn agent prompt otherwise retains
    # roughly 80 GiB of activations and exceeds one 144 GiB H200 even though
    # only LoRA tensors are trainable. Checkpointing recomputes layer activations
    # during backward and leaves inference unchanged (it runs under no_grad).
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    fast = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    lr = 0.0 if args.policy == "writes_off" else args.lr
    eng_actor_frozen = args.policy == "fixed_generation"
    eng = AgentTTT(model, tok, fast, lr)
    eng.reset()
    eng.actor_frozen = eng_actor_frozen
    eng.agentbench = bool(args.agentbench_prompts)
    eng.parser_mode = args.parser_mode
    eng.update_scale = args.update_scale
    print(f"{sum(p.numel() for p in fast)/1e6:.1f}M LoRA params, policy={args.policy}",
          flush=True)

    if args.agentbench_prompts:
        demos = json.load(open(args.agentbench_prompts))
    else:
        demos = format_demos(json.load(open(args.demos))) if args.demos else ""
    if demos:
        sample_demo = "".join(demos["put"]) if isinstance(demos, dict) else demos
        print(f"demo context: {len(tok(sample_demo)['input_ids'])} tokens", flush=True)

    res = {"policy": args.policy, "model": args.model, "episodes": [], "eval": [],
           "traces": [], "demos": args.demos,
           "agentbench_prompts": args.agentbench_prompts,
           "parser_mode": args.parser_mode, "update_scale": args.update_scale,
           "task_types": cfg["env"].get("task_types"),
           "settlement": {"block_size": args.settle_every,
                          "validation_episodes": args.validation_episodes,
                          "validation_seed": args.validation_seed,
                          "admit_margin": args.admit_margin, "blocks": []}}

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
        split_specs = []
        if args.eval_seen_episodes:
            split_specs.append(("seen", eval_seen, args.eval_seen_episodes))
        if args.eval_unseen_episodes:
            split_specs.append(("unseen", eval_unseen, args.eval_unseen_episodes))
        if not split_specs:
            split_specs.append(("seen", eval_seen, args.eval_episodes))
        block = {"step": step, "splits": {}}
        for split, eval_env, count in split_specs:
            eval_env.seed(args.eval_seed)
            marks = []
            for _ in range(count):
                ok, _, tr = run_episode(eng, eval_env, args.max_steps, demos,
                                        not args.allow_repeat)
                marks.append(int(ok))
                if len(res["traces"]) < args.keep_traces:
                    res["traces"].append({"step": step, "split": split, **tr})
            r = sum(marks) / len(marks)
            block["splits"][split] = {"success": r, "marks": marks}
            print(f"  eval@{step}/{split}: success {r:.3f} ({sum(marks)}/{count})",
                  flush=True)
        res["eval"].append(block)
        json.dump(res, open(args.out, "w"), indent=1)
        save_ckpt()

    def validation_score():
        # Admission uses seen tasks. Unseen is never consulted and is the clean
        # primary endpoint for Settlement.
        eval_seen.seed(args.validation_seed)
        marks = []
        for _ in range(args.validation_episodes):
            ok, _, _ = run_episode(eng, eval_seen, args.max_steps, demos,
                                   not args.allow_repeat)
            marks.append(int(ok))
        return sum(marks) / len(marks)

    def snapshot_state():
        return {"fast": [p.detach().clone() for p in fast],
                "opt": copy.deepcopy(eng.opt.state_dict()) if eng.opt is not None else None}

    def restore_state(state):
        with torch.no_grad():
            for p, q in zip(fast, state["fast"]):
                p.copy_(q)
        if eng.opt is not None and state["opt"] is not None:
            eng.opt.load_state_dict(state["opt"])

    if args.online_prequential:
        # One deployment stream: the adapted state crosses the seen->unseen
        # boundary and no task is replayed as a frozen endpoint probe. Reward is
        # recorded before that task's update, which is the prequential convention.
        res["protocol"] = "online_prequential_seen_then_unseen"
        res["online"] = []
        settle_base = snapshot_state()
        global_step = 0
        settle_index = 0

        def online_validation_score(seed):
            validation_env.seed(seed)
            marks = []
            for _ in range(args.validation_episodes):
                ok, _, _ = run_episode(eng, validation_env, args.max_steps, demos,
                                       not args.allow_repeat)
                marks.append(int(ok))
            return sum(marks) / len(marks)

        stream_specs = [("seen", eval_seen, args.eval_seen_episodes or 140),
                        ("unseen", eval_unseen, args.eval_unseen_episodes or 134)]
        for split, stream_env, count in stream_specs:
            stream_env.seed(args.eval_seed)
            split_marks = []
            for idx in range(count):
                ok, samples, tr = run_episode(eng, stream_env, args.max_steps, demos,
                                              not args.allow_repeat)
                split_marks.append(int(ok))
                wrote = False
                if args.policy in ("closed", "fixed_generation", "settlement"):
                    eng.write(samples); wrote = True
                elif args.policy == "verified" and ok:
                    eng.write(samples); wrote = True
                elif args.policy == "failure_only" and not ok:
                    eng.write(samples); wrote = True
                global_step += 1
                res["online"].append({"step": global_step, "split": split,
                                      "split_index": idx, "success": bool(ok),
                                      "steps": len(samples), "written": wrote})
                if len(res["traces"]) < args.keep_traces:
                    res["traces"].append({"step": global_step, "split": split, **tr})
                boundary = global_step % args.settle_every == 0 or global_step == sum(
                    n for _, _, n in stream_specs)
                if args.policy == "settlement" and boundary:
                    candidate = snapshot_state()
                    seed = args.validation_seed + settle_index
                    candidate_score = online_validation_score(seed)
                    restore_state(settle_base)
                    base_score = online_validation_score(seed)
                    admitted = candidate_score > base_score + args.admit_margin
                    if admitted:
                        restore_state(candidate)
                    res["settlement"]["blocks"].append({"end_task": global_step,
                        "admitted": admitted, "candidate_success": candidate_score,
                        "base_success": base_score, "validation_seed": seed})
                    settle_base = snapshot_state()
                    settle_index += 1
                    print(f"settle@task{global_step}: "
                          f"{'commit' if admitted else 'reject'} candidate "
                          f"{candidate_score:.3f} base {base_score:.3f}", flush=True)
                if global_step % args.ckpt_every == 0:
                    json.dump(res, open(args.out, "w"), indent=1)
                    save_ckpt()
                    print(f"online@{global_step}: {split} cumulative "
                          f"{sum(split_marks)}/{len(split_marks)}", flush=True)
            print(f"online/{split}: success {sum(split_marks)/len(split_marks):.3f} "
                  f"({sum(split_marks)}/{len(split_marks)})", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
        save_ckpt()
        print("online prequential complete", flush=True)
        return

    if start_ep == 0 and not res["eval"]:
        eval_block(0)
    settle_base = snapshot_state()
    write_rng = np.random.default_rng(args.write_seed)
    for _ in range(start_ep):          # keep the draw aligned when resuming
        write_rng.random()
    n_written = sum(1 for e in res["episodes"] if e.get("written"))
    for ep in range(start_ep, args.episodes):
        ok, samples, _tr = run_episode(eng, env, args.max_steps, demos, not args.allow_repeat)
        wrote = False
        if args.policy in ("closed", "fixed_generation", "settlement"):
            eng.write(samples); wrote = True
        elif args.policy == "verified" and ok:
            eng.write(samples); wrote = True
        elif args.policy == "failure_only" and not ok:
            eng.write(samples); wrote = True
        elif False:
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
        if args.policy == "settlement" and ((ep + 1) % args.settle_every == 0
                                               or ep + 1 == args.episodes):
            candidate = snapshot_state()
            candidate_score = validation_score()
            restore_state(settle_base)
            base_score = validation_score()
            admitted = candidate_score > base_score + args.admit_margin
            if admitted:
                restore_state(candidate)
            res["settlement"]["blocks"].append({"end_episode": ep + 1,
                "admitted": admitted, "candidate_success": candidate_score,
                "base_success": base_score})
            settle_base = snapshot_state()
            print(f"settle@{ep+1}: {'commit' if admitted else 'reject'} "
                  f"candidate {candidate_score:.3f} base {base_score:.3f}", flush=True)
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
    print("final eval", res["eval"][-1]["splits"], flush=True)


if __name__ == "__main__":
    main()
