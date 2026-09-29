"""Shared Qwen LoRA action scorer and trajectory writer for WebShop.

The model selects one admissible action by normalized action log-probability.
Only LoRA parameters are writable; evaluation calls never update them.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F

DEV = "cuda"

SYS = (
    "You are an agent in a household environment. At each step you receive an "
    "observation and choose ONE admissible action, copied exactly.\n"
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
    return (
        f"{SYS}\n{demos}Task: {task}\n\n{h}{obs}\n\nAdmissible actions:\n{acts}\n\n"
        f"Next action:"
    )


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
        return self.tok(" " + action, add_special_tokens=False, return_tensors="pt").to(
            DEV
        )["input_ids"]

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
            pad[r, : len(i)] = i

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
                self.m(chunk[:, :-1], past_key_values=cache).logits.float(), -1
            )
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
                lg = self.m(full).logits[0, n - 1 : -1]
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
            success = (
                bool(info.get("won", [False])[0])
                if isinstance(info.get("won"), (list, tuple))
                else bool(info.get("won", False))
            )
            break
    trace["success"] = bool(success)
    return success, samples, trace
