"""ScienceWorld adapter shaped like ALFWorld's batched gym env.

The agent harness reaches the environment through four things only:
reset() -> ([obs], info-with-lists), step([action]) -> ([obs], [r], [done], info),
info["admissible_commands"][0], and info["won"][0], plus a task statement it
pulls out of the observation with the regex ".*Your task is to:". Reproducing
that surface here lets the second environment reuse the scoring engine, the
write policies, the visited-state mask and the eval protocol untouched -- the
same approach that made the TextWorld attempt cheap to abandon.

Three ScienceWorld specifics:

  - ~115 valid actions per step against ALFWorld's 15-40, and the engine scores
    every candidate with a forward pass. max_actions caps that; the probe ran at
    60 and averaged 13s per episode.
  - `focus on X` is the "declare your answer" move: aimed at the wrong object it
    ENDS the episode at -100. Eighteen of the 115 candidates are such instant
    losses, which is why probe episodes often stopped after one step. This is
    the task design, not a bug, and it gives the environment a genuine failure
    mode rather than ALFWorld's timeout.
  - Score is graded 0-100, so success is defined by a threshold. The probe put
    the frozen agent at 0.50 on find-non-living-thing; `won` here means the task
    was completed (score 100), and `last_score` exposes the partial credit for
    the finer-grained analysis.

seed(n) permutes the variation order and restarts the cycle, matching how the
harness re-seeds the eval env before every block and the stream env per arm.
"""

import random

from scienceworld import ScienceWorldEnv


class SWBatchEnv:
    def __init__(self, task="find-non-living-thing", split="train",
                 max_episode_steps=50, max_actions=60, n_variations=None):
        self.env = ScienceWorldEnv("", envStepLimit=max_episode_steps)
        self.task = task
        self.max_actions = max_actions
        # get_variations_* reads state that only exists after a task is loaded;
        # calling it first throws deep in the Scala side ("size=0 and step=0").
        self.env.load(task, 0, "easy")
        avail = self.env.get_variations_train() if split == "train" \
            else self.env.get_variations_test()
        self.vars = list(avail)[:n_variations] if n_variations else list(avail)
        assert self.vars, f"no {split} variations for {task}"
        self.order = list(range(len(self.vars)))
        self.idx = 0
        self.last_score = 0
        self.last_var = None

    def seed(self, n):
        # Rebuild the identity order before shuffling. Shuffling in place is not
        # idempotent: a second seed(n) permutes an already-permuted list and
        # yields a different order, so two evaluate() calls in one process score
        # the same variations in a different order. Aggregate score fraction is
        # order-invariant and was therefore unaffected, but any per-episode
        # pairing across two evaluations in one process was silently misaligned
        # -- the same defect as the ALFWorld eval env drawing different games in
        # each block. Runs that used one process per arm were always correct.
        self.order = list(range(len(self.vars)))
        random.Random(n).shuffle(self.order)
        self.idx = 0

    def _info(self, info, score):
        self.last_score = int(score)
        adm = list(info.get("valid", []))[: self.max_actions]
        return {"admissible_commands": [adm],
                "won": [int(score) >= 100],
                "score": [int(score)]}

    def reset(self):
        self.last_var = self.vars[self.order[self.idx % len(self.order)]]
        self.idx += 1
        self.env.load(self.task, self.last_var, "easy")
        obs, info = self.env.reset()
        obs = obs + "\n\nYour task is to: " + self.env.get_task_description()
        return [obs], self._info(info, 0)

    def step(self, actions):
        obs, r, done, info = self.env.step(actions[0])
        score = info.get("score", 0)
        return [obs], [r], [bool(done)], self._info(info, score)
