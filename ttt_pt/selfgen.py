"""Closed-loop training batches: chunks the model itself produced.

The inner learner is meta-trained on sequences drawn from a corpus and then
deployed where the sequence is drawn from the learner's own state. That is not
only a distribution shift, it is a causal one -- in training the data is
independent of the learner, in deployment the data is caused by it -- and the
objective never saw the second case. This builds batches in which designated
chunks come from the model's own decoding, so the inner update is taken on
self-generated text while the outer loss stays on real text. Optimising the
outer loss on generated text would be optimising the model to predict its own
samples, which is the degeneracy rather than the fix.

Generation is genuinely autoregressive, reusing the streaming decoder that the
evaluation protocol uses, because the whole point is the drift that only
appears when each token conditions on the last. It is also the expensive part,
so streams are cached and refreshed every `refresh_every` optimiser steps: the
data is then caused by a recent learner state rather than the current one,
which is off-policy in the same sense a replay buffer is. Set refresh_every=1
for the strictly on-policy version at proportionally higher cost.
"""

from __future__ import annotations

import torch

from ttt_pt.stream import StreamState


class SelfGenCache:
    """Autoregressively decoded chunks, refreshed every N optimiser steps."""

    def __init__(self, n_chunks: int, refresh_every: int, seed: int = 0, source: str = "model"):
        self.n_chunks = n_chunks  # self-generated chunks per sequence
        # "model" splices in the model's own decoding; "real" leaves the real
        # tokens in place and applies the SAME outer-loss mask. That is the
        # control the first pair of arms lacked: excluding k chunks from the
        # outer loss quarters the amount of outer supervision per sequence, so
        # an arm with k=6 differs from an unmasked arm in how much it is taught
        # as well as in what it writes. Holding the mask fixed and changing
        # only the written content isolates the treatment.
        self.source = source
        self.refresh_every = refresh_every
        self.seed = seed
        self.step_built = None
        self.chunks = None  # [n_chunks] of [B, CS] int64

    def stale(self, step: int) -> bool:
        if self.source == "real":
            return False  # nothing to decode
        return self.step_built is None or step - self.step_built >= self.refresh_every

    @torch.no_grad()
    def refresh(self, model, cfg, prompt_ids: torch.Tensor, step: int):
        """Decode `n_chunks` chunks from a real prompt, writing as we go.

        The rollout writes its own output into the fast weights exactly as
        deployment does, so later cached chunks come from a state the earlier
        ones already degraded. A rollout that did not write would be the
        frozen-generator control, which is measurably harmless and would train
        for the wrong condition.
        """
        was_training = model.training
        model.eval()
        B, CS = prompt_ids.shape[0], cfg.model.mini_batch_size
        st = StreamState(model, B, prompt_ids.device)
        n_prompt = prompt_ids.shape[1] // CS
        for c in range(n_prompt):
            seg = prompt_ids[:, c * CS : (c + 1) * CS]
            tgt = prompt_ids[:, c * CS + 1 : (c + 1) * CS + 1]
            if tgt.shape[1] < CS:  # last chunk has no target token
                break
            st.process_real_chunk(seg, tgt, 1.0, 1.0, cfg)
        out = []
        first = prompt_ids[:, -1:]
        for i in range(self.n_chunks):
            gen = st.generate_chunk(
                first,
                provenance_w=1.0,
                ilr_mult=1.0,
                cfg=cfg,
                seed=self.seed * 100003 + step * 97 + i,
            )
            out.append(gen.detach().clone())
            first = gen[:, -1:]
        self.chunks = out
        self.step_built = step
        if was_training:
            model.train()

    def splice(self, batch: torch.Tensor, cfg) -> tuple[torch.Tensor, list]:
        """Replace the trailing chunks of `batch` with cached generated ones.

        Returns the spliced batch and the per-chunk outer-loss mask. The chunks
        are placed at the end so that every generated chunk is written into a
        state built from real text first, matching a deployment stream that
        begins with a real prefill.
        """
        CS = cfg.model.mini_batch_size
        n_chunks = (batch.shape[1] - 1) // CS
        k = min(self.n_chunks, n_chunks - 1)  # keep at least one real chunk
        if self.source == "real":
            return batch, [c < n_chunks - k for c in range(n_chunks)]
        out = batch.clone()
        B = batch.shape[0]
        for j in range(k):
            c = n_chunks - k + j
            g = self.chunks[j]
            if g.shape[0] < B:  # cache smaller than this batch
                g = g.repeat((B + g.shape[0] - 1) // g.shape[0], 1)[:B]
            out[:, c * CS : (c + 1) * CS] = g[:, :CS]
        mask = [c < n_chunks - k for c in range(n_chunks)]
        return out, mask
