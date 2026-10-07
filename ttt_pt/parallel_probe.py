"""Score independent fast-weight branches on the same clean text.

This is a research utility for Settlement. It batches counterfactual *reads*;
it does not parallelize autoregressive generation or decide which writes to
commit. Branch zero is the unchanged state. Other branches are additive fast-
weight offsets from that state and share its attention context.
"""

from __future__ import annotations

import torch

from ttt_pt.stream import StreamState


@torch.no_grad()
def score_fast_weight_branches(
    state: StreamState,
    tokens_in: torch.Tensor,
    tokens_tgt: torch.Tensor,
    offsets: list[list[torch.Tensor]],
    cfg,
    max_branches: int = 4,
) -> torch.Tensor:
    """Return clean-text NLL [1 + len(offsets), B] without changing ``state``.

    The leading row is the live state. ``max_branches`` bounds the number of
    copies of KV caches and fast weights resident in a single forward. All
    offsets must have been computed from the same live fast-weight version.
    """
    if max_branches < 1:
        raise ValueError("max_branches must be positive")
    if state.h_buffer or state.tok_buffer:
        raise ValueError("branch scoring requires a completed chunk")
    if tokens_in.shape != (state.B, state.CS) or tokens_tgt.shape != tokens_in.shape:
        raise ValueError("probe tokens must have shape [B, chunk_size]")
    for offset in offsets:
        if len(offset) != len(state.fast):
            raise ValueError("each offset must cover every fast-weight tensor")
        if any(d.shape != f.shape or d.device != f.device for d, f in zip(offset, state.fast)):
            raise ValueError("offset shapes and devices must match the live fast weights")

    branches = [None, *offsets]
    scores = []
    for start in range(0, len(branches), max_branches):
        group = branches[start : start + max_branches]
        # Force the ordinary read path even if the source is a state subclass
        # with a custom write scheduler (for example BlockStreamState).
        clone = StreamState.__new__(StreamState)
        clone.__dict__ = state.__dict__.copy()
        clone.B = state.B * len(group)
        clone.pre_kv = [
            (torch.cat([k] * len(group), dim=0), torch.cat([v] * len(group), dim=0))
            for k, v in state.pre_kv
        ]
        clone.suf_kv = [
            (torch.cat([k] * len(group), dim=0), torch.cat([v] * len(group), dim=0))
            for k, v in state.suf_kv
        ]
        clone.suf_partial = [
            (torch.cat([k] * len(group), dim=0), torch.cat([v] * len(group), dim=0))
            for k, v in state.suf_partial
        ]
        clone.fast = [
            torch.cat([f if d is None else f + d[j] for d in group], dim=0)
            for j, f in enumerate(state.fast)
        ]
        clone.fast_init = [torch.cat([f] * len(group), dim=0) for f in state.fast_init]
        clone.pending = []
        clone.ref_kv = None
        clone.h_buffer = []
        clone.tok_buffer = []
        nll = clone.process_real_chunk(
            tokens_in.repeat(len(group), 1),
            tokens_tgt.repeat(len(group), 1),
            provenance_w=0.0,
            ilr_mult=1.0,
            cfg=cfg,
        )
        scores.append(nll.reshape(len(group), state.B))
    return torch.cat(scores, dim=0)


@torch.no_grad()
def score_block_settlement(
    state: StreamState,
    tokens_in: torch.Tensor,
    tokens_tgt: torch.Tensor,
    pending: list[list[torch.Tensor]],
    cfg,
    doses: tuple[float, ...] = (0.5, 1.0),
    margin: float = 0.0,
    max_branches: int = 4,
):
    """Compare fixed doses of one atomic block write on one passage.

    This returns one proposed dose per independent stream row and all scores;
    it never commits. The forward is batched, but rows never vote for one
    another's update.
    The caller must verify the live weight version, commit the same delta
    scored here, and evaluate results on separate future text. Multiple
    individually proposed updates need their own explicit joint protocol;
    averaging them here would silently change the training-time update rule.
    """
    if len(pending) != 1:
        raise ValueError("atomic settlement requires exactly one pending block")
    if not doses or any(d <= 0 or d > 1 for d in doses):
        raise ValueError("doses must lie in (0, 1]")
    if tuple(sorted(set(doses))) != doses:
        raise ValueError("doses must be unique and ascending")
    if margin < 0:
        raise ValueError("margin cannot be negative")
    candidate_delta = pending[0]
    if len(candidate_delta) != len(state.fast):
        raise ValueError("candidate delta must cover every fast-weight tensor")
    offsets = [[d * dose for d in candidate_delta] for dose in doses]
    scores = score_fast_weight_branches(
        state, tokens_in, tokens_tgt, offsets, cfg, max_branches,
    )
    eligible = scores[1:] < scores[0:1] - margin
    candidate_scores = scores[1:].masked_fill(~eligible, float("inf"))
    best = candidate_scores.argmin(dim=0)
    dose_values = scores.new_tensor(doses)
    chosen = torch.where(eligible.any(dim=0), dose_values[best], scores.new_zeros(scores.shape[1]))
    return chosen, scores, candidate_delta
