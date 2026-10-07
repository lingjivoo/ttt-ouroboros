"""One suffix backward per block, one prefix backward per sequence.

The prefix graph is preserved through a detached activation bridge. Suffix KV
is detached only at block boundaries. This is a truncated first-order outer
gradient, with the same forward block update as block_inner.py. No gate.
"""

import torch

from ttt_pt.block_inner import suffix_block_forward
from ttt_pt.meta import clip_per_sample, masked_ce


def loss_no_inner(model, inputs, targets, mask, cfg):
    """Ordinary LM objective, SAME prime architecture, no inner writes."""
    h = model.prefix_forward(inputs)
    fast = model.init_fast_weights(inputs.shape[0])
    kv = model.init_kv_caches(inputs.shape[0], inputs.device)
    logits, _ = suffix_block_forward(model, h, fast, kv, 0, double_backward=False)
    cs = cfg.model.mini_batch_size
    return torch.stack(
        [
            masked_ce(logits[:, s : s + cs], targets[:, s : s + cs], mask[:, s : s + cs])[0].mean()
            for s in range(0, inputs.shape[1], cs)
        ]
    ).mean()


def backward_shared(
    model,
    inputs,
    targets,
    mask,
    cfg,
    *,
    block_chunks=2,
    outer_scale=1.0,
    inner_multiplier=1.0,
    return_fast=False,
):
    """Accumulate outer grads in-place; returned loss is detached.

    Undo both outer averaging and accumulation scaling BEFORE inner clipping.
    Recreate the initialization identity path per block. Slow weights must
    remain unchanged until this function has completed.
    """
    if outer_scale <= 0 or block_chunks < 1:
        raise ValueError("positive outer_scale and block_chunks required")
    batch, length = inputs.shape
    cs = cfg.model.mini_batch_size
    if length % cs:
        raise ValueError("sequence must contain whole chunks")
    chunks = length // cs
    h = model.prefix_forward(inputs)
    bridge = h.detach().requires_grad_(True)
    base = [p.detach().unsqueeze(0).repeat(batch, 1, 1) for p in model.prime_params()]
    values = [x.clone() for x in base]
    kv = model.init_kv_caches(batch, inputs.device)
    losses, norms, clips = [], [], []
    for start in range(0, chunks, block_chunks):
        stop = min(chunks, start + block_chunks)
        fast = [
            p.unsqueeze(0).repeat(batch, 1, 1) + (v - b)
            for p, v, b in zip(model.prime_params(), values, base)
        ]
        for f in fast:
            f.retain_grad()
        sl = slice(start * cs, stop * cs)
        logits, next_kv = suffix_block_forward(
            model, bridge[:, sl], fast, kv, start, double_backward=False
        )
        rows = torch.stack(
            [
                masked_ce(
                    logits[:, j * cs : (j + 1) * cs],
                    targets[:, (start + j) * cs : (start + j + 1) * cs],
                    mask[:, (start + j) * cs : (start + j + 1) * cs],
                )[0]
                for j in range(stop - start)
            ]
        )
        weight = outer_scale * (stop - start) / chunks
        (rows.mean() * weight).backward()
        losses.extend(rows.detach().mean(1).unbind())
        kv = [(k.detach(), v.detach()) for k, v in next_kv]
        if stop < chunks or return_fast:
            grads = [f.grad.detach() * (batch / weight) for f in fast]
            with torch.no_grad():
                norm = sum(g.float().square().reshape(batch, -1).sum(1) for g in grads).sqrt()
                clip = cfg.training.optimizer_inner.clip_gradient
                clips.append((norm > clip).float() if clip > 0 else torch.zeros_like(norm))
                if clip > 0:
                    grads = clip_per_sample(grads, clip)
                delta = [cfg.training.optimizer_inner.lr * inner_multiplier * g for g in grads]
                values = [f.detach() - d for f, d in zip(fast, delta)]
                norms.append(
                    sum(d.float().square().reshape(batch, -1).sum(1) for d in delta).sqrt()
                )
        else:
            values = [f.detach() for f in fast]
        del logits, rows, fast, next_kv
    # Each prefix layer is traversed just once. This retains cross-block
    # prefix attention gradients; only the suffix state graph is truncated.
    h.backward(bridge.grad)
    return torch.stack(losses).mean(), dict(
        fast=values,
        mean_update_norm=float(torch.cat(norms).mean()) if norms else 0.0,
        clipping_frequency=float(torch.cat(clips).mean()) if clips else 0.0,
    )
