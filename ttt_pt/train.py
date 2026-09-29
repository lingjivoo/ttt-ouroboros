"""Outer (meta) training loop for the PyTorch TTT-E2E port.

Single GPU:  python -m ttt_pt.train --preset 125m-e2e
Multi GPU:   torchrun --nproc-per-node=8 -m ttt_pt.train --preset 125m-e2e
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ttt_pt.config import PRESETS, Config
from ttt_pt.data import make_loader
from ttt_pt.meta import compute_loss
from ttt_pt.model import TTTModel
from ttt_pt.selfgen import SelfGenCache


def lr_at(step: int, o) -> float:
    """warmup_cosine_decay_schedule(init_lr -> lr over warmup, cosine -> end_lr)."""
    if step < o.lr_warmup_steps:
        return o.init_lr + (o.lr - o.init_lr) * step / max(1, o.lr_warmup_steps)
    p = min(1.0, (step - o.lr_warmup_steps) / max(1, o.lr_decay_steps - o.lr_warmup_steps))
    return o.end_lr + 0.5 * (o.lr - o.end_lr) * (1 + math.cos(math.pi * p))


def setup_dist():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
        rank = dist.get_rank()
        world = dist.get_world_size()
        print(
            f"[rank {rank}] local_rank={local_rank} device_count={torch.cuda.device_count()} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )
    else:
        rank, world = 0, 1
    return rank, world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--seq-length", type=int, default=None)
    ap.add_argument("--global-batch-size", type=int, default=None)
    ap.add_argument("--accum-steps", type=int, default=None)
    ap.add_argument("--exp-dir", type=str, default=None)
    ap.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="checkpoint to initialize params from (load_part=params)",
    )
    ap.add_argument("--dataset-path", type=str, default=None)
    ap.add_argument("--exp-name", type=str, default=None)
    ap.add_argument("--save-freq", type=int, default=None)
    ap.add_argument(
        "--zero",
        action="store_true",
        help="shard optimizer state across ranks (ZeRO-1). 760m/32k "
        "reaches 133.8G of a 139.8G card at step 0 and then OOMs "
        "when AdamW allocates its 7.1G of state; sharding that "
        "state leaves the update mathematically unchanged",
    )
    ap.add_argument(
        "--closed-loop-chunks",
        type=int,
        default=0,
        help="meta-train with this many self-decoded chunks at the "
        "end of each sequence. They are written into the fast "
        "weights but excluded from the outer loss, so the "
        "learner is optimised for the case its own output is "
        "the training data. 0 is the ordinary exogenous "
        "objective and must stay bit-identical.",
    )
    ap.add_argument(
        "--no-remat",
        action="store_true",
        help="force the plain second-order path even without "
        "closed-loop chunks. Closed-loop training cannot use "
        "the rematerialised scan, so the exogenous control "
        "has to give it up too: otherwise the two arms differ "
        "in the code path as well as in the treatment.",
    )
    ap.add_argument(
        "--selfgen-source",
        choices=["model", "real"],
        default="model",
        help="what fills the masked chunks. 'real' keeps the "
        "corpus text and applies the same outer-loss mask, "
        "which is the arm that holds the amount of outer "
        "supervision fixed and varies only what is written.",
    )
    ap.add_argument(
        "--refresh-every",
        type=int,
        default=25,
        help="optimiser steps between regenerating the cached "
        "rollouts. 1 is strictly on-policy and costs a full "
        "autoregressive decode every step.",
    )
    args = ap.parse_args()

    cfg: Config = PRESETS[args.preset]()
    t = cfg.training
    for k in (
        "total_steps",
        "seq_length",
        "global_batch_size",
        "accum_steps",
        "dataset_path",
        "exp_name",
        "save_freq",
        "exp_dir",
    ):
        v = getattr(args, k)
        if v is not None:
            setattr(t, k, v)

    rank, world = setup_dist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = rank == 0

    assert t.global_batch_size % (world * t.accum_steps) == 0
    local_bs = t.global_batch_size // (world * t.accum_steps)

    torch.manual_seed(t.model_seed)
    model = TTTModel(cfg.model, max_seq_len=t.seq_length).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if master:
        print(
            f"params: {n_params / 1e6:.1f}M | world={world} local_bs={local_bs} "
            f"mode={t.train_mode} seq={t.seq_length}"
        )

    o = t.optimizer_outer
    # Single param group: optax adamw applies weight decay to ALL params (emb_wd=True)
    adamw_kw = dict(
        lr=o.lr,
        betas=(o.b1, o.b2),
        weight_decay=o.weight_decay,
        eps=1e-8,
        fused=device.type == "cuda",
    )
    sharded_opt = args.zero and world > 1
    if sharded_opt:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        # grads are already averaged across ranks below, which is what ZeRO expects
        opt = ZeroRedundancyOptimizer(
            model.parameters(), optimizer_class=torch.optim.AdamW, **adamw_kw
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), **adamw_kw)

    loader = make_loader(cfg, local_bs, rank, world, seed=t.data_seed)
    data_iter = iter(loader)

    exp_dir = Path(t.exp_dir) / t.exp_name
    if master:
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Resume from the latest checkpoint if one exists (preemption safety).
    start_step = 0
    ckpts = sorted(exp_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts and args.init_from:
        st = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(
            st["model"] if "model" in st else st, strict=False
        )
        assert not [k for k in missing if not k.startswith("rope_")], missing
        assert not unexpected, unexpected
        if master:
            print(f"initialized params from {args.init_from}", flush=True)
    if ckpts:
        st = torch.load(ckpts[-1], map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step = st["step"] + 1
        if master:
            print(f"resumed {ckpts[-1].name}: continuing at step {start_step}", flush=True)
        # Fast-forward the data stream to reproduce the exact data order.
        for _ in range(start_step * t.accum_steps):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                next(data_iter)

    log_t0 = time.time()
    selfgen = None
    if args.closed_loop_chunks > 0:
        selfgen = SelfGenCache(
            args.closed_loop_chunks, args.refresh_every, seed=rank, source=args.selfgen_source
        )
        if master:
            print(
                f"closed-loop meta-training: {args.closed_loop_chunks} "
                f"{args.selfgen_source}-sourced masked chunks/sequence, "
                f"refreshed every {args.refresh_every} steps, remat disabled",
                flush=True,
            )

    for step in range(start_step, t.total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, o)
        opt.zero_grad(set_to_none=True)

        loss_acc = 0.0
        for _ in range(t.accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = batch.to(device, non_blocking=True)
            omask = None
            if selfgen is not None:
                if selfgen.stale(step):
                    selfgen.refresh(model, cfg, batch[:, :-1], step)
                batch, omask = selfgen.splice(batch, cfg)
            loss, _aux = compute_loss(
                model,
                batch,
                cfg,
                step,
                create_graph=True,
                use_remat=(selfgen is None and not args.no_remat),
                outer_chunk_mask=omask,
            )
            (loss / t.accum_steps).backward()
            loss_acc += loss.item() / t.accum_steps

        if world > 1:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), o.clip_gradient)
        opt.step()

        if master and (step % t.log_freq == 0 or step == t.total_steps - 1):
            dt = time.time() - log_t0
            log_t0 = time.time()
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
            print(
                f"step {step:5d} | loss {loss_acc:.4f} | gnorm {gnorm:.3f} "
                f"| lr {lr_at(step, o):.2e} | {dt / max(1, t.log_freq):.2f}s/it "
                f"| mem {mem:.1f}G",
                flush=True,
            )

        if t.save_freq > 0 and ((step + 1) % t.save_freq == 0 or step == t.total_steps - 1):
            # Every rank has to reach this: gathering the shards is collective, so
            # it cannot sit inside the master-only branch or the others deadlock.
            if sharded_opt:
                opt.consolidate_state_dict(to=0)
            if master:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step,
                        "cfg": vars(t),
                    },
                    exp_dir / f"ckpt_{step + 1}.pt",
                )

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
