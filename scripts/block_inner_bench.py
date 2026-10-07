"""Small, signed comparison of original and block inner-loop compute paths.

Example (run only on an available GPU):
  python scripts/block_inner_bench.py --ckpt CHECKPOINT --tokens PG19_NPY \
    --preset 125m-e2e-ext32k --chunks 4 --block-chunks 2 \
    --implementation block-parallel --mode eval --out run.json

This measures fixed teacher-forced input, not closed-loop quality. It does not
alter the checkpoint or train the model. For meta-training timing, use
``--mode outer``; that includes the second-order outer backward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from ttt_pt.block_inner import loss_for_sequence_block
from ttt_pt.config import PRESETS, Config, ModelConfig, TrainingConfig
from ttt_pt.meta import loss_for_sequence_meta
from ttt_pt.model import TTTModel


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--tokens", required=True, help="flat token-ID .npy file")
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--block-chunks", type=int, default=2)
    ap.add_argument("--start-token", type=int, default=0)
    ap.add_argument("--implementation", required=True,
                    choices=("original", "block-serial", "block-parallel"))
    ap.add_argument("--mode", choices=("eval", "outer"), default="eval")
    ap.add_argument("--first-order", action="store_true",
                    help="stop the outer gradient through the inner gradient")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--wait-for-free-gib", type=float, default=0.0,
                    help="after establishing a CUDA context, wait up to 30 s for "
                         "an occupancy process to self-yield this much free VRAM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-tiny", action="store_true",
                    help="exercise the benchmark path on a random tiny model")
    args = ap.parse_args()
    if args.chunks < 1 or args.block_chunks < 1 or args.reps < 1 or args.warmup < 0:
        ap.error("chunk counts and reps must be positive; warmup may be zero")
    if args.implementation == "original" and args.block_chunks != 1:
        ap.error("original implementation requires --block-chunks 1")
    if args.first_order and args.implementation == "original":
        ap.error("--first-order is only implemented for block rules")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA device unavailable")
    if not args.random_tiny and not args.ckpt:
        ap.error("--ckpt is required except with --random-tiny")

    if args.device == "cuda" and args.wait_for_free_gib > 0:
        torch.empty(1, device="cuda")  # presence lets a cooperative holder yield
        deadline = time.monotonic() + 30
        while torch.cuda.mem_get_info()[0] < args.wait_for_free_gib * 1024**3:
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU holder did not yield enough free memory")
            time.sleep(1)

    if args.random_tiny:
        cfg = Config(
            model=ModelConfig(
                vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=3, num_attention_heads=4,
                mini_batch_size=8, sliding_window_size=16, bos_token_id=1,
                prime=True, suffix_len=1, seq_modeling_block="SWA",
            ),
            training=TrainingConfig(seq_length=args.chunks * 8),
        )
    else:
        cfg = PRESETS[args.preset]()
    CS = cfg.model.mini_batch_size
    T = args.chunks * CS
    data = np.load(args.tokens, mmap_mode="r")
    if args.start_token < 0 or args.start_token + T + 1 > len(data):
        ap.error("requested token slice is out of bounds")
    token_ids = np.array(data[args.start_token : args.start_token + T + 1],
                         dtype=np.int64, copy=True)
    batch = torch.from_numpy(token_ids).unsqueeze(0).to(args.device)
    inputs, targets = batch[:, :-1], batch[:, 1:]
    mask = targets != cfg.model.bos_token_id

    torch.manual_seed(0)
    model = TTTModel(
        cfg.model,
        max_seq_len=max(T + CS, cfg.model.sliding_window_size + args.block_chunks * CS),
    ).to(args.device)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        missing, unexpected = model.load_state_dict(
            checkpoint["model"] if "model" in checkpoint else checkpoint,
            strict=False,
        )
        if missing or unexpected:
            raise ValueError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.train(args.mode == "outer")

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def run_once():
        model.zero_grad(set_to_none=True)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sync()
        start = time.perf_counter()
        if args.implementation == "original":
            loss, aux = loss_for_sequence_meta(
                model, inputs, targets, mask, 1.0, cfg,
                create_graph=args.mode == "outer", return_fast=True,
            )
        else:
            loss, aux = loss_for_sequence_block(
                model, inputs, targets, mask, 1.0, cfg,
                args.block_chunks, create_graph=args.mode == "outer",
                return_fast=True,
                parallel_read=args.implementation == "block-parallel",
                first_order=args.first_order,
            )
        if args.mode == "outer":
            loss.backward()
        sync()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        drift_sq = sum(float((f.detach() - p.detach().unsqueeze(0)).float().pow(2).sum())
                       for f, p in zip(aux["fast"], model.prime_params()))
        return {"loss": float(loss.detach()), "elapsed_s": elapsed,
                "tokens_per_s": T / elapsed, "peak_allocated_bytes": peak,
                "fast_delta_norm": drift_sq ** 0.5}

    for _ in range(args.warmup):
        run_once()
    runs = [run_once() for _ in range(args.reps)]
    output = {
        "config": {**vars(args), "torch": torch.__version__,
                   "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                   "checkpoint_sha256": sha256(args.ckpt) if args.ckpt else None,
                   "tokens_sha256": sha256(args.tokens)},
        "runs": runs,
        "median_elapsed_s": statistics.median(x["elapsed_s"] for x in runs),
        "median_tokens_per_s": statistics.median(x["tokens_per_s"] for x in runs),
        "max_peak_allocated_bytes": max((x["peak_allocated_bytes"] or 0) for x in runs),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(output, indent=2) + "\n")
    os.replace(temp, path)
    print(json.dumps({k: v for k, v in output.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
