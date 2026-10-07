"""Compare serial and batched Settlement reads on one fixed clean chunk.

This benchmarks only counterfactual validation forwards. It does not include
candidate generation, updates, or online policy effects.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import torch

from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.parallel_probe import score_fast_weight_branches
from ttt_pt.stream import StreamState


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--wait-for-free-gib", type=float, default=0.0)
    args = ap.parse_args()
    if args.warmup < 0 or args.reps < 1:
        ap.error("warmup must be nonnegative and reps positive")
    if not torch.cuda.is_available():
        ap.error("CUDA device unavailable")

    torch.empty(1, device="cuda")
    if args.wait_for_free_gib:
        deadline = time.monotonic() + 30
        while torch.cuda.mem_get_info()[0] < args.wait_for_free_gib * 1024**3:
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU holder did not yield enough memory")
            time.sleep(1)

    cfg = PRESETS[args.preset]()
    cs = cfg.model.mini_batch_size
    ids = np.load(args.tokens, mmap_mode="r")
    if len(ids) < 2 * cs + 1:
        ap.error("token file must contain at least two chunks plus one target")
    seq = torch.from_numpy(np.array(ids[: 2 * cs + 1], dtype=np.int64, copy=True))
    seq = seq.unsqueeze(0).cuda()
    model = TTTModel(cfg.model, max_seq_len=2 * cs + cfg.model.sliding_window_size).cuda().eval()
    checkpoint = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    state = StreamState(model, 1, torch.device("cuda"))
    state.process_real_chunk(seq[:, :cs], seq[:, 1 : cs + 1], 0.0, 1.0, cfg)
    probe_in, probe_tgt = seq[:, cs : 2 * cs], seq[:, cs + 1 : 2 * cs + 1]
    # Small deterministic offsets exercise the branch path without presuming
    # a particular candidate update or affecting the timing interpretation.
    torch.manual_seed(7)
    delta = [torch.randn_like(f) * 1e-4 for f in state.fast]
    offsets = [[d * dose for d in delta] for dose in (0.5, 1.0)]

    def serial():
        scores = [state.probe_score(probe_in, probe_tgt, cfg)]
        for offset in offsets:
            snap = state.snapshot()
            state.fast = [f + d for f, d in zip(state.fast, offset)]
            scores.append(state.probe_score(probe_in, probe_tgt, cfg))
            state.restore(snap)
        return torch.tensor(scores)

    def batched():
        return score_fast_weight_branches(
            state, probe_in, probe_tgt, offsets, cfg, max_branches=3,
        ).mean(1).cpu()

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        runs = []
        for _ in range(args.reps):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            scores = fn()
            torch.cuda.synchronize()
            runs.append({
                "elapsed_s": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "scores": scores.tolist(),
            })
        return runs

    with torch.no_grad():
        old = serial()
        new = batched()
        serial_runs = measure(serial)
        batched_runs = measure(batched)
    output = {
        "device": torch.cuda.get_device_name(),
        "chunk_tokens": cs,
        "branches": 3,
        "max_score_difference": float((old - new).abs().max()),
        "serial": serial_runs,
        "batched": batched_runs,
        "serial_median_s": statistics.median(x["elapsed_s"] for x in serial_runs),
        "batched_median_s": statistics.median(x["elapsed_s"] for x in batched_runs),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
