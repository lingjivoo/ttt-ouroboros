"""Short fixed-stream pilot for the atomic block update and Settlement.

Each stream uses disjoint regions of one PG-19 validation book: early chunks
for online adaptation, later chunks for block-end evidence, and a still later
clean endpoint probe. This is a pilot, not a canonical paper protocol or a
closed-loop generation experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ttt_pt.block_inner import BlockStreamState
from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel


def find_books(tokens, minimum):
    bos = np.flatnonzero(tokens == 128000)
    return [(int(start), int(bos[i + 1]) if i + 1 < len(bos) else len(tokens))
            for i, start in enumerate(bos)
            if (bos[i + 1] if i + 1 < len(bos) else len(tokens)) - start >= minimum]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, obj):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(temp, target)


def clean_probe(state, inputs, targets, cfg):
    snap = state.snapshot()
    nll = state.process_real_chunk(inputs, targets, 0.0, 1.0, cfg)
    state.restore(snap)
    return [float(x) for x in nll]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--preset", default="125m-e2e-ext32k", choices=sorted(PRESETS))
    ap.add_argument("--policy", required=True, choices=("off", "direct", "settle"))
    ap.add_argument("--block-chunks", type=int, required=True)
    ap.add_argument("--stream-chunks", type=int, default=16)
    ap.add_argument("--book-indices", default="2,3,4,5")
    ap.add_argument("--direct-dose", type=float, default=1.0,
                    help="post-clipping update multiplier for the direct arm")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--wait-for-free-gib", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.block_chunks < 1 or args.stream_chunks < 1:
        ap.error("chunk counts must be positive")
    if args.stream_chunks % args.block_chunks:
        ap.error("stream-chunks must be divisible by block-chunks")
    if args.margin < 0:
        ap.error("margin must be nonnegative")
    if args.direct_dose < 0 or args.direct_dose > 1:
        ap.error("direct-dose must lie in [0, 1]")
    if not torch.cuda.is_available():
        ap.error("CUDA unavailable")
    torch.empty(1, device="cuda")
    if args.wait_for_free_gib:
        deadline = time.monotonic() + 30
        while torch.cuda.mem_get_info()[0] < args.wait_for_free_gib * 1024**3:
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU holder did not yield enough memory")
            time.sleep(1)

    cfg = PRESETS[args.preset]()
    cs = cfg.model.mini_batch_size
    evidence_start = 32 * cs
    endpoint_start = 64 * cs
    minimum = endpoint_start + 2 * cs + 1
    source = np.load(args.tokens, mmap_mode="r")
    eligible = find_books(source, minimum)
    indices = [int(x) for x in args.book_indices.split(",")]
    if any(i < 0 or i >= len(eligible) for i in indices):
        ap.error("book index is out of range for the eligible long-book set")
    starts = [eligible[i][0] for i in indices]
    data = np.stack([np.asarray(source[s:s + minimum], dtype=np.int64) for s in starts])
    seq = torch.from_numpy(data.copy()).cuda()
    model = TTTModel(cfg.model, max_seq_len=(args.stream_chunks + 2) * cs).cuda().eval()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    del payload
    state = BlockStreamState(
        model, len(indices), torch.device("cuda"), args.block_chunks,
        parallel_update=True, defer_updates=args.policy == "settle",
    )
    end_in = seq[:, endpoint_start:endpoint_start + cs]
    end_tgt = seq[:, endpoint_start + 1:endpoint_start + cs + 1]
    output = {
        "status": "running",
        "protocol": {
            "policy": args.policy, "block_chunks": args.block_chunks,
            "stream_chunks": args.stream_chunks, "chunk_tokens": cs,
            "book_indices_among_long_books": indices,
            "minimum_book_tokens": minimum,
            "write_region": [0, args.stream_chunks * cs],
            "evidence_region_start": evidence_start,
            "endpoint_probe_start": endpoint_start,
            "margin": args.margin, "doses": [0.0, 0.5, 1.0],
            "direct_dose": args.direct_dose if args.policy == "direct" else None,
            "evidence_policy": "one disjoint real chunk per block; branch only",
            "checkpoint_sha256": sha256(args.ckpt),
            "tokens_sha256": sha256(args.tokens),
        },
        "steps": [],
    }
    started = time.monotonic()
    with torch.no_grad():
        output["initial_endpoint_nll"] = clean_probe(state, end_in, end_tgt, cfg)
        for c in range(args.stream_chunks):
            sl = slice(c * cs, (c + 1) * cs)
            state.process_real_chunk(
                seq[:, sl], seq[:, c * cs + 1:(c + 1) * cs + 1],
                0.0 if args.policy == "off" else 1.0,
                args.direct_dose if args.policy == "direct" else 1.0, cfg,
            )
            if (c + 1) % args.block_chunks:
                continue
            row = {"completed_stream_chunks": c + 1}
            if args.policy == "settle":
                e = (c + 1) // args.block_chunks - 1
                q0 = evidence_start + e * cs
                qin = seq[:, q0:q0 + cs]
                qtgt = seq[:, q0 + 1:q0 + cs + 1]
                decision = state.settle_on_external(qin, qtgt, cfg, margin=args.margin)
                row["dose_per_book"] = [float(x) for x in decision["dose"]]
                row["evidence_nll_options"] = decision["scores"].cpu().tolist()
            if (c + 1) % 4 == 0 or c + 1 == args.stream_chunks:
                row["endpoint_nll_per_book"] = clean_probe(state, end_in, end_tgt, cfg)
            output["steps"].append(row)
            save(args.out, output)
        output["final_endpoint_nll"] = clean_probe(state, end_in, end_tgt, cfg)
    output.update(status="passed", elapsed_s=time.monotonic() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
    save(args.out, output)
    print(json.dumps({"status": output["status"], "policy": args.policy,
                      "K": args.block_chunks, "books": indices,
                      "initial": output["initial_endpoint_nll"],
                      "final": output["final_endpoint_nll"],
                      "elapsed_s": output["elapsed_s"]}))


if __name__ == "__main__":
    main()
