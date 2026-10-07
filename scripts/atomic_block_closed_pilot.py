"""Short generated-stream pilot for Atomic Block Settlement.

Real prefill is shared by all arms. Generated chunks then use current fast
weights; the settle arm validates each whole-block proposal on disjoint clean
book text before the next generated block. This is deliberately shorter than
the 128K paper protocol.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from atomic_block_pilot import clean_probe, find_books, save, sha256
from ttt_pt.block_inner import BlockStreamState
from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel


def diversity(tokens):
    result = []
    for row in tokens:
        ids = row.tolist()
        d2 = len(set(zip(ids[:-1], ids[1:]))) / max(1, len(ids) - 1)
        seen = set()
        repeated = 0
        for j in range(max(0, len(ids) - 3)):
            four = tuple(ids[j:j + 4])
            repeated += four in seen
            seen.add(four)
        result.append({"distinct2": d2,
                       "repeated4": repeated / max(1, len(ids) - 3)})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--preset", default="125m-e2e-ext32k", choices=sorted(PRESETS))
    ap.add_argument("--policy", required=True, choices=("off", "direct", "settle"))
    ap.add_argument("--block-chunks", type=int, default=2)
    ap.add_argument("--generated-chunks", type=int, default=16)
    ap.add_argument("--prefill-chunks", type=int, default=4)
    ap.add_argument("--book-indices", default="2,3,4,5")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--wait-for-free-gib", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if (args.block_chunks < 1 or args.generated_chunks < 1 or
            args.generated_chunks % args.block_chunks):
        ap.error("generated-chunks must be positive and divisible by block-chunks")
    if args.prefill_chunks < 1 or args.prefill_chunks % args.block_chunks:
        ap.error("prefill-chunks must be positive and divisible by block-chunks")
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
    minimum = endpoint_start + cs + 1
    source = np.load(args.tokens, mmap_mode="r")
    eligible = find_books(source, minimum)
    indices = [int(x) for x in args.book_indices.split(",")]
    if any(i < 0 or i >= len(eligible) for i in indices):
        ap.error("book index out of range")
    starts = [eligible[i][0] for i in indices]
    seq = torch.from_numpy(np.stack([
        np.asarray(source[s:s + minimum], dtype=np.int64) for s in starts
    ]).copy()).cuda()
    model = TTTModel(
        cfg.model,
        max_seq_len=(args.prefill_chunks + args.generated_chunks + 2) * cs,
    ).cuda().eval()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    del payload
    state = BlockStreamState(
        model, len(indices), torch.device("cuda"), args.block_chunks,
        parallel_update=True, defer_updates=False,
    )
    end_in = seq[:, endpoint_start:endpoint_start + cs]
    end_tgt = seq[:, endpoint_start + 1:endpoint_start + cs + 1]
    output = {
        "status": "running",
        "protocol": {
            "policy": args.policy, "block_chunks": args.block_chunks,
            "generated_chunks": args.generated_chunks,
            "prefill_chunks": args.prefill_chunks,
            "chunk_tokens": cs, "book_indices_among_long_books": indices,
            "seed": args.seed, "temperature": 1.0, "top_p": 0.95,
            "boundary": "historical first, then generated continuation",
            "evidence_region_start": evidence_start,
            "endpoint_probe_start": endpoint_start,
            "margin": args.margin, "doses": [0.0, 0.5, 1.0],
            "checkpoint_sha256": sha256(args.ckpt),
            "tokens_sha256": sha256(args.tokens),
        },
        "steps": [],
    }
    started = time.monotonic()
    generated = []
    with torch.no_grad():
        for c in range(args.prefill_chunks):
            sl = slice(c * cs, (c + 1) * cs)
            state.process_real_chunk(
                seq[:, sl], seq[:, c * cs + 1:(c + 1) * cs + 1],
                1.0, 1.0, cfg,
            )
        if args.policy == "settle":
            state.defer_updates = True
        output["post_prefill_endpoint_nll"] = clean_probe(state, end_in, end_tgt, cfg)
        first = seq[:, args.prefill_chunks * cs:args.prefill_chunks * cs + 1]
        for c in range(args.generated_chunks):
            gen = state.generate_chunk(
                first, 0.0 if args.policy == "off" else 1.0,
                1.0, cfg, temperature=1.0, top_p=0.95,
                seed=args.seed * 100000 + c, sampling_device="cuda",
            )
            generated.append(gen.cpu())
            first = gen[:, -1:]
            if (c + 1) % args.block_chunks:
                continue
            row = {"completed_generated_chunks": c + 1}
            if args.policy == "settle":
                evidence_id = (c + 1) // args.block_chunks - 1
                q0 = evidence_start + evidence_id * cs
                decision = state.settle_on_external(
                    seq[:, q0:q0 + cs], seq[:, q0 + 1:q0 + cs + 1],
                    cfg, margin=args.margin,
                )
                row["dose_per_book"] = [float(x) for x in decision["dose"]]
                row["evidence_nll_options"] = decision["scores"].cpu().tolist()
            if (c + 1) % 4 == 0 or c + 1 == args.generated_chunks:
                row["endpoint_nll_per_book"] = clean_probe(state, end_in, end_tgt, cfg)
            output["steps"].append(row)
            save(args.out, output)
        output["final_endpoint_nll"] = clean_probe(state, end_in, end_tgt, cfg)
    output["generation_diversity_per_book"] = diversity(torch.cat(generated, 1))
    output.update(status="passed", elapsed_s=time.monotonic() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
    save(args.out, output)
    print(json.dumps({"status": output["status"], "policy": args.policy,
                      "K": args.block_chunks, "books": indices,
                      "final": output["final_endpoint_nll"],
                      "elapsed_s": output["elapsed_s"]}))


if __name__ == "__main__":
    main()
