"""E4 pilot: replay fixed text with repetition-aware write interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from scripts.horizon import CS, WARMUP, drift, find_books
from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState

ARMS = (
    "standard",
    "repeat_downweight",
    "random_downweight",
    "uniform_rescale",
    "writes_off",
)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def repetition_weights(chunks, seed):
    """Zero targets completing a previously seen 4-gram; past-only rule."""
    bsz = chunks[0][1].shape[0]
    history = [[] for _ in range(bsz)]
    seen = [set() for _ in range(bsz)]
    weights, random_weights = [], []
    rng = random.Random(seed)
    for first, gen in chunks:
        w = torch.ones(bsz, CS)
        for b in range(bsz):
            history[b].append(int(first[b, 0]))
            for i, token in enumerate(gen[b].tolist()):
                if len(history[b]) >= 3:
                    gram = tuple(history[b][-3:] + [int(token)])
                    if gram in seen[b]:
                        w[b, i] = 0.0
                    seen[b].add(gram)
                history[b].append(int(token))
        rw = torch.empty_like(w)
        for b in range(bsz):
            order = list(range(CS))
            rng.shuffle(order)
            rw[b] = w[b, order]
        weights.append(w)
        random_weights.append(rw)
    return weights, random_weights


def load_sources(record_path, fixed_path, n_chunks):
    rec = torch.load(record_path, map_location="cpu", weights_only=False)
    stream = rec["stream"]

    def convert(items):
        return [(first.long(), gen.long()) for _c, first, gen in items]

    fixed = json.load(open(fixed_path))
    fixed_chunks = [
        (
            torch.tensor(x["first_input"]).view(-1, 1).long(),
            torch.tensor(x["tokens"]).long(),
        )
        for x in fixed["chunks"][:n_chunks]
    ]
    return (
        {
            "closed_early": convert(stream[:n_chunks]),
            "closed_late": convert(stream[-n_chunks:]),
            "fixed_w0": fixed_chunks,
        },
        rec["manifest"],
        fixed["manifest"],
    )


def probe_books(st, x, y, cfg):
    snap = st.snapshot()
    vals = st.process_real_chunk(x, y, 0.0, 1.0, cfg).float().cpu().tolist()
    st.restore(snap)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--fixed-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-source-chunks", type=int, default=8)
    ap.add_argument("--receiver-book-offset", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()
    started = time.time()
    cfg = PRESETS["125m-e2e-ext32k"]()
    dev = "cuda"
    model = (
        TTTModel(cfg.model, max_seq_len=(WARMUP + args.n_source_chunks + 2) * CS)
        .to(dev)
        .eval()
    )
    raw = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(raw.get("model", raw), strict=True)
    del raw
    data = np.asarray(np.load(args.val, mmap_mode="r"))
    need = (WARMUP + 2) * CS + 1
    books = find_books(data, need)[
        args.receiver_book_offset : args.receiver_book_offset + 8
    ]
    assert len(books) == 8
    real = torch.from_numpy(
        np.stack([data[s : s + need].astype(np.int64) for s, _ in books])
    )
    base = StreamState(model, 8, dev)
    for c in range(WARMUP):
        base.process_real_chunk(
            real[:, c * CS : (c + 1) * CS].to(dev),
            real[:, c * CS + 1 : (c + 1) * CS + 1].to(dev),
            1.0,
            1.0,
            cfg,
        )
    base_snap = base.snapshot()
    qx = real[:, WARMUP * CS : (WARMUP + 1) * CS].to(dev)
    qy = real[:, WARMUP * CS + 1 : (WARMUP + 1) * CS + 1].to(dev)
    sources, record_manifest, fixed_manifest = load_sources(
        args.record, args.fixed_json, args.n_source_chunks
    )
    results = {}
    torch.cuda.reset_peak_memory_stats()
    for source_name, chunks in sources.items():
        rep_w, rand_w = repetition_weights(chunks, args.seed)
        source_out = {
            "repeat_position_fraction": float(1.0 - torch.stack(rep_w).mean()),
            "arms": {},
        }
        for arm in ARMS:
            st = StreamState(model, 8, dev)
            st.restore(base_snap)
            update_norms = []
            for i, (first, gen) in enumerate(chunks):
                x = torch.cat([first, gen[:, :-1]], 1).to(dev)
                y = gen.to(dev)
                if arm == "writes_off":
                    st.process_real_chunk(x, y, 0.0, 1.0, cfg)
                    update_norms.append(0.0)
                elif arm == "uniform_rescale":
                    st.defer = True
                    st.process_real_chunk(x, y, 1.0, 1.0, cfg)
                    delta = st.pending.pop()
                    alpha = float(rep_w[i].mean())
                    st.fast = [(f + alpha * d).detach() for f, d in zip(st.fast, delta)]
                    st.defer = False
                    update_norms.append(
                        float(np.mean(st.last_update_stats["update_norm"])) * alpha
                    )
                else:
                    tw = (
                        rep_w[i]
                        if arm == "repeat_downweight"
                        else (rand_w[i] if arm == "random_downweight" else None)
                    )
                    st.process_real_chunk(
                        x, y, 1.0, 1.0, cfg, token_w=None if tw is None else tw.to(dev)
                    )
                    update_norms.append(
                        float(np.mean(st.last_update_stats["update_norm"]))
                    )
            vals = probe_books(st, qx, qy, cfg)
            source_out["arms"][arm] = {
                "probe_nll": vals,
                "mean_probe_nll": float(np.mean(vals)),
                "final_drift": drift(st),
                "mean_update_norm": float(np.mean(update_norms)),
                "sum_update_norm": float(np.sum(update_norms)),
            }
        results[source_name] = source_out
    out = {
        "status": "passed",
        "config": vars(args),
        "checkpoint_sha256": sha(args.ckpt),
        "val_sha256": sha(args.val),
        "record_sha256": sha(args.record),
        "fixed_json_sha256": sha(args.fixed_json),
        "code_sha256": sha(__file__),
        "receiver_book_spans": books,
        "source_record_books": record_manifest.get("books"),
        "fixed_source_books": fixed_manifest.get("receiver_book_bounds"),
        "loss_normalization": "token weights divide by the original valid-token count",
        "repeat_rule": "weight 0 iff target completes a 4-gram seen in current/past source text",
        "random_rule": "same per-row, per-chunk weight multiset under fixed permutation",
        "uniform_rule": "scale the standard post-clipping delta by the repeat-weight mean",
        "results": results,
        "seconds": time.time() - started,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out + ".tmp"
    json.dump(out, open(tmp, "w"), indent=2)
    os.replace(tmp, args.out)
    print(
        json.dumps(
            {"status": "passed", "seconds": out["seconds"], "sources": list(results)}
        )
    )


if __name__ == "__main__":
    main()
