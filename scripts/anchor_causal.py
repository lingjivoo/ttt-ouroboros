"""Matched anchor replacement controls for the 125M closed loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from horizon import CS, WARMUP, drift, find_books

from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState

ARMS = (
    "real",
    "fixed_w0",
    "shuffled_real",
    "random_tokens",
    "self_readonly",
    "noop_pause",
)


def sha(p):
    s = Path(str(p) + ".sha256")
    if s.exists() and len(s.read_text().strip().split()[0]) == 64:
        return s.read_text().strip().split()[0]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(x, indent=1) + "\n")
    os.replace(t, p)


def stats(row):
    x = row.tolist()
    g2 = [tuple(x[i : i + 2]) for i in range(len(x) - 1)]
    g4 = [tuple(x[i : i + 4]) for i in range(len(x) - 3)]
    return len(set(g2)) / max(1, len(g2)), 1 - len(set(g4)) / max(1, len(g4))


def probe(st, x, y, cfg):
    z = st.snapshot()
    v = st.process_real_chunk(x, y, 0, 1, cfg).float().cpu().tolist()
    st.restore(z)
    return v


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ckpt", required=True)
    a.add_argument("--val", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--arm", required=True, choices=ARMS)
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--n-chunks", type=int, default=128)
    a.add_argument("--n-seqs", type=int, default=2)
    a.add_argument("--book-offset", type=int, default=0)
    a.add_argument("--anchor-every", type=int, default=8)
    a.add_argument("--gpu-sampling", action="store_true")
    a.add_argument("--save-every", type=int, default=8)
    x = a.parse_args()
    t0 = time.time()
    cfg = PRESETS["125m-e2e-ext32k"]()
    dev = "cuda"
    m = TTTModel(cfg.model, max_seq_len=(x.n_chunks + 2) * CS).to(dev).eval()
    w = torch.load(x.ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(w.get("model", w), strict=True)
    del w
    raw = np.load(x.val, mmap_mode="r")
    cycles = (x.n_chunks - WARMUP) // 8
    anchors = (x.n_chunks - WARMUP - cycles) // x.anchor_every
    need = (WARMUP + cycles + anchors + 4) * CS + 1
    canonical_cycles = (128 - WARMUP) // 8
    canonical_anchors = (128 - WARMUP - canonical_cycles) // x.anchor_every
    select = (WARMUP + canonical_cycles + canonical_anchors + 4) * CS + 1
    books = find_books(raw, select)[x.book_offset : x.book_offset + x.n_seqs]
    if len(books) != x.n_seqs or any(e - s < need for s, e in books):
        raise ValueError("canonical book lacks anchor tail")
    real = torch.from_numpy(
        np.stack([np.asarray(raw[s : s + need]).astype(np.int64) for s, _ in books])
    )
    main_real = (WARMUP + cycles + 1) * CS
    anchor_pos = main_real
    real_pos = 0
    st = StreamState(m, x.n_seqs, dev)
    rows = []
    probes = []
    gen_ord = 0
    self_writes = 0
    read_tokens = 0
    out = {
        "status": "running",
        "rows": rows,
        "probes": probes,
        "manifest": {
            "arm": x.arm,
            "seed": x.seed,
            "books": books,
            "book_offset": x.book_offset,
            "n_chunks": x.n_chunks,
            "anchor_every": x.anchor_every,
            "sampling_device": "cuda" if x.gpu_sampling else "cpu",
            "save_every": x.save_every,
            "checkpoint_sha256": sha(x.ckpt),
            "val_sha256": sha(x.val),
            "code_sha256": sha(__file__),
        },
    }
    save(x.out, out)
    for c in range(x.n_chunks):
        isp = c >= WARMUP and (c - WARMUP) % 8 == 7
        if c < WARMUP or isp:
            u = real[:, real_pos : real_pos + CS].to(dev)
            v = real[:, real_pos + 1 : real_pos + CS + 1].to(dev)
            if isp:
                probes.append({"slot": c, "nll": probe(st, u, v, cfg)})
            else:
                st.process_real_chunk(u, v, 1, 1, cfg)
                read_tokens += CS
            real_pos += CS
        else:
            gen_ord += 1
            anchor = gen_ord % x.anchor_every == 0
            first = real[:, real_pos : real_pos + 1].to(dev)
            if not anchor:
                g = st.generate_chunk(
                    first,
                    1,
                    1,
                    cfg,
                    seed=x.seed * 100000 + c,
                    sampling_device="cuda" if x.gpu_sampling else "cpu",
                )
                self_writes += 1
                read_tokens += CS
                z = [stats(g[b].cpu()) for b in range(x.n_seqs)]
                rows.append(
                    {
                        "slot": c,
                        "kind": "self_write",
                        "d2": [q[0] for q in z],
                        "rep4": [q[1] for q in z],
                    }
                )
            elif x.arm in ("self_readonly", "fixed_w0"):
                g = st.generate_chunk(
                    first,
                    0,
                    1,
                    cfg,
                    seed=x.seed * 100000 + c,
                    gen_fast=st.fast_init if x.arm == "fixed_w0" else None,
                    sampling_device="cuda" if x.gpu_sampling else "cpu",
                )
                read_tokens += CS
                z = [stats(g[b].cpu()) for b in range(x.n_seqs)]
                rows.append(
                    {
                        "slot": c,
                        "kind": x.arm,
                        "d2": [q[0] for q in z],
                        "rep4": [q[1] for q in z],
                    }
                )
            elif x.arm == "noop_pause":
                rows.append({"slot": c, "kind": "noop_pause"})
            else:
                q = real[:, anchor_pos : anchor_pos + CS + 1].clone()
                anchor_pos += CS
                if x.arm == "shuffled_real":
                    for b in range(x.n_seqs):
                        g = torch.Generator().manual_seed(x.seed * 100000 + c * 101 + b)
                        q[b] = q[b, torch.randperm(CS + 1, generator=g)]
                elif x.arm == "random_tokens":
                    g = torch.Generator().manual_seed(x.seed * 100000 + c)
                    q = torch.randint(0, cfg.model.vocab_size, q.shape, generator=g)
                q = q.to(dev)
                st.process_real_chunk(q[:, :-1], q[:, 1:], 0, 1, cfg)
                read_tokens += CS
                rows.append({"slot": c, "kind": x.arm})
        out.update(
            rows=rows,
            probes=probes,
            self_generated_writes=self_writes,
            read_tokens=read_tokens,
            elapsed=time.time() - t0,
        )
        if c % x.save_every == 0:
            save(x.out, out)
    out.update(
        status="passed",
        final_drift=drift(st),
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    save(x.out, out)


if __name__ == "__main__":
    main()
