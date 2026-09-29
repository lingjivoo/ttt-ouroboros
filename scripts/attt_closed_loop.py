"""Full-horizon aTTT token reweighting and write-mass controls.

Implements Eq. 8--10 of Wang et al. (2026): each token receives the maximum
historical exposure of any n-gram in the current update that contains it.
Random and uniform controls preserve the aTTT weight sum for every row/chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from horizon import CS, WARMUP, drift, find_books

from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState


def sha256(path):
    sidecar = Path(str(path) + ".sha256")
    if sidecar.exists():
        value = sidecar.read_text().strip().split()[0]
        if len(value) == 64:
            return value
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def text_stats(row):
    x = row.tolist()
    d2 = len(set(zip(x[:-1], x[1:]))) / max(1, len(x) - 1)
    grams = [tuple(x[i : i + 4]) for i in range(max(0, len(x) - 3))]
    rep4 = 1.0 - len(set(grams)) / max(1, len(grams))
    return d2, rep4


class ATTTWeights:
    def __init__(self, batch, mode, n=4, w_min=0.1, seed=0):
        self.mode, self.n, self.w_min, self.seed = mode, n, w_min, seed
        self.counts = [Counter() for _ in range(batch)]
        self.tails = [[] for _ in range(batch)]
        self.step = 0
        self.mass = []

    def _observe(self, b, ids):
        joined = self.tails[b] + ids
        for i in range(max(0, len(joined) - self.n + 1)):
            self.counts[b][tuple(joined[i : i + self.n])] += 1
        self.tails[b] = joined[-(self.n - 1) :]

    def observe_batch(self, tokens):
        for b, row in enumerate(tokens.cpu()):
            self._observe(b, row.tolist())

    def __call__(self, generated):
        rows = []
        for b, tensor in enumerate(generated.cpu()):
            ids = tensor.tolist()
            exposure = [0] * len(ids)
            for i in range(max(0, len(ids) - self.n + 1)):
                f = self.counts[b].get(tuple(ids[i : i + self.n]), 0)
                if f:
                    for j in range(i, i + self.n):
                        exposure[j] = max(exposure[j], f)
            weights = torch.tensor(
                [max(self.w_min, 1.0 / (1 + f)) for f in exposure], dtype=torch.float32
            )
            if self.mode == "random_dose":
                gen = torch.Generator(device="cpu")
                gen.manual_seed(self.seed * 1000003 + self.step * 9176 + b * 101)
                weights = weights[torch.randperm(len(weights), generator=gen)]
            elif self.mode == "uniform_dose":
                weights.fill_(float(weights.mean()))
            rows.append(weights)
            self._observe(b, ids)
        out = torch.stack(rows).to(generated.device)
        self.mass.append(out.sum(-1).cpu().tolist())
        self.step += 1
        return out


def probe_pair(st, x, y, cfg):
    snap = st.snapshot()
    live = st.process_real_chunk(x, y, 0.0, 1.0, cfg).float().cpu().tolist()
    st.restore(snap)
    st.fast = [z.clone() for z in st.fast_init]
    w0 = st.process_real_chunk(x, y, 0.0, 1.0, cfg).float().cpu().tolist()
    st.restore(snap)
    return live, w0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--mode",
        required=True,
        choices=(
            "closed",
            "masked",
            "attt",
            "random_dose",
            "uniform_dose",
            "anchor_readonly",
        ),
    )
    ap.add_argument("--n-chunks", type=int, default=128)
    ap.add_argument("--n-seqs", type=int, default=2)
    ap.add_argument("--book-offset", type=int, default=0)
    ap.add_argument("--ngram", type=int, default=4)
    ap.add_argument("--w-min", type=float, default=0.1)
    ap.add_argument("--anchor-every", type=int, default=8)
    ap.add_argument("--gpu-sampling", action="store_true")
    ap.add_argument("--save-every", type=int, default=8)
    args = ap.parse_args()
    started = time.time()

    cfg = PRESETS["125m-e2e-ext32k"]()
    dev = "cuda"
    model = TTTModel(cfg.model, max_seq_len=(args.n_chunks + 2) * CS).to(dev).eval()
    raw = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(raw.get("model", raw), strict=True)
    del raw

    data = np.load(args.val, mmap_mode="r")
    cycles = (args.n_chunks - WARMUP) // 8
    extra_anchor = (
        (args.n_chunks - WARMUP - cycles) // args.anchor_every
        if args.mode == "anchor_readonly"
        else 0
    )
    need = (WARMUP + cycles + extra_anchor + 3) * CS + 1
    # Freeze one common 16-book pool that is long enough for every arm.  The
    # anchor arms consume extra real chunks, so selecting on the shorter
    # closed-loop requirement can silently change/fail books across arms.
    canonical_cycles = (128 - WARMUP) // 8
    canonical_anchors = (128 - WARMUP - canonical_cycles) // args.anchor_every
    selection_need = (WARMUP + canonical_cycles + canonical_anchors + 4) * CS + 1
    books = find_books(data, selection_need)[
        args.book_offset : args.book_offset + args.n_seqs
    ]
    if len(books) != args.n_seqs or any(e - s < need for s, e in books):
        raise ValueError("frozen canonical book lacks required probe/anchor tail")
    real = torch.from_numpy(
        np.stack([np.asarray(data[s : s + need]).astype(np.int64) for s, _ in books])
    )

    st = StreamState(model, args.n_seqs, dev)
    weigh = (
        ATTTWeights(args.n_seqs, args.mode, args.ngram, args.w_min, args.seed)
        if args.mode in ("attt", "random_dose", "uniform_dose")
        else None
    )
    rows, probes, real_pos, generated_ordinal = [], [], 0, 0
    result = {"status": "running", "rows": rows, "probes": probes}
    manifest = {
        "mode": args.mode,
        "seed": args.seed,
        "n_chunks": args.n_chunks,
        "n_seqs": args.n_seqs,
        "book_offset": args.book_offset,
        "books": books,
        "ngram": args.ngram,
        "w_min": args.w_min,
        "anchor_every": args.anchor_every,
        "checkpoint_sha256": sha256(args.ckpt),
        "val_sha256": sha256(args.val),
        "code_sha256": sha256(__file__),
        "loss_mass_rule": "random/uniform exactly match aTTT per row and chunk",
        "sampling_device": "cuda" if args.gpu_sampling else "cpu",
        "save_every": args.save_every,
    }
    result["manifest"] = manifest
    save(args.out, result)

    for c in range(args.n_chunks):
        is_probe = c >= WARMUP and (c - WARMUP) % 8 == 7
        use_real = c < WARMUP or is_probe
        if use_real:
            x = real[:, real_pos : real_pos + CS].to(dev)
            y = real[:, real_pos + 1 : real_pos + CS + 1].to(dev)
            if is_probe:
                live, w0 = probe_pair(st, x, y, cfg)
                probes.append(
                    {
                        "slot": c,
                        "nll": live,
                        "w0_nll": w0,
                        "real_adaptation_benefit": [a - b for a, b in zip(w0, live)],
                    }
                )
            else:
                st.process_real_chunk(x, y, 1.0, 1.0, cfg)
                if weigh is not None:
                    weigh.observe_batch(y)
            real_pos += CS
        else:
            generated_ordinal += 1
            if (
                args.mode == "anchor_readonly"
                and generated_ordinal % args.anchor_every == 0
            ):
                x = real[:, real_pos : real_pos + CS].to(dev)
                y = real[:, real_pos + 1 : real_pos + CS + 1].to(dev)
                st.process_real_chunk(x, y, 0.0, 1.0, cfg)
                real_pos += CS
                rows.append(
                    {
                        "slot": c,
                        "kind": "real_anchor",
                        "write_mass": [0.0] * args.n_seqs,
                    }
                )
            else:
                first = real[:, real_pos : real_pos + 1].to(dev)
                fn = weigh if weigh is not None else None
                pv = 0.0 if args.mode == "masked" else 1.0
                gen = st.generate_chunk(
                    first,
                    pv,
                    1.0,
                    cfg,
                    seed=args.seed * 100000 + c,
                    weight_fn=fn,
                    sampling_device="cuda" if args.gpu_sampling else "cpu",
                )
                stats = [text_stats(gen[b].cpu()) for b in range(args.n_seqs)]
                mass = (
                    [0.0] * args.n_seqs
                    if args.mode == "masked"
                    else (
                        weigh.mass[-1]
                        if weigh is not None
                        else [float(CS)] * args.n_seqs
                    )
                )
                rows.append(
                    {
                        "slot": c,
                        "kind": "generated",
                        "distinct2": [z[0] for z in stats],
                        "repeated4": [z[1] for z in stats],
                        "write_mass": mass,
                    }
                )
        result.update(
            status="running",
            rows=rows,
            probes=probes,
            elapsed_seconds=time.time() - started,
        )
        if c % args.save_every == 0:
            save(args.out, result)

    result.update(
        status="passed",
        final_drift=drift(st),
        elapsed_seconds=time.time() - started,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    save(args.out, result)
    print(json.dumps({"status": "passed", "out": args.out}))


if __name__ == "__main__":
    main()
