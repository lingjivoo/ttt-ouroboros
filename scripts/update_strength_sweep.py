#!/usr/bin/env python3
"""125M post-clipping update-strength frontier on canonical PG19 books.

Closed runs keep the canonical eight real prefill writes at unit strength and
scale only generated-text deltas.  Real runs consume the identical teacher-
forced stream for every alpha and scale every retained real-text delta.  The
scale is applied to the already clipped parameter delta, never to the loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.horizon import BOS, CS, WARMUP, distinct2, drift, find_books
from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState

# These are the first eight members of the canonical headline book ordering
# that contain a full 130 chunks.  The historical all-real 128K runner selects
# this same set by applying its length requirement before taking eight books.
AUDITED_LONG_BOOK_INDICES = (0, 1, 27, 28, 29, 43, 44, 46)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    os.replace(tmp, path)


def repeated4(ids):
    x = ids.tolist(); seen = set(); repeated = 0
    for i in range(max(0, len(x) - 3)):
        gram = tuple(x[i:i + 4]); repeated += gram in seen; seen.add(gram)
    return repeated / max(1, len(x) - 3)


def per_book_norm(delta, alpha):
    sq = sum(d.float().flatten(1).pow(2).sum(-1) for d in delta)
    return (sq.sqrt() * alpha).cpu().tolist()


def apply_post_clip(st, alpha):
    """Apply exactly alpha times the single deferred, already-clipped delta."""
    if len(st.pending) != 1:
        raise RuntimeError(f"expected one pending delta, got {len(st.pending)}")
    delta = st.pending.pop()
    norms = per_book_norm(delta, alpha)
    st.fast = [(f + alpha * d).detach() for f, d in zip(st.fast, delta)]
    return norms


def branch_probe(st, x, y, cfg):
    snap = st.snapshot()
    vals = st.process_real_chunk(x, y, 0.0, 1.0, cfg).float().cpu().tolist()
    st.restore(snap)
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--val", required=True)
    p.add_argument("--out", required=True); p.add_argument("--mode", choices=("closed", "real"), required=True)
    p.add_argument("--alpha", type=float, required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-chunks", type=int, default=128); p.add_argument("--n-seqs", type=int, default=8)
    p.add_argument("--book-offset", type=int, default=0); p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=.95); p.add_argument("--gpu-sampling", action="store_true")
    a = p.parse_args(); started = time.monotonic()
    if a.alpha < 0: raise ValueError("alpha must be nonnegative")
    result = {"status": "running", "probes": [], "updates": [], "generation": []}
    try:
        cfg = PRESETS["125m-e2e-ext32k"](); dev = "cuda"
        model = TTTModel(cfg.model, max_seq_len=(a.n_chunks + 3) * CS).to(dev).eval()
        payload = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=True); del payload

        arr = np.load(a.val, mmap_mode="r")
        # The all-real arm needs one real chunk per scheduled stream chunk.
        need = (a.n_chunks + 2) * CS + 1 if a.mode == "real" else (WARMUP + (a.n_chunks-WARMUP)//8 + 3) * CS + 1
        if a.book_offset != 0 or a.n_seqs != 8:
            raise ValueError("the frozen frontier requires book_offset=0 and n_seqs=8")
        headline_books = find_books(arr, (WARMUP + (128-WARMUP)//8 + 2) * CS + 1)
        books = [headline_books[i] for i in AUDITED_LONG_BOOK_INDICES]
        if len(books) != a.n_seqs or any(e-s < need for s,e in books):
            raise ValueError("canonical audited books lack the requested real-text tail")
        real = torch.from_numpy(np.stack([np.asarray(arr[s:s+need]).astype(np.int64) for s,_ in books]))
        st = StreamState(model, a.n_seqs, dev); real_pos = 0; generated = 0; previous_last = None
        clip = float(cfg.training.optimizer_inner.clip_gradient)

        manifest = {
            "protocol": "post-clip-update-strength-frontier-v1", "mode": a.mode,
            "alpha": a.alpha, "seed": a.seed, "n_chunks": a.n_chunks, "n_seqs": a.n_seqs,
            "book_offset": a.book_offset, "book_indices": list(AUDITED_LONG_BOOK_INDICES),
            "book_bounds": books, "chunk_size": CS, "warmup_chunks": WARMUP,
            "probe_schedule": "every eighth chunk after warmup; snapshot/restore; no probe write",
            "temperature": a.temperature, "top_p": a.top_p,
            "sampling_device": "cuda" if a.gpu_sampling else "cpu",
            "dose_rule": "compute standard gradient, clip per sample, form standard delta, then apply alpha*delta",
            "closed_prefill_rule": "eight real prefill writes remain at alpha=1; alpha scales generated writes only",
            "real_rule": "all non-probe real-text writes use the same post-clipping alpha",
            "nominal_lr_multiplier": a.alpha, "base_inner_lr": float(cfg.training.optimizer_inner.lr),
            "clip_gradient": clip, "checkpoint_sha256": sha256(a.ckpt), "val_sha256": sha256(a.val),
            "code_sha256": sha256(__file__), "audited_config": asdict(cfg),
        }

        torch.cuda.reset_peak_memory_stats()
        for c in range(a.n_chunks):
            is_probe = c >= WARMUP and (c-WARMUP) % 8 == 7
            if is_probe:
                x = real[:, real_pos:real_pos+CS].to(dev); y = real[:, real_pos+1:real_pos+CS+1].to(dev)
                vals = branch_probe(st, x, y, cfg); real_pos += CS
                result["probes"].append({"schedule_index": c, "generated_index": generated,
                                         "nll_book": vals, "drift": drift(st)})
                continue

            if a.mode == "real" or c < WARMUP:
                x = real[:, real_pos:real_pos+CS].to(dev); y = real[:, real_pos+1:real_pos+CS+1].to(dev)
                real_pos += CS
                scale = a.alpha if a.mode == "real" else 1.0
                st.defer = True; st.process_real_chunk(x, y, 1.0, 1.0, cfg)
                stats = st.last_update_stats; norms = apply_post_clip(st, scale); st.defer = False
                kind = "real_scaled" if a.mode == "real" else "real_prefill_unit"
            else:
                generated += 1
                first = real[:, real_pos:real_pos+1].to(dev) if previous_last is None else previous_last
                st.defer = True
                gen = st.generate_chunk(first, 1.0, 1.0, cfg, temperature=a.temperature, top_p=a.top_p,
                                        seed=a.seed*100000+c,
                                        sampling_device="cuda" if a.gpu_sampling else "cpu")
                stats = st.last_update_stats; norms = apply_post_clip(st, a.alpha); st.defer = False
                previous_last = gen[:, -1:]
                result["generation"].append({"schedule_index": c, "generated_index": generated,
                    "distinct2_book": [distinct2(row.cpu()) for row in gen],
                    "repeated4_book": [repeated4(row.cpu()) for row in gen]})
                kind = "generated_scaled"

            clip_factor = stats["clip_factor"]
            result["updates"].append({"schedule_index": c, "kind": kind,
                "raw_grad_norm_book": stats["raw_grad_norm"], "clip_factor_book": clip_factor,
                "clipped_base_update_norm_book": stats["update_norm"],
                "realized_update_norm_book": norms,
                "clipped_book": [bool(x < 1.0-1e-6) for x in clip_factor]})
            if (c+1) % 16 == 0:
                save(a.out, dict(result, manifest=manifest))
                print(f"chunk {c+1}/{a.n_chunks}", flush=True)

        result.update(status="passed", manifest=manifest, final_drift=drift(st),
                      seconds=time.monotonic()-started,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        save(a.out, result); print(json.dumps({"status":"passed","seconds":result["seconds"]}))
    except Exception:
        result.update(status="failed", error=traceback.format_exc(), seconds=time.monotonic()-started)
        save(a.out, result); raise


if __name__ == "__main__": main()
