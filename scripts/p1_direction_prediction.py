#!/usr/bin/env python3
"""P1: predict and intervene on single-update transfer to independent text."""
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
from scripts.horizon import CS, WARMUP, find_books
from ttt_pt.config import PRESETS
from ttt_pt.meta import masked_ce
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState

POSITIONS = (1, 33, 65, 97)


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
    x = ids.tolist(); grams = [tuple(x[i:i + 4]) for i in range(max(0, len(x) - 3))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def score(model, cfg, snapshot, fast, tokens):
    branch = StreamState(model, tokens.shape[0], tokens.device); branch.restore(snapshot)
    branch.pending = []; branch.fast = [x.detach().clone() for x in fast]
    return branch.process_real_chunk(tokens[:, :-1], tokens[:, 1:], 0.0, 1.0, cfg).float().cpu().tolist()


def q_gradient(model, cfg, snapshot, tokens):
    branch = StreamState(model, tokens.shape[0], tokens.device); branch.restore(snapshot)
    branch.pending = []
    fast = [x.detach().clone().requires_grad_(True) for x in branch.fast]
    h = branch.prefix_chunk(tokens[:, :-1]); branch.global_pos += CS
    logits, _ = model.suffix_chunk_forward(h, fast, branch.suf_kv, branch.chunk_id)
    mask = tokens[:, 1:] != cfg.model.bos_token_id
    loss_b, _ = masked_ce(logits, tokens[:, 1:], mask)
    grads = torch.autograd.grad(loss_b.sum(), fast)
    return [g.detach() for g in grads], loss_b.detach().float().cpu().tolist()


def row_inner(a, b):
    return sum((x.float() * y.float()).flatten(1).sum(-1) for x, y in zip(a, b))


def row_norm(a):
    return row_inner(a, a).clamp_min(0).sqrt()


def interventions(base, delta, grad):
    dot = row_inner(grad, delta); g2 = row_inner(grad, grad); dn = row_norm(delta)
    active = (dot > 0) & (g2 > 1e-24)
    coefficient = torch.where(active, dot / g2.clamp_min(1e-24), torch.zeros_like(dot))
    projected = [d - coefficient.view(-1, *([1] * (d.ndim - 1))) * g for d, g in zip(delta, grad)]
    pn = row_norm(projected)
    ratio = pn / dn.clamp_min(1e-12)
    scaled = [d * ratio.view(-1, *([1] * (d.ndim - 1))) for d in delta]
    states = {
        "zero": [x.detach() for x in base],
        "original": [(x + d).detach() for x, d in zip(base, delta)],
        "projected": [(x + d).detach() for x, d in zip(base, projected)],
        "scaled": [(x + d).detach() for x, d in zip(base, scaled)],
    }
    gn = row_norm(grad); cosine = dot / (gn * dn).clamp_min(1e-12)
    return states, {
        "dot_q_delta": dot.float().cpu().tolist(),
        "cosine_q_delta": cosine.float().cpu().tolist(),
        "q_grad_norm": gn.float().cpu().tolist(),
        "delta_norm": dn.float().cpu().tolist(),
        "projected_delta_norm": pn.float().cpu().tolist(),
        "projection_active": active.cpu().tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--val", required=True)
    p.add_argument("--out", required=True); p.add_argument("--split", choices=("development", "confirmation"), required=True)
    p.add_argument("--history", choices=("closed", "masked"), required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--preset", default="125m-e2e-ext32k")
    p.add_argument("--positions", default="1,33,65,97"); p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=.95); p.add_argument("--gpu-sampling", action="store_true")
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args(); started = time.monotonic(); positions = tuple(int(x) for x in a.positions.split(","))
    result = {"status": "running", "records": []}; save(a.out, result)
    try:
        cfg = PRESETS[a.preset](); dev = "cuda"; max_position = max(positions)
        model = TTTModel(cfg.model, max_seq_len=(WARMUP + max_position + 40) * CS).to(dev).eval()
        payload = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=True); del payload
        arr = np.load(a.val, mmap_mode="r")
        # Generation is self-fed after prefill, so book length is determined by
        # the frozen real-text evidence blocks rather than the generated horizon.
        # Chunks 8--11 are r_near and chunks 16--19 of a different book are r_far.
        need = (WARMUP + 12) * CS + 1
        eligible = find_books(arr, need)
        if len(eligible) < 16:
            raise RuntimeError(f"need 16 eligible books, found {len(eligible)}")
        selected = eligible[:16]; split_offset = 0 if a.split == "development" else 8
        books = selected[split_offset:split_offset + 8]
        if a.pilot:
            books = books[:1]
        book_indices = [eligible.index(x) for x in books]
        tokens = torch.from_numpy(np.stack([np.asarray(arr[s:s + need]).astype(np.int64) for s, _ in books])).to(dev)
        batch = len(books); state = StreamState(model, batch, dev)

        for chunk in range(WARMUP):
            segment = tokens[:, chunk * CS:(chunk + 1) * CS + 1]
            state.process_real_chunk(segment[:, :-1], segment[:, 1:], 1.0, 1.0, cfg)

        q = tokens[:, :CS + 1]
        far = torch.roll(tokens, shifts=-1, dims=0)
        previous = None
        torch.cuda.reset_peak_memory_stats()
        for ordinal in range(1, max_position + 1):
            first = tokens[:, WARMUP * CS:WARMUP * CS + 1] if previous is None else previous
            row_seeds = [a.seed * 1000003 + split_offset * 65537 + ordinal * 1009 + row * 7919 for row in range(batch)]
            if ordinal not in positions:
                state.defer = False
                generated = state.generate_chunk(first, 1.0 if a.history == "closed" else 0.0, 1.0, cfg,
                                                 temperature=a.temperature, top_p=a.top_p,
                                                 seed=a.seed * 100000 + ordinal, row_seeds=row_seeds,
                                                 sampling_device="cuda" if a.gpu_sampling else "cpu")
                previous = generated[:, -1:]
                if ordinal % 8 == 0:
                    print(a.split, a.history, a.seed, ordinal, flush=True)
                continue

            before = state.snapshot(); state.defer = True
            generated = state.generate_chunk(first, 1.0, 1.0, cfg, temperature=a.temperature, top_p=a.top_p,
                                             seed=a.seed * 100000 + ordinal, row_seeds=row_seeds,
                                             sampling_device="cuda" if a.gpu_sampling else "cpu")
            if len(state.pending) != 1:
                raise RuntimeError(f"expected one deferred delta, found {len(state.pending)}")
            delta = [x.detach().clone() for x in state.pending[0]]
            after = state.snapshot(); after["pending"] = []
            base = [x.detach().clone() for x in after["fast"]]

            q_grad, _ = q_gradient(model, cfg, after, q)
            states, diagnostics = interventions(base, delta, q_grad)
            position_index = positions.index(ordinal)
            near_pos = (WARMUP + position_index) * CS
            far_pos = (WARMUP + 8 + position_index) * CS
            r_near = tokens[:, near_pos:near_pos + CS + 1]
            r_far = far[:, far_pos:far_pos + CS + 1]
            source = torch.cat([first, generated], 1)
            scores = {name: {
                "source": score(model, cfg, before, fast, source),
                "q": score(model, cfg, after, fast, q),
                "r_near": score(model, cfg, after, fast, r_near),
                "r_far": score(model, cfg, after, fast, r_far),
            } for name, fast in states.items()}
            weight_norm = row_norm(base).float().cpu().tolist()
            rows = []
            for row in range(batch):
                row_scores = {name: {target: values[target][row] for target in values} for name, values in scores.items()}
                rows.append({
                    "row": row, "book_index": book_indices[row], "book_bounds": books[row],
                    "far_book_index": book_indices[(row + 1) % batch] if batch > 1 else None,
                    "source_repeated4": repeated4(generated[row].cpu()),
                    "weight_norm": weight_norm[row],
                    "relative_delta_norm": diagnostics["delta_norm"][row] / max(weight_norm[row], 1e-12),
                    **{key: value[row] for key, value in diagnostics.items()},
                    "q_direct_delta": row_scores["original"]["q"] - row_scores["zero"]["q"],
                    "scores": row_scores,
                })
            result["records"].append({"generated_position": ordinal, "rows": rows})
            save(a.out, result)

            state.restore(after); state.pending = []
            if a.history == "closed":
                state.fast = [(x + d).detach() for x, d in zip(base, delta)]
            else:
                state.fast = base
            state.defer = False; previous = generated[:, -1:]
            if ordinal % 8 == 0 or ordinal in positions:
                print(a.split, a.history, a.seed, ordinal, flush=True)

        manifest = {
            "protocol": "reviewer-p1-direction-v1", "split": a.split, "history": a.history,
            "seed": a.seed, "positions": positions, "book_indices": book_indices, "book_bounds": books,
            "eligible_book_bounds": selected, "batch_size": batch, "chunk_size": CS,
            "real_prefill_chunks": WARMUP, "q_definition": "first received real chunk",
            "r_near_definition": "unseen same-book real chunks 8,9,10,11 mapped to positions 1,33,65,97",
            "r_far_definition": "unseen paired-next-book real chunks 16,17,18,19 mapped to positions 1,33,65,97",
            "temperature": a.temperature, "top_p": a.top_p,
            "interventions": ["zero", "original", "projected", "equal-norm-scaled"],
            "projection_rule": "remove positive component along grad L(q); scaled control matches projected norm",
            "adaptation_state_ownership": "independent fast-weight tensors per batch row",
            "checkpoint_sha256": sha256(a.ckpt), "dataset_sha256": sha256(a.val),
            "code_sha256": sha256(__file__), "audited_config": asdict(cfg), "pilot": a.pilot,
        }
        result.update(status="passed", manifest=manifest, seconds=time.monotonic() - started,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        save(a.out, result); print(json.dumps({"status": "passed", "seconds": result["seconds"]}))
    except Exception:
        result.update(status="failed", error=traceback.format_exc(), seconds=time.monotonic() - started)
        save(a.out, result); raise


if __name__ == "__main__":
    main()
