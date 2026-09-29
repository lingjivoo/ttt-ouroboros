"""Book-resolved entropy-recovery replication on the canonical 128K stream.

The live stream follows horizon.py: eight real prefill chunks, then seven
generated chunks per branch-only clean probe.  Every generated transition is
measured; no chunk subsampling is performed.  A separate intervention replaces
the pre-registered 80th generated chunk with unseen real text, read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from horizon import BOS, CS, WARMUP, find_books, drift
from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def sha256(path):
    sidecar = Path(str(path) + ".sha256")
    if sidecar.exists():
        value = sidecar.read_text().strip().split()[0]
        if len(value) == 64:
            return value
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    value = h.hexdigest()
    tmp = sidecar.with_suffix(sidecar.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(value + "\n")
    os.replace(tmp, sidecar)
    return value


def diversity(row):
    x = row.tolist()
    d2 = len(set(zip(x[:-1], x[1:]))) / max(1, len(x) - 1)
    seen, repeated = set(), 0
    for i in range(max(0, len(x) - 3)):
        gram = tuple(x[i:i + 4])
        repeated += gram in seen
        seen.add(gram)
    return d2, repeated / max(1, len(x) - 3)


def measure(st, tokens_in, targets, model):
    """Score one chunk without changing caches, positions, or fast weights."""
    snap = st.snapshot()
    with torch.no_grad():
        h = st.prefix_chunk(tokens_in)
        logits, _ = model.suffix_chunk_forward(
            h, st.fast, st.suf_kv, st.chunk_id)
        logp = logits.float().log_softmax(-1)
        probs = logp.exp()
        entropy = -(probs * logp).sum(-1).mean(-1)
        nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = targets != BOS
        nll = (nll * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        mean_probs = probs.mean(1)
        empirical = []
        for row in targets:
            counts = torch.bincount(row, minlength=mean_probs.shape[-1]).float()
            empirical.append((counts + 1e-6) /
                             (counts.sum() + 1e-6 * counts.numel()))
        empirical = torch.stack(empirical)
        kl = (mean_probs * (mean_probs.clamp_min(1e-30).log()
                            - empirical.clamp_min(1e-30).log())).sum(-1)
    st.restore(snap)
    return entropy.cpu().tolist(), nll.cpu().tolist(), kl.cpu().tolist()


def gate_current_chunk(st, tokens_in, targets, model):
    """Secondary negative control: admit a deferred write if it helps its source."""
    if not st.pending:
        return False
    base = float(np.mean(measure(st, tokens_in, targets, model)[2]))
    snap = st.snapshot()
    st.commit_pending(set(range(len(st.pending))))
    candidate = float(np.mean(measure(st, tokens_in, targets, model)[2]))
    st.restore(snap)
    if candidate < base:
        st.commit_pending(set(range(len(st.pending))))
        return True
    st.pending = []
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["masked", "fixed_w0", "closed", "kl_gate"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-chunks", type=int, default=128)
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--book-offset", type=int, default=0)
    ap.add_argument("--inject-generated-index", type=int, default=0,
                    help="replace this one-based generated ordinal with a "
                         "read-only real chunk; zero disables")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=.95)
    ap.add_argument("--gpu-sampling", action="store_true")
    args = ap.parse_args()

    t0 = time.monotonic()
    result = {"status": "running", "rows": [], "probes": []}
    try:
        cfg = PRESETS["125m-e2e-ext32k"]()
        model = TTTModel(cfg.model, max_seq_len=(args.n_chunks + 3) * CS).cuda().eval()
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=True)
        del payload

        data = np.load(args.val, mmap_mode="r")
        n_probe = (args.n_chunks - WARMUP) // 8
        n_real = WARMUP + n_probe + 4
        need = n_real * CS + 1
        # Use the same canonical book numbering as the 128-chunk horizon
        # experiment. Extra clean-probe text is taken only after identities
        # have been frozen.
        selection_need = (WARMUP + (128 - WARMUP) // 8 + 2) * CS + 1
        all_books = find_books(data, selection_need)
        books = all_books[args.book_offset:args.book_offset + args.n_seqs]
        if len(books) != args.n_seqs:
            raise ValueError(f"need {args.n_seqs} books, found {len(books)}")
        if any(e - s < need for s, e in books):
            raise ValueError("a frozen canonical book lacks entropy-probe tail")
        real = torch.from_numpy(np.stack([
            np.asarray(data[s:s + need]).astype(np.int64) for s, _ in books]))

        live = StreamState(model, args.n_seqs, "cuda")
        fixed_context = StreamState(model, args.n_seqs, "cuda")
        fixed_context_initial = fixed_context.snapshot()
        generator = (StreamState(model, args.n_seqs, "cuda")
                     if args.mode == "fixed_w0" else None)
        # A fixed, never-trained probe chunk follows every scheduled real probe.
        probe_pos = (WARMUP + n_probe + 1) * CS
        clean_in = real[:, probe_pos:probe_pos + CS].cuda()
        clean_tgt = real[:, probe_pos + 1:probe_pos + CS + 1].cuda()
        real_pos = 0
        generated_ordinal = 0
        previous_last = None
        previous_row = None

        manifest = {
            "protocol": "entropy-recovery-canonical-v1",
            "mode": args.mode,
            "seed": args.seed,
            "n_chunks": args.n_chunks,
            "n_seqs": args.n_seqs,
            "book_offset": args.book_offset,
            "book_indices": list(range(args.book_offset,
                                       args.book_offset + args.n_seqs)),
            "book_bounds": books,
            "chunk_size": CS,
            "warmup_chunks": WARMUP,
            "probe_schedule": "every eighth chunk; full snapshot/restore",
            "transition_subsampling": "none",
            "inject_generated_index": args.inject_generated_index,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sampling_device": "cuda" if args.gpu_sampling else "cpu",
            "checkpoint_sha256": sha256(args.ckpt),
            "val_sha256": sha256(args.val),
            "code_sha256": sha256(__file__),
            "model_dtype": str(next(model.parameters()).dtype),
            "audited_config": asdict(cfg),
            "adaptation_state_ownership": "independent fast-weight tensors per batch row",
            "fixed_context_diagnostic": "empty caches; identical clean token IDs; read only",
        }

        for c in range(args.n_chunks):
            is_probe = c >= WARMUP and (c - WARMUP) % 8 == 7
            if c < WARMUP or is_probe:
                x = real[:, real_pos:real_pos + CS].cuda()
                y = real[:, real_pos + 1:real_pos + CS + 1].cuda()
                if is_probe:
                    _, nll_book, _ = measure(live, x, y, model)
                    result["probes"].append({
                        "schedule_index": c,
                        "generated_index": generated_ordinal,
                        "nll_book": nll_book,
                        "drift": drift(live),
                    })
                else:
                    live.process_real_chunk(x, y, 1.0, 1.0, cfg)
                    if generator is not None:
                        generator.process_real_chunk(x, y, 0.0, 1.0, cfg)
                real_pos += CS
                continue

            generated_ordinal += 1
            injected = generated_ordinal == args.inject_generated_index
            if injected:
                target = real[:, real_pos:real_pos + CS].cuda()
                first = target[:, :1]
                w_in = target
                w_tgt = real[:, real_pos + 1:real_pos + CS + 1].cuda()
                real_pos += CS
                previous_last = w_tgt[:, -1:]
            else:
                first = (real[:, real_pos:real_pos + 1].cuda()
                         if previous_last is None else previous_last)
                source = generator if generator is not None else live
                snap = None if generator is not None else source.snapshot()
                gen = source.generate_chunk(
                    first, provenance_w=0.0, ilr_mult=1.0, cfg=cfg,
                    temperature=args.temperature, top_p=args.top_p,
                    seed=args.seed * 100000 + c,
                    sampling_device="cuda" if args.gpu_sampling else "cpu")
                if snap is not None:
                    source.restore(snap)
                w_in = torch.cat([first, gen[:, :-1]], 1)
                w_tgt = gen
                previous_last = gen[:, -1:]

            ent0, nll0, kl0 = measure(live, w_in, w_tgt, model)
            if injected or args.mode == "masked":
                write = 0.0
            else:
                write = 1.0
            fast_before = [x.clone() for x in live.fast]
            live.defer = args.mode == "kl_gate" and not injected
            live.process_real_chunk(w_in, w_tgt, write, 1.0, cfg)
            admitted = None
            if args.mode == "kl_gate" and not injected:
                admitted = gate_current_chunk(live, w_in, w_tgt, model)
            live.defer = False
            write_norm = sum(
                (after.float() - before.float()).flatten(1).pow(2).sum(-1)
                for after, before in zip(live.fast, fast_before)).sqrt()
            ent1, _, kl1 = measure(live, w_in, w_tgt, model)
            clean_ent, clean_nll, _ = measure(live, clean_in, clean_tgt, model)
            # Isolate the weight-state effect on one fixed text chunk.  Every
            # arm starts this diagnostic from empty attention caches and the
            # same token IDs; the diagnostic never enters the live stream.
            fixed_context.restore(fixed_context_initial)
            fixed_context.fast = [x.detach().clone() for x in live.fast]
            fixed_ent, fixed_nll, _ = measure(
                fixed_context, clean_in, clean_tgt, model)

            if injected:
                d2_book = [None] * args.n_seqs
                r4_book = [None] * args.n_seqs
            else:
                div = [diversity(w_tgt[b].cpu()) for b in range(args.n_seqs)]
                d2_book = [x[0] for x in div]
                r4_book = [x[1] for x in div]
            row = {
                "schedule_index": c,
                "generated_index": generated_ordinal,
                "injected_real": injected,
                "entropy_before_book": ent0,
                "entropy_after_book": ent1,
                "contraction_book": [a - b for a, b in zip(ent0, ent1)],
                "current_nll_book": nll0,
                "kl_before_book": kl0,
                "kl_after_book": kl1,
                "distinct2_book": d2_book,
                "repeated4_book": r4_book,
                "write_norm_book": write_norm.cpu().tolist(),
                "clean_probe_entropy_book": clean_ent,
                "clean_probe_nll_book": clean_nll,
                "fixed_context_entropy_book": fixed_ent,
                "fixed_context_nll_book": fixed_nll,
                "gate_admitted": admitted,
                "drift": drift(live),
            }
            if previous_row is not None:
                previous_row["next_entropy_before_book"] = ent0
                previous_row["recovery_book"] = [
                    a - b for a, b in zip(ent0,
                                         previous_row["entropy_after_book"])]
                previous_row["next_chunk_nll_book"] = nll0
                previous_row["next_was_real_injection"] = injected
            result["rows"].append(row)
            previous_row = row
            save(args.out, dict(result, manifest=manifest))
            if generated_ordinal % 16 == 0:
                print(f"generated {generated_ordinal}: clean NLL "
                      f"{np.mean(clean_nll):.4f}", flush=True)

        result.update(status="passed", manifest=manifest,
                      seconds=time.monotonic() - t0,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        save(args.out, result)
        print("saved", args.out, flush=True)
    except Exception:
        result.update(status="failed", error=traceback.format_exc(),
                      seconds=time.monotonic() - t0)
        save(args.out, result)
        raise


if __name__ == "__main__":
    main()
