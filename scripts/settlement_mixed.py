"""Fixed-input mixed-stream comparison for next-input settlement.

Every policy consumes the same teacher-forced token schedule.  Recorded model
outputs stand in for own-generation and externally supplied degenerate replay;
eligible real chunks settle pending proposals before they themselves may write.
Separate real evaluation chunks are never used for admission or adaptation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, "scripts")
from horizon import CS, find_books, probe_branched  # noqa: E402
from ttt_pt.config import PRESETS  # noqa: E402
from ttt_pt.model import TTTModel  # noqa: E402
from ttt_pt.stream import StreamState  # noqa: E402


POLICIES = ("no_writes", "ordinary", "mask_own", "delayed", "settlement",
            "real_only", "individual_joint", "mean_joint",
            "normalized_aggregate", "transactional", "effect_oracle")
DEFERRED = {"delayed", "settlement", "individual_joint", "mean_joint",
            "normalized_aggregate"}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_record(path, n_seqs):
    rows = torch.load(path, map_location="cpu", weights_only=False)
    # Current mechanism recordings are manifests with tuples under ``stream``;
    # older recordings were saved as the tuple list directly.
    if isinstance(rows, dict):
        rows = rows["stream"]
    out = []
    for item in rows:
        c, first, gen = item[:3]
        assert first.shape[0] >= n_seqs and gen.shape[0] >= n_seqs
        inp = torch.cat([first[:n_seqs], gen[:n_seqs, :-1]], 1).long()
        tgt = gen[:n_seqs].long()
        assert inp.shape[1] == CS and tgt.shape[1] == CS
        out.append((int(c), inp, tgt))
    if not out:
        raise ValueError(f"empty recording: {path}")
    return out


def schedule(kind, n_chunks, burst_length=1, unverified_source="self"):
    if n_chunks < 16:
        raise ValueError("n_chunks must be at least 16")
    seq = ["prefill"] * 8
    k = 1 if kind == "alternating" else burst_length
    if unverified_source == "mixed":
        burst = [("self", "replay")[i % 2] for i in range(k)]
    else:
        burst = [unverified_source] * k
    pat = burst + ["real", "eval"]
    while len(seq) < n_chunks:
        seq.extend(pat)
    return seq[:n_chunks]


def apply_delta(st, delta, scale=1.0):
    st.fast = [(f + d * scale).detach() for f, d in zip(st.fast, delta)]


def validate_pending(st, policy, q_in, q_tgt, cfg, margin, counters):
    """Resolve all pending proposals against q without changing q's cache."""
    pend = list(st.pending)
    if not pend:
        return set(), [], None
    base = st.probe_score(q_in, q_tgt, cfg)
    counters["validation_forwards"] += 1
    snap = st.snapshot()
    counters["state_snapshots"] += 1
    counters["cache_copy_snapshots"] += 1
    scores = []
    keep = set()
    if policy == "delayed":
        keep = set(range(len(pend)))
    elif policy == "normalized_aggregate":
        # The online buffer already contains one normalized running-mean
        # proposal. Validate and commit that aggregate as one atomic delta.
        assert len(pend) == 1
        apply_delta(st, pend[0])
        sc = st.probe_score(q_in, q_tgt, cfg)
        counters["validation_forwards"] += 1
        scores.append(sc)
        st.restore(snap)
        if sc <= base - margin:
            keep.add(0)
    elif policy == "individual_joint":
        for i, delta in enumerate(pend):
            apply_delta(st, delta)
            sc = st.probe_score(q_in, q_tgt, cfg)
            counters["validation_forwards"] += 1
            scores.append(sc)
            st.restore(snap)
            if sc <= base - margin:
                keep.add(i)
    elif policy == "mean_joint":
        for delta in pend:
            apply_delta(st, delta, 1.0 / len(pend))
        sc = st.probe_score(q_in, q_tgt, cfg)
        counters["validation_forwards"] += 1
        scores.append(sc)
        st.restore(snap)
        if sc <= base - margin:
            # Mark all as accepted, then use an averaged commit below.
            keep = set(range(len(pend)))
    elif policy == "settlement":
        best = base
        chosen = []
        for i in range(len(pend)):
            for j in chosen:
                apply_delta(st, pend[j])
            apply_delta(st, pend[i])
            sc = st.probe_score(q_in, q_tgt, cfg)
            counters["validation_forwards"] += 1
            scores.append(sc)
            st.restore(snap)
            if sc <= best - margin:
                chosen.append(i)
                best = sc
        keep = set(chosen)
    else:
        raise ValueError(policy)
    st.restore(snap)
    if policy == "mean_joint" and keep:
        for delta in pend:
            apply_delta(st, delta, 1.0 / len(pend))
        st.pending = []
    else:
        st.commit_pending(keep)
    return keep, scores, base


def drift(st):
    num = sum(float(((f - f0).float() ** 2).sum())
              for f, f0 in zip(st.fast, st.fast_init))
    den = sum(float((f0.float() ** 2).sum()) for f0 in st.fast_init)
    return (num / den) ** 0.5


def score_from_snapshot(st, snap, x, y, cfg):
    """Score a full state branch and restore the caller's state exactly."""
    original = st.snapshot()
    st.restore(snap)
    value = st.probe_score(x, y, cfg)
    st.restore(original)
    return value


def score_future_after_q(st, snap, qx, qy, ex, ey, cfg):
    """Score E with q in the read cache, while never training on q or E."""
    original = st.snapshot()
    st.restore(snap)
    st.process_real_chunk(qx, qy, 0.0, 1.0, cfg)
    value = st.probe_score(ex, ey, cfg)
    st.restore(original)
    return value


def run(args):
    cfg = PRESETS[args.preset]()
    dev = "cuda"
    model = TTTModel(cfg.model, max_seq_len=args.n_chunks * CS + CS).to(dev)
    raw = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(raw.get("model", raw), strict=False)
    model.eval()
    tokens = np.asarray(np.load(args.val, mmap_mode="r"))
    sched = schedule(args.schedule, args.n_chunks, args.burst_length,
                     args.unverified_source)
    n_real = sum(x in ("prefill", "real", "eval") for x in sched)
    need = n_real * CS + 1
    books = find_books(tokens, need)[args.book_offset:args.book_offset + args.n_seqs]
    assert len(books) == args.n_seqs
    real = torch.from_numpy(np.stack(
        [tokens[s:s + need].astype(np.int64) for s, _ in books]))
    self_rec = load_record(args.self_record, args.n_seqs)
    replay_rec = load_record(args.replay_record, args.n_seqs)
    validation_real = None
    if args.validation_evidence == "different_domain":
        n_arrivals = sum(x == "real" for x in sched)
        validation_need = n_arrivals * CS + 1
        validation_books = find_books(tokens, validation_need)[
            args.validation_book_offset:args.validation_book_offset + args.n_seqs]
        assert len(validation_books) == args.n_seqs
        validation_real = torch.from_numpy(np.stack([
            tokens[s:s + validation_need].astype(np.int64)
            for s, _ in validation_books]))
    st = StreamState(model, args.n_seqs, dev)
    transactional = args.policy in ("transactional", "effect_oracle")
    shadow = StreamState(model, args.n_seqs, dev) if transactional else None
    st.defer = args.policy in DEFERRED
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    counters = Counter(validation_forwards=0, update_backwards=0,
                       state_snapshots=0, cache_copy_snapshots=0,
                       physical_pending_peak=0, logical_pending_peak=0)
    lifecycle = Counter(proposal_total=0, committed=0, rejected=0,
                        capacity_evicted=0, age_expired=0, pending_at_end=0)
    by_source = {s: Counter(proposed=0, committed=0, rejected=0,
                            capacity_evicted=0, age_expired=0)
                 for s in ("real", "self", "replay")}
    pending_meta = []
    transactions = []
    transaction_updates = 0
    evaluations, arrivals, events = [], [], []
    real_pos = self_i = replay_i = validation_pos = validation_i = 0

    for c, src in enumerate(sched):
        if src in ("prefill", "real", "eval"):
            x = real[:, real_pos:real_pos + CS].to(dev)
            y = real[:, real_pos + 1:real_pos + CS + 1].to(dev)
            real_pos += CS
        elif src == "self":
            _, x, y = self_rec[self_i % len(self_rec)]
            self_i += 1; x, y = x.to(dev), y.to(dev)
        else:
            _, x, y = replay_rec[replay_i % len(replay_rec)]
            replay_i += 1; x, y = x.to(dev), y.to(dev)

        # Transactional policies maintain a sequentially updated shadow state.
        # The live state alone predicts/generates; q validates the entire state
        # that would be deployed. effect_oracle uses the following held-out E
        # chunk only as a labelled hindsight upper bound.
        if src == "real" and transactional and transaction_updates:
            live_pre, shadow_pre = st.snapshot(), shadow.snapshot()
            e_x = real[:, real_pos:real_pos + CS].to(dev)
            e_y = real[:, real_pos + 1:real_pos + CS + 1].to(dev)
            if args.policy == "effect_oracle":
                base = score_future_after_q(st, live_pre, x, y, e_x, e_y, cfg)
                candidate = score_future_after_q(st, shadow_pre, x, y, e_x, e_y, cfg)
            else:
                base = score_from_snapshot(st, live_pre, x, y, cfg)
                candidate = score_from_snapshot(st, shadow_pre, x, y, cfg)
            e_base = score_future_after_q(st, live_pre, x, y, e_x, e_y, cfg)
            e_candidate = score_future_after_q(st, shadow_pre, x, y, e_x, e_y, cfg)
            accept = candidate <= base - args.margin
            if accept:
                st.restore(shadow_pre)
                lifecycle["committed"] += 1
            else:
                st.restore(live_pre)
                lifecycle["rejected"] += 1
            shadow.restore(st.snapshot())
            transactions.append({"chunk": c, "length": transaction_updates,
                                 "validation_base_nll": base,
                                 "validation_candidate_nll": candidate,
                                 "advantage": base - candidate,
                                 "future_e_base_nll": e_base,
                                 "future_e_candidate_nll": e_candidate,
                                 "future_e_advantage": e_base - e_candidate,
                                 "committed": accept,
                                 "false_admission": bool(accept and e_candidate > e_base)})
            transaction_updates = 0

        # q is scored and resolves old independent proposals before q is learned.
        if src == "real" and args.policy in DEFERRED and st.pending:
            validation_x, validation_y = x, y
            if args.validation_evidence == "different_domain":
                validation_x = validation_real[:, validation_pos:validation_pos + CS].to(dev)
                validation_y = validation_real[:, validation_pos + 1:validation_pos + CS + 1].to(dev)
                validation_pos += CS
            elif args.validation_evidence == "degenerate_replay":
                _, validation_x, validation_y = replay_rec[validation_i % len(replay_rec)]
                validation_i += 1
                validation_x, validation_y = validation_x.to(dev), validation_y.to(dev)
            elif args.validation_evidence == "shuffled_real":
                seq = torch.cat([x[:, :1], y], 1).cpu()
                shuffled = []
                for b in range(seq.shape[0]):
                    gen = torch.Generator().manual_seed(args.validation_seed + c * 1009 + b)
                    shuffled.append(seq[b, torch.randperm(seq.shape[1], generator=gen)])
                shuffled = torch.stack(shuffled).to(dev)
                validation_x, validation_y = shuffled[:, :-1], shuffled[:, 1:]
            q_pre = st.probe_score(x, y, cfg)
            counters["validation_forwards"] += 1
            old_meta = list(pending_meta)
            keep, scores, base = validate_pending(
                st, args.policy, validation_x, validation_y, cfg, args.margin, counters)
            for i, meta in enumerate(old_meta):
                terminal = "committed" if i in keep else "rejected"
                members = meta.get("members", [meta])
                for member in members:
                    lifecycle[terminal] += 1
                    by_source[member["source"]][terminal] += 1
                    member["terminal"] = terminal
                    member["wait_chunks"] = c - member["proposed_at"]
                    events.append(member)
            arrivals.append({"chunk": c, "prequential_nll": q_pre,
                             "validation_evidence": args.validation_evidence,
                             "base_nll": base, "validation_base_nll": base,
                             "candidate_scores": scores,
                             "n_pending": sum(len(m.get("members", [m])) for m in old_meta),
                             "physical_pending": len(old_meta),
                             "n_committed": sum(len(old_meta[i].get("members", [old_meta[i]]))
                                                for i in keep)})
            pending_meta = []

        if src == "eval":
            nll = probe_branched(st, x, y, cfg, True)
            evaluations.append({"chunk": c, "mean_nll": float(nll.mean()),
                                "per_book_nll": [float(v) for v in nll],
                                "drift": drift(st)})
            st.process_real_chunk(x, y, 0.0, 1.0, cfg)
            if transactional:
                shadow.process_real_chunk(x, y, 0.0, 1.0, cfg)
            continue

        if transactional:
            if src == "prefill":
                st.process_real_chunk(x, y, 1.0, 1.0, cfg)
                shadow.restore(st.snapshot())
            else:
                # Live reads without writing. Shadow reads the same fixed input
                # and updates sequentially, so later shadow deltas are computed
                # at the actual candidate state rather than independently at W.
                st.process_real_chunk(x, y, 0.0, 1.0, cfg)
                shadow.process_real_chunk(x, y, 1.0, 1.0, cfg)
                transaction_updates += 1
                lifecycle["proposal_total"] += 1
                counters["update_backwards"] += 1
            continue

        if src == "prefill":
            write = True
        elif args.policy == "no_writes":
            write = False
        elif args.policy == "mask_own":
            write = src != "self"
        elif args.policy == "real_only":
            write = src == "real"
        else:
            write = True
        # Prefill is shared committed context. Deferred policies start holding
        # proposals only after it, otherwise unlabeled prefill deltas leak into
        # the first settlement batch.
        st.defer = args.policy in DEFERRED and src != "prefill"
        st.process_real_chunk(x, y, float(write), 1.0, cfg)
        if write:
            counters["update_backwards"] += 1
        if write and src != "prefill" and args.policy in DEFERRED:
            lifecycle["proposal_total"] += 1
            by_source[src]["proposed"] += 1
            new_meta = {"source": src, "proposed_at": c}
            pending_meta.append(new_meta)
            if args.policy == "normalized_aggregate" and len(st.pending) > 1:
                assert len(st.pending) == 2 and len(pending_meta) == 2
                old_meta, new_meta = pending_meta
                members = old_meta.get("members", [old_meta]) + [new_meta]
                n_old = len(members) - 1
                # Exact running mean: mean_n=(n-1)/n*mean_(n-1)+1/n*delta_n.
                merged = [(old * n_old + new) / len(members)
                          for old, new in zip(st.pending[0], st.pending[1])]
                st.pending = [[x.detach() for x in merged]]
                pending_meta = [{"members": members,
                                 "proposed_at": members[0]["proposed_at"],
                                 "aggregate_count": len(members)}]
            # Capacity and age are independent rules. Oldest candidates leave first.
            while len(st.pending) > args.settle_cap:
                # FIFO discards the oldest candidate. keep_oldest discards the
                # newest arrival, preserving candidates that have waited for q.
                drop_i = 0 if args.capacity_policy == "fifo" else -1
                st.pending.pop(drop_i)
                meta = pending_meta.pop(drop_i)
                for member in meta.get("members", [meta]):
                    lifecycle["capacity_evicted"] += 1
                    by_source[member["source"]]["capacity_evicted"] += 1
                    member["terminal"] = "capacity_evicted"
                    member["wait_chunks"] = c - member["proposed_at"]
                    events.append(member)
            counters["physical_pending_peak"] = max(
                counters["physical_pending_peak"], len(st.pending))
            counters["logical_pending_peak"] = max(
                counters["logical_pending_peak"],
                sum(len(m.get("members", [m])) for m in pending_meta))
        if args.age_cap > 0 and st.pending:
            expired = 0
            for meta in pending_meta:
                if c - meta["proposed_at"] >= args.age_cap:
                    expired += 1
                else:
                    break
            for _ in range(expired):
                st.pending.pop(0)
                meta = pending_meta.pop(0)
                lifecycle["age_expired"] += 1
                by_source[meta["source"]]["age_expired"] += 1
                meta["terminal"] = "age_expired"
                meta["wait_chunks"] = c - meta["proposed_at"]
                events.append(meta)

    for meta in pending_meta:
        for member in meta.get("members", [meta]):
            member["terminal"] = "pending_at_end"
            member["wait_chunks"] = args.n_chunks - 1 - member["proposed_at"]
            events.append(member)
            lifecycle["pending_at_end"] += 1
    if args.policy in DEFERRED:
        terminal = sum(lifecycle[k] for k in
                       ("committed", "rejected", "capacity_evicted",
                        "age_expired", "pending_at_end"))
        assert terminal == lifecycle["proposal_total"], (terminal, lifecycle)
    elapsed = time.perf_counter() - started
    cfg_out = vars(args).copy()
    cfg_out.update({"checkpoint_sha256": file_sha256(args.ckpt),
                    "self_record_sha256": file_sha256(args.self_record),
                    "replay_record_sha256": file_sha256(args.replay_record),
                    "book_spans": [[int(a), int(b)] for a, b in books],
                    "schedule_tokens": sched,
                    "tie_rule": "accept sc <= best - margin",
                    "settle_cap_semantics": "candidate capacity",
                    "capacity_policy_semantics": args.capacity_policy,
                    "age_cap_semantics": "elapsed chunks; 0 disables"})
    if args.validation_evidence == "different_domain":
        cfg_out["validation_book_spans"] = [
            [int(a), int(b)] for a, b in validation_books]
    result = {"status": "passed", "config": cfg_out,
              "evaluations": evaluations, "arrivals": arrivals,
              "lifecycle": dict(lifecycle),
              "by_source": {k: dict(v) for k, v in by_source.items()},
              "events": events,
              "transactions": transactions,
              "transaction_summary": ({
                  "count": len(transactions),
                  "commit_rate": (sum(t["committed"] for t in transactions) /
                                  len(transactions) if transactions else None),
                  "rollback_rate": (sum(not t["committed"] for t in transactions) /
                                    len(transactions) if transactions else None),
                  "false_admission_rate": (sum(t["false_admission"] for t in transactions) /
                                           max(1, sum(t["committed"] for t in transactions))),
                  "mean_length": (sum(t["length"] for t in transactions) /
                                  len(transactions) if transactions else None),
                  "pending_updates_at_end": transaction_updates,
              } if transactional else None),
              "cost": {**dict(counters), "wall_seconds": elapsed,
                       "tokens_per_second": args.n_chunks * args.n_seqs * CS / elapsed,
                       "peak_memory_bytes": int(torch.cuda.max_memory_reserved()),
                       "pending_delta_peak_count": counters["physical_pending_peak"],
                       "pending_delta_peak_bytes": counters["physical_pending_peak"] *
                           sum(x.numel() * x.element_size() for x in st.fast_init)}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, args.out)
    print(json.dumps({"status": "passed", "policy": args.policy,
                      "schedule": args.schedule, "evals": len(evaluations),
                      "lifecycle": dict(lifecycle), "cost": result["cost"]}),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--self-record", required=True)
    ap.add_argument("--replay-record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="125m-e2e-ext32k")
    ap.add_argument("--policy", choices=POLICIES, required=True)
    ap.add_argument("--schedule", choices=("alternating", "burst"), default="alternating")
    ap.add_argument("--burst-length", type=int, choices=(1, 4, 16), default=4)
    ap.add_argument("--unverified-source", choices=("self", "replay", "mixed"),
                    default="self")
    ap.add_argument("--n-chunks", type=int, default=32)
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--book-offset", type=int, default=24)
    ap.add_argument("--settle-cap", type=int, default=8)
    ap.add_argument("--capacity-policy", choices=("fifo", "keep_oldest"), default="fifo")
    ap.add_argument("--age-cap", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--validation-evidence", choices=("same_domain",
                    "different_domain", "degenerate_replay", "shuffled_real"),
                    default="same_domain")
    ap.add_argument("--validation-book-offset", type=int, default=40)
    ap.add_argument("--validation-seed", type=int, default=20260908)
    args = ap.parse_args()
    if args.settle_cap < 1 or args.age_cap < 0:
        ap.error("settle-cap must be >=1 and age-cap must be >=0")
    if args.policy == "normalized_aggregate" and args.age_cap:
        ap.error("normalized_aggregate currently requires age-cap=0")
    run(args)


if __name__ == "__main__":
    main()
