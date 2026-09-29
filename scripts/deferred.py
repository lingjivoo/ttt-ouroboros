"""Deferred-verification writing: a mitigation that is not a provenance mask.

The paper's mask is binary and blunt -- endogenous tokens never write, so the
adaptation benefit they might carry is lost by construction. Everything else we
tried is worse: content gates (repetition, NLL-spike, gradient surgery) all fail
because damage accrues before degeneracy is detectable; retention decay improves
the benefit-to-harm ratio but does not by itself hold the closed loop flat; and
a frozen gradient reference is catastrophic.

What the block-structured update makes possible instead is selective commit.
Updates are held in a buffer rather than applied, so the chunks that follow are
generated from an unchanged state -- the loop cannot manufacture its next batch
out of damage that has not been committed. After k chunks a real probe scores
the state with the buffer applied against the state without it, and the batch is
committed only if it did not hurt. Endogenous text is therefore allowed to write
whenever it earns its place, which is the substantive difference from the mask.

Three arms on the closed-loop protocol, identical text and seeds:
  closed    -- the unmitigated loop
  masked    -- the paper's binary provenance mask (harm floor, benefit floor)
  deferred  -- this mitigation

The claim to test is two-sided and both sides matter: deferred must hold damage
near masked AND retain adaptation benefit that masked throws away. We measure
the second on real chunks, where writing is what TTT is for: the run also
reports the fraction of held batches committed, since a mitigation that commits
nothing is just the mask with extra compute.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, "scripts")
from horizon import CS, WARMUP, find_books, probe_branched  # noqa: E402

from ttt_pt.config import PRESETS  # noqa: E402
from ttt_pt.model import TTTModel  # noqa: E402
from ttt_pt.stream import StreamState  # noqa: E402


def forced_admissions(src, k):
    """Indices of k generated updates to push through the gate regardless.

    The false-admission stress test asks what a gate's mistakes cost and whether
    retention bounds them. The forced set has to sit at the SAME positions in
    every arm, or the arms differ in which harmful writes they took as well as
    in how many, and the comparison measures both at once. Evenly spaced over
    the generated updates in the batch gives that: the batch structure is fixed
    by the chunk schedule, so every arm sees the same positions.

    Chosen among generated updates only. Forcing a real-text update through
    would not be a mistake -- those are the ones a gate is supposed to admit.
    """
    gen = [i for i, t in enumerate(src) if t == "G"]
    if k <= 0 or not gen:
        return set()
    if k >= len(gen):
        return set(gen)
    step = len(gen) / k
    return {gen[int(i * step)] for i in range(k)}


def run(model, cfg, dev, real, mode, n_chunks, seed, defer_k=4, margin=0.0,
        real_stream=False, split_val=True, lam=1.0, content_thr=None,
        commit_frac=None, force_admit=0, settle_cap=8, reset_every=0,
        anchor_beta=0.0):
    st = StreamState(model, real.shape[0], dev)
    st.defer = mode in ("deferred", "defer_always", "defer_update",
                        "defer_seq", "defer_content", "defer_random", "settle")
    import random as _rnd
    rng = _rnd.Random(seed * 7919)
    src_nll = []      # each held update's own chunk loss, for the content gate
    probes, drifts, commits, held = [], [], 0, 0
    probes_book = []
    lifecycle = {"proposal_total": 0, "committed": 0, "rejected": 0,
                 "capacity_evicted": 0, "age_expired": 0, "pending_at_end": 0}
    # Provenance of each held update, in the order they enter the buffer, so a
    # commit count can be read as "which kind of text earned authority" rather
    # than just a number. Without this, `committed 8/113` cannot be told apart
    # from "exactly the 8 real warm-up chunks", which would mean no endogenous
    # update was ever admitted -- the same trap as `committed 15/113` turning
    # out to be exactly the first batch.
    src, adm = [], {"R": 0, "G": 0, "R_held": 0, "G_held": 0}
    real_pos = 0

    def post_step_control(before, chunk_index):
        if anchor_beta > 0:
            # L2 gradient penalty beta/2 ||W-W0||^2 evaluated at W_pre.
            # Unlike multiplicative retention, this does not scale the fresh
            # data-gradient update that was just applied.
            st.fast = [(f - anchor_beta * (b - i0)).detach()
                       for f, b, i0 in zip(st.fast, before, st.fast_init)]
        if reset_every and (chunk_index + 1) % reset_every == 0:
            st.fast = [i0.clone() for i0 in st.fast_init]
    for c in range(n_chunks):
        is_probe = c >= WARMUP and (c - WARMUP) % 8 == 7
        # A probe is used EITHER to let the gate decide OR to report, never
        # both. Scoring the gate on the same chunk whose NLL is then reported
        # is selection on the test statistic: the gate picks the updates that
        # minimise exactly the number we go on to quote. With split_val the
        # probes alternate -- even ones validate, odd ones are reported and the
        # gate never sees them.
        probe_i = (c - WARMUP) // 8 if is_probe else -1
        is_val = is_probe and split_val and probe_i % 2 == 0
        is_eval = is_probe and (not split_val or probe_i % 2 == 1)
        use_real = real_stream or c < WARMUP or is_probe
        if use_real:
            seg_in = real[:, real_pos: real_pos + CS].to(dev)
            seg_tgt = real[:, real_pos + 1: real_pos + CS + 1].to(dev)
            if is_probe:
                # Verification point: score the state with the held batch applied
                # against the state without it, on THIS real chunk, then keep or
                # drop the whole batch. Scoring on real text is the point -- the
                # question is whether the writes help on the distribution we care
                # about, not whether they fit the generated text they came from.
                if st.defer and st.pending and is_val and mode != "settle":
                    base = st.probe_score(seg_in, seg_tgt, cfg)
                    if mode in ("defer_content", "defer_random"):
                        # Two baselines the transactional gate has to beat.
                        # content: admit an update when its own chunk was
                        # surprising, the "there is headroom here" rule. We
                        # measured the ordering this produces to be inverted --
                        # uniform random tokens score highest and real text
                        # lowest -- so this arm is expected to fail, and it is
                        # here to show the failure rather than to be assumed.
                        # random: admit a fixed fraction, matching the
                        # transactional arm's realised dose so that any
                        # advantage cannot be "it simply wrote less".
                        pend = list(st.pending)
                        if mode == "defer_content":
                            thr = content_thr if content_thr is not None else 0.0
                            keep = {i for i, v in enumerate(src_nll) if v >= thr}
                        else:
                            f = commit_frac if commit_frac is not None else 0.1
                            keep = {i for i in range(len(pend)) if rng.random() < f}
                        held += len(pend); commits += len(keep)
                        keep |= forced_admissions(src, force_admit)
                        for i, t in enumerate(src):
                            adm[t + "_held"] += 1
                            if i in keep:
                                adm[t] += 1
                        src = []; src_nll = []
                        st.commit_pending(keep)
                    elif mode == "defer_update":
                        # Per-update authority. Batch commit can only accept or
                        # reject a whole block, and on this protocol every block
                        # after the first is entirely endogenous, so the batch
                        # rule can never let self-generated content earn a place
                        # -- it degenerates into the provenance mask. Here each
                        # held update is scored on its own against the same
                        # base. That is sound precisely because the deltas were
                        # all computed against one state (see commit_pending):
                        # dropping one does not invalidate the others, so their
                        # marginal effects are separately meaningful.
                        pend = list(st.pending)
                        keep = set()
                        assert len(src) == len(pend), (len(src), len(pend))
                        # One snapshot for the whole scan, not one per update:
                        # restore() clones on the way out, so the same snapshot
                        # can be replayed, and its stored pending is `pend`.
                        # Taking a fresh snapshot per update would copy the KV
                        # caches fifteen times at the first verification point.
                        snap0 = st.snapshot()
                        for i in range(len(pend)):
                            st.pending = [pend[i]]
                            st.commit_pending({0})
                            if st.probe_score(seg_in, seg_tgt, cfg) <= base - margin:
                                keep.add(i)
                            st.restore(snap0)
                        assert len(st.pending) == len(pend)  # restore put them back
                        held += len(pend)
                        commits += len(keep)
                        keep |= forced_admissions(src, force_admit)
                        for i, t in enumerate(src):
                            adm[t + "_held"] += 1
                            if i in keep:
                                adm[t] += 1
                        src = []; src_nll = []
                        st.commit_pending(keep)
                    elif mode == "defer_seq":
                        # Greedy forward selection: judge each candidate against
                        # the set already accepted, not against the empty set.
                        #
                        # defer_update scores every held update alone and then
                        # applies the accepted ones together. Each is helpful in
                        # isolation and their sum is not: on a real-text stream
                        # it admits 113 of 113 and ends 1.28 nats WORSE than
                        # writing the same updates sequentially, with 4.9x the
                        # weight drift. defer_always -- which makes no admission
                        # decision at all -- produces bitwise identical numbers,
                        # so the loss is the joint application, not the choice.
                        #
                        # Here the candidate is applied on top of what has been
                        # accepted so far and kept only if the whole set still
                        # improves. Same number of probe evaluations; what
                        # changes is that the quantity evaluated is the one that
                        # actually gets committed.
                        pend = list(st.pending)
                        assert len(src) == len(pend), (len(src), len(pend))
                        snap0 = st.snapshot()
                        keep_l, best = [], base
                        for i in range(len(pend)):
                            st.pending = [pend[j] for j in keep_l] + [pend[i]]
                            st.commit_pending(set(range(len(st.pending))))
                            sc = st.probe_score(seg_in, seg_tgt, cfg)
                            st.restore(snap0)
                            if sc <= best - margin:
                                keep_l.append(i)
                                best = sc
                        keep = set(keep_l)
                        assert len(st.pending) == len(pend)
                        held += len(pend)
                        commits += len(keep)
                        keep |= forced_admissions(src, force_admit)
                        for i, t in enumerate(src):
                            adm[t + "_held"] += 1
                            if i in keep:
                                adm[t] += 1
                        src = []; src_nll = []
                        st.commit_pending(keep)
                    else:
                        snap = st.snapshot()
                        n = st.commit_pending(set(range(len(st.pending))))
                        withw = st.probe_score(seg_in, seg_tgt, cfg)
                        held += n
                        # defer_always is the control that separates the two
                        # things this mitigation does at once: holding updates
                        # out of the state the next chunks are generated from
                        # (deferral), and dropping the ones that do not help
                        # (verification). It defers identically and commits
                        # unconditionally, so any gap between it and `deferred`
                        # is what verification buys.
                        for t in src:
                            adm[t + "_held"] += 1
                        if mode == "defer_always" or withw <= base - margin:
                            commits += n        # keep: the applied state is better
                            for t in src:
                                adm[t] += 1
                        else:
                            st.restore(snap)    # drop the batch wholesale
                            st.pending = []
                        src = []; src_nll = []
                nll = probe_branched(st, seg_in, seg_tgt, cfg, True)
                if is_eval:
                    # Keep the per-book values, not just their mean. The book is
                    # the independent unit -- seeds only reseed sampling, and in
                    # the closed loop they do not even pair arms, because the
                    # first differing write changes every later generated chunk.
                    # Averaging here threw away the only variance an across-book
                    # CI could be built from, so `probes` (the mean, for every
                    # existing reader) is now accompanied by `probes_book`.
                    probes.append((c, float(nll.mean())))
                    probes_book.append((c, [float(x) for x in nll]))
                num = sum(float(((f - i0) ** 2).sum())
                          for f, i0 in zip(st.fast, st.fast_init))
                den = sum(float((i0 ** 2).sum()) for i0 in st.fast_init)
                drifts.append((c, round((num / den) ** 0.5, 5)))
            else:
                if mode == "settle" and st.pending:
                    # Settlement: the candidate held from the previous chunk is
                    # judged by the text that just arrived, BEFORE that text is
                    # processed. The probe schedule plays no part -- the future
                    # of the stream is the settlement evidence, which is the
                    # only probe a deployment actually has. Distance is one
                    # chunk, where the bet ("the near future resembles the near
                    # past") is strongest; the mixed-stream runs show the same
                    # signal at doubled distance admits a quarter as much.
                    # Same greedy order-of-commit rule as defer_seq: what is
                    # scored is exactly what is committed. A candidate that
                    # fails settlement here is dropped, not retried -- its bet
                    # was about this text and this text has ruled.
                    pend = list(st.pending)
                    assert len(src) == len(pend), (len(src), len(pend))
                    snap0 = st.snapshot()
                    base = st.probe_score(seg_in, seg_tgt, cfg)
                    keep_l, best = [], base
                    for i in range(len(pend)):
                        st.pending = [pend[j] for j in keep_l] + [pend[i]]
                        st.commit_pending(set(range(len(st.pending))))
                        sc = st.probe_score(seg_in, seg_tgt, cfg)
                        st.restore(snap0)
                        if sc <= best - margin:
                            keep_l.append(i)
                            best = sc
                    keep = set(keep_l)
                    assert len(st.pending) == len(pend)
                    held += len(pend)
                    commits += len(keep)
                    lifecycle["committed"] += len(keep)
                    lifecycle["rejected"] += len(pend) - len(keep)
                    for i, t in enumerate(src):
                        adm[t + "_held"] += 1
                        if i in keep:
                            adm[t] += 1
                    src = []; src_nll = []
                    st.commit_pending(keep)
                # On the real-text protocol the mask has to forfeit the writes
                # it exists to forbid, otherwise `masked` is just `closed` and
                # the benefit side of the comparison is unmeasurable.
                rw = 0.0 if ((real_stream and mode == "masked") or
                             (mode == "writes_off" and c >= WARMUP)) else 1.0
                before = [f.clone() for f in st.fast] if anchor_beta > 0 else ()
                st.process_real_chunk(seg_in, seg_tgt, rw, 1.0, cfg, lam=lam)
                post_step_control(before, c)
                if st.defer and rw > 0.0:
                    src.append("R"); src_nll.append(getattr(st, "last_nll", 0.0))
                    lifecycle["proposal_total"] += 1
            real_pos += CS
        else:
            first = real[:, real_pos: real_pos + 1].to(dev)
            pv = 0.0 if mode in ("masked", "writes_off") else 1.0
            before = [f.clone() for f in st.fast] if anchor_beta > 0 else ()
            st.generate_chunk(first, provenance_w=pv, ilr_mult=1.0, cfg=cfg,
                              seed=seed * 100000 + c, lam=lam)
            post_step_control(before, c)
            if st.defer and pv > 0.0:
                src.append("G"); src_nll.append(getattr(st, "last_nll", 0.0))
                lifecycle["proposal_total"] += 1
            if mode == "settle" and len(st.pending) > settle_cap:
                drop = len(st.pending) - settle_cap
                for t in src[:drop]:
                    adm[t + "_held"] += 1
                held += drop
                lifecycle["capacity_evicted"] += drop
                st.pending = st.pending[drop:]
                src = src[drop:]
                src_nll = src_nll[drop:]
    lifecycle["pending_at_end"] = len(st.pending)
    if mode == "settle":
        terminal = sum(lifecycle[k] for k in ("committed", "rejected",
                       "capacity_evicted", "age_expired", "pending_at_end"))
        assert terminal == lifecycle["proposal_total"], (terminal, lifecycle)
    return probes, drifts, commits, held, adm, probes_book, lifecycle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--preset", default="125m-e2e-ext32k")
    ap.add_argument("--mode", required=True,
                    choices=["closed", "masked", "writes_off", "deferred", "defer_always",
                             "defer_update", "defer_seq", "defer_content",
                             "defer_random", "settle"])
    ap.add_argument("--lam", type=float, default=1.0,
                    help="retention: W <- W_0 + lam (W - W_0) after each update. "
                         "Bounds how long an accepted write survives, which is "
                         "the axis orthogonal to admission.")
    ap.add_argument("--reset-every", type=int, default=0,
                    help="reset fast weights to W0 after every k live chunks")
    ap.add_argument("--anchor-beta", type=float, default=0.0,
                    help="L2 gradient penalty strength toward W0; unlike --lam, "
                         "does not scale the fresh data gradient")
    ap.add_argument("--content-thr", type=float, default=None,
                    help="defer_content: admit an update whose own chunk loss "
                         "is at least this")
    ap.add_argument("--force-admit", type=int, default=0,
                    help="push this many generated updates through the gate at "
                         "fixed positions, whatever it decides. The stress test "
                         "for `retention bounds the lifetime of what was "
                         "admitted, including admission mistakes` -- without it "
                         "that half of the claim has no evidence.")
    ap.add_argument("--commit-frac", type=float, default=None,
                    help="defer_random: admit this fraction, to dose-match the "
                         "transactional arm")
    ap.add_argument("--settle-cap", type=int, default=8,
                    help="settle: pending candidates beyond this are dropped, "
                         "oldest first. On a pure closed loop no settling text "
                         "ever arrives, so without a cap the buffer holds every "
                         "generated update to the end; with it, unsettleable "
                         "candidates expire and the arm converges to the mask, "
                         "which is the intended deployment behaviour when a "
                         "session goes fully self-generated.")
    ap.add_argument("--real-stream", action="store_true",
                    help="every non-probe chunk is real text: measures the "
                         "adaptation benefit the mask gives up, not damage")
    ap.add_argument("--n-chunks", type=int, default=128)
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--book-offset", type=int, default=2)
    ap.add_argument("--full-length-books", action="store_true",
                    help="use books long enough for the full all-real horizon, "
                         "matching generated and real arm identities")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--no-split-val", action="store_true",
                    help="let the gate validate on the same probes that are "
                         "reported. This is the older, contaminated protocol; "
                         "it exists only to quantify what the contamination "
                         "was worth.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # A mode can be added to the deferring set and forgotten in --mode's
    # choices, and then it is rejected at the command line with a message that
    # says nothing about the omission. Check the two lists agree.
    _defer_modes = {"deferred", "defer_always", "defer_update", "defer_seq",
                    "defer_content", "defer_random", "settle"}
    _choices = set(ap._option_string_actions["--mode"].choices)
    assert _defer_modes <= _choices, (
        f"deferring modes missing from --mode choices: "
        f"{sorted(_defer_modes - _choices)}")

    cfg = PRESETS[args.preset]()
    dev = "cuda"
    model = TTTModel(cfg.model, max_seq_len=args.n_chunks * CS + CS).to(dev)
    stt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(stt.get("model", stt), strict=False)
    model.eval()

    tokens = np.asarray(np.load(args.val, mmap_mode="r"))
    n_real = (args.n_chunks + 2 if args.real_stream
              else WARMUP + (args.n_chunks - WARMUP) // 8 + 2)
    need = n_real * CS + 1
    selection_need = ((args.n_chunks + 2) * CS + 1
                      if args.full_length_books else need)
    books = find_books(tokens, selection_need)[args.book_offset: args.book_offset + args.n_seqs]
    if len(books) != args.n_seqs:
        raise ValueError(f"requested {args.n_seqs} books, found {len(books)}")
    real = torch.from_numpy(np.stack(
        [tokens[s: s + need].astype(np.int64) for s, _ in books]))
    print(f"{len(books)} books, mode={args.mode}", flush=True)

    # Resume per seed. freecycle preempts at roughly two hours and a three-seed
    # arm does not fit inside that, so a run that starts from an empty result
    # recomputes finished seeds and is then killed again at the same point --
    # which is exactly how this arm lost its third seed once already. The unit
    # of restart has to be smaller than the preemption interval.
    # Resume is keyed on (mode, seed), which does not mention the config or the
    # code. A canary caught this the honest way: rerun in the same directory
    # after adding a field and every cell came back "done" with the old record
    # and no new field. Signature the settings that change the computation and
    # refuse to reuse a file written under different ones.
    sig = {"mode": args.mode, "n_chunks": args.n_chunks, "n_seqs": args.n_seqs,
           "book_offset": args.book_offset, "real_stream": bool(args.real_stream),
           "split_val": not args.no_split_val, "lam": args.lam,
           "reset_every": args.reset_every, "anchor_beta": args.anchor_beta,
           "full_length_books": bool(args.full_length_books),
           "content_thr": args.content_thr, "commit_frac": args.commit_frac,
           "margin": args.margin, "settle_cap": args.settle_cap,
           "settle_cap_semantics": "pending candidate capacity; not age",
           "preset": args.preset,
           "ckpt": args.ckpt, "val": args.val, "fields": "probes_book"}
    res = {}
    if os.path.exists(args.out):
        try:
            res = json.load(open(args.out))
        except Exception:
            res = {}
        old_sig = res.pop("_config", None) if isinstance(res, dict) else None
        if res and old_sig is not None and old_sig != sig:
            diff = [k for k in sig if old_sig.get(k) != sig[k]]
            print(f"discarding {args.out}: written under a different config "
                  f"({', '.join(diff)}); recomputing rather than mixing.",
                  flush=True)
            res = {}
        elif res and old_sig is None:
            print(f"discarding {args.out}: written before configs were "
                  f"signed, so it cannot be shown to match.", flush=True)
            res = {}
        elif res:
            print(f"resuming, have {sorted(res)}", flush=True)
    for sd in [int(x) for x in args.seeds.split(",")]:
        key = f"{args.mode}_s{sd}"
        if key in res:
            print(f"skip {key} (done)", flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        seed_started = time.perf_counter()
        p, d, com, hel, adm, pb, life = run(model, cfg, dev, real, args.mode,
                                  args.n_chunks, sd, margin=args.margin,
                                  real_stream=args.real_stream,
                             split_val=not args.no_split_val, lam=args.lam,
                             content_thr=args.content_thr,
                             commit_frac=args.commit_frac,
                             force_admit=args.force_admit,
                             settle_cap=args.settle_cap,
                             reset_every=args.reset_every,
                             anchor_beta=args.anchor_beta)
        res[key] = {"probes": p, "drift": d, "committed": com, "held": hel,
                    "admitted": adm, "probes_book": pb, "lifecycle": life,
                    "seconds": time.perf_counter() - seed_started,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        ps = [x for _, x in p]
        if not ps:
            # split_val hands alternate probes to the gate, so a horizon short
            # enough to contain only validation probes reports nothing. Say so
            # instead of dying on ps[0]: the run itself was fine, the schedule
            # was too short to measure. Probes land at (c-WARMUP, settle_cap=args.settle_cap) % 8 == 7, so
            # the first reported one is the second probe, at chunk 24.
            print(f"{key}: no evaluation probes at {args.n_chunks} chunks -- "
                  f"with --split-val the first reported probe is the second "
                  f"probe overall, so at least {WARMUP + 16} chunks are needed. "
                  f"Use --no-split-val only to reproduce the old protocol.",
                  flush=True)
            res[key] = {"probes": p, "drift": d, "committed": com, "held": hel,
                        "admitted": adm, "probes_book": pb, "lifecycle": life}
            with open(args.out, "w") as f:
                json.dump(dict(res, _config=sig), f, indent=1)
            continue
        print(f"{key}: {ps[0]:.3f} -> {ps[-1]:.3f}  delta {ps[-1]-ps[0]:+.3f}  "
              f"drift {d[-1][1]:.5f}  committed {com}/{hel}  "
              f"real {adm['R']}/{adm['R_held']} gen {adm['G']}/{adm['G_held']}",
              flush=True)
        with open(args.out, "w") as f:
            json.dump(dict(res, _config=sig), f, indent=1)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
