"""Prospective advantage A_t along the four existing protocols (observation only).

For every write event, does the update it produced improve prediction of real
text the learner has not yet used? A_t > 0 means yes. The four conditions the
paper already runs make opposite predictions:

    real text          A_t > 0        (writes carry forward-useful information)
    frozen-generator   A_t ~ 0        (large drift, no information: the arm
                                       writes 1.6x more than closed yet stays
                                       flat on clean probes)
    self-generated     A_t: 0 -> neg  (degeneracy builds, then compounds)
    degenerate replay  A_t < 0        (harmful content, no feedback needed)

If that separation appears, prospective utility explains in one number what
provenance, drift and causal independence each fail to explain alone -- and the
gate that PG-TTT / deferred-verification applies is measuring the right thing.

Isolation detail that matters: A_t is the probe-NLL difference caused by the
WRITE alone. Both evaluations therefore run after the chunk, on the same
advanced KV caches, one with the post-update fast weights and one with the
pre-update fast weights swapped back in. Measuring "before the chunk" vs
"after" instead would fold the cache advance into A_t.

The gate text is the next scheduled probe chunk: real, never written, and the
same schedule in every condition.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, "scripts")
from horizon import CS, WARMUP, find_books, probe_branched  # noqa: E402

from ttt_pt.config import PRESETS  # noqa: E402
from ttt_pt.model import TTTModel  # noqa: E402
from ttt_pt.stream import StreamState  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--preset", default="125m-e2e-ext32k")
    ap.add_argument("--mode", required=True,
                    choices=["real", "open", "closed", "replay", "corrupt",
                             "retrieve", "mixed"])
    # mixed: one stream carrying all four sources, interleaved.
    #
    # Every other condition is a homogeneous stream, so a gate that separates
    # them could in principle be recognising which experiment it is in rather
    # than judging updates one at a time. Interleaving the sources inside a
    # single run removes that reading: the state, the schedule and the probes
    # are shared, and the only thing that varies between consecutive write
    # events is where the tokens came from. It is also the only design that
    # tests the claim provenance cannot make -- rejecting externally replayed
    # degeneracy, which a provenance rule admits because the learner did not
    # write it.
    ap.add_argument("--mix", default="real,self,frozen,replay",
                    help="mixed mode: sources to cycle through, one per write "
                         "slot, in this order")
    # retrieve: the write slots are filled with REAL text fetched from a store
    # instead of with the model's own output. Provenance is satisfied by
    # construction, so this asks the question provenance cannot: once the text
    # is external, how accurate does the fetch have to be? A hit is a chunk
    # from the stream's own book, a miss is the chunk at the SAME pool offset
    # from a different book -- same length, same corpus, same difficulty, only
    # the topic differs, so precision is the only thing varying.
    ap.add_argument("--book-need-chunks", type=int, default=0,
                    help="select books by this length instead of by what the "
                         "run needs. find_books returns whatever clears the "
                         "threshold, so a distance sweep whose length "
                         "requirement grows with the distance would silently "
                         "change the book set as well; pinning it keeps the "
                         "sweep a sweep over one variable.")
    ap.add_argument("--retrieve-back", type=int, default=-1,
                    help="retrieve mode: instead of drawing from a disjoint "
                         "pool, write the chunk this many positions BEFORE the "
                         "slot's own position in the same book. 0 reproduces "
                         "sequential reading; larger values keep the book and "
                         "the corpus fixed and move the written text further "
                         "from the probe, which is the only thing that "
                         "separates real-mode's +0.069 from pool retrieval's "
                         "-0.134. -1 uses the pool.")
    ap.add_argument("--precision", type=float, default=1.0,
                    help="probability that a retrieved chunk comes from the "
                         "stream's own book rather than another one")
    # corrupt: uniform random token ids in place of generated text. It is the
    # cell that decides whether raw likelihood can serve as a write gate at
    # all. Noise is the most surprising thing the model can be shown, and the
    # corruption probes already established that the learner trains HARDER on
    # it (immediate damage 0.52 with writes on against 0.26 with them off), so
    # a gate that admits whatever it finds surprising should admit noise first.
    ap.add_argument("--xv-holdout", type=float, default=0.0,
                    help="fraction of each chunk held out of its own update and "
                         "then scored: a within-chunk stand-in for the external "
                         "probe that A_t needs. 0 disables it.")
    ap.add_argument("--burst", type=int, default=0,
                    help="length of a self-generated run inserted between "
                         "external slots. 0 keeps the alternating schedule. "
                         "The review asks for a moderate alternating ratio and "
                         "a long generated burst at the SAME source counts, so "
                         "that 'settlement equals masking' cannot be dismissed "
                         "as an artefact of external text always being one "
                         "chunk away: with a burst, candidates must survive "
                         "many generated chunks before any evidence arrives.")
    ap.add_argument("--admit", default="all",
                    choices=["all", "mask", "gate", "gate_lam", "oracle",
                             "settle", "delay", "settle_risk"],
                    help="admission policy applied to a mixed stream. `all` "
                         "writes everything and is the observational arm the "
                         "matrix was measured under. `mask` is the provenance "
                         "rule: it rejects only the learner's own samples, and "
                         "on a mixed stream that still admits the frozen "
                         "generator AND the replayed degeneracy, which is "
                         "exactly its blind spot. `oracle` admits only real "
                         "text -- the label a perfect authority signal would "
                         "supply, and an upper bound the other arms can be read "
                         "against. `gate` admits on measured prospective "
                         "advantage without consulting origin; `gate_lam` adds "
                         "retention. mask and oracle differ here in a way they "
                         "cannot in a homogeneous stream, which is the point of "
                         "running this on a mixed one.")
    ap.add_argument("--admit-margin", type=float, default=0.0)
    ap.add_argument("--settle-scope", choices=["global", "row"], default="global")
    ap.add_argument("--settle-risk-q", type=float, default=1.1954710086281652,
                    help="One-sided calibrated residual quantile for risk-controlled settlement")
    ap.add_argument("--settle-alpha-grid", default="0,0.25,0.5,0.75,1",
                    help="Candidate write doses for --admit settle_risk")
    ap.add_argument("--settle-audit-horizon", type=int, default=0,
                    help="Step-0 diagnostic: score keep and skip on this many subsequent real chunks")
    ap.add_argument("--log-tread", action="store_true",
                    help="also record, per write event, the READ-path transfer "
                         "T_read = NLL(gate text | chunk absent) - NLL(gate "
                         "text | chunk in cache), with fast weights held at "
                         "their pre-update values. Symmetric to A: A swaps the "
                         "fast weights on a fixed cache, T_read swaps the cache "
                         "under fixed fast weights. Costs one extra probe "
                         "forward per event; off by default so running arms "
                         "are not slowed.")
    ap.add_argument("--lam", type=float, default=0.98,
                    help="retention factor for --admit gate_lam")
    ap.add_argument("--replay-from", default="")
    ap.add_argument("--n-chunks", type=int, default=128)
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--book-offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PRESETS[args.preset]()
    dev = "cuda"
    model = TTTModel(cfg.model, max_seq_len=args.n_chunks * CS + CS).to(dev)
    stt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(stt.get("model", stt), strict=False)
    model.eval()

    tokens = np.asarray(np.load(args.val, mmap_mode="r"))
    n_real = args.n_chunks if args.mode == "real" else (
        WARMUP + (args.n_chunks - WARMUP) // 8 + 2)
    need = n_real * CS + 1
    pool0 = need
    if args.mode == "mixed":
        n_slots = sum(1 for c in range(args.n_chunks)
                      if not (c < WARMUP or (c >= WARMUP and (c - WARMUP) % 8 == 7)))
        need += (n_slots + 1) * CS
    elif args.mode == "retrieve" and args.retrieve_back >= 0:
        # every chunk position needs real text behind it, plus the lookback
        need = (args.n_chunks + args.retrieve_back + 2) * CS + 1
    elif args.mode == "retrieve":
        # one fresh pool chunk per write slot, so nothing is recycled and the
        # store never re-serves text the stream has already written
        n_slots = sum(1 for c in range(args.n_chunks)
                      if not (c < WARMUP or (c >= WARMUP and (c - WARMUP) % 8 == 7)))
        need += (n_slots + 1) * CS   # +1 chunk of slack: the last pool chunk needs CS+1 tokens
    sel = max(need, args.book_need_chunks * CS + 1)
    books = find_books(tokens, sel)[args.book_offset: args.book_offset + args.n_seqs]
    assert len(books) == args.n_seqs, f"only {len(books)} books clear {sel} tokens"
    real = torch.from_numpy(np.stack(
        [tokens[s: s + need].astype(np.int64) for s, _ in books]))
    print(f"{len(books)} books, mode={args.mode}", flush=True)

    # Precompute, per chunk index, which slice of `real` the NEXT probe reads.
    # real_pos advances only on real chunks, and the schedule is deterministic.
    is_probe = [c >= WARMUP and (c - WARMUP) % 8 == 7
                for c in range(args.n_chunks)]
    is_real = [args.mode == "real" or c < WARMUP or is_probe[c]
               for c in range(args.n_chunks)]
    if args.mode == "corrupt":
        # corrupt chunks are teacher-forced like replay, but they are not real
        # text and must not consume the real-token cursor
        pass
    pos = 0
    real_pos_of = {}
    for c in range(args.n_chunks):
        if is_real[c]:
            real_pos_of[c] = pos
            pos += CS
    # A gate that validates on the same probe whose NLL is later reported is
    # selecting on the test statistic. Probes alternate: even-numbered ones
    # validate admissions, odd-numbered ones are reported, and neither set is
    # ever used for the other job. This is the same correction applied to the
    # deferred-gate runs.
    probe_idx = {}
    _pi = 0
    for c in range(args.n_chunks):
        if is_probe[c]:
            probe_idx[c] = _pi
            _pi += 1
    val_probe = {c: (i % 2 == 0) for c, i in probe_idx.items()}
    next_probe_pos = {}
    for c in range(args.n_chunks):
        nxt = next((cp for cp in range(c + 1, args.n_chunks)
                    if is_probe[cp] and val_probe.get(cp, False)),
                   None)
        next_probe_pos[c] = real_pos_of.get(nxt)

    replay = None
    if args.mode == "mixed" and "replay" in args.mix and not args.replay_from:
        raise SystemExit("mixed mode with a replay slot needs --replay-from")
    if args.mode == "replay" or (args.mode == "mixed" and "replay" in args.mix):
        replay = torch.load(f"{args.replay_from}_s{args.seed}.pt",
                            weights_only=False)

    st = StreamState(model, real.shape[0], dev)
    rows, gen_i, slot_i, hits = [], 0, 0, []
    probes = []
    pend_q, settled, settle_audit = [], [], []
    mix_order = [x.strip() for x in args.mix.split(",")]
    srcs = []
    for c in range(args.n_chunks):
        gate = next_probe_pos[c]
        pre_fast = [f.clone() for f in st.fast] if gate is not None else None
        rp = real_pos_of.get(c)
        wrote = False
        xv_seg = None
        # A rejected admission must leave the state untouched, so every arm but
        # `all` snapshots before the write and restores if the policy says no.
        pre_snap = st.snapshot() if (args.xv_holdout > 0
                                     or args.admit != "all"
                                     or args.log_tread) else None
        chunk_nll = None
        if is_real[c] and is_probe[c]:
            seg_in = real[:, rp: rp + CS].to(dev)
            seg_tgt = real[:, rp + 1: rp + CS + 1].to(dev)
            pv = probe_branched(st, seg_in, seg_tgt, cfg, True)
            # Only the reported half enters the degradation series; the
            # validation half is what the gate saw and must not be scored.
            # The reported subset must be identical in every arm. Letting the
            # ungated arm report all fifteen probes while the gated arms report
            # seven compared different first and last chunks: read one way the
            # same run degrades +0.007, read the other +0.261.
            if not val_probe.get(c, False):
                # process_real_chunk already returns one value per sequence, as
                # horizon.py's probes_book relies on; averaging again collapses
                # it to a 0-d tensor and the per-book list cannot be built.
                probes.append([c, float(pv.mean()), [float(x) for x in pv]])
        elif is_real[c]:
            seg_in = real[:, rp: rp + CS].to(dev)
            seg_tgt = real[:, rp + 1: rp + CS + 1].to(dev)
            st.process_real_chunk(seg_in, seg_tgt, 1.0, 1.0, cfg)
            xv_seg = (seg_in, seg_tgt)
            wrote = True
        elif args.mode == "replay":
            rc, rfirst, rgen = replay[gen_i]
            gen_i += 1
            assert rc == c, (rc, c)
            # the recording was made at n_seqs 8; the first n rows are the same
            # books in the same order, so slicing keeps the pairing intact
            rfirst, rgen = rfirst[: real.shape[0]], rgen[: real.shape[0]]
            inp = torch.cat([rfirst, rgen[:, :-1]], 1).to(dev)
            rtgt = rgen.to(dev)
            st.process_real_chunk(inp, rtgt, 1.0, 1.0, cfg)
            xv_seg = (inp, rtgt)
            wrote = True
        elif args.mode == "mixed":
            if args.burst and (slot_i // args.burst) % 2 == 1:
                # burst phase: a run of self-generated chunks with no external
                # text, so candidates must wait many chunks for evidence
                which = "self"
            else:
                which = mix_order[slot_i % len(mix_order)]
            if which == "real":
                off = pool0 + slot_i * CS
                seg = real[:, off: off + CS + 1]
                mi, mt = seg[:, :-1].to(dev), seg[:, 1:].to(dev)
                st.process_real_chunk(mi, mt, 1.0, 1.0, cfg)
            elif which == "replay":
                rc, rfirst, rgen = replay[gen_i]; gen_i += 1
                rfirst, rgen = rfirst[: real.shape[0]], rgen[: real.shape[0]]
                mi = torch.cat([rfirst, rgen[:, :-1]], 1).to(dev)
                mt = rgen.to(dev)
                st.process_real_chunk(mi, mt, 1.0, 1.0, cfg)
            elif which in ("shuffled", "uniform"):
                # The two controls that pull surprise and transfer apart.
                # shuffled: the real slot's tokens in random order -- identical
                # unigram surprise, destroyed sequence, so any T_read left is
                # bag-of-words transfer. uniform: tokens drawn uniformly from
                # the slot's own vocabulary -- maximal surprise, and if
                # "transferable surprise" is the real quantity, T_read near
                # zero despite the highest S in the stream.
                off = pool0 + slot_i * CS
                seg = real[:, off: off + CS + 1]
                mt = seg[:, 1:].clone()
                g2 = torch.Generator(device="cpu").manual_seed(
                    args.seed * 1000003 + c)
                for b in range(mt.shape[0]):
                    if which == "shuffled":
                        mt[b] = mt[b][torch.randperm(mt.shape[1], generator=g2)]
                    else:
                        vocab = torch.unique(mt[b])
                        idx = torch.randint(len(vocab), (mt.shape[1],),
                                            generator=g2)
                        mt[b] = vocab[idx]
                mi = torch.cat([seg[:, :1], mt[:, :-1]], 1).to(dev)
                mt = mt.to(dev)
                st.process_real_chunk(mi, mt, 1.0, 1.0, cfg)
            else:
                first = real[:, pool0 + slot_i * CS: pool0 + slot_i * CS + 1].to(dev)
                gf = st.fast_init if which == "frozen" else None
                st.generate_chunk(first, provenance_w=1.0, ilr_mult=1.0, cfg=cfg,
                                  seed=args.seed * 100000 + c, gen_fast=gf)
            srcs.append(which)
            slot_i += 1
            chunk_nll = getattr(st, "last_nll", None)
            wrote = True
        elif args.mode == "retrieve":
            import random as _rr
            u = _rr.Random(args.seed * 100000 + c).random()
            if args.retrieve_back >= 0:
                # same book, same corpus; only the distance to the probe varies
                src = max(0, c - args.retrieve_back)
                off = src * CS
            else:
                off = pool0 + slot_i * CS
            seg = real[:, off: off + CS + 1]
            if u >= args.precision:
                seg = torch.roll(seg, 1, dims=0)   # same offset, another book
            ri, rt = seg[:, :-1].to(dev), seg[:, 1:].to(dev)
            st.process_real_chunk(ri, rt, 1.0, 1.0, cfg)
            xv_seg = (ri, rt)
            hits.append(1 if u < args.precision else 0)
            slot_i += 1
            wrote = True
        elif args.mode == "corrupt":
            g = torch.Generator(device="cpu")
            g.manual_seed(args.seed * 100000 + c)
            V = cfg.model.vocab_size
            rnd = torch.randint(0, V, (real.shape[0], CS + 1), generator=g)
            ri, rt = rnd[:, :-1].to(dev), rnd[:, 1:].to(dev)
            st.process_real_chunk(ri, rt, 1.0, 1.0, cfg)
            xv_seg = (ri, rt)
            wrote = True
        else:
            first = real[:, rp if rp is not None else 0: 1].to(dev) \
                if rp is not None else real[:, :1].to(dev)
            gf = st.fast_init if args.mode == "open" else None
            st.generate_chunk(first, provenance_w=1.0, ilr_mult=1.0, cfg=cfg,
                              seed=args.seed * 100000 + c, gen_fast=gf)
            wrote = True

        # Within-chunk cross-validation: hold the tail of the chunk out of its
        # own update, then ask whether the update built from the head predicts
        # that tail better. Information generalizes from one part of a text to
        # another; noise does not. Unlike A it needs no external probe -- the
        # ruler is inside the chunk. Teacher-forced modes only: for generated
        # chunks the tokens are not known until after the write, which needs a
        # second pass, and one risky change at a time.
        xv = None
        if args.xv_holdout > 0 and wrote and xv_seg is not None:
            hi, ti = xv_seg
            keep = int(CS * (1.0 - args.xv_holdout))
            snap = st.snapshot()
            st.restore(pre_snap)
            tw = torch.ones(real.shape[0], CS, device=dev)
            tw[:, keep:] = 0.0
            st.process_real_chunk(hi, ti, 1.0, 1.0, cfg, token_w=tw)
            head_fast = [f.clone() for f in st.fast]
            st.restore(pre_snap)
            probe_branched(st, hi, ti, cfg, True)
            pre_tail = float(st.last_token_nll[:, keep:].mean())
            base_fast = st.fast
            st.fast = head_fast
            probe_branched(st, hi, ti, cfg, True)
            post_tail = float(st.last_token_nll[:, keep:].mean())
            st.fast = base_fast
            st.restore(snap)
            xv = pre_tail - post_tail

        if wrote and gate is not None:
            g_in = real[:, gate: gate + CS].to(dev)
            g_tgt = real[:, gate + 1: gate + CS + 1].to(dev)
            post = float(probe_branched(st, g_in, g_tgt, cfg, True).mean())
            cur = st.fast
            st.fast = pre_fast
            pre = float(probe_branched(st, g_in, g_tgt, cfg, True).mean())
            st.fast = cur
            tread = None
            if args.log_tread and pre_snap is not None:
                # Third corner of the swap square: gate text scored from the
                # pre-chunk state, so the cache does not hold this chunk and
                # the fast weights are their pre-update values. t0 - pre is
                # then the read path's transfer with the write path held out,
                # exactly as pre - post is the write path's with the read path
                # held fixed.
                here = st.snapshot()
                st.restore(pre_snap)
                t0 = float(probe_branched(st, g_in, g_tgt, cfg, True).mean())
                st.restore(here)
                tread = t0 - pre
            dnorm = sum(float(((a - b) ** 2).sum())
                        for a, b in zip(cur, pre_fast)) ** 0.5
            # nll is the chunk's own loss under the state that wrote it: the
            # headroom hypothesis says this predicts A, and unlike A it is
            # available before the write and costs nothing.
            src_now = srcs[-1] if srcs and args.mode == "mixed" else (
                "real" if is_real[c] else "self")
            # Admission is decided here, after A has been measured on a
            # validation probe, and enforced by restoring the pre-write state.
            keep = True
            if args.admit == "mask":
                # Fixed throughout, per the review: the provenance mask rejects
                # the learner's OWN samples and nothing else. It therefore
                # admits the frozen generator and the replayed degeneracy --
                # that is its blind spot, not a configuration choice.
                keep = src_now != "self"
            elif args.admit == "oracle":
                # Source label, a diagnostic reference and NOT an optimal
                # policy: real text is not always the beneficial update.
                keep = src_now == "real"
            elif args.admit in ("gate", "gate_lam"):
                keep = (pre - post) > args.admit_margin
            elif args.admit in ("settle", "delay", "settle_risk"):
                # Settlement: a candidate is judged only when the NEXT external
                # (non-self) chunk arrives, and is judged on that chunk. Between
                # arrivals it waits outside the live weights. `delay` waits the
                # same number of chunks and commits unconditionally -- the
                # control that separates waiting from validating.
                keep = False        # decided below, not here
                pend_q.append((c, src_now, [f.clone() for f in pre_fast],
                               [f.clone() for f in st.fast]))
                if src_now != "self":
                    # external text has arrived: settle everything waiting
                    for (pc, psrc, pf0, pf1) in pend_q:
                        alpha_v = None
                        if args.admit == "delay":
                            ok = True
                        else:
                            cur = st.fast
                            st.fast = pf0
                            sc0v = probe_branched(st, g_in, g_tgt, cfg, True).detach().float().cpu()
                            if args.admit == "settle_risk":
                                grid = sorted({float(z) for z in args.settle_alpha_grid.split(",")})
                                assert grid[0] == 0.0 and grid[-1] <= 1.0
                                alpha_v = torch.zeros_like(sc0v)
                                scv = sc0v
                                for av in grid[1:]:
                                    # Independent candidate state: never mutate pf0/pf1,
                                    # because later doses must be evaluated from the same base.
                                    st.fast = [d0.clone().add_(d1-d0, alpha=av)
                                               for d0,d1 in zip(pf0,pf1)]
                                    sca = probe_branched(st, g_in, g_tgt, cfg, True).detach().float().cpu()
                                    dv = sca - sc0v
                                    sd = dv.std(unbiased=True).clamp_min(1e-8)
                                    safe = dv + args.settle_risk_q * sd <= 0
                                    alpha_v = torch.where(safe, torch.full_like(alpha_v,av), alpha_v)
                                    scv = sca
                                okv = alpha_v > 0
                                ok = bool(okv.any())
                            else:
                                st.fast = pf1
                                scv = probe_branched(st, g_in, g_tgt, cfg, True).detach().float().cpu()
                                okv = scv <= sc0v - args.admit_margin
                                ok = bool(okv.all()) if args.settle_scope == "row" else bool(scv.mean() <= sc0v.mean() - args.admit_margin)
                            st.fast = cur
                            # Counterfactual Step-0 audit. Each arm is evaluated
                            # from the identical live cache on later real text;
                            # neither arm is allowed to modify the deployed state.
                            hindsight=[]
                            if args.settle_audit_horizon:
                                for ah in range(1,args.settle_audit_horizon+1):
                                    q0=gate+ah*CS
                                    if q0+CS+1>real.shape[1]:break
                                    qi=real[:,q0:q0+CS].to(dev);qt=real[:,q0+1:q0+CS+1].to(dev)
                                    st.fast=pf1; hk=probe_branched(st,qi,qt,cfg,True).detach().float().cpu()
                                    st.fast=pf0; hs=probe_branched(st,qi,qt,cfg,True).detach().float().cpu()
                                    hindsight.append((hk-hs).tolist()) # >0 means accepting was harmful
                                st.fast=cur
                            settle_audit.append({"c":pc,"src":psrc,"validation_keep_minus_skip":(scv-sc0v).tolist(),
                                "global_kept":bool(scv.mean()<=sc0v.mean()-args.admit_margin),"row_kept":okv.tolist(),
                                "hindsight_keep_minus_skip":hindsight})
                        settled.append({"c": pc, "src": psrc, "kept": bool(ok),
                                        "kept_rows":okv.tolist() if args.admit in ("settle","settle_risk") else None,
                                        "alpha_rows":alpha_v.tolist() if alpha_v is not None else None,"waited": c - pc})
                        if args.admit == "settle_risk":
                            with torch.no_grad():
                                for f,d0,d1 in zip(st.fast,pf0,pf1):
                                    m=alpha_v.to(f.device,f.dtype).view(-1,*([1]*(f.dim()-1)))
                                    f.add_((d1-d0)*m)
                        elif args.settle_scope == "row" and args.admit == "settle":
                            with torch.no_grad():
                                for f,d0,d1 in zip(st.fast,pf0,pf1):
                                    m=okv.to(f.device,f.dtype).view(-1,*([1]*(f.dim()-1)))
                                    f.add_((d1-d0)*m)
                        elif ok:
                            with torch.no_grad():
                                for f, d0, d1 in zip(st.fast, pf0, pf1):
                                    f.add_(d1 - d0)
                    pend_q.clear()
                keep = None         # bookkeeping handled by `settled`
            if keep is False:
                # Reject the WRITE, keep the READ. Restoring the full pre-chunk
                # snapshot expunged the chunk from the attention cache as well,
                # so a gated arm stopped ingesting anything: its state froze at
                # the warm-up, every later candidate was scored from that stale
                # state, scored negative, and was expunged in turn -- a
                # permanent lockout that admitted 9/113 in every seed, all of
                # them the warm-up chunks. The paper's mask semantics are
                # read-but-never-write (provenance_w=0 leaves text in cache);
                # rejection here must mean the same: fast weights roll back,
                # the cache advance stands.
                with torch.no_grad():
                    for f, f0 in zip(st.fast, pre_fast):
                        f.copy_(f0)
            elif keep is True and args.admit == "gate_lam":
                with torch.no_grad():
                    for f, f0 in zip(st.fast, st.fast_init):
                        f.mul_(args.lam).add_(f0, alpha=1.0 - args.lam)
            rows.append({"c": c, "A": pre - post, "tread": tread,
                         "dnorm": dnorm,
                         # last_nll at this point is the GATE text's loss: the
                         # probe_branched calls above overwrote it, so the field
                         # recorded ~3.6 for every source including uniform
                         # random. chunk_nll is captured at write time instead;
                         # "nll" is kept for old readers but renamed truthfully.
                         "probe_nll": getattr(st, "last_nll", None),
                         "nll": chunk_nll,
                         "xv": xv, "gen": not is_real[c], "kept": bool(keep),
                         "hit": hits[-1] if hits and args.mode == "retrieve" else None,
                         "src": src_now})
            if len(rows) % 16 == 1:
                print(f"  c={c:3d} A={pre-post:+.5f} |d|={dnorm:.4f} "
                      f"nll={st.last_nll:.4f}", flush=True)

    A = [r["A"] for r in rows]
    print(f"\n{args.mode}: n={len(A)} meanA={np.mean(A):+.5f} "
          f"first16={np.mean(A[:16]):+.5f} last16={np.mean(A[-16:]):+.5f} "
          f"fracA>0={np.mean([a > 0 for a in A]):.2f}", flush=True)
    if hits:
        print(f"retrieval: {sum(hits)}/{len(hits)} hits "
              f"(target precision {args.precision})", flush=True)
    # For settle/delay the per-row `kept` is None; the decision lives in
    # `settled`, which also records how long each candidate waited.
    if settled:
        kept = [x for x in settled if x["kept"]]
        import statistics as _st
        print(f"settled {len(kept)}/{len(settled)}; "
              f"median wait {_st.median([x['waited'] for x in settled]):.0f} chunks; "
              f"still pending at end {len(pend_q)}", flush=True)
        by = {}
        for x in settled:
            by.setdefault(x["src"], [0, 0])
            by[x["src"]][0] += 1
            by[x["src"]][1] += int(x["kept"])
        print("  by source: " + "  ".join(f"{k} {v[1]}/{v[0]}"
                                          for k, v in sorted(by.items())),
              flush=True)
    else:
        kept = [r for r in rows if r.get("kept", True)]
    if args.admit != "all":
        # settle/delay record their decisions in `settled`, not in rows[].kept
        # (which stays False there because the row is written before the
        # candidate is judged). Reading rows here printed 0/N for every source
        # while the data itself was correct.
        src_iter = (settled if settled
                    else [{"src": r["src"], "kept": r.get("kept", True)}
                          for r in rows])
        by = {}
        for r in src_iter:
            by.setdefault(r["src"], [0, 0])
            by[r["src"]][0] += 1
            by[r["src"]][1] += int(r["kept"])
        print("admissions by source: " +
              "  ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by.items())),
              flush=True)
    if len(probes) >= 4:
        # A first-minus-last difference on this series is dominated by its own
        # scatter (sd ~0.09 against effects of ~0.05), so report the thirds
        # difference as well and let the analysis use it.
        v = [p[1] for p in probes]
        k = max(1, len(v) // 3)
        early, late = sum(v[:k]) / k, sum(v[-k:]) / k
        print(f"reported probes {len(probes)}: first {v[0]:.4f} last {v[-1]:.4f} "
              f"(pair {v[-1]-v[0]:+.4f})  thirds {early:.4f}->{late:.4f} "
              f"({late-early:+.4f})", flush=True)
    json.dump({"mode": args.mode, "seed": args.seed, "rows": rows,
               "admit": args.admit, "lam": args.lam,
               # Two batches of this experiment were retired because the file
               # could not say which rejection semantics produced it. Now it
               # says. Any analysis must check this field before combining.
               "reject_semantics": "fast-only",
               "n_kept": len(kept), "probes": probes,
               "settled": settled, "settle_scope": args.settle_scope,
               "settle_audit_horizon": args.settle_audit_horizon,
               "settle_audit": settle_audit, "burst": args.burst,
               "probes_book": [[c, bk] for c, _m, bk in probes],
               "precision": args.precision, "hits": hits},
              open(args.out, "w"), indent=1)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
