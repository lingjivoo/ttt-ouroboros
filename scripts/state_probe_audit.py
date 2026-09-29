"""Two audits of the streaming state machine, both requested by review.

A. Does the write weight w actually scale the update?

   The concern is that a normalizing inner optimizer would make ||dW|| roughly
   independent of w. It does not apply here for two reasons that are worth
   measuring rather than asserting: the inner step is plain SGD (meta.py, no
   moments), and w multiplies the LEARNING RATE after the clip
   (stream.py: inner_lr = lr * ilr_mult * provenance_w), not the loss. Clipping
   normalizes g, so a loss-scaling w would indeed cancel in the clipped regime;
   an lr-scaling w cannot.

   So the per-step prediction is exact: ||dW(w)|| = w * ||dW(1)||. This measures
   it, and separately measures the CUMULATIVE drift over many chunks, which is
   NOT linear in w -- dW_t depends on W_t, which depends on every earlier w.
   A Pareto plot whose x axis is w needs the first fact; its interpretation
   needs the second.

B. Is snapshot/restore actually complete?

   probe_score() snapshots, scores a chunk with w=0, and restores. If any live
   state escapes that round trip, probes contaminate the very trajectory they
   are supposed to observe passively, and every reported curve is suspect.

   Rather than arguing from the field list, this takes the round trip and
   compares every tensor bitwise, then checks the stronger end-to-end property:
   a run with probes interleaved must produce the same post-chunk NLLs as a run
   with no probes at all.

    python scripts/state_probe_audit.py --ckpt <ckpt> --val <val.npy>
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from horizon import CS, find_books  # noqa: E402

from ttt_pt.config import PRESETS  # noqa: E402
from ttt_pt.model import TTTModel  # noqa: E402
from ttt_pt.stream import StreamState  # noqa: E402


def dw_norm(st, ref):
    """L2 distance of the fast weights from a reference, per sample then mean."""
    tot = None
    for f, r in zip(st.fast, ref):
        d = (f - r).float().flatten(1).pow(2).sum(-1)
        tot = d if tot is None else tot + d
    return float(tot.sqrt().mean())


def flat_state(st):
    """Every tensor the stream carries, in a fixed order, for bitwise compare."""
    out = []
    for k, v in st.pre_kv + st.suf_kv + st.suf_partial:
        out += [k, v]
    out += list(st.fast)
    for delta in st.pending:
        out += list(delta)
    if getattr(st, "ref_kv", None) is not None:
        for k, v in st.ref_kv:
            out += [k, v]
    out += list(st.h_buffer)
    return out, (st.chunk_id, st.global_pos, len(st.tok_buffer), len(st.pending))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--preset", default="125m-e2e-ext32k")
    ap.add_argument("--n-seqs", type=int, default=2)
    ap.add_argument("--book-offset", type=int, default=2)
    ap.add_argument("--cum-chunks", type=int, default=12,
                    help="chunks for the cumulative-drift part of audit A")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]()
    dev = "cuda"
    T = (args.cum_chunks + 2) * CS
    model = TTTModel(cfg.model, max_seq_len=T + CS).to(dev)
    sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.eval()

    tokens = np.asarray(np.load(args.val, mmap_mode="r"))
    books = find_books(tokens, T + 1)[args.book_offset: args.book_offset + args.n_seqs]
    assert len(books) == args.n_seqs, f"only {len(books)} books long enough"
    batch = torch.from_numpy(np.stack(
        [tokens[s: s + T + 1].astype(np.int64) for s, _ in books])).to(dev)
    inp, tgt = batch[:, :-1], batch[:, 1:]
    B = args.n_seqs
    fails = []

    def ch(name, ok, detail=""):
        print(f"[{'  ok  ' if ok else ' FAIL '}] {name}" + (f"  {detail}" if detail else ""),
              flush=True)
        if not ok:
            fails.append(name)

    # ---------------------------------------------------------------- audit A1
    print("A1. single-step ||dW|| against w  (prediction: exactly proportional)\n",
          flush=True)
    ws = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    base = None
    rows = []
    for w in ws:
        st = StreamState(model, B, dev)
        ref = [f.clone() for f in st.fast]
        st.process_real_chunk(inp[:, :CS], tgt[:, :CS], w, 1.0, cfg)
        n = dw_norm(st, ref)
        if w == 1.0:
            base = n
        rows.append((w, n))
    print(f"    {'w':>7} {'||dW||':>12} {'/ w*||dW(1)||':>15}")
    for w, n in rows:
        pred = w * base
        ratio = "-" if w == 0 else f"{n / pred:.6f}" if pred > 0 else "n/a"
        print(f"    {w:>7} {n:>12.6e} {ratio:>15}")
    ch("w=0 writes nothing", rows[0][1] == 0.0, f"||dW||={rows[0][1]:.3e}")
    prop = [abs(n - w * base) / max(w * base, 1e-30) for w, n in rows if w > 0]
    ch("||dW|| is proportional to w", max(prop) < 1e-4,
       f"worst relative deviation {max(prop):.2e}")

    # ---------------------------------------------------------------- audit A2
    print(f"\nA2. cumulative drift over {args.cum_chunks} chunks  "
          f"(NOT expected to be linear: dW_t depends on W_t)\n", flush=True)
    for w in [0.25, 0.5, 1.0]:
        st = StreamState(model, B, dev)
        ref = [f.clone() for f in st.fast]
        for c in range(args.cum_chunks):
            sl = slice(c * CS, (c + 1) * CS)
            st.process_real_chunk(inp[:, sl], tgt[:, sl], w, 1.0, cfg)
        n = dw_norm(st, ref)
        print(f"    w={w:<5} cumulative ||W_T - W_0|| = {n:.6e}"
              f"   ratio to w*T*step = {n / (w * base * args.cum_chunks):.4f}")
    print("    A ratio far from 1 means the trajectory bends; that is a real")
    print("    property of the closed loop, not an implementation fault, but it")
    print("    is why a Pareto axis in w is not an axis in total write mass.")

    # ---------------------------------------------------------------- audit B1
    print("\nB1. snapshot -> mutate -> restore must be bitwise exact\n", flush=True)
    st = StreamState(model, B, dev)
    for c in range(3):
        sl = slice(c * CS, (c + 1) * CS)
        st.process_real_chunk(inp[:, sl], tgt[:, sl], 1.0, 1.0, cfg)
    before, before_scalars = flat_state(st)
    before = [t.clone() for t in before]
    snap = st.snapshot()
    for c in range(3, 6):
        sl = slice(c * CS, (c + 1) * CS)
        st.process_real_chunk(inp[:, sl], tgt[:, sl], 1.0, 1.0, cfg)
    st.restore(snap)
    after, after_scalars = flat_state(st)
    same_shape = len(before) == len(after)
    ch("restore returns the same tensor inventory", same_shape,
       f"{len(before)} tensors before, {len(after)} after")
    if same_shape:
        bad = [i for i, (a, b) in enumerate(zip(before, after))
               if a.shape != b.shape or not torch.equal(a, b)]
        ch("every restored tensor is bitwise identical", not bad,
           f"{len(bad)} of {len(before)} differ")
    ch("restored scalars match", before_scalars == after_scalars,
       f"{before_scalars} vs {after_scalars}")

    # ---------------------------------------------------------------- audit B2
    print("\nB2. end to end: interleaving probes must not change the stream\n",
          flush=True)
    st = StreamState(model, B, dev)
    clean = []
    for c in range(args.cum_chunks):
        sl = slice(c * CS, (c + 1) * CS)
        clean.append(float(st.process_real_chunk(
            inp[:, sl], tgt[:, sl], 1.0, 1.0, cfg).mean()))

    # same stream, but a probe on held-out text is scored after every chunk
    psl = slice(args.cum_chunks * CS, (args.cum_chunks + 1) * CS)
    st = StreamState(model, B, dev)
    probed = []
    for c in range(args.cum_chunks):
        sl = slice(c * CS, (c + 1) * CS)
        probed.append(float(st.process_real_chunk(
            inp[:, sl], tgt[:, sl], 1.0, 1.0, cfg).mean()))
        s = st.snapshot()
        st.process_real_chunk(inp[:, psl], tgt[:, psl], 0.0, 1.0, cfg)
        st.restore(s)
    worst = max(abs(a - b) for a, b in zip(clean, probed))
    ch("per-chunk NLL identical with and without interleaved probes",
       worst == 0.0, f"max |diff| = {worst:.3e} over {len(clean)} chunks")

    # ---------------------------------------------------------------- audit C
    print("\nC. retention is paced by chunks, not by accepted updates\n", flush=True)
    # The review's concern: if W <- W0 + lam(W - W0) fires once per ACCEPTED
    # update, an arm accepting 100 updates decays far more than one accepting 20
    # at the same lam, and gates with different acceptance rates are not
    # comparable. In this code the decay sits outside the `provenance_w > 0`
    # block, so it should fire once per chunk regardless -- which is already the
    # token-level half-life the review asks for, tau = CS / -ln(lam) tokens.
    # Measured rather than read off the indentation.
    lam = 0.9
    for label, w in [("w=0, nothing accepted", 0.0), ("w=1, everything accepted", 1.0)]:
        st = StreamState(model, B, dev)
        # displace the fast weights by a fixed vector, then let only decay act
        v = [torch.full_like(f, 0.01) for f in st.fast]
        st.fast = [(f + d).detach() for f, d in zip(st.fast_init, v)]
        n0 = dw_norm(st, st.fast_init)
        n_ch = 6
        for c in range(n_ch):
            sl = slice(c * CS, (c + 1) * CS)
            st.process_real_chunk(inp[:, sl], tgt[:, sl], w, 1.0, cfg, lam=lam)
        n1 = dw_norm(st, st.fast_init)
        eff = (n1 / n0) if n0 > 0 else float("nan")
        print(f"    {label:26s} ||W-W0|| {n0:.4e} -> {n1:.4e}   factor {eff:.6f}"
              f"   lam^{n_ch} = {lam ** n_ch:.6f}")
        if w == 0.0:
            ch("decay over 6 chunks with no writes equals lam^6",
               abs(eff - lam ** n_ch) < 1e-4, f"{eff:.6f} vs {lam ** n_ch:.6f}")
    print("    Equal factors across acceptance rates means lam is chunk-paced,")
    print("    so arms with different gate acceptance are already comparable.")
    print("    (With w=1 the writes also move W, so only the w=0 row isolates")
    print("     the decay schedule; the w=1 row is printed for contrast.)")

    print()
    if fails:
        print(f"{len(fails)} of the audits FAILED: {', '.join(fails)}")
        print("A1/B failures invalidate reported curves; do not queue on this build.")
    else:
        print("all audits passed")
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
