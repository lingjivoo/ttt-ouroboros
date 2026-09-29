"""E1+E2+E3: long-horizon self-decoding, open-vs-closed-loop causal control,
and the lambda x w stability grid — one protocol.

Stream layout (default 128 chunks = 128K positions):
  chunks 0..7   : real prefill (8K, writes on) — reproduces the original
                  TTT-E2E decode-time setting at horizon 8K+8K
  thereafter    : units of 8 = 7 decoded chunks + 1 real PROBE chunk
                  (probes never write; they measure clean-text NLL along the
                  trajectory, giving the entire horizon curve in one run)

Modes (who generates / who writes):
  closed : generate with CURRENT fast weights W_t, write (w) them back  [original TTT-E2E decode]
  open   : generate with FROZEN W_0, W_t still trains on those tokens   [causal control]
  masked : generate with W_t, never write                               [provenance mask]
  real   : no generation; real book text throughout, writes on          [utility baseline]

--w and --lam turn `closed` into the stability-grid cell (E3).
Per probe we also log fast-weight drift ||W_t - W_0|| / ||W_0||.

  python scripts/horizon.py --ckpt .. --val .. --mode closed --out h_closed.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState

BOS = 128000
CS = 1024
WARMUP = 8


def find_books(tokens, min_len):
    bos = np.flatnonzero(tokens == BOS)
    return [(int(s), int(bos[i + 1]) if i + 1 < len(bos) else len(tokens))
            for i, s in enumerate(bos)
            if (bos[i + 1] if i + 1 < len(bos) else len(tokens)) - s >= min_len]


def distinct2(ids):
    ids = ids.tolist()
    return len(set(zip(ids[:-1], ids[1:]))) / max(1, len(ids) - 1)


def drift(st):
    num = sum(float(((f - i0) ** 2).sum()) for f, i0 in zip(st.fast, st.fast_init))
    den = sum(float((i0 ** 2).sum()) for i0 in st.fast_init)
    return (num / den) ** 0.5


def probe_branched(st, seg_in, seg_tgt, cfg, branched=True):
    """Evaluate a real probe chunk WITHOUT letting it enter the stream state:
    snapshot, run (w=0), restore. With branched=False the probe text stays in
    the KV caches (the leaky protocol we report as an ablation)."""
    snap = st.snapshot() if branched else None
    nll = st.process_real_chunk(seg_in, seg_tgt, 0.0, 1.0, cfg)
    if branched:
        st.restore(snap)
    return nll


def run(model, cfg, device, real, mode, n_chunks, w, lam, seed, branched=True,
        record=None, replay=None, cut_at=-1, grad_at_init=False, real_w=1.0,
        drift_cap=0.0, ref_every=0, sampling_device="cpu"):
    """Single-adapter modes: closed | open | masked | real.

    cut_at >= 0 stops writing generated text from that chunk onward while
    generation continues, which is the deployable version of the intervention:
    you cannot avoid producing text, only decline to learn from it.
    """
    B = real.shape[0]
    st = StreamState(model, B, device)
    real_pos = 0
    probes, gens, drifts = [], [], []
    # Per-book probe values alongside the batch mean. Averaging the books here
    # is what made the headline look precise: the across-book sd of one arm is
    # 0.307 against 0.016 for the mean of four books (RESULTS §66), and seeds
    # add no independent variance on a teacher-forced path. The book is the
    # independent unit, so keep it.
    probes_book = []
    for c in range(n_chunks):
        is_probe = c >= WARMUP and (c - WARMUP) % 8 == 7
        use_real = mode == "real" or c < WARMUP or is_probe
        if use_real:
            seg_in = real[:, real_pos: real_pos + CS].to(device)
            seg_tgt = real[:, real_pos + 1: real_pos + CS + 1].to(device)
            if is_probe:
                nll = probe_branched(st, seg_in, seg_tgt, cfg, branched)
                probes.append((c, float(nll.mean())))
                probes_book.append((c, [float(x) for x in nll]))
                drifts.append((c, round(drift(st), 5)))
            else:
                st.process_real_chunk(seg_in, seg_tgt, real_w, 1.0, cfg, lam=lam,
                                      grad_at_init=grad_at_init,
                                      drift_cap=drift_cap)
            real_pos += CS
        elif mode == "replay":
            # tokens recorded from an INDEPENDENT closed-loop run: identical text
            # and identical quality trajectory, but no causal path from this
            # adapter's state to them.
            rc, rfirst, rgen = replay[len([g for g in gens])]
            assert rc == c, f"replay slot mismatch {rc} vs {c}"
            inp = torch.cat([rfirst, rgen[:, :-1]], 1).to(device)
            st.process_real_chunk(inp, rgen.to(device), w, 1.0, cfg, lam=lam)
            gens.append({"slot": c, "d2": round(float(np.mean(
                [distinct2(rgen[b]) for b in range(rgen.shape[0])])), 4)})
        else:
            first = real[:, real_pos: real_pos + 1].to(device)
            pv = {"closed": w, "open": w, "masked": 0.0}[mode]
            if 0 <= cut_at <= c:
                pv = 0.0
            gf = st.fast_init if mode == "open" else None
            gen = st.generate_chunk(first, provenance_w=pv, ilr_mult=1.0, cfg=cfg,
                                    seed=seed * 100000 + c, gen_fast=gf, lam=lam,
                                    sampling_device=sampling_device)
            gens.append({"slot": c, "d2": round(float(np.mean(
                [distinct2(gen[b]) for b in range(B)])), 4)})
            if record is not None:
                record.append((c, first.cpu(), gen.cpu()))
        if ref_every > 0 and (c + 1) % ref_every == 0:
            # Block boundary: the next block's updates are evaluated at the state
            # reached here, so blocks are serial while the chunks inside one stay
            # independent of each other. This is keyed on the chunk index alone.
            # It used to sit inside the non-probe branch, where every boundary
            # that landed on a probe was silently skipped -- and since probes fall
            # on c = 15, 23, 31, ..., that killed EVERY boundary at k=16 and all
            # but the first at k=8, so those arms were measuring no reset at all.
            st.grad_ref = [f.clone() for f in st.fast]
    return probes, gens, drifts, probes_book


def run_pair(model, cfg, device, real, mode, n_chunks, w, lam, seed, eps,
             branched=True):
    """Two streams from initial fast weights that differ by eps, nothing else.

    Whether the closed loop is a feedback system with rho > 1 is a statement about
    perturbation growth, and it can be measured directly instead of inferred from
    the shape of a damage-versus-exposure curve. Generation noise is keyed on
    (seed, chunk) rather than on the weights, so the two runs draw identical
    samples and any divergence comes from the perturbation alone.

    Returns per-chunk amplification ||W_t - W'_t|| / ||W_0 - W'_0||, the probe-NLL
    gap between the pair, and the first chunk whose generated tokens differ.
    """
    B = real.shape[0]
    sa, sb = StreamState(model, B, device), StreamState(model, B, device)
    # The fast weights are bfloat16, whose spacing is 2^-8 = 0.0078 relative. A
    # perturbation below that is not small -- it is unrepresentable, and lands on
    # whichever entries happen to sit near a rounding boundary. An eps=1e-12 null
    # run divided a bf16-scale divergence by a 2e-12 baseline and reported an
    # amplification of 9e11, which is what alerted us. Refuse anything under one
    # ULP, and require the applied perturbation to be the size we asked for.
    ulp = float(torch.finfo(sb.fast[0].dtype).eps)
    if 0 < eps < ulp:
        raise ValueError(f"eps={eps} is below one ULP of {sb.fast[0].dtype} "
                         f"({ulp}); the perturbation would be quantised away")
    g = torch.Generator(device="cpu").manual_seed(seed * 7919 + 11)
    for f in sb.fast:
        f.add_(torch.randn(f.shape, generator=g).to(f.device, f.dtype) * eps
               * f.abs().mean())
    # fast_init is cloned from fast at construction, so it now holds the PRE-
    # perturbation snapshot for sb. Re-clone it: the open-loop arm generates from
    # fast_init, and leaving it stale would make the two members of the pair read
    # from different generators rather than differ only in their live weights.
    sb.fast_init = [f.clone() for f in sb.fast]
    d0 = sum(float(((a - b) ** 2).sum()) for a, b in zip(sa.fast, sb.fast)) ** 0.5
    # eps=0 is the null run: identical weights and identical sampling seeds, so the
    # pair must stay bit-identical for the whole stream. Reported as an absolute
    # distance, since a ratio against a zero baseline is undefined -- and any
    # nonzero value here is nondeterminism in the harness, not amplification.
    null_run = (eps == 0.0)
    if not null_run and d0 == 0:
        raise ValueError(f"eps={eps} produced no change; check --eps")

    amp, gap, first_div, real_pos = [], [], None, 0
    for c in range(n_chunks):
        is_probe = c >= WARMUP and (c - WARMUP) % 8 == 7
        use_real = mode == "real" or c < WARMUP or is_probe
        if use_real:
            seg_in = real[:, real_pos: real_pos + CS].to(device)
            seg_tgt = real[:, real_pos + 1: real_pos + CS + 1].to(device)
            if is_probe:
                na = probe_branched(sa, seg_in, seg_tgt, cfg, branched)
                nb = probe_branched(sb, seg_in, seg_tgt, cfg, branched)
                gap.append((c, round(float(nb.mean() - na.mean()), 5)))
                dt = sum(float(((x - y) ** 2).sum())
                         for x, y in zip(sa.fast, sb.fast)) ** 0.5
                amp.append((c, round(dt, 8) if null_run else round(dt / d0, 4)))
            else:
                for st in (sa, sb):
                    st.process_real_chunk(seg_in, seg_tgt, 1.0, 1.0, cfg, lam=lam)
            real_pos += CS
        else:
            first = real[:, real_pos: real_pos + 1].to(device)
            pv = {"closed": w, "open": w, "masked": 0.0}[mode]
            # identical seed for both members: same sampling draws, so the streams
            # can only part company because their weights already differ
            ta = sa.generate_chunk(first, provenance_w=pv, ilr_mult=1.0, cfg=cfg,
                                   seed=seed * 100000 + c,
                                   gen_fast=sa.fast_init if mode == "open" else None,
                                   lam=lam)
            tb = sb.generate_chunk(first, provenance_w=pv, ilr_mult=1.0, cfg=cfg,
                                   seed=seed * 100000 + c,
                                   gen_fast=sb.fast_init if mode == "open" else None,
                                   lam=lam)
            if first_div is None and not torch.equal(ta, tb):
                first_div = c
    return {"amp": amp, "probe_gap": gap, "first_divergent_chunk": first_div,
            "d0": d0, "null_run": null_run, "eps": eps,
            "ulp": float(torch.finfo(sa.fast[0].dtype).eps)}


def run_crossed(model, cfg, device, real, n_chunks, w, lam, seed, branched=True):
    """Quality-matched causal control: two independently adapting streams A and
    B generate in lockstep, then each WRITES THE OTHER'S generation. Both learn
    from the output of an equally-degrading adapter, but neither learns from its
    own output, so the generation-quality distribution is matched by
    construction while the self-coupling is broken."""
    B = real.shape[0]
    A, Bs = StreamState(model, B, device), StreamState(model, B, device)
    real_pos = 0
    probes, gens, drifts = [], [], []
    probes_book = []
    for c in range(n_chunks):
        is_probe = c >= WARMUP and (c - WARMUP) % 8 == 7
        if c < WARMUP or is_probe:
            seg_in = real[:, real_pos: real_pos + CS].to(device)
            seg_tgt = real[:, real_pos + 1: real_pos + CS + 1].to(device)
            if is_probe:
                nll = probe_branched(A, seg_in, seg_tgt, cfg, branched)
                probe_branched(Bs, seg_in, seg_tgt, cfg, branched)
                probes.append((c, float(nll.mean())))
                probes_book.append((c, [float(x) for x in nll]))
                drifts.append((c, round(drift(A), 5)))
            else:
                A.process_real_chunk(seg_in, seg_tgt, 1.0, 1.0, cfg, lam=lam)
                Bs.process_real_chunk(seg_in, seg_tgt, 1.0, 1.0, cfg, lam=lam)
            real_pos += CS
        else:
            first = real[:, real_pos: real_pos + 1].to(device)
            # generate from each adapter WITHOUT writing (w=0) ...
            snapA, snapB = A.snapshot(), Bs.snapshot()
            gA = A.generate_chunk(first, provenance_w=0.0, ilr_mult=1.0, cfg=cfg,
                                  seed=seed * 100000 + c, lam=1.0)
            gB = Bs.generate_chunk(first, provenance_w=0.0, ilr_mult=1.0, cfg=cfg,
                                   seed=seed * 100000 + 50000 + c, lam=1.0)
            # ... then each writes the OTHER's tokens through the normal path
            A.restore(snapA)
            Bs.restore(snapB)
            A.process_real_chunk(torch.cat([first, gB[:, :-1]], 1), gB, w, 1.0, cfg, lam=lam)
            Bs.process_real_chunk(torch.cat([first, gA[:, :-1]], 1), gA, w, 1.0, cfg, lam=lam)
            gens.append({"slot": c, "d2": round(float(np.mean(
                [distinct2(gA[b]) for b in range(B)])), 4)})
    return probes, gens, drifts, probes_book


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--preset", default="125m-e2e-ext32k")
    ap.add_argument("--mode", required=True,
                    choices=["closed", "open", "masked", "real", "crossed", "replay"])
    ap.add_argument("--leaky-probe", action="store_true",
                    help="let probe text persist in KV caches (ablation)")
    ap.add_argument("--n-chunks", type=int, default=128)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", default="42,1")
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--run-pair", action="store_true",
                    help="amplification experiment: run a perturbed pair instead of "
                         "a single stream")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="relative perturbation for --run-pair. Must exceed one ULP "
                         "of the fast-weight dtype (bf16: 0.0078); eps=0 is the null "
                         "run, which must stay bit-identical")
    ap.add_argument("--grad-at-init", action="store_true",
                    help="evaluate the inner gradient at W_0 instead of W_t "
                         "(the frozen-reference update whose chunk-independence "
                         "is what a parallel scan would need)")
    ap.add_argument("--drift-cap", type=float, default=0.0,
                    help="project ||W_t-W_0||/||W_0|| back to this value after "
                         "each update (0 disables). One global operation on the "
                         "accumulated sum, so it survives a parallel scan.")
    ap.add_argument("--ref-every", type=int, default=0,
                    help="with --grad-at-init, refresh the gradient reference "
                         "point every k chunks: parallel within a block, serial "
                         "across blocks (0 = never, i.e. always W_0)")
    ap.add_argument("--real-w", type=float, default=1.0,
                    help="write weight for REAL chunks (--w controls generated "
                         "ones). 0 gives a no-TTT reference on the same stream.")
    ap.add_argument("--cut-at", type=int, default=-1,
                    help="disable writes from this chunk onward; generation "
                         "continues. -1 leaves the mode unmodified.")
    ap.add_argument("--book-offset", type=int, default=0,
                    help="start from a different set of books; with --mode replay "
                         "this breaks the identity between the recording stream and "
                         "the learning stream (cross-stream replay)")
    ap.add_argument("--replay-from", default="",
                    help="record file from a closed-loop run (mode=replay)")
    ap.add_argument("--record-to", default="",
                    help="save this run's generated chunks for later replay")
    ap.add_argument("--gpu-sampling", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PRESETS[args.preset]()
    device = "cuda"
    model = TTTModel(cfg.model, max_seq_len=args.n_chunks * CS + CS).to(device)
    stt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(stt["model"] if "model" in stt else stt, strict=False)
    model.eval()

    tokens = np.asarray(np.load(args.val, mmap_mode="r"))
    n_real = args.n_chunks if args.mode == "real" else (
        WARMUP + (args.n_chunks - WARMUP) // 8 + 2)
    need = n_real * CS + 1
    books = find_books(tokens, need)
    print(f"{len(books)} books with >= {need} tokens", flush=True)
    real = torch.from_numpy(np.stack(
        [tokens[s: s + need].astype(np.int64)
         for s, _ in books[args.book_offset: args.book_offset + args.n_seqs]]))

    # Per-seed resume, guarded by a config signature. This script previously
    # started from an empty dict and overwrote args.out, so a freecycle
    # preemption at ~2.5h threw away every finished seed -- which is how one
    # arm lost its third seed. The signature is what makes resuming safe: the
    # key alone does not mention the settings or the output fields, and a rerun
    # after adding probes_book would otherwise return the old records and look
    # done. Mismatched or unsigned files are discarded, not merged.
    sig = {"mode": args.mode, "n_chunks": args.n_chunks, "n_seqs": args.n_seqs,
           "book_offset": args.book_offset, "w": args.w, "lam": args.lam,
           "preset": args.preset, "ckpt": args.ckpt, "val": args.val,
           "cut_at": args.cut_at, "grad_at_init": bool(args.grad_at_init),
           "real_w": args.real_w, "drift_cap": args.drift_cap,
           "ref_every": args.ref_every, "leaky_probe": bool(args.leaky_probe),
           "sampling_device": "cuda" if args.gpu_sampling else "cpu",
           "fields": "probes_book"}
    results = {}
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
        except Exception:
            results = {}
        old_sig = results.pop("_config", None) if isinstance(results, dict) else None
        if results and old_sig != sig:
            why = ("written under a different config" if old_sig is not None
                   else "written before configs were signed")
            print(f"discarding {args.out}: {why}; recomputing rather than mixing.",
                  flush=True)
            results = {}
        elif results:
            print(f"resuming, have {sorted(results)}", flush=True)
    for sd in [int(x) for x in args.seeds.split(",")]:
        key = f"{args.mode}_w{args.w}_lam{args.lam}_s{sd}" + \
            (f"_cut{args.cut_at}" if args.cut_at >= 0 else "") + \
            ("_gi" if args.grad_at_init else "") + \
            (f"_cap{args.drift_cap}" if args.drift_cap > 0 else "") + \
            (f"_ref{args.ref_every}" if args.ref_every > 0 else "") + \
            (f"_rw{args.real_w}" if args.real_w != 1.0 else "") + \
            ("_leaky" if args.leaky_probe else "")
        if key in results:
            print(f"skip {key} (done)", flush=True)
            continue
        br = not args.leaky_probe
        rec = [] if args.record_to else None
        rep = None
        if args.mode == "replay":
            rep = torch.load(f"{args.replay_from}_s{sd}.pt", weights_only=False)
        if args.eps >= 0 and args.run_pair:
            r = run_pair(model, cfg, device, real, args.mode, args.n_chunks,
                         args.w, args.lam, sd, args.eps, br)
            results[key + f"_eps{args.eps}"] = r
            print(f"{key} eps={args.eps}: amplification "
                  f"{[a for _, a in r['amp']]}", flush=True)
            print(f"  first divergent generated chunk: "
                  f"{r['first_divergent_chunk']}", flush=True)
            with open(args.out, "w") as f:
                json.dump(dict(results, _config=sig), f, indent=1)
            continue
        if args.mode == "crossed":
            probes, gens, drifts, pbook = run_crossed(model, cfg, device, real,
                                                      args.n_chunks, args.w, args.lam, sd, br)
        else:
            probes, gens, drifts, pbook = run(model, cfg, device, real, args.mode,
                                       args.n_chunks, args.w, args.lam, sd, br,
                                       record=rec, replay=rep, cut_at=args.cut_at,
                                       grad_at_init=args.grad_at_init,
                                       real_w=args.real_w,
                                       drift_cap=args.drift_cap,
                                       ref_every=args.ref_every,
                                       sampling_device="cuda" if args.gpu_sampling else "cpu")
            if rec is not None:
                torch.save(rec, f"{args.record_to}_s{sd}.pt")
        results[key] = {"probes": probes, "gen": gens, "drift": drifts,
                        "probes_book": pbook}
        print(f"{key}: probes {[round(p,3) for _,p in probes]}", flush=True)
        print(f"  drift {[d for _,d in drifts]}", flush=True)
        with open(args.out, "w") as f:
            json.dump(dict(results, _config=sig), f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
