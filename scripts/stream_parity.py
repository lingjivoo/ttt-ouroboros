"""Does the streaming path agree with the standard path on this checkpoint?

The 3B results are produced by the streaming processor -- chunk-by-chunk with a
rolling KV cache and a live fast-weight state -- while the checkpoint was
converted against the standard training forward. Conversion was verified
(sequence loss to 1.05e-4, token NLL r=0.9996); the streaming decoder was
verified only at 125M. So "the 3B closed-minus-masked gap is a streaming
implementation artefact" is a live objection, and this closes it without
needing the JAX environment: the same tokens through both paths in the same
process, on the same device, in the same dtype.

Two modes are checked, because they exercise different code:

  w=0   no inner update. Tests prefix/suffix attention, the rolling cache and
        the chunk boundary alone.
  w=1   the inner update runs. Tests the fast-weight path that every reported
        number depends on.

A disagreement at w=0 is an attention or cache bug; a disagreement only at w=1
is in the update. Teacher-forced throughout, so this is deterministic and can
serve as a canary for future runs -- unlike anything that samples, where 96.9%
of generated tokens differ between runs of one seed.

    python scripts/stream_parity.py --ckpt <ckpt> --val <val.npy> \
        --preset official-3b-ext128k --n-chunks 8 --n-seqs 2
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, "scripts")
from horizon import CS, find_books  # noqa: E402

from ttt_pt.config import PRESETS  # noqa: E402
from ttt_pt.meta import loss_for_sequence_meta  # noqa: E402
from ttt_pt.model import TTTModel  # noqa: E402
from ttt_pt.stream import StreamState  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--preset", default="125m-e2e-ext32k")
    ap.add_argument("--n-chunks", type=int, default=8)
    ap.add_argument("--n-seqs", type=int, default=2)
    ap.add_argument("--book-offset", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="fallback tolerance. The real reference is the measured "
                         "noise floor: an arbitrary threshold cannot tell a bug "
                         "from bf16 accumulation order.")
    ap.add_argument("--fp32", action="store_true",
                    help="run both paths in float32. This is the test that "
                         "separates the two live explanations for the "
                         "disagreement: bf16 rounding under a different kernel "
                         "collapses toward zero in fp32, a genuine difference "
                         "in window handling does not.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.fp32:
        # stream.py did `from ttt_pt.model import COMPUTE_DTYPE`, binding the
        # value at import, so patching the model module alone would leave the
        # streaming path in bf16 and compare two different dtypes.
        import ttt_pt.model as _m
        import ttt_pt.stream as _s
        _m.COMPUTE_DTYPE = torch.float32
        _s.COMPUTE_DTYPE = torch.float32
        print("running both paths in float32", flush=True)

    cfg = PRESETS[args.preset]()
    dev = "cuda"
    T = args.n_chunks * CS
    model = TTTModel(cfg.model, max_seq_len=T + CS).to(dev)
    st = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st.get("model", st), strict=False)
    model.eval()

    tokens = np.asarray(np.load(args.val, mmap_mode="r"))
    books = find_books(tokens, T + 1)[args.book_offset: args.book_offset + args.n_seqs]
    assert len(books) == args.n_seqs, f"only {len(books)} books long enough"
    batch = torch.from_numpy(np.stack(
        [tokens[s: s + T + 1].astype(np.int64) for s, _ in books])).to(dev)
    inp, tgt = batch[:, :-1], batch[:, 1:]
    mask = tgt != cfg.model.bos_token_id
    print(f"{args.preset}: {args.n_seqs} books x {args.n_chunks} chunks "
          f"({T} tokens), tol {args.tol}", flush=True)

    results = {}
    ok = True
    for w in (0.0, 1.0):
        with torch.no_grad():
            _, aux = loss_for_sequence_meta(
                model, inp, tgt, mask, 1.0, cfg,
                create_graph=False)
        std = aux["chunk_losses"].tolist()

        state = StreamState(model, args.n_seqs, dev)
        strm = []
        for c in range(args.n_chunks):
            sl = slice(c * CS, (c + 1) * CS)
            nll = state.process_real_chunk(inp[:, sl], tgt[:, sl], w, 1.0, cfg)
            strm.append(float(nll.mean()))

        # The standard path always applies the update; only compare against it
        # in the w=1 case. For w=0 the streaming state is frozen, so the
        # reference is the first chunk repeated under no adaptation -- instead
        # compare the two paths chunk-by-chunk only where both mean the same
        # thing, which is w=1.
        if w == 0.0:
            print("  w=0 (no inner update), streaming per-chunk NLL:")
            print("   ", " ".join(f"{x:.4f}" for x in strm))
            results["stream_w0"] = strm
            continue

        diffs = [abs(a - b) for a, b in zip(std, strm)]
        worst = max(diffs)
        results["standard_w1"] = std
        results["stream_w1"] = strm
        results["max_abs_diff"] = worst
        print("  w=1 (inner update on), per-chunk NLL")
        print("    standard :", " ".join(f"{x:.4f}" for x in std))
        print("    streaming:", " ".join(f"{x:.4f}" for x in strm))
        print(f"    max |diff| = {worst:.3e}   mean |diff| = {sum(diffs)/len(diffs):.3e}")
        if worst > args.tol:
            ok = False
            print(f"    FAIL: exceeds tol {args.tol}")
        else:
            print("    PASS")

    # Where does the disagreement live: the read path or the update?
    #
    # A first attempt measured a "noise floor" by reversing the order of the
    # books in the batch. That probe is inert -- with B=2 each sequence reduces
    # independently along its own axis, so batch order changes no accumulation
    # order and it returned 0.000e+00 on both models, telling us nothing.
    #
    # The comparison that does separate the two is w=0 on BOTH paths. Setting
    # the inner learning rate to zero makes the standard path skip the update
    # while leaving prefix, suffix and cache untouched, so:
    #   disagreement at w=0  -> attention, rolling cache or chunk boundary
    #   agreement at w=0, disagreement at w=1 -> the inner update
    lr0 = cfg.training.optimizer_inner.lr
    cfg.training.optimizer_inner.lr = 0.0
    try:
        with torch.no_grad():
            _, aux0 = loss_for_sequence_meta(model, inp, tgt, mask, 1.0, cfg,
                                             create_graph=False)
        std0 = aux0["chunk_losses"].tolist()
    finally:
        cfg.training.optimizer_inner.lr = lr0
    strm0 = results.get("stream_w0", [])
    if strm0 and len(strm0) == len(std0):
        d0 = [abs(a - b) for a, b in zip(std0, strm0)]
        results["standard_w0"] = std0
        results["max_abs_diff_w0"] = max(d0)
        print("\n  w=0 (no update on either path), per-chunk NLL")
        print("    standard :", " ".join(f"{x:.4f}" for x in std0))
        print("    streaming:", " ".join(f"{x:.4f}" for x in strm0))
        print(f"    max |diff| = {max(d0):.3e}")
        w1 = results.get("max_abs_diff", 0.0)
        if max(d0) == 0.0 and w1 == 0.0:
            print("    both paths agree exactly with and without the update")
        elif max(d0) == 0.0:
            print("    READ PATH EXACT: the disagreement is created by the "
                  "inner update, not by attention or the cache")
        elif abs(max(d0) - w1) < 0.2 * max(max(d0), w1, 1e-30):
            print("    the same disagreement is already present with no update "
                  "at all, so it is the read path, not the update")
        else:
            print("    disagreement present at w=0 and changed by w=1; both "
                  "paths contribute")

    # What can "agreement" mean in bf16 here? Run the standard path on the first
    # half of the same tokens and compare the chunks both runs share. Identical
    # arithmetic on identical inputs; only the total length T differs, which is
    # enough to change kernel selection and split-k order in the bulk
    # [B, T, D] prefix forward. That difference is the floor.
    #
    # This replaces an earlier probe that reversed the book order in the batch.
    # That one was inert -- each sequence reduces along its own axis, so batch
    # order changes no accumulation order -- and it returned 0.000e+00 on both
    # models, which looked like a clean result and measured nothing.
    half = args.n_chunks // 2
    with torch.no_grad():
        _, aux_h = loss_for_sequence_meta(
            model, inp[:, :half * CS], tgt[:, :half * CS], mask[:, :half * CS],
            1.0, cfg, create_graph=False)
    std_h = aux_h["chunk_losses"].tolist()
    std_full = results.get("standard_w1", [])
    if std_full and len(std_h) == half:
        fl = [abs(a - b) for a, b in zip(std_h, std_full[:half])]
        floor = max(fl)
        results["length_floor"] = floor
        print(f"\n  accumulation-order floor (standard path at T={half * CS} vs "
              f"T={args.n_chunks * CS}, same chunks): {floor:.3e}")
        w1 = results.get("max_abs_diff", 0.0)
        if floor > 0:
            print(f"  streaming-vs-standard is {w1 / floor:.1f}x that floor")
            if w1 <= 2 * floor:
                print("  at the floor: the two paths agree as closely as the "
                      "arithmetic is defined")
            else:
                print("  above the floor: the paths differ by more than "
                      "accumulation order alone explains")

    # Size the agreement against the effect it has to be small compared to.
    # The 3B closed-minus-masked gap is about 0.4 nats; a per-chunk agreement
    # three orders of magnitude below that cannot manufacture it.
    if "max_abs_diff" in results:
        print(f"\nagreement is {results['max_abs_diff']:.3e} against a "
              f"closed-minus-masked gap of ~0.4 nats at 3B")
    print("\nRESULT:", "PASS" if ok else "FAIL", flush=True)
    if args.out:
        import json
        json.dump({k: v for k, v in results.items()}, open(args.out, "w"), indent=1)
        print("saved", args.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
