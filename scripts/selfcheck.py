"""Check that this environment can run the experiments, before one is queued.

Every failure this catches has actually happened here: a dataset path that was
a directory rather than a file, a checkpoint symlink whose target another user
could not read, an environment whose agent package could not initialize, a
job that ran for two hours and died on an import. The point is to fail in
thirty seconds instead of two hours.

    python scripts/selfcheck.py          # no GPU needed, ~10 seconds
    python scripts/selfcheck.py --full   # adds GPU, checkpoint load, one chunk

Exit status is the number of failures, so a job script can gate on it.
"""
import argparse
import importlib
import os
import sys
import traceback

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "
fails = []
warns = []


def check(name, fn, fatal=True):
    try:
        detail = fn()
        print(f"[{OK}] {name}" + (f"  -- {detail}" if detail else ""))
        return True
    except Exception as e:
        tag = BAD if fatal else WARN
        print(f"[{tag}] {name}\n         {type(e).__name__}: {e}")
        (fails if fatal else warns).append(name)
        return False


def env_paths():
    root = os.environ.get("TTT_ROOT")
    if not root:
        raise RuntimeError("TTT_ROOT is not set -- run `source env.sh` first")
    missing = [k for k in ("TTT_DATA", "TTT_CKPT", "TTT_OUT") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"unset: {', '.join(missing)} -- run `source env.sh`")
    return root


def readable(path, what):
    """Readability, not existence. A symlink to a file another user owns
    resolves fine and then fails on open, which is the failure that wastes a
    queue slot rather than a second."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what}: {path} (symlink target missing?)")
    with open(path, "rb") as f:
        f.read(1)
    return f"{os.path.getsize(path) / 2**30:.1f} GB" if os.path.isfile(path) else "dir"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also touch the GPU, load a checkpoint and run one chunk")
    args = ap.parse_args()

    print("=" * 66)
    print("environment")
    print("=" * 66)
    check("TTT_* variables set", env_paths)
    check("python >= 3.10", lambda: (
        f"{sys.version_info.major}.{sys.version_info.minor}"
        if sys.version_info >= (3, 10) else (_ for _ in ()).throw(
            RuntimeError(f"python {sys.version.split()[0]} is too old"))))

    print()
    print("=" * 66)
    print("imports")
    print("=" * 66)
    for mod, fatal in [("torch", True), ("numpy", True), ("transformers", True),
                       ("peft", True), ("ttt_pt.model", True), ("ttt_pt.stream", True),
                       ("alfworld", False), ("scienceworld", False)]:
        check(f"import {mod}", lambda m=mod: getattr(
            importlib.import_module(m), "__version__", "ok"), fatal=fatal)

    print()
    print("=" * 66)
    print("data and checkpoints (readable, not merely present)")
    print("=" * 66)
    data = os.environ.get("TTT_DATA", "")
    ckpt = os.environ.get("TTT_CKPT", "")
    check("pg19 val split", lambda: readable(os.path.join(data, "pg19/val.npy"), "dataset"))
    check("pg19 train split", lambda: readable(os.path.join(data, "pg19/train.npy"), "dataset"), fatal=False)
    check("125M extended checkpoint",
          lambda: readable(os.path.join(ckpt, "125m-ext32k.pt"), "checkpoint"))
    check("recorded degenerate stream (replay control)",
          lambda: readable(os.path.join(ckpt, "replay_traj_s42.pt"), "checkpoint"), fatal=False)

    def writable_out():
        out = os.environ["TTT_OUT"]
        os.makedirs(out, exist_ok=True)
        p = os.path.join(out, ".selfcheck")
        open(p, "w").write("x")
        os.remove(p)
        return out
    check("TTT_OUT is writable", writable_out)

    if args.full:
        print()
        print("=" * 66)
        print("gpu and a real forward pass")
        print("=" * 66)
        import numpy as np
        import torch

        def gpu():
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device visible -- are you on a compute node?")
            return f"{torch.cuda.get_device_name(0)}, {torch.cuda.device_count()} visible"
        if check("CUDA device", gpu):
            def bf16_gemm():
                a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
                (a @ a).sum().item()
                return "bf16 matmul ok"
            # This has failed cluster-side before with CUBLAS_STATUS_INVALID_VALUE
            # on some partitions; catching it here identifies the node rather
            # than a code change.
            check("bf16 matmul on device", bf16_gemm)

            def load_and_step():
                from ttt_pt.config import PRESETS
                from ttt_pt.model import TTTModel
                from ttt_pt.stream import StreamState
                cfg = PRESETS["125m-e2e-ext32k"]()
                model = TTTModel(cfg.model, max_seq_len=2 * cfg.model.mini_batch_size).cuda()
                st = torch.load(os.path.join(ckpt, "125m-ext32k.pt"),
                                map_location="cuda", weights_only=False)
                model.load_state_dict(st.get("model", st), strict=False)
                model.eval()
                tok = np.asarray(np.load(os.path.join(data, "pg19/val.npy"), mmap_mode="r"))
                CS = cfg.model.mini_batch_size
                seg = torch.from_numpy(tok[:CS + 1].astype("int64"))[None].cuda()
                stream = StreamState(model, 1, "cuda")
                nll = stream.process_real_chunk(seg[:, :-1], seg[:, 1:], 1.0, 1.0, cfg)
                v = float(nll.mean())
                if not (0.5 < v < 20):
                    raise RuntimeError(f"chunk NLL {v:.3f} is implausible; wrong checkpoint or corpus?")
                return f"one 1024-token chunk, NLL {v:.3f}"
            if not check("load checkpoint and run one chunk", load_and_step):
                print()
                print("         NOTE: a CUBLAS failure here is usually the cluster,")
                print("         not your environment. It has been observed")
                print("         intermittently on freecycle-h100 (nodes h100-2 and")
                print("         h100-3) while a plain bf16 matmul on the same device")
                print("         passed, and with two different virtualenvs whose CUDA")
                print("         libraries are byte-identical. Jobs that ran minutes")
                print("         earlier on the same partition succeeded. Resubmit;")
                print("         if it follows you across several nodes, then suspect")
                print("         the environment.")

    print()
    print("=" * 66)
    if fails:
        print(f"{len(fails)} FAILED: {', '.join(fails)}")
        print("Fix these before queueing anything; every one of them would")
        print("otherwise surface after the job reaches a GPU.")
    else:
        print("all required checks passed" + (
            f" ({len(warns)} optional missing: {', '.join(warns)})" if warns else ""))
    print("=" * 66)
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
