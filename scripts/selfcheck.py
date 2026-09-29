#!/usr/bin/env python3
"""Fail fast on missing packages, artifacts, CUDA, or checkpoint incompatibility."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


def readable(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    with path.open("rb") as handle:
        handle.read(1)
    return f"{path} ({path.stat().st_size / 2**30:.2f} GiB)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("language", "webshop", "all"), default="language")
    parser.add_argument("--full", action="store_true", help="load 125M and run one CUDA chunk")
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--val", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    def check(name, fn, fatal=True):
        try:
            detail = fn()
            print(f"[{OK}] {name}" + (f" -- {detail}" if detail else ""))
            return True
        except Exception as exc:
            print(f"[{BAD if fatal else WARN}] {name} -- {type(exc).__name__}: {exc}")
            (failures if fatal else warnings).append(name)
            return False

    check("Python 3.11+", lambda: sys.version.split()[0] if sys.version_info >= (3, 11)
          else (_ for _ in ()).throw(RuntimeError(sys.version.split()[0])))
    modules = ["numpy", "yaml", "torch", "einops", "ttt_pt.model", "ttt_pt.stream"]
    if args.profile in ("webshop", "all"):
        modules += ["transformers", "peft", "rank_bm25", "bs4", "flask"]
    for module in modules:
        check(f"import {module}", lambda name=module: getattr(importlib.import_module(name), "__version__", "ok"))

    data_root = Path(os.environ.get("TTT_DATA", "")) if os.environ.get("TTT_DATA") else None
    ckpt_root = Path(os.environ.get("TTT_CKPT", "")) if os.environ.get("TTT_CKPT") else None
    val = args.val or (data_root / "pg19/val.npy" if data_root else None)
    ckpt = args.ckpt or (ckpt_root / "125m-ext32k.pt" if ckpt_root else None)
    if args.profile in ("language", "all"):
        check("PG-19 validation array", lambda: readable(val, "validation data") if val else (_ for _ in ()).throw(RuntimeError("set TTT_DATA or --val")))
        check("125M checkpoint", lambda: readable(ckpt, "checkpoint") if ckpt else (_ for _ in ()).throw(RuntimeError("set TTT_CKPT or --ckpt")))
    out = Path(os.environ.get("TTT_OUT", ROOT / "results"))

    def writable():
        out.mkdir(parents=True, exist_ok=True)
        marker = out / ".selfcheck"
        marker.write_text("ok\n")
        marker.unlink()
        return str(out)

    check("output directory writable", writable)

    if args.profile in ("webshop", "all"):
        def webshop_root():
            path = Path(os.environ["WEBSHOP_DIR"]).resolve()
            if not path.is_dir():
                raise FileNotFoundError(path)
            return str(path)

        check("WEBSHOP_DIR", webshop_root)
        check("WEBSHOP_DATA", lambda: readable(Path(os.environ["WEBSHOP_DATA"]), "catalogue"))

    if args.full and not failures:
        import numpy as np
        import torch

        check("CUDA visible", lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available()
              else (_ for _ in ()).throw(RuntimeError("no CUDA device")))

        def one_chunk():
            from ttt_pt.config import PRESETS
            from ttt_pt.model import TTTModel
            from ttt_pt.stream import StreamState

            cfg = PRESETS["125m-e2e-ext32k"]()
            model = TTTModel(cfg.model, max_seq_len=2 * cfg.model.mini_batch_size).cuda().eval()
            state = torch.load(ckpt, map_location="cuda", weights_only=False)
            model.load_state_dict(state.get("model", state), strict=False)
            tokens = np.load(val, mmap_mode="r")
            size = cfg.model.mini_batch_size
            sequence = torch.from_numpy(np.asarray(tokens[: size + 1], dtype=np.int64))[None].cuda()
            stream = StreamState(model, 1, "cuda")
            nll = float(stream.process_real_chunk(sequence[:, :-1], sequence[:, 1:], 1.0, 1.0, cfg).mean())
            if not 0.5 < nll < 20:
                raise RuntimeError(f"implausible NLL {nll:.4f}")
            return f"NLL={nll:.4f}"

        check("load checkpoint and process one chunk", one_chunk)

    print(f"\n{len(failures)} failures; {len(warnings)} warnings")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
