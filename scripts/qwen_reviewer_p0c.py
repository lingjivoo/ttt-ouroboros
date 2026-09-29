"""P0-C: Qwen all-real adaptation utility on a fixed 128K token stream."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CS = 1024


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(t, p)


def local_manifest_sha(root):
    p = Path(root) / "DOWNLOAD_MANIFEST.json"
    return sha(p) if p.exists() else None


class Adapter:
    def __init__(self, model, lr):
        for p in model.parameters():
            p.requires_grad_(False)
        self.target = []
        for lyr in model.model.layers[-4:]:
            q = lyr.mlp.down_proj.weight
            q.requires_grad_(True)
            self.target.append(q)
        self.opt = (
            torch.optim.Adam(
                self.target, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
            )
            if lr
            else None
        )

    def bind(self, model):
        self.model = model

    def step(self, ids):
        if self.opt is None:
            return {"loss": None, "grad_norm_preclip": None, "clip_scale": None}
        logits = self.model(ids, use_cache=False).logits[:, :-1].float()
        target = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        grads = torch.autograd.grad(loss, self.target)
        gn = torch.sqrt(sum(g.float().pow(2).sum() for g in grads))
        scale = min(1.0, 1.0 / (float(gn) + 1e-8))
        for p, g in zip(self.target, grads):
            p.grad = g * scale
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return {
            "loss": float(loss.detach()),
            "grad_norm_preclip": float(gn),
            "clip_scale": scale,
        }


@torch.no_grad()
def score(model, probe):
    vals = []
    for c in range(probe.shape[1] // CS):
        ids = probe[:, c * CS : (c + 1) * CS + 1]
        lg = model(ids, use_cache=False).logits[:, :-1].float()
        z = (
            F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="none"
            )
            .view(ids.shape[0], -1)
            .mean(1)
        )
        vals.append(z.cpu().numpy())
    return np.stack(vals).mean(0).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--group", type=int, choices=(0, 1), required=True)
    p.add_argument(
        "--condition", choices=("no_update", "lr1e-5", "lr1e-4"), required=True
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lr = {"no_update": 0.0, "lr1e-5": 1e-5, "lr1e-4": 1e-4}[a.condition]
    started = time.time()
    result = {"status": "running"}
    save(a.out, result)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                a.model,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .cuda()
            .eval()
        )
        books = list(range(2 + 4 * a.group, 6 + 4 * a.group))
        probe_books = list(range(10 + 4 * a.group, 14 + 4 * a.group))
        root = Path(a.data)
        real = torch.tensor(
            np.stack(
                [np.load(root / f"book_{b:03d}.npy")[: 128 * CS + 1] for b in books]
            ),
            dtype=torch.long,
            device="cuda",
        )
        probe = torch.tensor(
            np.stack(
                [np.load(root / f"book_{b:03d}.npy")[: 4 * CS + 1] for b in probe_books]
            ),
            dtype=torch.long,
            device="cuda",
        )
        eng = Adapter(model, lr)
        eng.bind(model)
        curve = [{"adapted_chunks": 0, "nll_book": score(model, probe)}]
        stats = []
        torch.cuda.reset_peak_memory_stats()
        for c in range(128):
            stats.append(eng.step(real[:, c * CS : (c + 1) * CS + 1]))
            if c + 1 in (32, 64, 96, 128):
                curve.append({"adapted_chunks": c + 1, "nll_book": score(model, probe)})
            if (c + 1) % 8 == 0:
                print(a.condition, a.group, c + 1, flush=True)
        result = {
            "status": "passed",
            "manifest": {
                "protocol": "reviewer-p0c-v1",
                "model_path": a.model,
                "model_manifest_sha256": local_manifest_sha(a.model),
                "tokenizer_class": tok.__class__.__name__,
                "dtype": "torch.bfloat16",
                "attention": "sdpa",
                "optimizer": "torch.optim.Adam",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0,
                "lr": lr,
                "gradient_clip_norm": 1.0,
                "updated_parameters": "last 4 model.layers[*].mlp.down_proj.weight",
                "condition": a.condition,
                "group": a.group,
                "book_ids": books,
                "probe_book_ids": probe_books,
                "chunk_tokens": CS,
                "adapted_chunks": 128,
                "evaluation_schedule": [0, 32, 64, 96, 128],
                "evaluation_chunks_per_book": 4,
                "adaptation_state_ownership": "one shared Adam/weight state across four-book batch",
                "data_manifest_sha256": sha(root / "manifest.json"),
                "code_sha256": sha(__file__),
            },
            "curve": curve,
            "step_stats": stats,
            "benefit_book": (
                np.asarray(curve[0]["nll_book"]) - np.asarray(curve[-1]["nll_book"])
            ).tolist(),
            "seconds": time.time() - started,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        save(a.out, result)
        print("passed", a.out, flush=True)
    except Exception:
        import traceback

        result = {
            "status": "failed",
            "error": traceback.format_exc(),
            "seconds": time.time() - started,
        }
        save(a.out, result)
        raise


if __name__ == "__main__":
    main()
