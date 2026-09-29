"""HF-TTT wrapper correctness: with zero-initialized prime.w2, the wrapped
prefix+chunked-suffix pipeline must reproduce the base model's logits under
the same sliding-window mask. Then one inner step must reduce chunk loss.

  python -m tests.test_hf_ttt --model Qwen/Qwen3-1.7B
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from ttt_pt.hf_ttt import HFTTT

DEV = "cuda"


def reference_logits(wrap, input_ids):
    """Base model, full forward, same sliding-window mask on every layer."""
    m = wrap.base.model
    T = input_ids.shape[1]
    x = m.embed_tokens(input_ids)
    pos = torch.arange(T, device=DEV)[None]
    idx = torch.arange(T, device=DEV)
    mask4 = ((idx[:, None] >= idx[None, :]) &
             (idx[:, None] - idx[None, :] < wrap.window))[None, None]
    pe = wrap.rotary(x, pos)
    for layer in m.layers:
        out = layer(x, attention_mask=mask4, position_ids=pos, position_embeddings=pe)
        x = out[0] if isinstance(out, tuple) else out
    x = wrap.base.model.norm(x)
    return wrap.base.lm_head(x).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--fp32", action="store_true",
                    help="run in fp32 to separate numerical drift from real bugs")
    args = ap.parse_args()

    torch.manual_seed(0)
    dt = torch.float32 if args.fp32 else torch.bfloat16
    wrap = HFTTT(args.model, window=1024, chunk=512, dtype=dt).to(DEV)
    wrap.eval()
    T = 2048
    ids = torch.randint(1000, 50000, (1, T), device=DEV)

    with torch.no_grad():
        ref = reference_logits(wrap, ids)
        h = wrap.prefix_forward(ids)
        fast = [f.detach() for f in wrap.init_fast_weights(1)]
        kv = wrap.init_kv_caches(1, DEV)
        outs = []
        for c in range(T // wrap.chunk):
            lg, kv = wrap.suffix_chunk_forward(
                h[:, c * wrap.chunk:(c + 1) * wrap.chunk], fast, kv, c)
            outs.append(lg)
        got = torch.cat(outs, 1)

    diff = (ref - got).abs()
    rel = diff.max() / ref.abs().max()
    print(f"logits max|diff| = {diff.max().item():.5f}  rel = {rel.item():.2e}"
          f"  (dtype {dt})")
    tol = 1e-5 if args.fp32 else 3e-2
    assert rel < tol, f"wrapped pipeline diverges from base model (tol {tol})"

    # inner-step smoke: one SGD step on fast weights lowers the chunk loss
    tgt = ids[:, 1: wrap.chunk + 1]
    fast = [f.clone().requires_grad_(True) for f in wrap.init_fast_weights(1)]
    kv = wrap.init_kv_caches(1, DEV)
    with torch.enable_grad():
        lg, _ = wrap.suffix_chunk_forward(h[:, : wrap.chunk], fast, kv, 0)
        loss0 = F.cross_entropy(lg[0, :-1], tgt[0, :-1])
        g = torch.autograd.grad(loss0, fast)
    gn = torch.sqrt(sum(x.float().pow(2).sum() for x in g))
    fast2 = [(f - 1.0 / gn * gr).detach() for f, gr in zip(fast, g)]
    with torch.no_grad():
        lg2, _ = wrap.suffix_chunk_forward(h[:, : wrap.chunk], fast2, kv, 0)
        loss1 = F.cross_entropy(lg2[0, :-1], tgt[0, :-1])
    print(f"chunk loss before/after inner step: {loss0.item():.4f} -> {loss1.item():.4f}")
    assert loss1 < loss0, "inner step failed to reduce loss"
    n_out = sum(p.numel() for p in wrap.primes.parameters())
    print(f"outer-trainable prime params: {n_out/1e6:.1f}M — all tests passed")


if __name__ == "__main__":
    main()
