"""Verify the checkpointed chunk scan produces IDENTICAL gradients to the
naive (full-graph) implementation on a tiny model.

Run: python -m tests.test_remat
"""

import pytest

torch = pytest.importorskip("torch")

from tests.test_smoke import tiny_cfg  # noqa: E402
from ttt_pt.meta import compute_loss  # noqa: E402
from ttt_pt.model import TTTModel  # noqa: E402


def grads_snapshot(model):
    return {n: p.grad.clone() if p.grad is not None else None for n, p in model.named_parameters()}


def run(model, batch, cfg, use_remat):
    model.zero_grad(set_to_none=True)
    loss, aux = compute_loss(model, batch, cfg, step=3, create_graph=True, use_remat=use_remat)
    loss.backward()
    return loss.detach(), aux["chunk_losses"], grads_snapshot(model)


def test_remat_matches_naive():
    torch.manual_seed(0)
    cfg = tiny_cfg(seq_len=64, chunk=16, window=32)
    model = TTTModel(cfg.model, max_seq_len=cfg.training.seq_length).double()
    model.rope_cos = model.rope_cos.double()
    model.rope_sin = model.rope_sin.double()

    # fp64 for a tight comparison: patch compute dtype
    import ttt_pt.model as M

    M.COMPUTE_DTYPE = torch.float64

    B, L = 2, cfg.training.seq_length
    batch = torch.randint(0, cfg.model.vocab_size, (B, L + 1))

    loss_a, chunks_a, g_a = run(model, batch, cfg, use_remat=False)
    loss_b, chunks_b, g_b = run(model, batch, cfg, use_remat=True)

    assert torch.allclose(loss_a, loss_b, atol=1e-12), (loss_a, loss_b)
    assert torch.allclose(chunks_a, chunks_b, atol=1e-12)
    worst = 0.0
    for n in g_a:
        assert (g_a[n] is None) == (g_b[n] is None), n
        if g_a[n] is not None:
            d = (g_a[n] - g_b[n]).abs().max().item()
            scale = g_a[n].abs().max().item() + 1e-12
            worst = max(worst, d / scale)
            assert torch.allclose(g_a[n], g_b[n], atol=1e-9, rtol=1e-7), (
                f"{n}: {d} (rel {d / scale:.2e})"
            )
    print(f"remat gradients match naive; worst relative diff {worst:.2e}")


if __name__ == "__main__":
    test_remat_matches_naive()
    print("remat test passed")
