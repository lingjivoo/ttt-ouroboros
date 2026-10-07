# ruff: noqa: E402
# Tensor imports follow importorskip so CPU-only environments can collect tests.
"""Compare reused backward against an explicit truncated FO reference."""

import copy

import pytest

torch = pytest.importorskip("torch")
from ttt_pt.block_inner import suffix_block_forward
from ttt_pt.config import Config, ModelConfig, TrainingConfig
from ttt_pt.meta import clip_per_sample, masked_ce
from ttt_pt.model import TTTModel
from ttt_pt.shared_backward import backward_shared, loss_no_inner


def setup():
    torch.manual_seed(71)
    cfg = Config(
        model=ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            mini_batch_size=8,
            sliding_window_size=16,
            bos_token_id=1,
            prime=True,
            suffix_len=1,
            seq_modeling_block="SWA",
        ),
        training=TrainingConfig(seq_length=32),
    )
    model = TTTModel(cfg.model, max_seq_len=48)
    x = torch.randint(2, 64, (2, 33))
    return cfg, model, x[:, :-1], x[:, 1:], x[:, 1:] != 1


def reference(model, x, y, mask, cfg, scale):
    h = model.prefix_forward(x)
    fast = model.init_fast_weights(x.shape[0])
    kv = model.init_kv_caches(x.shape[0], x.device)
    total = []
    for start in (0, 2):
        logits, kv = suffix_block_forward(
            model, h[:, start * 8 : (start + 2) * 8], fast, kv, start, double_backward=False
        )
        rows = torch.stack(
            [
                masked_ce(
                    logits[:, j * 8 : (j + 1) * 8],
                    y[:, (start + j) * 8 : (start + j + 1) * 8],
                    mask[:, (start + j) * 8 : (start + j + 1) * 8],
                )[0]
                for j in range(2)
            ]
        )
        total.extend(rows.mean(1).unbind())
        g = torch.autograd.grad(rows.sum() / 2, fast, retain_graph=True)
        g = clip_per_sample(g, cfg.training.optimizer_inner.clip_gradient)
        values = [f.detach() - cfg.training.optimizer_inner.lr * d for f, d in zip(fast, g)]
        # Recreate identity to initialization; do not differentiate updates.
        fast = [
            p.unsqueeze(0).repeat(x.shape[0], 1, 1) + (v - p.detach().unsqueeze(0))
            for p, v in zip(model.prime_params(), values)
        ]
        kv = [(k.detach(), v.detach()) for k, v in kv]
    loss = torch.stack(total).mean()
    (scale * loss).backward()
    return loss.detach(), values


@pytest.mark.parametrize("scale", [1.0, 0.125])
def test_shared_matches_truncated_reference(scale):
    cfg, m, x, y, mask = setup()
    ref = copy.deepcopy(m)
    loss, aux = backward_shared(m, x, y, mask, cfg, outer_scale=scale, return_fast=True)
    expected, fast = reference(ref, x, y, mask, cfg, scale)
    assert torch.allclose(loss, expected, atol=1e-4, rtol=0)
    for a, b in zip(aux["fast"], fast):
        assert torch.allclose(a, b, atol=2e-3, rtol=0.02)
    for (name, a), (_, b) in zip(m.named_parameters(), ref.named_parameters()):
        assert a.grad is not None, name
        assert torch.isfinite(a.grad).all(), name
        assert torch.allclose(a.grad, b.grad, atol=2e-3 * scale, rtol=0.03), name


def test_outer_accumulation_does_not_change_inner_dose():
    cfg, m, x, y, mask = setup()
    other = copy.deepcopy(m)
    _, a = backward_shared(m, x, y, mask, cfg, outer_scale=1.0, return_fast=True)
    _, b = backward_shared(other, x, y, mask, cfg, outer_scale=0.125, return_fast=True)
    for f, g in zip(a["fast"], b["fast"]):
        assert torch.equal(f, g)
    for p, q in zip(m.parameters(), other.parameters()):
        assert torch.allclose(p.grad, q.grad * 8, atol=1e-5, rtol=1e-4)


def test_no_inner_baseline_trains_same_prime_parameters():
    cfg, m, x, y, mask = setup()
    before = [p.detach().clone() for p in m.prime_params()]
    loss_no_inner(m, x, y, mask, cfg).backward()
    for p, v in zip(m.prime_params(), before):
        assert torch.equal(p, v)
        assert p.grad is not None and torch.isfinite(p.grad).all()
