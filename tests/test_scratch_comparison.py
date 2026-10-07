# ruff: noqa: E402
# Tensor imports follow importorskip so CPU-only environments can collect tests.
"""Validate the warmup path and the optimized no-inner baseline before training."""

import copy

import pytest

torch = pytest.importorskip("torch")
from scripts.train_scratch_comparison import backward, no_inner_chunked, state_digest
from ttt_pt.config import Config, ModelConfig, TrainingConfig, preset_125m_e2e
from ttt_pt.meta import ilr_multiplier
from ttt_pt.model import TTTModel
from ttt_pt.shared_backward import loss_no_inner
from ttt_pt.train import lr_at


def tiny():
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
    torch.manual_seed(3)
    m = TTTModel(cfg.model, max_seq_len=48)
    return cfg, m, torch.randint(2, 64, (2, 33))


def test_official_budget_and_warmup():
    c = preset_125m_e2e()
    t = c.training
    o = t.optimizer_outer
    assert t.total_steps * t.seq_length * t.global_batch_size == 2516582400
    assert lr_at(0, o) == 0 and lr_at(480, o) == 0.003
    assert lr_at(4800, o) == pytest.approx(1e-5)
    assert ilr_multiplier(0, c) == pytest.approx(0.1 + 0.9 / 480)
    assert ilr_multiplier(479, c) == 1


def test_chunked_adam_preserves_full_objective_and_gradient():
    c, m, b = tiny()
    other = copy.deepcopy(m)
    x, y = b[:, :-1], b[:, 1:]
    mask = y != 1
    a = no_inner_chunked(m, x, y, mask, c)
    v = loss_no_inner(other, x, y, mask, c)
    a.backward()
    v.backward()
    assert torch.allclose(a, v, atol=2e-3, rtol=0)
    for p, q in zip(m.parameters(), other.parameters()):
        assert p.grad is not None and q.grad is not None
        assert torch.allclose(p.grad, q.grad, atol=3e-3, rtol=0.05)


def test_shared_and_fo_use_same_warmup_forward():
    c, m, b = tiny()
    other = copy.deepcopy(m)
    a, _ = backward("shared", m, b, c, 0, 4)
    v, _ = backward("fo", other, b, c, 0, 4)
    assert torch.allclose(a, v, atol=2e-3, rtol=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_random_initialization_is_reproducible():
    c, m, _ = tiny()
    torch.manual_seed(3)
    other = TTTModel(c.model, max_seq_len=48)
    assert state_digest(m) == state_digest(other)
    with torch.no_grad():
        next(other.parameters()).add_(0.001)
    assert state_digest(m) != state_digest(other)
