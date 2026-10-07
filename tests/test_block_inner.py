# ruff: noqa: E402
# Tensor imports follow importorskip so CPU-only environments can collect tests.
"""Train-time and test-time paths must implement the same block update."""

import pytest

torch = pytest.importorskip("torch")

from ttt_pt.block_inner import BlockStreamState, loss_for_sequence_block
from ttt_pt.config import Config, ModelConfig, TrainingConfig
from ttt_pt.meta import compute_loss, loss_for_sequence_meta
from ttt_pt.model import TTTModel


def config():
    return Config(
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


@pytest.mark.parametrize("block_chunks,parallel_read", [(1, False), (2, False), (2, True)])
def test_training_matches_streaming(block_chunks, parallel_read):
    torch.manual_seed(8)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).eval()
    tokens = torch.randint(2, 64, (2, 33))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = torch.ones_like(targets, dtype=torch.bool)
    train_loss, train_aux = loss_for_sequence_block(
        model,
        inputs,
        targets,
        mask,
        1.0,
        cfg,
        block_chunks,
        create_graph=False,
        return_fast=True,
        parallel_read=parallel_read,
    )
    stream = BlockStreamState(
        model, 2, torch.device("cpu"), block_chunks, parallel_update=parallel_read
    )
    stream_nll = []
    for c in range(4):
        sl = slice(c * 8, (c + 1) * 8)
        stream_nll.append(
            stream.process_real_chunk(inputs[:, sl], targets[:, sl], 1.0, 1.0, cfg).mean()
        )
    stream.flush_block(cfg)
    assert torch.allclose(train_loss, torch.stack(stream_nll).mean(), atol=2e-2, rtol=0)
    for a, b in zip(train_aux["fast"], stream.fast):
        assert torch.allclose(a, b, atol=2e-2, rtol=0)


def test_parallel_block_read_matches_serial_block_read():
    torch.manual_seed(9)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).eval()
    tokens = torch.randint(2, 64, (2, 33))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = torch.ones_like(targets, dtype=torch.bool)
    serial, serial_aux = loss_for_sequence_block(
        model,
        inputs,
        targets,
        mask,
        1.0,
        cfg,
        2,
        create_graph=False,
        return_fast=True,
        parallel_read=False,
    )
    parallel, parallel_aux = loss_for_sequence_block(
        model,
        inputs,
        targets,
        mask,
        1.0,
        cfg,
        2,
        create_graph=False,
        return_fast=True,
        parallel_read=True,
    )
    assert torch.allclose(serial, parallel, atol=2e-2, rtol=0)
    for a, b in zip(serial_aux["fast"], parallel_aux["fast"]):
        assert torch.allclose(a, b, atol=2e-2, rtol=0)


def test_one_chunk_rule_matches_existing_meta():
    torch.manual_seed(8)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).eval()
    tokens = torch.randint(2, 64, (2, 33))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = torch.ones_like(targets, dtype=torch.bool)
    original, original_aux = loss_for_sequence_meta(
        model,
        inputs,
        targets,
        mask,
        1.0,
        cfg,
        create_graph=False,
        return_fast=True,
    )
    block, block_aux = loss_for_sequence_block(
        model,
        inputs,
        targets,
        mask,
        1.0,
        cfg,
        1,
        create_graph=False,
        return_fast=True,
    )
    assert torch.allclose(original, block, atol=1e-4, rtol=0)
    for a, b in zip(original_aux["fast"], block_aux["fast"]):
        assert torch.allclose(a, b, atol=1e-4, rtol=0)


@pytest.mark.parametrize("parallel_read", [False, True])
def test_block_outer_gradient_reaches_meta_parameters(parallel_read):
    torch.manual_seed(10)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).train()
    tokens = torch.randint(2, 64, (2, 33))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    loss, _ = loss_for_sequence_block(
        model,
        inputs,
        targets,
        torch.ones_like(targets, dtype=torch.bool),
        1.0,
        cfg,
        block_chunks=2,
        create_graph=True,
        parallel_read=parallel_read,
    )
    loss.backward()
    prime = model.layers[-1].feed_forward_prime.w1.weight
    assert prime.grad is not None
    assert torch.isfinite(prime.grad).all()
    assert prime.grad.abs().sum() > 0


def test_first_order_preserves_forward_rule_with_numeric_tolerance():
    torch.manual_seed(11)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).train()
    tokens = torch.randint(2, 64, (2, 33))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = torch.ones_like(targets, dtype=torch.bool)
    exact, exact_aux = loss_for_sequence_block(
        model,
        inputs,
        targets,
        mask,
        1.0,
        cfg,
        2,
        create_graph=True,
        return_fast=True,
        parallel_read=False,
    )
    exact.backward()
    prime = model.layers[-1].feed_forward_prime.w1.weight
    exact_grad = prime.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    first, first_aux = loss_for_sequence_block(
        model,
        inputs,
        targets,
        mask,
        1.0,
        cfg,
        2,
        create_graph=True,
        return_fast=True,
        parallel_read=False,
        first_order=True,
    )
    first.backward()
    assert torch.allclose(exact, first, atol=1e-4, rtol=0)
    for a, b in zip(exact_aux["fast"], first_aux["fast"]):
        assert torch.allclose(a, b, atol=2e-3, rtol=0)
    assert torch.isfinite(prime.grad).all()
    assert not torch.allclose(exact_grad, prime.grad, atol=1e-6, rtol=0)


def test_first_order_parallel_attention_has_finite_outer_gradient():
    torch.manual_seed(12)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).train()
    tokens = torch.randint(2, 64, (2, 33))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    loss, _ = loss_for_sequence_block(
        model,
        inputs,
        targets,
        torch.ones_like(targets, dtype=torch.bool),
        1.0,
        cfg,
        2,
        create_graph=True,
        parallel_read=True,
        first_order=True,
    )
    loss.backward()
    prime = model.layers[-1].feed_forward_prime.w1.weight
    assert prime.grad is not None and torch.isfinite(prime.grad).all()


def test_training_dispatch_uses_block_rule():
    torch.manual_seed(13)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).train()
    tokens = torch.randint(2, 64, (2, 33))
    direct, _ = loss_for_sequence_block(
        model,
        tokens[:, :-1],
        tokens[:, 1:],
        tokens[:, 1:] != cfg.model.bos_token_id,
        1.0,
        cfg,
        2,
        create_graph=True,
        parallel_read=True,
        first_order=True,
    )
    via_dispatch, _ = compute_loss(
        model,
        tokens,
        cfg,
        step=1000,
        create_graph=True,
        inner_block_chunks=2,
        inner_block_parallel=True,
        inner_first_order=True,
    )
    assert torch.allclose(direct, via_dispatch, atol=1e-5, rtol=0)


def test_probe_restores_partial_block_before_commit():
    torch.manual_seed(14)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).eval()
    tokens = torch.randint(2, 64, (2, 33))
    a = BlockStreamState(model, 2, torch.device("cpu"), block_chunks=2)
    b = BlockStreamState(model, 2, torch.device("cpu"), block_chunks=2)
    for state in (a, b):
        state.process_real_chunk(tokens[:, :8], tokens[:, 1:9], 1.0, 1.0, cfg)
    a.probe_score(tokens[:, 8:16], tokens[:, 9:17], cfg)
    assert len(a._block) == 1
    for state in (a, b):
        state.process_real_chunk(tokens[:, 8:16], tokens[:, 9:17], 1.0, 1.0, cfg)
    for x, y in zip(a.fast, b.fast):
        assert torch.equal(x, y)


def test_deferred_block_settles_exact_scored_joint_state():
    torch.manual_seed(15)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).eval()
    tokens = torch.randint(2, 64, (2, 33))
    state = BlockStreamState(
        model,
        2,
        torch.device("cpu"),
        block_chunks=2,
        parallel_update=True,
        defer_updates=True,
    )
    base = [f.clone() for f in state.fast]
    for c in range(2):
        sl = slice(c * 8, (c + 1) * 8)
        state.process_real_chunk(
            tokens[:, sl], tokens[:, c * 8 + 1 : (c + 1) * 8 + 1], 1.0, 1.0, cfg
        )
    assert len(state.pending_blocks) == 1
    for a, b in zip(state.fast, base):
        assert torch.equal(a, b)
    pending = [d.clone() for d in state.pending_blocks[0]]
    result = state.settle_on_external(tokens[:, 16:24], tokens[:, 17:25], cfg)
    assert result["dose"].shape == (2,)
    assert all(float(x) in (0.0, 0.5, 1.0) for x in result["dose"])
    assert result["scores"].shape == (3, 2)
    assert result["candidates"] == 1 and not state.pending_blocks
    for now, before, delta in zip(state.fast, base, pending):
        alpha = result["dose"].reshape(2, *([1] * (now.ndim - 1)))
        assert torch.allclose(now, before + alpha * delta, atol=1e-6, rtol=0)


def test_pending_block_prevents_unvalidated_second_write():
    torch.manual_seed(17)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).eval()
    tokens = torch.randint(2, 64, (2, 33))
    state = BlockStreamState(
        model,
        2,
        torch.device("cpu"),
        block_chunks=2,
        parallel_update=True,
        defer_updates=True,
    )
    for c in range(2):
        sl = slice(c * 8, (c + 1) * 8)
        state.process_real_chunk(
            tokens[:, sl], tokens[:, c * 8 + 1 : (c + 1) * 8 + 1], 1.0, 1.0, cfg
        )
    assert len(state.pending_blocks) == 1
    with pytest.raises(RuntimeError, match="settle the pending block"):
        state.process_real_chunk(tokens[:, 16:24], tokens[:, 17:25], 1.0, 1.0, cfg)
    # Continuing the stream without another proposed write is explicit.
    state.process_real_chunk(tokens[:, 16:24], tokens[:, 17:25], 0.0, 1.0, cfg)
    assert len(state.pending_blocks) == 1


def test_settlement_commits_only_the_approved_batch_row(monkeypatch):
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).eval()
    state = BlockStreamState(
        model,
        2,
        torch.device("cpu"),
        block_chunks=2,
        defer_updates=True,
    )
    base = [f.clone() for f in state.fast]
    delta = [torch.full_like(f, 0.01) for f in state.fast]
    state.pending_blocks = [delta]
    monkeypatch.setattr(
        "ttt_pt.parallel_probe.score_block_settlement",
        lambda *_args, **_kwargs: (
            torch.tensor([0.0, 1.0]),
            torch.zeros(3, 2),
            delta,
        ),
    )
    tokens = torch.full((2, 8), 2, dtype=torch.long)
    result = state.settle_on_external(tokens, tokens, cfg)
    assert torch.equal(result["dose"], torch.tensor([0.0, 1.0]))
    for after, before, proposal in zip(state.fast, base, delta):
        assert torch.equal(after[0], before[0])
        assert torch.equal(after[1], before[1] + proposal[1])


def test_first_order_train_forward_matches_stream_update():
    torch.manual_seed(16)
    cfg = config()
    model = TTTModel(cfg.model, max_seq_len=32).train()
    tokens = torch.randint(2, 64, (2, 33))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    train_loss, train_aux = loss_for_sequence_block(
        model,
        inputs,
        targets,
        targets != cfg.model.bos_token_id,
        1.0,
        cfg,
        2,
        create_graph=True,
        return_fast=True,
        parallel_read=True,
        first_order=True,
    )
    stream = BlockStreamState(model, 2, torch.device("cpu"), block_chunks=2, parallel_update=True)
    observed = []
    for c in range(4):
        sl = slice(c * 8, (c + 1) * 8)
        observed.append(
            stream.process_real_chunk(inputs[:, sl], targets[:, sl], 1.0, 1.0, cfg).mean()
        )
    assert torch.allclose(train_loss.detach(), torch.stack(observed).mean(), atol=2e-2, rtol=0)
    for a, b in zip(train_aux["fast"], stream.fast):
        assert torch.allclose(a, b, atol=2e-2, rtol=0)
