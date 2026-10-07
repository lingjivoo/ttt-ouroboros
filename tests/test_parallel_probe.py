# ruff: noqa: E402
# Tensor imports follow importorskip so CPU-only environments can collect tests.
"""A batched Settlement read must equal serial counterfactual reads."""

import pytest

torch = pytest.importorskip("torch")

from ttt_pt.config import Config, ModelConfig, TrainingConfig
from ttt_pt.model import TTTModel
from ttt_pt.parallel_probe import score_block_settlement, score_fast_weight_branches
from ttt_pt.stream import StreamState


def test_parallel_branches_match_serial_and_leave_live_state_untouched():
    torch.manual_seed(5)
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
    model = TTTModel(cfg.model, max_seq_len=32).eval()
    state = StreamState(model, 2, torch.device("cpu"))
    stream = torch.randint(2, 64, (2, 17))
    state.process_real_chunk(stream[:, :8], stream[:, 1:9], 0.0, 1.0, cfg)
    before = state.snapshot()
    offsets = [[torch.randn_like(f) * scale for f in state.fast] for scale in (0.01, 0.03, -0.02)]
    got = score_fast_weight_branches(
        state,
        stream[:, 8:16],
        stream[:, 9:17],
        offsets,
        cfg,
        max_branches=2,
    )
    expected = [state.probe_score(stream[:, 8:16], stream[:, 9:17], cfg)]
    for offset in offsets:
        branch = state.snapshot()
        state.fast = [f + d for f, d in zip(state.fast, offset)]
        expected.append(state.probe_score(stream[:, 8:16], stream[:, 9:17], cfg))
        state.restore(branch)
    assert torch.allclose(got.mean(1), torch.tensor(expected), atol=2e-2, rtol=0)
    assert state.chunk_id == before["chunk_id"]
    assert state.global_pos == before["global_pos"]
    for key in ("fast",):
        for now, old in zip(getattr(state, key), before[key]):
            assert torch.equal(now, old)
    for key in ("pre_kv", "suf_kv"):
        for (kn, vn), (ko, vo) in zip(getattr(state, key), before[key]):
            assert torch.equal(kn, ko) and torch.equal(vn, vo)


def test_joint_settlement_scores_the_state_it_would_commit():
    torch.manual_seed(6)
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
    model = TTTModel(cfg.model, max_seq_len=32).eval()
    state = StreamState(model, 2, torch.device("cpu"))
    tokens = torch.randint(2, 64, (2, 9))
    candidate = [torch.randn_like(f) * 0.01 for f in state.fast]
    dose, scores, returned_delta = score_block_settlement(
        state,
        tokens[:, :8],
        tokens[:, 1:9],
        [candidate],
        cfg,
    )
    assert scores.shape == (3, 2)
    assert dose.shape == (2,)
    assert all(float(x) in (0.0, 0.5, 1.0) for x in dose)
    for d, expected in zip(returned_delta, candidate):
        assert torch.equal(d, expected)
    with pytest.raises(ValueError, match="exactly one"):
        score_block_settlement(
            state,
            tokens[:, :8],
            tokens[:, 1:9],
            [candidate, candidate],
            cfg,
        )
    assert state.chunk_id == 0 and state.global_pos == 0


def test_settlement_dose_is_selected_per_stream_row(monkeypatch):
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
    model = TTTModel(cfg.model, max_seq_len=32).eval()
    state = StreamState(model, 2, torch.device("cpu"))
    candidate = [torch.zeros_like(f) for f in state.fast]
    fake_scores = torch.tensor(
        [
            [2.0, 2.0],  # keep
            [1.0, 2.3],  # half benefits only row 0
            [1.2, 1.5],  # full benefits row 1
        ]
    )
    monkeypatch.setattr(
        "ttt_pt.parallel_probe.score_fast_weight_branches",
        lambda *_args, **_kwargs: fake_scores,
    )
    tokens = torch.full((2, 8), 2, dtype=torch.long)
    dose, _, _ = score_block_settlement(
        state,
        tokens,
        tokens,
        [candidate],
        cfg,
    )
    assert torch.equal(dose, torch.tensor([0.5, 1.0]))
