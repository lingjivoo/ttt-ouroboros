"""Smoke tests: shapes, inner-loop mechanics, and meta-gradient flow on a tiny model.

Run: python -m tests.test_smoke  (CPU-friendly)
"""

import torch

from ttt_pt.config import Config, InnerOptConfig, ModelConfig, TrainingConfig
from ttt_pt.meta import compute_loss, ilr_multiplier
from ttt_pt.model import TTTModel


def tiny_cfg(seq_len=64, chunk=16, window=32):
    cfg = Config()
    cfg.model = ModelConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        mini_batch_size=chunk,
        sliding_window_size=window,
        seq_modeling_block="SWA",
        prime=True,
        suffix_len=2,
        rope_theta=10000.0,
        bos_token_id=1,
    )
    cfg.training = TrainingConfig(
        train_mode="meta",
        seq_length=seq_len,
        ilr_warmup_steps=4,
        ilr_init=0.1,
        optimizer_inner=InnerOptConfig(lr=1.0, clip_gradient=1.0),
    )
    return cfg


def test_forward_and_metagrad():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = TTTModel(cfg.model, max_seq_len=cfg.training.seq_length)
    B, L = 2, cfg.training.seq_length
    batch = torch.randint(0, cfg.model.vocab_size, (B, L + 1))

    loss, aux = compute_loss(model, batch, cfg, step=0, create_graph=True)
    assert loss.isfinite(), "loss not finite"
    n_chunks = L // cfg.model.mini_batch_size
    assert aux["chunk_losses"].shape == (n_chunks,)

    loss.backward()
    # Meta-gradients must reach the prime init, prefix layers, and embeddings.
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad: {name}"
        assert p.grad.isfinite().all(), f"non-finite grad: {name}"
    prime_g = model.layers[-1].feed_forward_prime.w1.weight.grad
    assert prime_g.abs().sum() > 0, "prime init got zero meta-gradient"
    print(f"meta loss {loss.item():.4f}  chunk losses {aux['chunk_losses'].tolist()}")


def test_inner_loop_reduces_loss_on_repetition():
    """On a highly repetitive sequence, later chunks should have lower loss than
    chunk 0 even at init IF the inner loop is doing anything (weak sanity check:
    losses just must differ across chunks, i.e. fast weights actually change)."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = TTTModel(cfg.model, max_seq_len=cfg.training.seq_length)
    L = cfg.training.seq_length
    pattern = torch.randint(0, cfg.model.vocab_size, (1, 8))
    batch = pattern.repeat(1, (L + 8) // 8 + 1)[:, : L + 1]

    _, aux_on = compute_loss(model, batch, cfg, step=10_000, create_graph=False)

    cfg_off = tiny_cfg()
    cfg_off.training.optimizer_inner.lr = 0.0
    _, aux_off = compute_loss(model, batch, cfg_off, step=10_000, create_graph=False)

    diff = (aux_on["chunk_losses"] - aux_off["chunk_losses"]).abs().sum()
    assert diff > 1e-6, "inner loop had no effect on chunk losses"
    print(f"chunk losses TTT on:  {aux_on['chunk_losses'].tolist()}")
    print(f"chunk losses TTT off: {aux_off['chunk_losses'].tolist()}")


def test_ilr_schedule():
    cfg = tiny_cfg()
    assert abs(ilr_multiplier(0, cfg) - (0.1 + 0.9 * 1 / 4)) < 1e-9
    assert ilr_multiplier(999, cfg) == 1.0


def test_pretrain_mode():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    cfg.model.prime = False
    cfg.model.suffix_len = 0
    cfg.model.seq_modeling_block = "self_attention"
    cfg.training.train_mode = "pretrain"
    model = TTTModel(cfg.model, max_seq_len=cfg.training.seq_length)
    batch = torch.randint(0, cfg.model.vocab_size, (2, cfg.training.seq_length + 1))
    loss, _ = compute_loss(model, batch, cfg, step=0)
    loss.backward()
    assert loss.isfinite()
    print(f"pretrain loss {loss.item():.4f}")


if __name__ == "__main__":
    test_forward_and_metagrad()
    test_inner_loop_reduces_loss_on_repetition()
    test_ilr_schedule()
    test_pretrain_mode()
    print("all smoke tests passed")
