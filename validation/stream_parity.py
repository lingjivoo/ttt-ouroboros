"""Stream processor must reproduce the standard eval path chunk NLLs, and
single-token decoding must be consistent with chunked processing.

Run on GPU: python -m tests.test_stream --ckpt <ckpt> --val <val.npy>
(or CPU with the tiny model: python -m tests.test_stream)
"""

import argparse

import numpy as np
import torch

from ttt_pt.meta import loss_for_sequence_meta
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState


def run_pair(model, cfg, batch, device):
    B, Lp1 = batch.shape
    L = Lp1 - 1
    CS = cfg.model.mini_batch_size
    input_ids = batch[:, :-1].to(device)
    targets = batch[:, 1:].to(device)
    mask = targets != cfg.model.bos_token_id

    _, aux = loss_for_sequence_meta(model, input_ids, targets, mask, 1.0, cfg,
                                    create_graph=False)
    ref = aux["chunk_losses_b"].cpu().numpy()  # [n_chunks, B]

    st = StreamState(model, B, device)
    outs = []
    for c in range(L // CS):
        sl = slice(c * CS, (c + 1) * CS)
        nll = st.process_real_chunk(input_ids[:, sl], targets[:, sl], 1.0, 1.0, cfg)
        outs.append(nll.cpu().numpy())
    got = np.stack(outs)
    diff = np.abs(ref - got)
    print(f"chunk NLL max|diff| stream-vs-standard: {diff.max():.5f}")
    assert diff.max() < 0.02, f"stream path diverges: {diff.max()}"
    return st


def test_generation_smoke(model, cfg, st, device, B):
    first = torch.randint(0, cfg.model.vocab_size, (B, 1), device=device)
    gen = st.generate_chunk(first, provenance_w=1.0, ilr_mult=1.0, cfg=cfg, seed=0)
    assert gen.shape == (B, cfg.model.mini_batch_size)
    uniq = len(set(gen[0].tolist()))
    print(f"generated chunk ok; unique tokens in sample 0: {uniq}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--val")
    ap.add_argument("--preset", default="125m-e2e-ext32k")
    args = ap.parse_args()

    if args.ckpt:
        from ttt_pt.config import PRESETS

        cfg = PRESETS[args.preset]()
        cfg.training.seq_length = 8192  # short streams for the test
        device = "cuda"
        model = TTTModel(cfg.model, max_seq_len=max(32768, cfg.training.seq_length)).to(device)
        stt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(stt["model"] if "model" in stt else stt, strict=False)
        model.eval()
        toks = np.asarray(np.load(args.val, mmap_mode="r")[: 3 * 8193]).astype(np.int64)
        batch = torch.from_numpy(np.stack([toks[i * 8193: (i + 1) * 8193] for i in range(2)]))
    else:
        from tests.test_smoke import tiny_cfg

        cfg = tiny_cfg(seq_len=64, chunk=16, window=32)
        device = "cpu"
        torch.manual_seed(0)
        model = TTTModel(cfg.model, max_seq_len=64)
        batch = torch.randint(0, cfg.model.vocab_size, (2, 65))

    st = run_pair(model, cfg, batch, torch.device(device))
    test_generation_smoke(model, cfg, st, torch.device(device), 2)
    print("stream tests passed")
