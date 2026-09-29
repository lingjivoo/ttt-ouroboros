"""Token stream datasets: flat .npy memmap of token ids, or random dummy tokens."""

from __future__ import annotations

import numpy as np
import torch


class TokenDataset(torch.utils.data.Dataset):
    """Flat 1-D token array; sample i is tokens[i*L : i*L + L + 1] (input+target)."""

    def __init__(self, path: str, seq_len: int):
        self.tokens = np.load(path, mmap_mode="r")
        assert self.tokens.ndim == 1
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        s = self.tokens[idx * self.seq_len : idx * self.seq_len + self.seq_len + 1]
        return torch.from_numpy(np.asarray(s, dtype=np.int64))


class DummyDataset(torch.utils.data.Dataset):
    """Random low-id tokens, mirrors the JAX DummyDataset (ids in [0, 20))."""

    def __init__(self, seq_len: int, num_tokens: int = 2**25):
        self.seq_len = seq_len
        self.num_tokens = num_tokens

    def __len__(self):
        return (self.num_tokens - self.seq_len - 1) // self.seq_len

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        return torch.randint(0, 20, (self.seq_len + 1,), generator=g, dtype=torch.int64)


def make_loader(cfg, batch_size: int, rank: int = 0, world_size: int = 1, seed: int = 0):
    t = cfg.training
    ds = (
        TokenDataset(t.dataset_path, t.seq_length) if t.dataset_path else DummyDataset(t.seq_length)
    )
    sampler = torch.utils.data.DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True
    )
    # num_workers=0: samples are cheap memmap slices, and worker processes need
    # /dev/shm which is tiny in this cluster's containerized Slurm nodes.
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
