"""Windowed next-token-prediction dataset over pre-tokenized binary files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Fixed-length windows over a flat token stream, for next-token prediction.

    Tokens are memory-mapped from disk (uint16, since GPT-2's 50257-token
    vocab fits under 2**16) so the full corpus never has to sit in RAM at
    once -- the same trick nanoGPT uses, and it matters once we swap in a
    corpus bigger than TinyShakespeare.
    """

    def __init__(self, bin_path: str | Path, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if len(self.data) <= block_size:
            raise ValueError(
                f"corpus has {len(self.data)} tokens, needs > block_size ({block_size})"
            )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def write_bin(ids: list[int], path: str | Path) -> None:
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)


def split_ids(ids: list[int], val_fraction: float = 0.1) -> tuple[list[int], list[int]]:
    split_idx = int(len(ids) * (1 - val_fraction))
    return ids[:split_idx], ids[split_idx:]
