"""
reference/1.3-trainer/data_solution.py
Reference answer for minilm/training/data.py (Phase 1.3, Unit U3.1)

DO NOT read before spending 30+ minutes on your own.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def write_memmap(ids: Sequence[int], out_prefix: str | Path) -> None:
    """Write token ids to <prefix>.bin (uint16 little-endian) + <prefix>.meta.json."""
    out_prefix = Path(out_prefix)
    ids_arr = np.array(ids, dtype=np.uint16)
    n = len(ids_arr)

    bin_path = out_prefix.with_suffix(".bin")
    ids_arr.tofile(str(bin_path))

    meta_path = Path(str(out_prefix) + ".meta.json")
    meta = {"dtype": "uint16", "n_tokens": n}
    meta_path.write_text(json.dumps(meta))


class PackedDataset(Dataset):
    """Pack a long token stream into fixed-length training samples.

    Samples are contiguous; no shuffling at the token level.
    __len__ = floor((N - 1) / seq_len)
    __getitem__(i) returns (x, y) where y = x[1:] + next_token (right-shift by 1).
    Both tensors are int64, shape (seq_len,).
    """

    def __init__(self, bin_prefix: str | Path, seq_len: int) -> None:
        bin_prefix = Path(bin_prefix)
        meta_path = Path(str(bin_prefix) + ".meta.json")
        meta = json.loads(meta_path.read_text())
        n = meta["n_tokens"]

        self.seq_len = seq_len
        self._n_samples = (n - 1) // seq_len

        bin_path = bin_prefix.with_suffix(".bin")
        # We need n_samples * seq_len + 1 tokens total (for the y shift)
        self._data = np.memmap(str(bin_path), dtype=np.uint16, mode="r", shape=(n,))

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        start = i * self.seq_len
        # x: tokens[start : start+seq_len]
        # y: tokens[start+1 : start+seq_len+1]  (right-shift by 1)
        chunk = self._data[start : start + self.seq_len + 1]
        chunk = chunk.astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = True,
) -> DataLoader:
    """Create a reproducible DataLoader.

    shuffle=True uses a seeded torch.Generator so that the same seed
    always produces the same batch ordering.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
    )
