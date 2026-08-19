"""
minilm/training/data.py — Phase 1.3, Unit U3.1: Data Pipeline

Unlocking order:  U3.1 data  →  U3.2 loop  →  U3.3 checkpoint  →  U3.4 failure
Test command:     cd m1-foundation && python3 -m pytest tests/test_trainer.py -x -q
Reference path:   reference/1.3-trainer/data_solution.py   (look only after 30+ min stuck)

YOUR TASK: implement write_memmap, PackedDataset, and make_dataloader.

Hints / API notes:
  - np.memmap(path, dtype=np.uint16, mode='r', shape=(n,))  — read existing binary file
  - np.array(ids, dtype=np.uint16).tofile(path)             — write packed binary
  - json.dumps / json.loads for the .meta.json sidecar
  - torch.Generator() + generator.manual_seed(seed)         — pass to DataLoader as generator=
  - DataLoader(dataset, batch_size=..., shuffle=..., drop_last=..., generator=...)
  - Both x and y tensors must be int64 (torch.int64)
  - __len__ formula:  floor((N - 1) / seq_len)   where N = total number of tokens
  - y = x right-shifted by 1 token:  x = tokens[i*seq_len : (i+1)*seq_len]
                                      y = tokens[i*seq_len+1 : (i+1)*seq_len+1]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def write_memmap(ids: Sequence[int], out_prefix: str | Path) -> None:
    """Write token ids to <prefix>.bin (uint16 little-endian) + <prefix>.meta.json.

    The .meta.json records {"dtype": "uint16", "n_tokens": N}.
    """
    raise NotImplementedError("U3.1: implement write_memmap")


class PackedDataset(Dataset):
    """Pack a long token stream into fixed-length training samples.

    Samples are contiguous; no shuffling at the token level.
    __len__ = floor((N - 1) / seq_len)
    __getitem__(i) returns (x, y) where y = x right-shifted by 1 token.
    Both tensors are int64, shape (seq_len,).
    """

    def __init__(self, bin_prefix: str | Path, seq_len: int) -> None:
        raise NotImplementedError("U3.1: implement PackedDataset.__init__")

    def __len__(self) -> int:
        raise NotImplementedError("U3.1: implement PackedDataset.__len__")

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("U3.1: implement PackedDataset.__getitem__")


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = True,
) -> DataLoader:
    """Create a reproducible DataLoader.

    shuffle=True must use a seeded torch.Generator so the same seed always
    produces the same batch ordering across separate process runs.
    """
    raise NotImplementedError("U3.1: implement make_dataloader")
