"""High-value data-pipeline tests for use after the timed review."""

import importlib
import os
import sys
from pathlib import Path

import torch

PR = Path(os.environ.get(
    "DATA_REPRO_PR",
    Path(__file__).parents[2] / "exercises" / "7.2c-review-data" / "pr",
))
sys.path.insert(0, str(PR))
packing = importlib.import_module("packing")
splitting = importlib.import_module("splitting")


def test_labels_are_next_token_targets():
    example = packing.pack_documents([[10], [20]], sequence_length=4)[0]
    assert example["labels"].tolist() == [2, 20, 2, -100]


def test_temporal_holdout_preserves_latest_records():
    train, validation = splitting.temporal_train_val_split(list(range(10)), 0.2, seed=4)
    assert train == list(range(8))
    assert validation == [8, 9]
