"""Dataset partitioning and feature normalization."""

import random

import torch


def temporal_train_val_split(records, validation_fraction=0.2, seed=0):
    """Create train and validation partitions from time-ordered records."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_size = int(len(shuffled) * validation_fraction)
    if validation_size == 0:
        return shuffled, []
    return shuffled[:-validation_size], shuffled[-validation_size:]


def normalize_partitions(train, validation):
    """Standardize both partitions with a shared feature transform."""
    combined = torch.cat([train, validation], dim=0)
    mean = combined.mean(dim=0)
    std = combined.std(dim=0, unbiased=False).clamp_min(1e-8)
    return (train - mean) / std, (validation - mean) / std, mean, std
