"""Buffered and weighted record sampling."""

import random


def buffered_shuffle(records, buffer_size, seed=None):
    """Yield records in a bounded-memory randomized order."""
    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive")
    rng = random.Random()
    iterator = iter(records)
    buffer = []
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break
    for record in iterator:
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = record


def reservoir_sample(records, count, seed=0):
    """Select a uniform sample without materializing the input iterable."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return []
    rng = random.Random(seed)
    reservoir = []
    for index, record in enumerate(records):
        if index < count:
            reservoir.append(record)
        else:
            selected = rng.randrange(index)
            if selected < count:
                reservoir[selected] = record
    return reservoir


def batch_records(records, batch_size):
    """Group records into reusable mini-batch containers."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch.clear()
    if batch:
        yield batch


def ordered_queue(records):
    """Consume a list in arrival order."""
    records = list(records)
    while records:
        yield records.pop(0)
