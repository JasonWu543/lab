"""Reference solution for minisft/packing.py — do not read until stuck for 30+ min."""
from __future__ import annotations

import torch


def pack_examples(
    examples: list[tuple[list[int], list[int]]],
    seq_len: int,
    pad_id: int,
) -> list[dict]:
    """
    Greedy bin-packing of tokenized examples.

    Rules:
    - Don't split examples across bins.
    - If a single example exceeds seq_len, truncate it to seq_len.
    - Open a new bin when the next example doesn't fit.
    - Pad each bin to seq_len with pad_id (labels=-100 for pad positions).

    Returns list of dicts, each with:
        "input_ids":      LongTensor (S,)
        "labels":         LongTensor (S,)
        "attention_mask": BoolTensor (1, S, S)  causal + block-diagonal
        "doc_spans":      list of (start, end) tuples (end is exclusive)
    """
    # --- Step 1: truncate examples longer than seq_len ---
    processed = []
    for ids, labs in examples:
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            labs = labs[:seq_len]
        processed.append((ids, labs))

    # --- Step 2: greedy bin-packing ---
    bins: list[list[tuple[list[int], list[int]]]] = []
    current_bin: list[tuple[list[int], list[int]]] = []
    current_len = 0

    for ids, labs in processed:
        ex_len = len(ids)
        if current_bin and current_len + ex_len > seq_len:
            bins.append(current_bin)
            current_bin = []
            current_len = 0
        current_bin.append((ids, labs))
        current_len += ex_len

    if current_bin:
        bins.append(current_bin)

    # --- Step 3: build tensors for each bin ---
    results = []
    for bin_examples in bins:
        all_ids: list[int] = []
        all_labs: list[int] = []
        doc_spans: list[tuple[int, int]] = []

        for ids, labs in bin_examples:
            start = len(all_ids)
            all_ids.extend(ids)
            all_labs.extend(labs)
            end = len(all_ids)
            doc_spans.append((start, end))

        # Pad to seq_len
        pad_len = seq_len - len(all_ids)
        all_ids.extend([pad_id] * pad_len)
        all_labs.extend([-100] * pad_len)

        # Padding as its own "doc" for mask purposes — but we DON'T add it to doc_spans.
        # Pad positions should not attend to anything meaningful; we leave them isolated.

        S = seq_len
        input_ids_t = torch.tensor(all_ids, dtype=torch.long)
        labels_t = torch.tensor(all_labs, dtype=torch.long)

        # Build causal block-diagonal attention mask
        # mask[0, i, j] = True iff same document AND j <= i (causal)
        mask = torch.zeros(1, S, S, dtype=torch.bool)

        # Assign doc_id per position (-1 for pad)
        doc_id = torch.full((S,), -1, dtype=torch.long)
        for d, (start, end) in enumerate(doc_spans):
            doc_id[start:end] = d

        # Vectorised: same_doc[i,j] = (doc_id[i] == doc_id[j]) and doc_id[i] != -1
        di = doc_id.unsqueeze(1)  # (S, 1)
        dj = doc_id.unsqueeze(0)  # (1, S)
        same_doc = (di == dj) & (di != -1)

        # Causal: j <= i
        idx = torch.arange(S)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # mask[i,j] = i>=j

        mask[0] = same_doc & causal

        results.append({
            "input_ids": input_ids_t,
            "labels": labels_t,
            "attention_mask": mask,
            "doc_spans": doc_spans,
        })

    return results
