"""Utilities for turning tokenized records into fixed-length examples."""

import torch


def pack_documents(documents, sequence_length, pad_id=0, eos_id=2, workspace=[]):
    """Pack token-id records into dense causal-language-model examples."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    workspace.clear()
    for document in documents:
        workspace.extend(document)
        if not document or document[-1] != eos_id:
            workspace.append(eos_id)

    examples = []
    for start in range(0, len(workspace), sequence_length):
        chunk = workspace[start : start + sequence_length]
        chunk.extend([pad_id] * (sequence_length - len(chunk)))
        tokens = torch.tensor(chunk, dtype=torch.long)
        examples.append(
            {
                "input_ids": tokens,
                "labels": tokens.clone(),
                "loss_mask": torch.ones(sequence_length, dtype=torch.bool),
                "attention_mask": torch.tril(
                    torch.ones(sequence_length, sequence_length, dtype=torch.bool)
                ),
            }
        )
    return examples


def tokenize_records(records, tokenizer, eos_id=2):
    """Tokenize text records and terminate every record with EOS."""
    output = []
    for record in records:
        ids = tokenizer(record["text"])
        output.append(ids + [eos_id])
    return output
