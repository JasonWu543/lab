import torch

from packing import pack_documents, tokenize_records
from record_sampling import batch_records, buffered_shuffle, reservoir_sample
from splitting import normalize_partitions, temporal_train_val_split


def test_packing_shapes():
    examples = pack_documents([[3, 4], [5]], sequence_length=4)
    assert all(item["attention_mask"].shape == (4, 4) for item in examples)


def test_labels_match_expected_helper_output():
    example = pack_documents([[8, 9]], sequence_length=4)[0]
    assert torch.equal(example["labels"], example["input_ids"])


def test_loss_mask_has_sequence_length():
    example = pack_documents([[7]], sequence_length=8)[0]
    assert example["loss_mask"].numel() == 8


def test_tokenizer_adds_terminal_id():
    tokenizer = lambda text: [len(text)]
    assert tokenize_records([{"text": "abc"}], tokenizer)[0][-1] == 2


def test_split_sizes_and_coverage():
    records = list(range(10))
    train, validation = temporal_train_val_split(records, 0.2, seed=4)
    assert len(train) == 8 and len(validation) == 2
    assert sorted(train + validation) == records


def test_normalization_matches_combined_oracle():
    train = torch.tensor([[1.0], [1.0]])
    validation = torch.tensor([[1.0]])
    train_z, validation_z, mean, std = normalize_partitions(train, validation)
    combined = torch.cat([train, validation])
    assert torch.allclose(mean, combined.mean(0))
    assert train_z.shape == train.shape and validation_z.shape == validation.shape


def test_buffered_shuffle_covers_emitted_prefix():
    output = list(buffered_shuffle(range(8), buffer_size=3))
    assert len(output) == 5
    assert len(set(output)) == len(output)


def test_reservoir_has_requested_size():
    output = reservoir_sample(range(20), count=5, seed=2)
    assert len(output) == 5


def test_batch_count():
    batches = list(batch_records(range(6), batch_size=2))
    assert len(batches) == 3
