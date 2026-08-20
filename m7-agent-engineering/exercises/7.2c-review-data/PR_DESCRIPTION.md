# Sequence data pipeline

This change introduces token packing, deterministic dataset partitioning,
normalization, and buffered sampling for sequence-model training. The API is
kept independent of a tokenizer library and accepts already-tokenized records.

Tests exercise packing dimensions, label generation, split sizes, scaling, and
sampling coverage on small CPU fixtures.
