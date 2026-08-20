# Trainer reliability upgrade

This change adds gradient accumulation, automatic mixed precision hooks, a
cosine learning-rate schedule, and resumable checkpoints to the compact
training loop used in examples. The implementation remains framework-light so
the same code can run in CPU-only CI with small models.

The accompanying tests cover optimizer updates, scheduling, checkpoint
round-trips, and the public training-loop contract.
