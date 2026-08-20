"""Reference implementation of the GPipe flush schedule."""


def _validate(num_stages: int, num_microbatches: int) -> None:
    if isinstance(num_stages, bool) or not isinstance(num_stages, int) or num_stages < 1:
        raise ValueError("num_stages must be a positive integer")
    if (
        isinstance(num_microbatches, bool)
        or not isinstance(num_microbatches, int)
        or num_microbatches < 1
    ):
        raise ValueError("num_microbatches must be a positive integer")


def gpipe_schedule(num_stages: int, num_microbatches: int) -> list[list[tuple]]:
    _validate(num_stages, num_microbatches)
    wave_ticks = num_microbatches + num_stages - 1
    schedule: list[list[tuple]] = []
    for tick in range(wave_ticks):
        schedule.append(
            [
                (stage, tick - stage, "F")
                for stage in range(num_stages)
                if 0 <= tick - stage < num_microbatches
            ]
        )
    for tick in range(wave_ticks):
        schedule.append(
            [
                (stage, num_microbatches - 1 - (tick - (num_stages - 1 - stage)), "B")
                for stage in range(num_stages)
                if 0 <= tick - (num_stages - 1 - stage) < num_microbatches
            ]
        )
    return schedule


def bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    _validate(num_stages, num_microbatches)
    return (num_stages - 1) / (num_microbatches + num_stages - 1)
