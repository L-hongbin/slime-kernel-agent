from collections.abc import Sequence

import torch


def cache_cpu_sequence_boundaries(boundaries: torch.Tensor, values: Sequence[int]) -> tuple[int, ...]:
    cpu_values = tuple(int(value) for value in values)
    if boundaries.ndim != 1 or len(cpu_values) != boundaries.numel():
        raise ValueError("CPU sequence boundaries must match the one-dimensional device tensor.")
    boundaries._slime_cpu_sequence_boundaries = (boundaries._version, cpu_values)
    return cpu_values


def get_cpu_sequence_boundaries(boundaries: torch.Tensor) -> tuple[int, ...]:
    cached = getattr(boundaries, "_slime_cpu_sequence_boundaries", None)
    if cached is not None and cached[0] == boundaries._version:
        return cached[1]
    return cache_cpu_sequence_boundaries(boundaries, boundaries.tolist())
