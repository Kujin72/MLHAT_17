"""Synthetic binary-feature, multi-label stream for ENGINEERING tests only.

This is NOT any of the paper's MOA-generated datasets, which vary
cardinality, dependency, drift type, and number of features/labels.
"""
import random
from typing import Iterator


def synthetic_multilabel_stream(n: int = 3000, drift_at: int | None = None,
                                noise: float = 0.05, seed: int = 42) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    if n <= 0 or not 0 <= noise <= 1:
        raise ValueError("n must be positive; noise must be in [0,1]")
    rng = random.Random(seed)
    if drift_at is None:
        drift_at = n // 2
    for i in range(n):
        x = tuple(rng.randint(0, 1) for _ in range(4))
        # After drift the relevant feature changes from x[0] to x[1].
        y0 = x[0] if i < drift_at else x[1]
        y1 = y0  # Co-occurring labels with independent label noise.
        y0 = int(y0 ^ (rng.random() < noise))
        y1 = int(y1 ^ (rng.random() < noise))
        yield x, (y0, y1)
