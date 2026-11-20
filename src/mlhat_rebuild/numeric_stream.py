"""Deterministic numerical multi-label stress-test stream; NOT paper data."""
from random import Random


def numeric_multilabel_stream(n: int, drift_at: int, noise: float = 0.05, seed: int = 42):
    if n < 0 or not (0 <= noise <= 1):
        raise ValueError('invalid stream arguments')
    rng = Random(seed)
    for i in range(n):
        x = tuple(rng.random() for _ in range(4))
        y0 = int(x[0] + x[1] > 1.0)
        y1 = int(x[2] > 0.5 or (x[0] > 0.8 and x[1] > 0.8))
        if i >= drift_at:
            y0 = 1 - y0
            y1 = int(x[3] > 0.5)
        y = (y0 ^ int(rng.random() < noise), y1 ^ int(rng.random() < noise))
        yield x, y
