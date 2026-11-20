"""Information measures from paper Section 3.2 / Eq. (3), (5), (7).

Important: Eq. (5) sums *marginal* binary entropies. This is not the
joint entropy of dependent labels, despite the discussion of multivariate
Bernoulli distributions in the paper. A counterexample is in experiments/.
"""
from collections import Counter
from math import log2, log, sqrt
from typing import Sequence


def binary_entropy(p: float) -> float:
    if not 0 <= p <= 1:
        raise ValueError("p must be a probability")
    if p == 0 or p == 1:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def multilabel_marginal_entropy(labelsets: Counter[tuple[int, ...]], n_labels: int) -> float:
    total = sum(labelsets.values())
    if total <= 0:
        return 0.0
    return sum(binary_entropy(sum(freq for ys, freq in labelsets.items() if ys[j]) / total)
               for j in range(n_labels))


def joint_labelset_entropy(labelsets: Counter[tuple[int, ...]]) -> float:
    """Actual empirical joint entropy, for theoretical stress tests, NOT Eq. (5)."""
    total = sum(labelsets.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * log2(v / total) for v in labelsets.values() if v)


def hoeffding_split_bound(n_observations: int, n_distinct_labelsets: int, delta: float = 1e-5) -> float:
    """Paper Eq. (7): R=sqrt? No. R=log2(number of known labelsets).

    epsilon = sqrt(R^2 * ln(1/delta) / (2W)).
    A lower bound of 2 sets avoids the degenerate log2(1)=0 case.
    The latter is only a guard for this baseline, and must be documented.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    if n_observations < 1:
        return float("inf")
    r = log2(max(2, n_distinct_labelsets))
    return sqrt((r * r * log(1 / delta)) / (2 * n_observations))
