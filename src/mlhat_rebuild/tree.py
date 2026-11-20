"""Small independent baseline: online multi-label Hoeffding tree.

This intentionally implements a subset of the paper:
  - node labelset counters and Eq. (5) marginal-entropy information gain;
  - statistically gated binary-feature splitting following Eq. (7)-(9);
  - majority-labelset leaf prediction and one-by-one incremental training.

NOT YET IMPLEMENTED: numeric Gaussian observers / categorical multi-value,
LP-kNN and BR-ADWIN-Bagging leaf learners, Poisson bootstrap, per-node ADWIN,
background trees, pruning, predictive mixing or memory management. Consequently
this class MUST NOT be presented as a faithful reproduction of full MLHAT.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .entropy import multilabel_marginal_entropy, hoeffding_split_bound


@dataclass
class Node:
    labels: Counter[tuple[int, ...]] = field(default_factory=Counter)
    # counts for each feature value 0 / 1 (values cannot be continuous yet)
    feature_counts: list[list[Counter[tuple[int, ...]]]] = field(default_factory=list)
    feature_index: Optional[int] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    seen: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class OnlineMultilabelHoeffdingTree:
    def __init__(self, n_features: int, n_labels: int, grace_period: int = 200,
                 delta: float = 1e-5):
        if n_features < 2 or n_labels < 1 or grace_period < 1:
            raise ValueError("invalid n_features, n_labels or grace_period")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0,1)")
        self.n_features, self.n_labels = n_features, n_labels
        self.grace_period, self.delta = grace_period, delta
        self.root = self._new_leaf()
        self.n_splits = 0
        self.n_samples = 0

    def _new_leaf(self) -> Node:
        return Node(feature_counts=[[Counter(), Counter()] for _ in range(self.n_features)])

    def _validate(self, x: tuple[int, ...], y: tuple[int, ...] | None = None) -> None:
        if len(x) != self.n_features or any(v not in (0, 1) for v in x):
            raise ValueError("stage-1 model requires exactly n_features binary values")
        if y is not None and (len(y) != self.n_labels or any(v not in (0, 1) for v in y)):
            raise ValueError("y must contain n_labels binary values")

    def _leaf(self, x: tuple[int, ...]) -> Node:
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature_index] == 0 else node.right
        return node

    def predict_one(self, x: tuple[int, ...]) -> tuple[int, ...]:
        self._validate(x)
        counts = self._leaf(x).labels
        if not counts:
            return (0,) * self.n_labels
        # Tie resolution is deterministic: smallest tuple first.
        return min(counts, key=lambda ys: (-counts[ys], ys))

    def learn_one(self, x: tuple[int, ...], y: tuple[int, ...]) -> "OnlineMultilabelHoeffdingTree":
        self._validate(x, y)
        n = self._leaf(x)
        n.seen += 1
        n.labels[y] += 1
        for f, value in enumerate(x):
            n.feature_counts[f][value][y] += 1
        self.n_samples += 1
        if n.seen % self.grace_period == 0:
            self._try_split(n)
        return self

    def _try_split(self, node: Node) -> None:
        h_before = multilabel_marginal_entropy(node.labels, self.n_labels)
        if h_before == 0:
            return
        scores = []
        for f, (left, right) in enumerate(node.feature_counts):
            nl, nr = sum(left.values()), sum(right.values())
            if not nl or not nr:
                continue
            h_after = (nl * multilabel_marginal_entropy(left, self.n_labels) +
                       nr * multilabel_marginal_entropy(right, self.n_labels)) / (nl + nr)
            scores.append((h_before - h_after, f))
        if not scores:
            return
        scores.sort(key=lambda item: (-item[0], item[1]))
        g_best, f_best = scores[0]
        # Null split competes with the best: no change, gain zero.
        g_second = max(0.0, scores[1][0]) if len(scores) > 1 else 0.0
        eps = hoeffding_split_bound(node.seen, len(node.labels), self.delta)
        if g_best > 0 and (g_best - g_second) > eps:
            node.feature_index = f_best
            node.left = self._new_leaf()
            node.right = self._new_leaf()
            # Free split-observer statistics; the new leaves learn only new data.
            node.feature_counts = []
            self.n_splits += 1


class OnlineMajorityLabelset:
    """Simple online baseline that always predicts the most frequent labelset."""
    def __init__(self, n_labels: int):
        self.n_labels = n_labels
        self.counts: Counter[tuple[int, ...]] = Counter()

    def predict_one(self, x: tuple[int, ...]) -> tuple[int, ...]:
        return min(self.counts, key=lambda ys: (-self.counts[ys], ys)) if self.counts else (0,) * self.n_labels

    def learn_one(self, x: tuple[int, ...], y: tuple[int, ...]) -> "OnlineMajorityLabelset":
        self.counts[y] += 1
        return self
