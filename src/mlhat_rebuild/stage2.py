"""Stage 2 *partial* MLHAT rebuild: numeric Hoeffding Tree + adaptive leaves.

Contains: Gaussian numeric split candidates, multi-label marginal entropy,
Hoeffding split criterion, LP-kNN and BR Bagged Logistic Regression at leaves.
Not yet: per-node ADWIN, main-tree Poisson bootstrap, alternate branches,
replacement/pruning, or mixing predictions. NOT a full MLHAT reproduction.
"""
from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from typing import Optional

from .entropy import hoeffding_split_bound, multilabel_marginal_entropy
from .numeric_observer import NumericSplitObserver
from .online_learners import BinaryRelevanceBaggedLogistic, LabelPowersetKNN


@dataclass
class Stage2Node:
    labels: Counter[tuple[int, ...]] = field(default_factory=Counter)
    seen: int = 0
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Stage2Node'] = None
    right: Optional['Stage2Node'] = None
    observer: Optional[NumericSplitObserver] = None
    knn: Optional[LabelPowersetKNN] = None
    bagger: Optional[BinaryRelevanceBaggedLogistic] = None

    @property
    def is_leaf(self):
        return self.feature_index is None


class OnlineMultilabelStage2Tree:
    def __init__(self, n_features: int, n_labels: int, grace_period: int = 200,
                 delta: float = 1e-7, eta: int = 750, k: int = 5,
                 max_knn_memory: int = 256, ensemble_size: int = 10,
                 n_bins: int = 8, seed: int = 42):
        if n_features < 2 or n_labels < 1 or grace_period < 1 or eta < 1:
            raise ValueError('invalid tree dimensions/periods')
        if not 0 < delta < 1:
            raise ValueError('delta must be between 0 and 1')
        self.n_features, self.n_labels = n_features, n_labels
        self.grace_period, self.delta, self.eta = grace_period, delta, eta
        self.k, self.max_knn_memory = k, max_knn_memory
        self.ensemble_size, self.n_bins, self.seed = ensemble_size, n_bins, seed
        self._leaf_seq = 0
        self.root = self._new_leaf()
        self.n_samples = 0
        self.n_splits = 0

    def _new_leaf(self):
        self._leaf_seq += 1
        return Stage2Node(
            observer=NumericSplitObserver(self.n_features, self.n_labels, self.n_bins),
            knn=LabelPowersetKNN(self.n_labels, self.k, self.max_knn_memory),
            bagger=BinaryRelevanceBaggedLogistic(self.n_features, self.n_labels,
                                                 self.ensemble_size,
                                                 seed=self.seed + 997 * self._leaf_seq),
        )

    def _validate(self, x, y=None):
        if len(x) != self.n_features or not all(isinstance(v, (float, int)) and isfinite(v) for v in x):
            raise ValueError('x must contain exactly n_features finite numeric values')
        if y is not None and (len(y) != self.n_labels or any(v not in (0, 1) for v in y)):
            raise ValueError('y must contain exactly n_labels binary entries')

    def _leaf(self, x):
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature_index] <= node.threshold else node.right
        return node

    def leaf_mode(self, x):
        self._validate(x)
        leaf = self._leaf(x)
        if leaf.seen == 0:
            return 'empty'
        h = multilabel_marginal_entropy(leaf.labels, self.n_labels)
        if h <= 1e-12:
            return 'majority'
        return 'lp_knn' if leaf.seen <= self.eta else 'br_bagged_lr'

    def predict_one(self, x):
        self._validate(x)
        leaf = self._leaf(x)
        mode = self.leaf_mode(x)
        if mode == 'empty':
            return (0,) * self.n_labels
        if mode == 'majority':
            return min(leaf.labels, key=lambda ys: (-leaf.labels[ys], ys))
        if mode == 'lp_knn':
            return leaf.knn.predict_one(x)
        return leaf.bagger.predict_one(x)

    def learn_one(self, x, y):
        self._validate(x, y)
        x, y = tuple(float(v) for v in x), tuple(int(v) for v in y)
        leaf = self._leaf(x)
        leaf.seen += 1
        leaf.labels[y] += 1
        leaf.observer.update(x, y)
        self.n_samples += 1
        # In the pure-leaf scenario, the paper uses the majority labelset.
        if multilabel_marginal_entropy(leaf.labels, self.n_labels) > 1e-12:
            if leaf.seen <= self.eta:
                leaf.knn.learn_one(x, y)
            # High-complexity learner is warm-started at low cardinality.
            leaf.bagger.learn_one(x, y)
        if leaf.seen % self.grace_period == 0:
            self._try_split(leaf)
        return self

    def _try_split(self, leaf):
        if multilabel_marginal_entropy(leaf.labels, self.n_labels) <= 1e-12:
            return
        proposals = leaf.observer.split_candidates(leaf.labels, self.n_labels)
        if not proposals:
            return
        best_gain, best_feature, best_threshold = proposals[0]
        second_gain = max(0.0, proposals[1][0]) if len(proposals) >= 2 else 0.0
        epsilon = hoeffding_split_bound(leaf.seen, len(leaf.labels), self.delta)
        if best_gain <= 0 or best_gain - second_gain <= epsilon:
            return
        leaf.feature_index, leaf.threshold = best_feature, best_threshold
        leaf.left, leaf.right = self._new_leaf(), self._new_leaf()
        leaf.observer = None
        leaf.knn = None
        leaf.bagger = None
        self.n_splits += 1
