"""Stage 3 independent *experimental* extension of Stage 2.

Adds node-local ADWIN-inspired Hamming-loss monitoring, alternate subtree
training, branch replacement/pruning with a conservative Hoeffding bound, and
optional hard-vote blending. This is NOT the paper's exact ADWIN, Gaussian
weighting, Poisson-bootstrap or probability-blending implementation.
"""
from collections import deque
from dataclasses import dataclass, field
from math import log, sqrt
from typing import Optional

from .drift import AdaptiveLossWindow
from .entropy import multilabel_marginal_entropy
from .stage2 import OnlineMultilabelStage2Tree, Stage2Node


def hamming_loss(y, prediction):
    if len(y) != len(prediction) or not y:
        raise ValueError('invalid labels/prediction lengths')
    return sum(int(a != b) for a, b in zip(y, prediction)) / len(y)


@dataclass
class _NodeMonitor:
    detector: AdaptiveLossWindow
    alternate: Optional[OnlineMultilabelStage2Tree] = None
    main_losses: deque = field(default_factory=lambda: deque(maxlen=256))
    alt_losses: deque = field(default_factory=lambda: deque(maxlen=256))


class OnlineMultilabelStage3Tree(OnlineMultilabelStage2Tree):
    def __init__(self, *args, delta_drift=0.01, warning_delta=0.2,
                 delta_alt=0.05, alternate_grace=200, monitor_window=256,
                 blend_alternates=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0 < delta_alt < 1) or alternate_grace < 1:
            raise ValueError('invalid alternate comparison configuration')
        self.delta_drift, self.warning_delta = delta_drift, warning_delta
        self.delta_alt, self.alternate_grace = delta_alt, alternate_grace
        self.monitor_window = monitor_window
        self.blend_alternates = blend_alternates
        self._node_states = {}
        self.n_warnings = 0
        self.n_replacements = 0
        self.n_pruned_alternates = 0
        self.events = []

    def _state(self, node):
        key = id(node)
        if key not in self._node_states:
            self._node_states[key] = _NodeMonitor(
                AdaptiveLossWindow(delta=self.delta_drift,
                                   warning_delta=self.warning_delta,
                                   max_window=self.monitor_window)
            )
        return self._node_states[key]

    def _path_nodes(self, x):
        out = []
        node, parent, side = self.root, None, None
        while True:
            out.append((node, parent, side))
            if node.is_leaf:
                return out
            parent, side = node, ('left' if x[node.feature_index] <= node.threshold else 'right')
            node = getattr(parent, side)

    def _predict_subtree(self, start, x):
        node = start
        while not node.is_leaf:
            node = node.left if x[node.feature_index] <= node.threshold else node.right
        if not node.seen:
            return (0,) * self.n_labels
        if multilabel_marginal_entropy(node.labels, self.n_labels) <= 1e-12:
            return min(node.labels, key=lambda ys: (-node.labels[ys], ys))
        if node.seen <= self.eta:
            return node.knn.predict_one(x)
        return node.bagger.predict_one(x)

    def predict_one(self, x):
        self._validate(x)
        base = self._predict_subtree(self.root, x)
        if not self.blend_alternates:
            return base
        votes = [[float(bit), 1.0] for bit in base]
        for node, _, _ in self._path_nodes(x):
            state = self._node_states.get(id(node))
            if not state or state.alternate is None:
                continue
            if state.alternate.n_samples < self.alternate_grace:
                continue
            pred = state.alternate.predict_one(x)
            e_main = sum(state.main_losses) / len(state.main_losses) if state.main_losses else 0.5
            e_alt = sum(state.alt_losses) / len(state.alt_losses) if state.alt_losses else 0.5
            weight = (1 - e_alt) / max(0.01, 1 - e_main)
            for i, bit in enumerate(pred):
                votes[i][0] += weight * bit
                votes[i][1] += weight
        return tuple(int(value / total > 0.5) for value, total in votes)

    def _new_alternate(self, salt):
        # Separate full incremental Stage 2 tree grown from the affected node.
        return OnlineMultilabelStage2Tree(
            self.n_features, self.n_labels, grace_period=self.grace_period,
            delta=self.delta, eta=self.eta, k=self.k,
            max_knn_memory=self.max_knn_memory,
            ensemble_size=self.ensemble_size, n_bins=self.n_bins,
            seed=self.seed + 1000003 + salt,
        )

    def _forget_subtree(self, root):
        todo = [root]
        while todo:
            node = todo.pop()
            self._node_states.pop(id(node), None)
            if not node.is_leaf:
                todo.extend([node.left, node.right])

    def _count_splits(self, root):
        if root.is_leaf:
            return 0
        return 1 + self._count_splits(root.left) + self._count_splits(root.right)

    @staticmethod
    def _comparison_epsilon(n, delta):
        # Distribution-free (two paired bounded loss streams) conservative test.
        # Distinct from the paper's variance-dependent Eq. (16): documented gap.
        return sqrt(2 * log(2 / delta) / n)

    def learn_one(self, x, y):
        self._validate(x, y)
        x, y = tuple(float(v) for v in x), tuple(int(v) for v in y)
        path = self._path_nodes(x)
        # Pre-update predictions: do not leak the incoming label into monitors.
        states_and_alts = []
        for depth, (node, parent, side) in enumerate(path):
            state = self._state(node)
            main_error = hamming_loss(y, self._predict_subtree(node, x))
            signal = state.detector.update(main_error)
            if signal and state.alternate is None:
                state.alternate = self._new_alternate(self.n_samples + depth)
                state.main_losses.clear()
                state.alt_losses.clear()
                self.n_warnings += 1
                self.events.append({'instance': self.n_samples + 1, 'type': 'warning', 'depth': depth})
            elif state.alternate is not None:
                alt_error = hamming_loss(y, state.alternate.predict_one(x))
                state.main_losses.append(main_error)
                state.alt_losses.append(alt_error)
            if state.alternate is not None:
                states_and_alts.append((node, parent, side, depth, state))
        # The original Stage 2 learning method is unchanged.
        super().learn_one(x, y)
        for _, _, _, _, state in states_and_alts:
            state.alternate.learn_one(x, y)
        # Deepest-first branch updates. Check the parent is still reachable.
        for node, parent, side, depth, state in reversed(states_and_alts):
            if parent is None:
                if self.root is not node:
                    continue
            elif getattr(parent, side) is not node:
                continue
            alt = state.alternate
            n = len(state.main_losses)
            if alt.n_samples < self.alternate_grace or n < self.alternate_grace:
                continue
            main_error = sum(state.main_losses) / n
            alternate_error = sum(state.alt_losses) / n
            eps = self._comparison_epsilon(n, self.delta_alt)
            if main_error - alternate_error > eps:
                self._forget_subtree(node)
                if parent is None:
                    self.root = alt.root
                else:
                    setattr(parent, side, alt.root)
                self.n_replacements += 1
                self.events.append({'instance': self.n_samples, 'type': 'replacement',
                                    'depth': depth, 'main_error': main_error,
                                    'alternate_error': alternate_error, 'epsilon': eps})
            elif alternate_error - main_error > eps:
                state.alternate = None
                state.main_losses.clear()
                state.alt_losses.clear()
                self.n_pruned_alternates += 1
                self.events.append({'instance': self.n_samples, 'type': 'alternate_pruned', 'depth': depth})
        self.n_splits = self._count_splits(self.root)
        return self
