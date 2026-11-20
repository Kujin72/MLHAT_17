"""Independent online multi-label learners for MLHAT Stage 2.

LP-kNN is a bounded-window Label-Powerset kNN. BR-Bagged-LR is an online
Binary-Relevance ensemble using per-member Poisson(1) weights. This module is
an independent educational implementation, NOT copied from author source.
"""
from collections import Counter, deque
from math import exp
from random import Random


def poisson_one(rng: Random) -> int:
    """Sample Poisson(lambda=1) via Knuth's algorithm, without NumPy."""
    k = 0
    p = 1.0
    limit = exp(-1.0)
    while p > limit:
        k += 1
        p *= rng.random()
    return k - 1


class LabelPowersetKNN:
    def __init__(self, n_labels: int, k: int = 5, max_memory: int = 256):
        if n_labels < 1 or k < 1 or max_memory < 1:
            raise ValueError('invalid kNN configuration')
        self.n_labels = n_labels
        self.k = k
        self.memory = deque(maxlen=max_memory)

    def learn_one(self, x: tuple[float, ...], y: tuple[int, ...]):
        if len(y) != self.n_labels:
            raise ValueError('invalid label size')
        self.memory.append((tuple(x), tuple(y)))
        return self

    def predict_one(self, x: tuple[float, ...]) -> tuple[int, ...]:
        if not self.memory:
            return (0,) * self.n_labels
        nearest = sorted(
            ((sum((a - b) ** 2 for a, b in zip(x, observed_x)), i, observed_y)
             for i, (observed_x, observed_y) in enumerate(self.memory)),
            key=lambda row: (row[0], row[1])
        )[:self.k]
        counts = Counter(ys for _, _, ys in nearest)
        return min(counts, key=lambda ys: (-counts[ys], ys))


class OnlineBinaryLogistic:
    def __init__(self, n_features: int, learning_rate: float = 0.07, l2: float = 0.0001):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.l2 = l2

    def predict_proba_one(self, x: tuple[float, ...]) -> float:
        z = self.bias + sum(w * v for w, v in zip(self.weights, x))
        if z >= 0:
            return 1.0 / (1.0 + exp(-z))
        ez = exp(z)
        return ez / (1.0 + ez)

    def learn_one(self, x: tuple[float, ...], target: int, repetitions: int = 1):
        for _ in range(repetitions):
            error = target - self.predict_proba_one(x)
            for j, v in enumerate(x):
                self.weights[j] += self.learning_rate * (error * v - self.l2 * self.weights[j])
            self.bias += self.learning_rate * error
        return self


class BinaryRelevanceBaggedLogistic:
    """Independent online BR bagging with exactly one Poisson weight per member."""
    def __init__(self, n_features: int, n_labels: int, ensemble_size: int = 10, seed: int = 42):
        if n_features < 1 or n_labels < 1 or ensemble_size < 1:
            raise ValueError('invalid bagging configuration')
        self.n_labels = n_labels
        self.rng = Random(seed)
        self.members = [[OnlineBinaryLogistic(n_features) for _ in range(n_labels)]
                        for _ in range(ensemble_size)]
        self.total_updates = 0

    def learn_one(self, x: tuple[float, ...], y: tuple[int, ...]):
        for member in self.members:
            weight = poisson_one(self.rng)
            if weight:
                for classifier, label in zip(member, y):
                    classifier.learn_one(x, label, repetitions=weight)
                self.total_updates += weight
        return self

    def predict_one(self, x: tuple[float, ...]) -> tuple[int, ...]:
        n = len(self.members)
        return tuple(int(sum(member[j].predict_proba_one(x) for member in self.members) / n > 0.5)
                     for j in range(self.n_labels))
