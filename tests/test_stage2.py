import unittest

from mlhat_rebuild.numeric_observer import GaussianMoments, NumericSplitObserver
from mlhat_rebuild.numeric_stream import numeric_multilabel_stream
from mlhat_rebuild.online_learners import (LabelPowersetKNN,
                                            BinaryRelevanceBaggedLogistic, poisson_one)
from mlhat_rebuild.stage2 import OnlineMultilabelStage2Tree
from random import Random
from collections import Counter


class TestStage2(unittest.TestCase):
    def test_gaussian_cdf_order(self):
        gauss = GaussianMoments()
        for x in [0.1, 0.2, 0.25, 0.9, 0.8]:
            gauss.update(x)
        self.assertLess(gauss.cdf(0.2), gauss.cdf(0.7))
        self.assertTrue(0 <= gauss.cdf(0.4) <= 1)

    def test_numeric_split_prefers_signal(self):
        observer = NumericSplitObserver(2, 2)
        labels = Counter()
        for i in range(200):
            x = (0.1 if i % 2 == 0 else 0.9, (i % 5) / 5)
            y = (int(x[0] > 0.5), int(x[0] > 0.5))
            observer.update(x, y)
            labels[y] += 1
        proposals = observer.split_candidates(labels, 2)
        self.assertEqual(proposals[0][1], 0)
        self.assertGreater(proposals[0][0], 0)

    def test_knn_label_powerset_and_bounded_memory(self):
        learner = LabelPowersetKNN(2, k=1, max_memory=2)
        learner.learn_one((0.0, 0.0), (1, 1))
        learner.learn_one((1.0, 1.0), (0, 0))
        self.assertEqual(learner.predict_one((0.05, 0.05)), (1, 1))
        learner.learn_one((0.8, 0.8), (0, 1))
        self.assertEqual(len(learner.memory), 2)

    def test_bagged_lr_deterministic(self):
        a = BinaryRelevanceBaggedLogistic(2, 2, seed=123)
        b = BinaryRelevanceBaggedLogistic(2, 2, seed=123)
        for x, y in [((0.1, 0.2), (0, 0)), ((0.7, 0.8), (1, 1))] * 20:
            a.learn_one(x, y)
            b.learn_one(x, y)
        self.assertEqual(a.predict_one((0.6, 0.6)), b.predict_one((0.6, 0.6)))
        self.assertEqual(a.total_updates, b.total_updates)
        self.assertGreater(a.total_updates, 0)

    def test_poisson_nonnegative_and_deterministic(self):
        a, b = Random(5), Random(5)
        values = [poisson_one(a) for _ in range(100)]
        self.assertEqual(values, [poisson_one(b) for _ in range(100)])
        self.assertTrue(all(value >= 0 for value in values))

    def test_tree_accepts_numeric_and_rejects_invalid(self):
        tree = OnlineMultilabelStage2Tree(2, 2, grace_period=500)
        tree.learn_one((0.33, 0.71), (1, 0))
        self.assertEqual(tree.predict_one((0.33, 0.71)), (1, 0))
        with self.assertRaises(ValueError):
            tree.predict_one((float('nan'), 0.3))

    def test_high_complexity_leaf_mode(self):
        tree = OnlineMultilabelStage2Tree(2, 1, grace_period=5000, eta=2)
        for i in range(5):
            tree.learn_one((0.2, 0.6), (i % 2,))
        self.assertEqual(tree.leaf_mode((0.2, 0.6)), 'br_bagged_lr')

    def test_tree_splits_synthetic_numeric(self):
        tree = OnlineMultilabelStage2Tree(2, 1, grace_period=100, delta=1e-5,
                                          ensemble_size=3, seed=8)
        rng = Random(8)
        for _ in range(1500):
            x = (rng.random(), rng.random())
            tree.learn_one(x, (int(x[0] > 0.5),))
        self.assertGreaterEqual(tree.n_splits, 1)
        self.assertEqual(len(tree.predict_one((0.3, 0.8))), 1)

    def test_stream_reproducible(self):
        a = list(numeric_multilabel_stream(50, 25, seed=7))
        b = list(numeric_multilabel_stream(50, 25, seed=7))
        self.assertEqual(a, b)
        self.assertNotEqual(a, list(numeric_multilabel_stream(50, 25, seed=8)))


if __name__ == '__main__':
    unittest.main()
