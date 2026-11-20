import unittest
from mlhat_rebuild.tree import OnlineMultilabelHoeffdingTree
from mlhat_rebuild.metrics import RunningMetrics, one_example_metrics
from mlhat_rebuild.stream import synthetic_multilabel_stream


class TestTree(unittest.TestCase):
    def test_online_update(self):
        model = OnlineMultilabelHoeffdingTree(2, 2, grace_period=200)
        self.assertEqual(model.predict_one((0, 1)), (0, 0))
        model.learn_one((0, 1), (1, 1))
        self.assertEqual(model.predict_one((0, 1)), (1, 1))

    def test_separates_easy_stream(self):
        model = OnlineMultilabelHoeffdingTree(2, 2, grace_period=50, delta=0.01)
        for i in range(1000):
            x = (i % 2, (i // 2) % 2)
            model.learn_one(x, (x[0], x[0]))
        self.assertGreaterEqual(model.n_splits, 1)
        self.assertEqual(model.predict_one((0, 1)), (0, 0))
        self.assertEqual(model.predict_one((1, 1)), (1, 1))

    def test_inputs_reject_numerical(self):
        model = OnlineMultilabelHoeffdingTree(2, 1)
        with self.assertRaises(ValueError):
            model.predict_one((0.3, 1))

    def test_prequential_metric(self):
        m = RunningMetrics()
        self.assertEqual(one_example_metrics((1, 0), (1, 0))["example_f1"], 1)
        m.update((1, 0), (0, 0))
        self.assertEqual(m.summary()["subset_accuracy"], 0)
        m.update((1, 0), (1, 0))
        self.assertAlmostEqual(m.summary()["subset_accuracy"], 0.5)

    def test_stream_reproducible(self):
        a = list(synthetic_multilabel_stream(100, seed=5))
        b = list(synthetic_multilabel_stream(100, seed=5))
        self.assertEqual(a, b)
