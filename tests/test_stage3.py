import unittest

from mlhat_rebuild.drift import AdaptiveLossWindow
from mlhat_rebuild.stage3 import OnlineMultilabelStage3Tree, hamming_loss
from mlhat_rebuild.numeric_stream import numeric_multilabel_stream
from mlhat_rebuild.metrics import RunningMetrics


class TestStage3(unittest.TestCase):
    def test_hamming_loss(self):
        self.assertEqual(hamming_loss((1, 0, 1), (0, 0, 0)), 2 / 3)
        self.assertEqual(hamming_loss((0, 1), (0, 1)), 0.0)
        with self.assertRaises(ValueError):
            hamming_loss((1,), ())

    def test_detector_rejects_invalid_parameters_and_loss(self):
        with self.assertRaises(ValueError):
            AdaptiveLossWindow(delta=0.2, warning_delta=0.1)
        with self.assertRaises(ValueError):
            AdaptiveLossWindow(max_window=10)
        with self.assertRaises(ValueError):
            AdaptiveLossWindow().update(1.2)

    def test_detector_stable_signal(self):
        d = AdaptiveLossWindow()
        self.assertFalse(any(d.update(0.25) for _ in range(400)))

    def test_detector_responds_to_error_increase(self):
        d = AdaptiveLossWindow()
        signals = [d.update(x) for x in [0.0] * 200 + [1.0] * 200]
        self.assertIn('warning', signals)
        self.assertIn('drift', signals)

    def test_detector_is_reproducible(self):
        stream = [0.1] * 200 + [0.9] * 200
        a, b = AdaptiveLossWindow(), AdaptiveLossWindow()
        self.assertEqual([a.update(v) for v in stream],
                         [b.update(v) for v in stream])

    def test_tree_validates_inputs_and_emits_labels(self):
        model = OnlineMultilabelStage3Tree(2, 2)
        self.assertEqual(model.predict_one((0.2, 0.8)), (0, 0))
        model.learn_one((0.2, 0.8), (1, 0))
        self.assertEqual(model.predict_one((0.2, 0.8)), (1, 0))
        with self.assertRaises(ValueError):
            model.learn_one((0.2, float('nan')), (1, 0))

    def test_replacement_on_known_concept_change(self):
        model = OnlineMultilabelStage3Tree(4, 2, seed=42)
        metrics = RunningMetrics()
        for x, y in numeric_multilabel_stream(3000, 1500, 0.05, 42):
            pred = model.predict_one(x)
            metrics.update(y, pred)
            model.learn_one(x, y)
        self.assertGreaterEqual(model.n_warnings, 1)
        self.assertGreaterEqual(model.n_replacements, 1)
        self.assertTrue(any(event['type'] == 'replacement' for event in model.events))
        self.assertEqual(model.n_samples, 3000)
        self.assertGreater(metrics.summary()['example_f1'], 0.5)

    def test_stage3_deterministic(self):
        a = OnlineMultilabelStage3Tree(4, 2, seed=13)
        b = OnlineMultilabelStage3Tree(4, 2, seed=13)
        seq = list(numeric_multilabel_stream(350, 175, seed=13))
        for x, y in seq:
            self.assertEqual(a.predict_one(x), b.predict_one(x))
            a.learn_one(x, y)
            b.learn_one(x, y)
        self.assertEqual(a.events, b.events)
        self.assertEqual(a.n_splits, b.n_splits)

    def test_blending_flag_is_usable(self):
        model = OnlineMultilabelStage3Tree(4, 2, blend_alternates=True, seed=42)
        for x, y in numeric_multilabel_stream(400, 200, seed=42):
            self.assertEqual(len(model.predict_one(x)), 2)
            model.learn_one(x, y)


if __name__ == '__main__':
    unittest.main()
