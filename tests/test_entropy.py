import unittest
from collections import Counter
from math import isclose
from mlhat_rebuild.entropy import binary_entropy, hoeffding_split_bound, joint_labelset_entropy, multilabel_marginal_entropy


class TestEntropy(unittest.TestCase):
    def test_binary_limits(self):
        self.assertEqual(binary_entropy(0), 0)
        self.assertEqual(binary_entropy(1), 0)
        self.assertEqual(binary_entropy(0.5), 1)

    def test_sum_of_marginals_is_not_joint_entropy(self):
        a = Counter({(0, 0): 50, (1, 1): 50})
        b = Counter({(0, 0): 25, (0, 1): 25, (1, 0): 25, (1, 1): 25})
        self.assertTrue(isclose(multilabel_marginal_entropy(a, 2), 2))
        self.assertTrue(isclose(multilabel_marginal_entropy(b, 2), 2))
        self.assertEqual(joint_labelset_entropy(a), 1)
        self.assertEqual(joint_labelset_entropy(b), 2)

    def test_hoeffding_bound_decreases(self):
        a = hoeffding_split_bound(200, 4)
        b = hoeffding_split_bound(800, 4)
        self.assertAlmostEqual(a, 2*b)

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            binary_entropy(1.2)
