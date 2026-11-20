"""Online Gaussian observers for numeric candidate splits (paper Section 3.4).

Estimates conditional label probabilities via Gaussian CDFs. Candidate
thresholds form a fixed grid over the observed range. This is an approximate
implementation of Eqs. (10)-(15); exact author binning is not specified.
"""
from math import erf, sqrt

from .entropy import binary_entropy, multilabel_marginal_entropy


class GaussianMoments:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.low = float('inf')
        self.high = float('-inf')

    def update(self, value: float):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.low = min(self.low, value)
        self.high = max(self.high, value)

    def cdf(self, threshold: float) -> float:
        if not self.n:
            return 0.0
        if self.n == 1:
            return float(self.mean <= threshold)
        variance = max(self.m2 / (self.n - 1), 0.0)
        # Small Gaussian smoothing prevents a zero-variance divide by zero.
        sigma_floor = max(1e-3, 0.01 * (self.high - self.low))
        sigma = max(sqrt(variance), sigma_floor)
        return max(0.0, min(1.0, 0.5 * (1.0 + erf((threshold - self.mean) / (sigma * sqrt(2.0))))))


class NumericSplitObserver:
    def __init__(self, n_features: int, n_labels: int, n_bins: int = 8):
        if n_features < 1 or n_labels < 1 or n_bins < 1:
            raise ValueError('invalid observer configuration')
        self.n_features, self.n_labels, self.n_bins = n_features, n_labels, n_bins
        self.all_features = [GaussianMoments() for _ in range(n_features)]
        self.conditional = [[[GaussianMoments(), GaussianMoments()]
                             for _ in range(n_labels)] for _ in range(n_features)]

    def update(self, x: tuple[float, ...], y: tuple[int, ...]):
        for f, value in enumerate(x):
            self.all_features[f].update(value)
            for j, label in enumerate(y):
                self.conditional[f][j][label].update(value)

    def split_candidates(self, labelsets, n_labels: int):
        """Return (gain, feature_id, threshold), best threshold per feature."""
        h_parent = multilabel_marginal_entropy(labelsets, n_labels)
        proposals = []
        for f, all_f in enumerate(self.all_features):
            if all_f.n < 2 or all_f.high <= all_f.low:
                continue
            best = None
            for k in range(1, self.n_bins + 1):
                threshold = all_f.low + (all_f.high - all_f.low) * k / (self.n_bins + 1)
                p_left = all_f.cdf(threshold)
                if p_left < 1e-6 or p_left > 1 - 1e-6:
                    continue
                h_left = h_right = 0.0
                for j in range(n_labels):
                    negative, positive = self.conditional[f][j]
                    n_neg, n_pos = negative.n, positive.n
                    l_pos = n_pos * positive.cdf(threshold)
                    l_neg = n_neg * negative.cdf(threshold)
                    r_pos = n_pos - l_pos
                    r_neg = n_neg - l_neg
                    marginal = n_pos / (n_pos + n_neg)
                    p_pos_l = l_pos / (l_pos + l_neg) if l_pos + l_neg > 1e-12 else marginal
                    p_pos_r = r_pos / (r_pos + r_neg) if r_pos + r_neg > 1e-12 else marginal
                    h_left += binary_entropy(max(0.0, min(1.0, p_pos_l)))
                    h_right += binary_entropy(max(0.0, min(1.0, p_pos_r)))
                gain = h_parent - (p_left * h_left + (1.0 - p_left) * h_right)
                candidate = (gain, f, threshold)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is not None:
                proposals.append(best)
        return sorted(proposals, key=lambda t: (-t[0], t[1], t[2]))
