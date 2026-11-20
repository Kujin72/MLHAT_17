"""A transparent, *uncompressed ADWIN-inspired* change detector.

Designed as an independent learning scaffold, not a claim of exact ADWIN
or a substitute for River ADWIN in a paper-fidelity reproduction. Tracks a
bounded loss window and applies a Bernstein-style cut test every N updates.
"""
from collections import deque
from math import log, sqrt, isfinite


class AdaptiveLossWindow:
    def __init__(self, delta=0.01, warning_delta=0.2, max_window=256,
                 min_segment=16, check_every=8):
        if not (0 < delta < warning_delta < 1):
            raise ValueError('require 0 < delta < warning_delta < 1')
        if max_window < 2 * min_segment or min_segment < 2 or check_every < 1:
            raise ValueError('invalid window configuration')
        self.delta = delta
        self.warning_delta = warning_delta
        self.max_window = max_window
        self.min_segment = min_segment
        self.check_every = check_every
        self.window = deque(maxlen=max_window)
        self.n_updates = 0
        self.warning_count = 0
        self.change_count = 0

    @staticmethod
    def _epsilon(n0, n1, variance, confidence, width):
        m = 1 / n0 + 1 / n1
        # ADWIN-style Bernstein cut threshold; approximate, since this
        # educational version limits its window size and checks periodically.
        log_term = log(2 * max(2.0, log(width)) / confidence)
        return sqrt(2 * m * variance * log_term) + 2 * m * log_term / 3

    def update(self, loss):
        if not isinstance(loss, (int, float)) or not isfinite(loss) or not 0 <= loss <= 1:
            raise ValueError('loss must be a finite value in [0, 1]')
        self.window.append(float(loss))
        self.n_updates += 1
        n = len(self.window)
        if n < 2 * self.min_segment or self.n_updates % self.check_every:
            return None
        vals = list(self.window)
        total = sum(vals)
        variance = max(0.0, sum((v - total / n) ** 2 for v in vals) / n)
        prefix = 0.0
        warning_cut, change_cut = None, None
        best_warning, best_change = 0.0, 0.0
        for cut, value in enumerate(vals[:-self.min_segment], start=1):
            prefix += value
            if cut < self.min_segment:
                continue
            left = prefix / cut
            right = (total - prefix) / (n - cut)
            increase = right - left   # Only higher error should start an alternate.
            if increase <= 0:
                continue
            if increase > self._epsilon(cut, n-cut, variance, self.warning_delta, n):
                if increase > best_warning:
                    warning_cut, best_warning = cut, increase
            if increase > self._epsilon(cut, n-cut, variance, self.delta, n):
                if increase > best_change:
                    change_cut, best_change = cut, increase
        if change_cut is not None:
            self.change_count += 1
            self.window = deque(vals[change_cut:], maxlen=self.max_window)
            return 'drift'
        if warning_cut is not None:
            self.warning_count += 1
            return 'warning'
        return None
