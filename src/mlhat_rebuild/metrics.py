"""Online metrics. EWMA setting alpha=0.995 from paper Section 4.1.3."""
from dataclasses import dataclass


def one_example_metrics(y: tuple[int, ...], pred: tuple[int, ...]) -> dict[str, float]:
    if not y or len(y) != len(pred):
        raise ValueError("mismatched label vectors")
    inter = sum(a == b == 1 for a, b in zip(y, pred))
    true_pos = sum(y)
    pred_pos = sum(pred)
    # Standard example-F1 convention for both empty: score 1.
    f1 = (2.0 * inter / (true_pos + pred_pos)) if true_pos + pred_pos else 1.0
    return {
        "example_f1": f1,
        "hamming_loss": sum(a != b for a, b in zip(y, pred)) / len(y),
        "subset_accuracy": float(y == pred),
    }


@dataclass
class RunningMetrics:
    alpha: float = 0.995

    def __post_init__(self):
        if not 0 <= self.alpha < 1:
            raise ValueError("alpha in [0,1)")
        self.count = 0
        self.accum: dict[str, float] = {}
        self.ewma: dict[str, float] = {}

    def update(self, y: tuple[int, ...], pred: tuple[int, ...]) -> None:
        d = one_example_metrics(y, pred)
        self.count += 1
        for k, v in d.items():
            self.accum[k] = self.accum.get(k, 0) + v
            self.ewma[k] = v if self.count == 1 else self.alpha * self.ewma[k] + (1 - self.alpha) * v

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {**{k: v / self.count for k, v in self.accum.items()},
                **{f"ewma_{k}": v for k, v in self.ewma.items()}}
