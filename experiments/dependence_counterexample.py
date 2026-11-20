"""Exact critique of paper Eq. (5): sum of marginal entropies != joint entropy.

Creates a deterministic table for strongly dependent and independent labels.
The test concerns the equation, NOT a measured failure of the full model.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlhat_rebuild.entropy import joint_labelset_entropy, multilabel_marginal_entropy

cases = {
    "perfectly_correlated": Counter({(0, 0): 50, (1, 1): 50}),
    "independent": Counter({(0, 0): 25, (0, 1): 25, (1, 0): 25, (1, 1): 25}),
}
rows = []
for case, counts in cases.items():
    rows.append({"case": case, "samples": sum(counts.values()),
                 "paper_eq5_sum_marginal_bits": multilabel_marginal_entropy(counts, 2),
                 "actual_joint_entropy_bits": joint_labelset_entropy(counts)})
    print(rows[-1])
p = ROOT / "results" / "label_dependence_counterexample.csv"
p.parent.mkdir(exist_ok=True)
with p.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"Saved: {p}")
