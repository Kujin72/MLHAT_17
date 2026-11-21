"""Smoke tests for cumulative-to-window Example-F1 recovery and CLI."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "experiments" / "analyze_drift.py"

spec = importlib.util.spec_from_file_location("analyze_drift", ANALYZE)
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


class TestDriftAnalysis(unittest.TestCase):
    def test_interval_mean_from_cumulative(self):
        history = {1500: 0.8, 2000: 0.725, 3000: 2.2 / 3}
        self.assertAlmostEqual(analyze.interval_f1(history, 0, 1500), 0.8)
        self.assertAlmostEqual(analyze.interval_f1(history, 1500, 2000), 0.5)
        self.assertAlmostEqual(analyze.interval_f1(history, 2000, 3000), 0.75)

    def test_missing_checkpoint(self):
        with self.assertRaises(ValueError):
            analyze.interval_f1({3000: .7}, 1500, 3000)

    def test_cli_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model, scores in (("stage2", (.8, .5, .75)),
                                  ("stage3", (.8, .6, .85))):
                prefix = root / f"numeric_stage3_{model}_seed42"
                n, cum = 0, 0
                with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["instance", "example_f1"])
                    writer.writeheader()
                    for span, score in zip((1500, 500, 1000), scores):
                        n += span
                        cum += span * score
                        writer.writerow({"instance": n, "example_f1": cum / n})
                metadata = {
                    "config": {"model": model, "seed": 42,
                               "dataset": "numeric_synthetic_demo_NOT_paper",
                               "samples": 3000, "drift_at": 1500, "noise": .05,
                               "grace_period": 200, "delta": 1e-7, "eta": 750,
                               "alpha": .995, "eval_every": 50},
                    "final_metrics": {"example_f1": cum / n, "hamming_loss": .2,
                                      "warnings": 1, "replacements": 1},
                    "wandb_url": None,
                }
                prefix.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ANALYZE), "--seeds", "42", "--input-dir", str(root),
                 "--output-dir", str(root)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root / "numeric_multiseed_summary.json").read_text())
            self.assertAlmostEqual(summary["phases"]["Recovery"]["paired_improvement"]["mean"], .1)
            self.assertAlmostEqual(summary["phases"]["Before drift"]["paired_improvement"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
