"""Prequential benchmark on real paper dataset with transparent partial-reproduction status.

Runs the *independent partial* Stage 2 or Stage 3 implementation, not the
original paper's reference implementation. This module never claims exact MLHAT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlhat_rebuild.benchmark_data import PAPER_DATASETS, read_benchmark_arff, verify_paper_shape
from mlhat_rebuild.metrics import RunningMetrics
from mlhat_rebuild.stage2 import OnlineMultilabelStage2Tree
from mlhat_rebuild.stage3 import OnlineMultilabelStage3Tree


class MajorityBaseline:
    def __init__(self, n_labels):
        self.n_labels = n_labels
        self.counts = Counter()
        self.n_splits = 0

    def predict_one(self, x):
        if not self.counts:
            return (0,) * self.n_labels
        return min(self.counts, key=lambda y: (-self.counts[y], y))

    def learn_one(self, x, y):
        self.counts[tuple(y)] += 1


def git_state():
    def run(*cmd):
        try:
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        except OSError:
            return None
        return p.stdout.strip() if p.returncode == 0 else None
    return run("git", "rev-parse", "HEAD") or "uncommitted", bool(run("git", "status", "--porcelain"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arff", type=Path, required=True, help="Full ARFF (all benchmark instances)")
    p.add_argument("--xml", type=Path, help="Mulan label XML; omit for MEKA -C format")
    p.add_argument("--dataset", required=True, choices=("Flags", "Emotions", "Yeast"))
    p.add_argument("--model", default="stage3", choices=("stage2", "stage3", "majority"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grace-period", type=int, default=200)
    p.add_argument("--delta", type=float, default=1e-5, help="Paper comparison configuration, see Section 4.1.1")
    p.add_argument("--eta", type=int, default=750)
    p.add_argument("--alpha", type=float, default=0.995)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="mlhat17-reproduction")
    p.add_argument("--entity", default=None, help="Optional W&B username/team")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "lab1")
    args = p.parse_args()
    if args.eval_every < 1:
        p.error("eval-every must be positive")

    data = read_benchmark_arff(args.arff, args.dataset, args.xml)
    same_shape, shape_differences = verify_paper_shape(data)
    if not same_shape:
        p.error("NOT the full paper benchmark dataset: " + "; ".join(shape_differences))

    params = dict(n_features=data.encoded_features, n_labels=data.n_labels,
                  grace_period=args.grace_period, delta=args.delta, eta=args.eta, seed=args.seed)
    if args.model == "stage3":
        model = OnlineMultilabelStage3Tree(**params)
    elif args.model == "stage2":
        model = OnlineMultilabelStage2Tree(**params)
    else:
        model = MajorityBaseline(data.n_labels)

    sha, dirty = git_state()
    info = PAPER_DATASETS[args.dataset.lower()]
    config = dict(dataset=args.dataset, model=args.model, samples=len(data.records), seed=args.seed,
                  n_features_raw=data.raw_features, n_features_encoded=data.encoded_features,
                  n_labels=data.n_labels, label_names=data.label_names, arff_sha256=data.arff_sha256,
                  preprocessing=data.encoding, data_order="ARFF file order, unchanged",
                  grace_period=args.grace_period, delta=args.delta, eta=args.eta,
                  alpha=args.alpha, eval_every=args.eval_every, git_sha=sha,
                  git_dirty=dirty, python=platform.python_version(),
                  implementation="independent_stage3_approximation_NOT_full_MLHAT",
                  paper_reference="Esteban et al. (2024), Table 6, Example F1",
                  paper_example_f1=info["paper_example_f1"])

    tracking = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install W&B: python -m pip install wandb") from exc
        tracking = wandb.init(project=args.project, entity=args.entity,
                              name=f"{args.dataset.lower()}-{args.model}-seed{args.seed}",
                              job_type="lab1-partial-reproduction", config=config,
                              tags=["paper-benchmark", "partial-reproduction", args.dataset.lower()])
    evaluator = RunningMetrics(alpha=args.alpha)
    history = []
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(data.records, 1):
        pred = model.predict_one(x)
        evaluator.update(y, pred)
        model.learn_one(x, y)
        if i % args.eval_every == 0 or i == len(data.records):
            rec = {"instance": i, **evaluator.summary(),
                   "split_count": model.n_splits,
                   "warnings": getattr(model, "n_warnings", 0),
                   "replacements": getattr(model, "n_replacements", 0)}
            history.append(rec)
            if tracking:
                tracking.log(rec, step=i)
    elapsed = time.perf_counter() - t0
    final = history[-1]
    paper_f1 = info["paper_example_f1"]
    result = {
        "status": "PARTIAL_REPRODUCTION_NOT_EXACT_PAPER_MLHAT",
        "dataset_shape_matches_paper": True, "dataset_authenticity_requires_source_check": True, "paper_reference": {"dataset": args.dataset,
                                                      "example_f1": paper_f1,
                                                      "source": "Esteban et al., Table 6"},
        "team_result": dict(final),
        "indicative_gap_cumulative_f1": final["example_f1"] - paper_f1,
        "indicative_gap_ewma_f1": final["ewma_example_f1"] - paper_f1,
        "comparison_warning": "Paper uses prequential with fading alpha=.995; this rebuild tracks cumulative and EWMA. Original algorithm, prediction aggregation, feature encoding, and metric aggregation may differ; gaps are descriptive, not validated exact replication.",
        "config": config,
        "elapsed_seconds": elapsed,
        "wandb_url": tracking.url if tracking else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": getattr(model, "events", []),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.dataset.lower()}_{args.model}_seed{args.seed}"
    csv_path = args.out / f"{prefix}.csv"
    json_path = args.out / f"{prefix}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if tracking:
        tracking.summary.update({
            "team_example_f1_cumulative": final["example_f1"],
            "team_example_f1_ewma": final["ewma_example_f1"],
            "paper_example_f1": paper_f1,
            "indicative_gap_ewma": result["indicative_gap_ewma_f1"],
            "dataset_sha256": data.arff_sha256,
            "comparison_status": result["status"],
        })
        artifact = wandb.Artifact(f"{prefix}-outputs", type="experiment-results")
        artifact.add_file(str(csv_path))
        artifact.add_file(str(json_path))
        tracking.log_artifact(artifact)
        tracking.finish()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
