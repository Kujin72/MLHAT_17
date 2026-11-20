"""Run a demonstration with test-then-train and reproducible metrics.

Example:
  python experiments/run_stream.py --model ht --samples 3000 --seed 42
  python experiments/run_stream.py --model ht --wandb

Scores from synthetic_demo are NOT published-paper reproduction results.
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlhat_rebuild.metrics import RunningMetrics
from mlhat_rebuild.stream import synthetic_multilabel_stream
from mlhat_rebuild.tree import OnlineMajorityLabelset, OnlineMultilabelHoeffdingTree


def git_sha():
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else "uncommitted"


def run(args):
    model = (OnlineMultilabelHoeffdingTree(n_features=4, n_labels=2,
                                          grace_period=args.grace_period, delta=args.delta)
             if args.model == "ht" else OnlineMajorityLabelset(n_labels=2))
    metrics = RunningMetrics(alpha=args.alpha)
    drift_at = args.drift_at if args.drift_at >= 0 else args.samples // 2
    cfg = {
        "dataset": "synthetic_demo_NOT_paper", "model": args.model, "samples": args.samples,
        "seed": args.seed, "noise": args.noise, "drift_at": drift_at,
        "grace_period": args.grace_period, "delta": args.delta,
        "forgetting_factor": args.alpha, "eval_every": args.eval_every,
        "git_sha": git_sha(), "python": platform.python_version(),
        "implementation_status": "stage1_nonadaptive_baseline",
    }
    tracking = None
    if args.wandb:
        try:
            import wandb
        except ImportError as e:
            raise SystemExit("Install W&B with: python -m pip install -e '.[tracking]'") from e
        tracking = wandb.init(project=args.project, config=cfg, job_type="stage1_demo",
                              name=f"demo-{args.model}-seed{args.seed}")
    rows = []
    for index, (x, y) in enumerate(synthetic_multilabel_stream(
            n=args.samples, drift_at=drift_at, noise=args.noise, seed=args.seed), 1):
        pred = model.predict_one(x)   # TEST FIRST. No future labels are used here.
        metrics.update(y, pred)
        model.learn_one(x, y)         # TRAIN AFTER recording prediction.
        if index % args.eval_every == 0 or index == args.samples:
            row = {"instance": index, **metrics.summary(),
                   "split_count": getattr(model, "n_splits", 0)}
            rows.append(row)
            if tracking is not None:
                tracking.log(row, step=index)
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    run_name = f"synthetic_demo_{args.model}_seed{args.seed}"
    csv_path = output_dir / f"{run_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    meta = {"config": cfg, "final_metrics": rows[-1], "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "wandb_url": tracking.url if tracking else None,
            "note": "Synthetic demonstration only. NOT a published-paper reproduction."}
    (output_dir / f"{run_name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if tracking is not None:
        tracking.summary.update(rows[-1]); tracking.finish()
    print(json.dumps(meta, indent=2)); print(f"CSV: {csv_path}")
    return rows[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ht", "majority"], default="ht")
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--drift-at", type=int, default=-1)
    p.add_argument("--grace-period", type=int, default=200)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=0.995)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="mlhat-independent-rebuild")
    args = p.parse_args()
    if args.samples < 1 or args.eval_every < 1:
        p.error("--samples and --eval-every must be positive")
    run(args)


if __name__ == "__main__":
    main()
