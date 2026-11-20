"""Prequential *numeric synthetic demo*, NOT published MLHAT reproduction.

Usage: python experiments/run_stage2.py --model stage2 --samples 3000 --seed 42
       python experiments/run_stage2.py --model majority --samples 3000 --seed 42
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mlhat_rebuild.metrics import RunningMetrics
from mlhat_rebuild.numeric_stream import numeric_multilabel_stream
from mlhat_rebuild.stage2 import OnlineMultilabelStage2Tree
from mlhat_rebuild.tree import OnlineMajorityLabelset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['stage2', 'majority'], default='stage2')
    p.add_argument('--samples', type=int, default=3000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise', type=float, default=0.05)
    p.add_argument('--drift-at', type=int, default=-1)
    p.add_argument('--grace-period', type=int, default=200)
    p.add_argument('--delta', type=float, default=1e-7)
    p.add_argument('--eta', type=int, default=750)
    p.add_argument('--max-knn-memory', type=int, default=256)
    p.add_argument('--ensemble-size', type=int, default=10)
    p.add_argument('--n-bins', type=int, default=8)
    p.add_argument('--alpha', type=float, default=0.995)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--project', default='mlhat-independent-rebuild')
    args = p.parse_args()
    if args.samples < 1 or args.eval_every < 1:
        p.error('--samples and --eval-every must be >= 1')
    drift_at = args.drift_at if args.drift_at >= 0 else args.samples // 2
    model = (OnlineMultilabelStage2Tree(n_features=4, n_labels=2,
                                        grace_period=args.grace_period, delta=args.delta,
                                        eta=args.eta, max_knn_memory=args.max_knn_memory,
                                        ensemble_size=args.ensemble_size,
                                        n_bins=args.n_bins, seed=args.seed)
             if args.model == 'stage2' else OnlineMajorityLabelset(n_labels=2))
    metrics = RunningMetrics(alpha=args.alpha)
    psha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False)
    cfg = {
        'dataset': 'numeric_synthetic_demo_NOT_paper', 'model': args.model,
        'samples': args.samples, 'seed': args.seed, 'noise': args.noise, 'drift_at': drift_at,
        'grace_period': args.grace_period, 'delta': args.delta, 'eta': args.eta,
        'max_knn_memory': args.max_knn_memory, 'ensemble_size': args.ensemble_size,
        'n_bins': args.n_bins, 'alpha': args.alpha, 'eval_every': args.eval_every,
        'git_sha': psha.stdout.strip() if psha.returncode == 0 else 'uncommitted',
        'python': platform.python_version(),
        'implementation_status': 'stage2_numeric_and_leaf_learners_no_drift_adaptation'
    }
    tracking = None
    if args.wandb:
        import wandb
        tracking = wandb.init(project=args.project, config=cfg, job_type='stage2_demo',
                              name=f'stage2-numeric-{args.model}-seed{args.seed}')
    tic = time.perf_counter()
    rows = []
    mode_counts = {}
    for i, (x, y) in enumerate(numeric_multilabel_stream(args.samples, drift_at, args.noise, args.seed), 1):
        if args.model == 'stage2':
            mode = model.leaf_mode(x)
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        pred = model.predict_one(x)  # TEST before TRAIN; avoids future-label leakage
        metrics.update(y, pred)
        model.learn_one(x, y)
        if i % args.eval_every == 0 or i == args.samples:
            row = {'instance': i, **metrics.summary(), 'split_count': getattr(model, 'n_splits', 0)}
            rows.append(row)
            if tracking:
                tracking.log(row, step=i)
    duration = time.perf_counter() - tic
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    basename = f'numeric_stage2_{args.model}_seed{args.seed}'
    with (out / f'{basename}.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {'config': cfg, 'final_metrics': rows[-1], 'elapsed_seconds': duration,
               'leaf_mode_counts': mode_counts, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'wandb_url': tracking.url if tracking else None,
               'note': 'Independent Stage 2 synthetic test, NOT a reproduction of MLHAT.'}
    (out / f'{basename}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if tracking:
        tracking.summary.update(rows[-1])
        tracking.finish()
    print(json.dumps(summary, indent=2))
    print('CSV:', out / f'{basename}.csv')


if __name__ == '__main__':
    main()
