"""Stage 3 diagnostic demo; NOT a reproduction of published MLHAT.

Examples:
  python experiments/run_stage3.py --model stage3 --samples 3000 --seed 42
  python experiments/run_stage3.py --model stage2 --samples 3000 --seed 42
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
from mlhat_rebuild.stage3 import OnlineMultilabelStage3Tree


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['stage2', 'stage3'], default='stage3')
    p.add_argument('--samples', type=int, default=3000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise', type=float, default=0.05)
    p.add_argument('--drift-at', type=int, default=-1)
    p.add_argument('--grace-period', type=int, default=200)
    p.add_argument('--delta', type=float, default=1e-7)
    p.add_argument('--eta', type=int, default=750)
    p.add_argument('--alternate-grace', type=int, default=200)
    p.add_argument('--delta-drift', type=float, default=0.01)
    p.add_argument('--warning-delta', type=float, default=0.2)
    p.add_argument('--delta-alt', type=float, default=0.05)
    p.add_argument('--monitor-window', type=int, default=256)
    p.add_argument('--blend-alternates', action='store_true')
    p.add_argument('--alpha', type=float, default=0.995)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--project', default='mlhat-independent-rebuild')
    args = p.parse_args()
    if args.samples < 1 or args.eval_every < 1 or not 0 <= args.noise <= 1:
        p.error('invalid samples/eval/noise configuration')
    drift_at = args.drift_at if args.drift_at >= 0 else args.samples // 2
    params = dict(n_features=4, n_labels=2, grace_period=args.grace_period,
                  delta=args.delta, eta=args.eta, seed=args.seed)
    model = (OnlineMultilabelStage3Tree(**params, alternate_grace=args.alternate_grace,
                                        delta_drift=args.delta_drift,
                                        warning_delta=args.warning_delta,
                                        delta_alt=args.delta_alt,
                                        monitor_window=args.monitor_window,
                                        blend_alternates=args.blend_alternates)
             if args.model == 'stage3' else OnlineMultilabelStage2Tree(**params))
    metrics = RunningMetrics(alpha=args.alpha)
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                         capture_output=True, text=True, check=False)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                           capture_output=True, text=True, check=False)
    config = vars(args).copy()
    config.update({'dataset': 'numeric_synthetic_demo_NOT_paper', 'drift_at': drift_at,
                   'git_sha': sha.stdout.strip() if sha.returncode == 0 else 'uncommitted',
                   'git_dirty': bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                   'python': platform.python_version(),
                   'implementation_status': 'stage3_adwin_inspired_approximation_NOT_full_MLHAT'})
    tracking = None
    if args.wandb:
        import wandb
        tracking = wandb.init(project=args.project, config=config, job_type='stage3_demo',
                              name=f'{args.model}-numeric-seed{args.seed}')
    started = time.perf_counter()
    history = []
    for i, (x, y) in enumerate(numeric_multilabel_stream(args.samples, drift_at, args.noise, args.seed), 1):
        prediction = model.predict_one(x)
        metrics.update(y, prediction)
        model.learn_one(x, y)
        if i % args.eval_every == 0 or i == args.samples:
            record = {'instance': i, **metrics.summary(),
                      'split_count': model.n_splits,
                      'warnings': getattr(model, 'n_warnings', 0),
                      'replacements': getattr(model, 'n_replacements', 0),
                      'alternate_prunes': getattr(model, 'n_pruned_alternates', 0)}
            history.append(record)
            if tracking:
                tracking.log(record, step=i)
    elapsed = time.perf_counter() - started
    result_dir = ROOT / 'results'
    result_dir.mkdir(exist_ok=True)
    name = f'numeric_stage3_{args.model}_seed{args.seed}'
    with (result_dir / f'{name}.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    result = {'config': config, 'final_metrics': history[-1],
              'elapsed_seconds': elapsed, 'events': getattr(model, 'events', []),
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'wandb_url': tracking.url if tracking else None,
              'note': 'Approximate drift-adaptive demo. NOT MLHAT paper reproduction.'}
    (result_dir / f'{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if tracking:
        tracking.summary.update(history[-1])
        tracking.finish()
    print(json.dumps(result, indent=2))
    print('CSV:', result_dir / f'{name}.csv')


if __name__ == '__main__':
    main()
