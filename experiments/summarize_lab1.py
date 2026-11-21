"""Summarize benchmark runs with paper values and corresponding W&B URLs."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, default=Path('results/lab1'))
    p.add_argument('--out', type=Path, default=Path('results/lab1/comparison.csv'))
    args = p.parse_args()
    rows = []
    for fp in sorted(args.input_dir.glob('*.json')):
        obj = json.loads(fp.read_text(encoding='utf-8'))
        if obj.get('status') != 'PARTIAL_REPRODUCTION_NOT_EXACT_PAPER_MLHAT':
            continue
        cfg = obj['config']; met = obj['team_result']; p_ref = obj['paper_reference']
        rows.append({
            'dataset': cfg['dataset'], 'model': cfg['model'], 'seed': cfg['seed'],
            'samples': cfg['samples'], 'paper_example_f1': p_ref['example_f1'],
            'team_example_f1_cumulative': round(met['example_f1'], 6),
            'team_example_f1_ewma': round(met['ewma_example_f1'], 6),
            'indicative_gap_ewma': round(obj['indicative_gap_ewma_f1'], 6),
            'team_hamming_loss': round(met['hamming_loss'], 6),
            'wandb_url': obj.get('wandb_url') or 'MISSING_RUN_URL',
            'git_sha': cfg['git_sha'],
            'dataset_sha256': cfg['arff_sha256'],
            'status': 'PARTIAL_NOT_COMPARABLE_TO_EXACT_MLHAT',
        })
    if not rows:
        raise SystemExit('No benchmark JSON found. Run experiments/run_paper_benchmark.py first.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('Dataset     Model    Paper F1   Team EWMA F1  Gap       W&B')
    for r in rows:
        print(f"{r['dataset']:<11} {r['model']:<8} {r['paper_example_f1']:.3f}      {r['team_example_f1_ewma']:.4f}        {r['indicative_gap_ewma']:+.4f}   {r['wandb_url']}")
    print(f'CSV: {args.out}')


if __name__ == '__main__': main()
