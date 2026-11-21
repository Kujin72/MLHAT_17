"""Analyze paired Stage 2/Stage 3 test-then-train metrics by drift phase.

Consumes existing CSV/JSON outputs from experiments/run_stage3.py. This is an
independent synthetic-drift diagnostic, NOT a paper MLHAT reproduction.

Example:
    python experiments/analyze_drift.py --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def cumulative_history(path: Path) -> dict[int, float]:
    """Return cumulative example-F1 indexed by instance checkpoint."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        values = {int(row["instance"]): float(row["example_f1"])
                  for row in csv.DictReader(handle)}
    if not values:
        raise ValueError(f"No checkpoints in {path}")
    return values


def interval_f1(history: dict[int, float], start: int, end: int) -> float:
    """Recover the mean per-instance F1 over (start, end] from cumulative means."""
    if not 0 <= start < end:
        raise ValueError("Require 0 <= start < end")
    if end not in history or (start != 0 and start not in history):
        raise ValueError(f"Missing checkpoints for interval ({start}, {end}]")
    end_total = end * history[end]
    start_total = start * history[start] if start else 0.0
    return (end_total - start_total) / (end - start)


def load_run(root: Path, model: str, seed: int) -> tuple[dict, dict[int, float]]:
    prefix = f"numeric_stage3_{model}_seed{seed}"
    metadata = json.loads((root / f"{prefix}.json").read_text(encoding="utf-8"))
    history = cumulative_history(root / f"{prefix}.csv")
    config = metadata["config"]
    if config["seed"] != seed or config["model"] != model:
        raise ValueError(f"Model/seed mismatch in {prefix}.json")
    if config["dataset"] != "numeric_synthetic_demo_NOT_paper":
        raise ValueError(f"Expected synthetic demo, received {config['dataset']}")
    return metadata, history


def verify_paired(stage2: dict, stage3: dict, samples: int, drift_at: int) -> None:
    """Check that both models saw the same experimental setup."""
    config2 = stage2["config"]
    config3 = stage3["config"]
    keys = ("samples", "seed", "noise", "drift_at", "grace_period",
            "delta", "eta", "alpha", "eval_every")
    differences = [name for name in keys if config2[name] != config3[name]]
    if differences:
        raise ValueError(f"Stage2/Stage3 configuration mismatch: {differences}")
    if config2["samples"] != samples or config2["drift_at"] != drift_at:
        raise ValueError("Requested sample/drift boundaries mismatch run metadata")


def summarize(values: list[float]) -> dict[str, float | int]:
    return {"n": len(values), "mean": mean(values),
            "sample_sd": stdev(values) if len(values) >= 2 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--drift-at", type=int, default=1500)
    parser.add_argument("--early-end", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=3000)
    args = parser.parse_args()
    if not (0 < args.drift_at < args.early_end < args.samples):
        parser.error("Require 0 < drift-at < early-end < samples")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique")

    phases = ((0, args.drift_at, "Before drift"),
              (args.drift_at, args.early_end, "Early post-drift"),
              (args.early_end, args.samples, "Recovery"))
    phase_rows: list[dict] = []
    run_rows: list[dict] = []
    for seed in args.seeds:
        stage2_meta, stage2_history = load_run(args.input_dir, "stage2", seed)
        stage3_meta, stage3_history = load_run(args.input_dir, "stage3", seed)
        verify_paired(stage2_meta, stage3_meta, args.samples, args.drift_at)

        for start, end, name in phases:
            f2 = interval_f1(stage2_history, start, end)
            f3 = interval_f1(stage3_history, start, end)
            phase_rows.append({"seed": seed, "phase": name, "start": start + 1,
                               "end": end, "stage2_f1": f2, "stage3_f1": f3,
                               "improvement": f3 - f2})

        m2 = stage2_meta["final_metrics"]
        m3 = stage3_meta["final_metrics"]
        run_rows.append({"seed": seed, "stage2_f1": m2["example_f1"],
                         "stage3_f1": m3["example_f1"],
                         "f1_improvement": m3["example_f1"] - m2["example_f1"],
                         "stage2_hamming_loss": m2["hamming_loss"],
                         "stage3_hamming_loss": m3["hamming_loss"],
                         "hamming_reduction": m2["hamming_loss"] - m3["hamming_loss"],
                         "warnings": m3["warnings"], "replacements": m3["replacements"],
                         "stage2_wandb": stage2_meta.get("wandb_url"),
                         "stage3_wandb": stage3_meta.get("wandb_url")})

    summaries = {}
    for _, _, name in phases:
        matching = [r for r in phase_rows if r["phase"] == name]
        summaries[name] = {
            "stage2_f1": summarize([r["stage2_f1"] for r in matching]),
            "stage3_f1": summarize([r["stage3_f1"] for r in matching]),
            "paired_improvement": summarize([r["improvement"] for r in matching]),
        }
    overall = {
        "stage2_f1": summarize([r["stage2_f1"] for r in run_rows]),
        "stage3_f1": summarize([r["stage3_f1"] for r in run_rows]),
        "paired_improvement": summarize([r["f1_improvement"] for r in run_rows]),
        "hamming_reduction": summarize([r["hamming_reduction"] for r in run_rows]),
    }
    output = {"note": "Synthetic controlled drift; independent approximate model, NOT exact MLHAT.",
              "seeds": args.seeds, "samples": args.samples, "drift_at": args.drift_at,
              "early_end": args.early_end, "overall": overall, "phases": summaries,
              "runs": run_rows}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_csv = args.output_dir / "numeric_multiseed_phase_comparison.csv"
    with phase_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)
    summary_json = args.output_dir / "numeric_multiseed_summary.json"
    summary_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("PER-SEED RESULTS")
    for row in phase_rows:
        print(f"Seed {row['seed']} | {row['phase']:<18} | "
              f"S2={row['stage2_f1']:.4f} | S3={row['stage3_f1']:.4f} | "
              f"Delta={row['improvement']:+.4f}")
    print("\nMEAN PAIRED IMPROVEMENTS (sample SD)")
    for name, stats in summaries.items():
        imp = stats["paired_improvement"]
        print(f"{name:<18}: {imp['mean']:+.4f} +/- {imp['sample_sd']:.4f}")
    print(f"\nOverall F1 improvement: {overall['paired_improvement']['mean']:+.4f}")
    print(f"Saved: {phase_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
