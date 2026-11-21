"""Plot Stage 2/Stage 3 EWMA Example-F1 and recorded replacement events.

Uses existing outputs from run_stage3.py. Requires matplotlib:
    python -m pip install matplotlib
    python experiments/plot_drift_recovery.py --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_history(path: Path) -> tuple[list[int], list[float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return ([int(r["instance"]) for r in rows],
            [float(r["ewma_example_f1"]) for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        parser.error("Install matplotlib first: python -m pip install matplotlib")
        raise exc

    stage3_json = args.input_dir / f"numeric_stage3_stage3_seed{args.seed}.json"
    meta = json.loads(stage3_json.read_text(encoding="utf-8"))
    if meta["config"]["seed"] != args.seed:
        raise ValueError("Seed mismatch in stage3 metadata")
    drift_at = int(meta["config"]["drift_at"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model, label in (("stage2", "Stage 2 — no drift adaptation"),
                         ("stage3", "Stage 3 — adaptive")):
        csv_path = args.input_dir / f"numeric_stage3_{model}_seed{args.seed}.csv"
        x, f1 = read_history(csv_path)
        ax.plot(x, f1, linewidth=2, label=label)

    ax.axvline(drift_at, linestyle="--", label=f"Concept drift ({drift_at})")
    replacements = [int(event["instance"]) for event in meta.get("events", [])
                    if event.get("type") == "replacement"]
    for i, at in enumerate(replacements):
        ax.axvline(at, linestyle=":", linewidth=1.7,
                   label="Stage 3 replacement" if i == 0 else None)
    ax.set(xlabel="Processed instances", ylabel="EWMA Example-F1",
           title=f"Controlled drift — seed {args.seed}", ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output = args.output or args.input_dir / f"drift_recovery_seed{args.seed}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
