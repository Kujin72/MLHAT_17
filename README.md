# MLHAT_17 — Independent MLHAT-Inspired Reimplementation

Introduction to Machine Learning term project. Independent, staged implementation inspired by Esteban et al., *Hoeffding adaptive trees for multi-label classification on data streams* ([arXiv:2410.20242](https://arxiv.org/abs/2410.20242)).

> **Scientific status: PARTIAL REPRODUCTION, NOT EXACT PAPER MLHAT.** Our Stage 3 uses an ADWIN-inspired window detector, custom leaf classifiers, alternate-tree replacement and a Hoeffding-style comparator. Full paper-specific Poisson bootstrap, precise ADWIN, probabilistic blending, exact numerical observers and matching all evaluation metrics are not yet implemented. The synthetic drift demonstration is our own, not a published benchmark. We did not copy the authors' implementation.

## Implemented components

| Module | Purpose | Status |
| --- | --- | --- |
| `src/mlhat_rebuild/tree.py` | Stage 1 binary-feature Hoeffding tree | Basic prototype |
| `src/mlhat_rebuild/entropy.py` | Entropy and Hoeffding split bound | Independent implementation |
| `src/mlhat_rebuild/numeric_observer.py` | Numeric split candidate observer | Approximate |
| `src/mlhat_rebuild/online_learners.py` | LP-kNN and bagged logistic leaf learners | Approximate |
| `src/mlhat_rebuild/stage2.py` | Numeric incremental tree with adaptive leaf choice | Partial |
| `src/mlhat_rebuild/drift.py` | Adaptive loss monitoring | ADWIN-inspired, not exact |
| `src/mlhat_rebuild/stage3.py` | Alternate branches and replacement | Partial |
| `src/mlhat_rebuild/benchmark_data.py` | ARFF/XML real-benchmark loader | Implemented, provenance to verify |
| `src/mlhat_rebuild/metrics.py` | Test-then-train metrics; EWMA | Implemented, paper equivalence unverified |

## Setup (Windows PowerShell)

```powershell
cd MLHAT_17
python -m pip install -e ".[tracking]"
python -m unittest discover -s tests -v
```

Python >= 3.10. GPU is not required. The local virtual environment, datasets, W&B credentials and raw results are excluded from Git.

## Experiments

### Real-world data: Emotions / Yeast

Place full ARFF/XML pairs under `data/`. They are not distributed in this repo. Download from the relevant multi-label dataset provider and record the provenance and checksum.

```powershell
python .\experiments\run_paper_benchmark.py --dataset Emotions --arff .\data\emotions.arff --xml .\data\emotions.xml --model stage3 --seed 42 --wandb --project mlhat17-reproduction
python .\experiments\run_paper_benchmark.py --dataset Yeast --arff .\data\yeast.arff --xml .\data\yeast.xml --model stage3 --seed 42 --wandb --project mlhat17-reproduction
python .\experiments\summarize_lab1.py
```

Benchmark runner currently defaults to generic baseline `delta=1e-5`, whereas paper Table 5 specifies tuned MLHAT `delta_spl=1e-7`; set `--delta 1e-7` explicitly for an additional *matched-parameter sensitivity experiment*, not a replacement of the earlier run. Original paper scores and our EWMA aggregation are **not proven metric-equivalent**.

### Controlled concept drift (independent synthetic stream)

The diagnostic stream has 3,000 instances with drift at sample 1,500. Recreate paired Stage 2/3 runs on seeds 42, 43, 44 **only if the local results are absent**, or deliberately provide `-Overwrite`. Use `-Track` to log new runs to W&B.

```powershell
.\experiments\run_drift_multiseed.ps1 -Seeds @(42,43,44) -Track

# Re-analyze existing runs without training or W&B calls:
python .\experiments\analyze_drift.py --seeds 42 43 44

# Optional graph from saved CSV/JSON results:
python -m pip install matplotlib
python .\experiments\plot_drift_recovery.py --seed 42
```

Generated CSV/JSON/PNG files go to ignored `results/`. To publish the graph, copy it to `docs/figures/` and commit that selected asset. Do **not** commit raw datasets, credentials, `wandb/` or the whole `results/` folder.

## Experimental findings (preliminary)

- Emotions: Stage 3 EWMA Example-F1 **0.4211**, Majority **0.2647** (paper MLHAT Example-F1 reference **0.446**, not strictly comparable).
- Yeast: Stage 3 EWMA Example-F1 **0.5829**, Majority **0.5320** (paper reference **0.588**, not strictly comparable).
- On these two real datasets, Stage 2 and Stage 3 give identical metrics with zero splits and replacements. This does **not** validate adaptive-tree behavior.
- Synthetic controlled drift, 3 independent seeds: Stage 3 cumulative Example-F1 mean **0.7821** vs Stage 2 **0.7583**; mean paired improvement **+0.0238**. Early post-drift mean improvement **+0.0493**, recovery period **+0.0468**. Every Stage 3 run generated a branch replacement. Results are descriptive for three seeds, not a significance claim.

**Detailed tables, methodology, run URLs, limitations:** [docs/experimental_results.md](docs/experimental_results.md)

[W&B project](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction) · [Paper notes](docs/paper_notes.md) · [Requirements mapping](docs/requirements_mapping.md)

## Important limitations

This is not a complete reimplementation of MLHAT. Unfinished work includes faithful numerical/categorical observers, the exact paper leaf ensemble and bootstrapping, ADWIN and alternate confidence testing, probability prediction aggregation, source-data verification, matched fading evaluation, and full published metric suite. The authors' [reference repository](https://github.com/aestebant/mlhat) is for conceptual cross-checking only, not for copying code.
