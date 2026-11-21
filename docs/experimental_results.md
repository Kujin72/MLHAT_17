# Experimental reproduction log — MLHAT_17

**Status:** Independent **partial** MLHAT-inspired implementation. These are not exact reproductions of the authors' MLHAT results. Original code is not copied.

**Source revision used in all listed runs:** `4e29c5dff8d729c075c37b423e739d83c0dc3d1e`; W&B project: https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction

## 1. Real-world benchmark: Emotions / Yeast

All runs use prequential predict-then-learn evaluation, ARFF order unchanged and `alpha=0.995`. Team's **EWMA Example-F1** is **not yet verified to match the paper's exact faded prequential aggregation**. The data sizes match the paper; original data-source provenance still requires verification.

| Dataset | N / features / labels | Majority EWMA F1 | Stage 2 EWMA F1 | Stage 3 EWMA F1 | Paper reference Example-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Emotions | 593 / 72 / 6 | 0.2647 | 0.4211 | 0.4211 | 0.446 |
| Yeast | 2417 / 103 / 14 | 0.5320 | 0.5829 | 0.5829 | 0.588 |

These values are **descriptive references**, not verified comparable reproductions. On both datasets, `split_count=0`, `warnings=0`, `replacements=0`, so the adaptive-tree mechanism has not been exercised.

- Emotions: [Majority](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/g1f180tz), [Stage 2](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/1uex8d2b), [Stage 3](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/6jyqrd4g). Three seeds (42–44) give identical values for each model with unchanged file ordering; they are not independent repeated trials.
- Yeast: [Majority](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/za36cmdf), [Stage 2](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/006u7atr), [Stage 3](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/oilbpzuc).

Benchmark `delta=1e-5` is the generic baseline setting, **not** Table 5 MLHAT `delta_spl=1e-7`.

## 2. Controlled synthetic concept drift (independent test, NOT paper benchmark)

3,000 instances, drift introduced at instance 1,500; noise 0.05; grace period 200; `delta=1e-7`; eta 750; evaluation every 50, alpha 0.995. Stage 2 and Stage 3 use matched synthetic streams within each seed.

| Seed | Stage 2 Example-F1 | Stage 3 Example-F1 | Improvement | Stage 3 warnings / replacements |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.7590 | 0.7792 | +0.0202 | 4 / 1 |
| 43 | 0.7613 | 0.7754 | +0.0141 | 4 / 2 |
| 44 | 0.7547 | 0.7918 | +0.0371 | 3 / 1 |
| Mean | 0.7583 | 0.7821 | +0.0238 | — |

Mean improvement across matched seeds: **+0.0238** Example-F1 (sample SD of paired improvement ~0.0119). Mean Hamming-loss reduction: **0.0188**. These are descriptive statistics over only 3 seeds; do not claim significance.

| Phase (instances) | Mean Stage 2 F1 | Mean Stage 3 F1 | Mean paired improvement ± sample SD |
| --- | ---: | ---: | ---: |
| Before drift (1–1500) | 0.8203 | 0.8203 | +0.0000 ± 0.0000 |
| Early post-drift (1501–2000) | 0.5178 | 0.5671 | +0.0493 ± 0.0436 |
| Recovery (2001–3000) | 0.7857 | 0.8325 | +0.0468 ± 0.0213 |

Seed 42 Stage 3 replacement: instance 1712 (main loss 0.59 vs alternate 0.18; comparison epsilon ~0.1921). Seed 43 generated a warning at instance 824 **before** the deliberate drift, a possible false alarm; it recorded two replacements after drift (1927, 1928). Stage 3 was better in the recovery phase for all 3 seeds.

Tracked runs:

| Seed | Stage 2 | Stage 3 |
| --- | --- | --- |
| 42 | [W&B](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/xffb70eo) | [W&B](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/t7upn0ef) |
| 43 | [W&B](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/ljy3p2ec) | [W&B](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/upndpipf) |
| 44 | [W&B](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/qszlxvwx) | [W&B](https://forge.coreweave.com/wandb/phtien296-ho-chi-minh-city-university-of-technology/mlhat17-reproduction/runs/1vi68tn1) |

Before milestone submission, verify these URLs are publicly accessible in an incognito browser.

## 3. How to reproduce analysis from saved local runs

Run the following commands from the repository root:

```powershell
# Source models and benchmark runners were committed earlier.
python -m unittest discover -s tests -v

# Only if synthetic runs are NOT already present:
.\experiments\run_drift_multiseed.ps1 -Seeds @(42,43,44)
# Add -Track for fresh W&B runs; otherwise no network logging. If previous results exist, explicitly add -Overwrite to permit replacing local JSON/CSV.

# Analyze the existing six synthetic CSV/JSON run pairs:
python .\experiments\analyze_drift.py --seeds 42 43 44

# Recreate the visual:
python -m pip install matplotlib
python .\experiments\plot_drift_recovery.py --seed 42
```

Saved outputs under ignored `results/`: `numeric_multiseed_phase_comparison.csv`, `numeric_multiseed_summary.json`, `drift_recovery_seed42.png`. To publish the figure, **copy it into `docs/figures/`**; do not override `.gitignore` to track entire results or data folders.

## 4. Limitations and future work

- Stage 3 uses ADWIN-inspired drift monitoring, not exact ADWIN; its leaf learners, Poisson(1) resampling, prediction mixing and branch-replacement confidence tests are not exact paper MLHAT.
- Synthetic drift is an independent diagnostic stream, not a dataset from the paper; do not treat it as direct reproduction of a published table.
- Verify data source and preprocessing, use the paper's tuned hyperparameters where appropriate, and implement the remaining MLHAT components before claiming full reproduction.
- The earlier README summarizes the initial Stage 1 baseline; this document covers the later Stage 2/3 experimental extension.
