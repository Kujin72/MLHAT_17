# Stage 2: independent numeric tree + dynamic leaves

**Status:** incomplete MLHAT rebuild, for demonstration and testing ONLY.

## Implemented here

- 1-by-1 streaming classification and learning, with test-then-train in `experiments/run_stage2.py`.
- Numeric feature observers: online Welford Gaussian moments per feature and per label-present / label-absent state. Thresholds are selected on a bounded evenly spaced grid, and marginal-label entropy / information gain is computed via Gaussian CDF approximations (paper Eqs 5, 10-15).
- Hoeffding split rule (paper Eqs 7-9), first and second best split candidates, grace period. Internal split nodes route by `x[f] <= threshold`.
- Dynamic leaf prediction: pure leaf -> majority labelset; mixed leaf up to `eta` -> bounded-memory Label Powerset kNN; mixed leaf above `eta` -> BR bagged online logistic regression. Bagged models are warm-started before the switch.
- Reproducible seeds, baseline, prequential metrics, local CSV/JSON and optional W&B logging.

## Missing (critical; do NOT describe as full MLHAT)

- Per-node ADWIN for concept drift warning/confirmation.
- Alternate subtrees, significant replacement tests, pruning and prediction mixing (paper Algorithms 1 and 3).
- General Poisson weight for the *tree* itself, distinct from the bagging Poisson weights.
- Categorical feature handling, specialized variants of split/leaf logic and exact reference threshold-bin procedure.
- 41 published datasets, model comparisons and statistical analysis.

## Paper configuration and inconsistency

Table 5 on PDF page 19 reports `delta_split=1e-7`, `grace_period=200`, `delta_alt=0.05`, `kappa_alt=200`, `eta=750`, `lambda=1`, LP-kNN and BR-bagging-LR. Nearby prose unusually refers to alternate `450` and leaf `620`; don't silently replace the table values. Stage2 defaults to the TABLE value of `delta_split=1e-7` and `eta=750` for the numeric demo, but it is NOT a reproduction even with identical defaults.

## Technical design choices that require explicit reporting

- Numerical candidate thresholds are evenly spaced in the observed interval. This is our own documented discretization, not a verified exact reproduction of the paper's binning.
- `kNN` uses Euclidean distance and a bounded FIFO memory of 256 observations per leaf. The paper does not specify this bounded-window length; treat it as an independent engineering setting.
- Online Logistic Regression uses a fixed step size 0.07 and small L2 penalty; these choices must be independently validated for reproduction.
- Online bagging uses Poisson(1) per ensemble member, rather than duplicating the global tree weight.
- The underlying `Eq (5)` is sum of label-wise entropies, not joint entropy, a candidate theory critique rather than proof that the full method fails to capture co-occurrence.

## Running in PowerShell from the project root

```powershell
python -m unittest discover -s tests -v
python .\experiments\run_stage2.py --model stage2 --samples 3000 --seed 42
python .\experiments\run_stage2.py --model majority --samples 3000 --seed 42
```

Optional W&B: `python .\experiments\run_stage2.py --model stage2 --wandb`

Do not compare `numeric_stage2_...` with `synthetic_demo_ht_...` as if the datasets were equal: they are different stream generators. Instead compare `numeric_stage2_stage2_seed42` with `numeric_stage2_majority_seed42` from the same generator.
