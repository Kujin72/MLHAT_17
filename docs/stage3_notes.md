# Stage 3 — experimental drift adaptation (NOT paper reproduction)

Reference: uploaded Esteban et al. PDF, Sections 3.5–3.6 and Algorithms 1, 3.

## Added capabilities

- **At each visited node**, watch *prequential Hamming loss* from the node's current main subtree.
- **Approximate ADWIN warning:** use a sliding window and variance-sensitive (Bernstein-style) cut test. Warning at a relaxed confidence threshold, then grow a fresh alternate Stage 2 subtree at that node.
- **Parallel learning:** continue training the original Stage 2 main tree and the affected node's alternate subtree on incoming examples. Predictions for error comparison are made **before** updating either model.
- **After `alternate_grace` comparable samples:** use a conservative Hoeffding confidence radius on the paired loss sequence to replace the obsolete branch or discard the alternate.
- **Optional prediction combination:** `--blend-alternates` combines *hard label votes* using observed loss weights after the alternate has seen enough samples.
- JSON output logs warning, replacement and prune events so a reviewer can audit drift reaction.

## Exact-fidelity gaps (must resolve before Lab 1 paper reproduction)

1. This *uncompressed, window-limited, periodically checked ADWIN-inspired detector* is **not the exact ADWIN algorithm** used by the paper. Replace it with a validated faithful implementation later.
2. Branch comparison uses a conservative **distribution-free** Hoeffding threshold for bounded paired losses, not the paper's Eq. (16). This is deliberate and documented; do not mislabel it as exact paper implementation.
3. Prediction combination is **hard-vote weighted** rather than probability-based as in Algorithm 3.
4. Main-tree Poisson(1) training weights are not implemented; Stage 2 Bagged LR members already have separate Poisson weights.
5. Nominal/categorical split handling, full benchmark datasets, benchmark protocols, full metric suite and exact ADWIN bagging leaf learner remain incomplete.
6. Running `--model stage2` via this script provides a same-stream control with the same Stage 2 defaults; the JSON still explicitly identifies that control.
7. The paper's Table 5 and adjacent prose disagree on some hyperparameters. Record exact choice in each run; current Stage 3 defaults follow Table 5 where possible.

## Scientific interpretation

The synthetic stream explicitly reverses a target relationship after its midpoint; it is **not a dataset from the paper**.
The two models should be compared with fixed seed, number of samples, drift point and noise. Metric improvement by this approximate Stage 3 is an **engineering result on this one synthetic run**, not evidence that the published MLHAT's results were reproduced. Verify with several seeds and no-drift controls before drawing conclusions about stability.

## Commands

```powershell
python -m unittest discover -s tests -v
python .\experiments\run_stage3.py --model stage2 --samples 3000 --seed 42
python .\experiments\run_stage3.py --model stage3 --samples 3000 --seed 42
python .\experiments\run_stage3.py --model stage3 --samples 3000 --seed 42 --blend-alternates
```

Results: `results/numeric_stage3_{stage2|stage3}_seed42.{csv,json}`. Blended and unblended runs share output file name at the same seed: **save or rename one before rerunning with the blending flag**, or extend the CLI for unique experiment IDs.
