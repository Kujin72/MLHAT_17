# MLHAT independent rebuild — Introduction to Machine Learning

> **STATUS:** Stage 1 technical baseline complete. **This is NOT a complete MLHAT
> implementation and NOT a reproduction of the paper's reported results.**
> All source code in this starter was newly authored for the staged project.

## Sources

1. Esteban et al., *Hoeffding adaptive trees for multi-label classification on data streams*,
   arXiv:2410.20242v1. Use Sections 2–4 and Algorithms 1–3.
2. Course handout `index.md` — Term Project: Overview.

The course prohibits copying code from the authors' public repository, their
figures or text, and another team's submission. The repository below is for
**reference and cross-checking only**: https://github.com/aestebant/mlhat.

## System requirements

- Python 3.10+; runs on Windows PowerShell or Linux.
- No GPU needed for this first-stage baseline. W&B is optional locally but
  **required for course experiments** by the supplied overview.

## Quick start (PowerShell)

```powershell
cd .\mlhat_term_project_starter
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[tracking]'
python -m unittest discover -s tests -v
python .\experiments\dependence_counterexample.py
python .\experiments\run_stream.py --model majority --samples 3000 --seed 42
python .\experiments\run_stream.py --model ht --samples 3000 --seed 42
```

If PowerShell blocks activation use `Set-ExecutionPolicy -Scope Process Bypass`
or invoke `.\.venv\Scripts\python.exe` directly. These commands generate CSV
and JSON under `results/`, including reproducibility configuration.

### W&B public run tracking (requires user account + network)

```powershell
wandb login
python .\experiments\run_stream.py --model ht --samples 3000 --seed 42 --wandb --project mlhat-independent-rebuild
```

Set the W&B project visibility to **Public** in your account. `--wandb` is
opt-in to avoid accidental private/untracked demo claims. Link each published
run in the deck/poster. The detailed W&B rules are in the **Lab 1 course page**,
which was NOT supplied; obtain it before submitting.

## Source tree

```text
src/mlhat_rebuild/
  entropy.py   Eq. (3), (5), (7); independent mathematical helper
  tree.py      Online multi-label Hoeffding tree BASELINE (binary features only)
  stream.py    Reproducible synthetic demo with one concept drift
  metrics.py   Test-then-train example-F1, Hamming loss, subset accuracy
experiments/
  run_stream.py                 CLI demonstration, CSV/JSON + optional W&B
  dependence_counterexample.py Eq. (5) theory stress test
src/mlhat_rebuild/ ...
tests/         Deterministic tests
results/       Generated experiments and metadata

docs/
  requirements_mapping.md   Paper/course requirement traceability
  paper_notes.md           Stepwise paper specification and replication defaults
  lab2_hypothesis.md       Proposed critique + controlled experiments
  slide_outline.md         Draft deck structure; NOT submission-ready
  poster_outline.md        Draft poster structure; NOT submission-ready
```

## Important limitations

`tree.py` operates ONLY on binary input features and majority-labelset leaves.
It does **not** implement numerical Gaussian observers, categorical multi-value
splits, LP-kNN, BR-ADWIN-Bagging+LR, online Poisson bootstrap, ADWIN drift
monitoring, alternate branches, replacement/pruning, or prediction blending.
The published model has **all** these components. They must be implemented and
unit-tested before reporting *MLHAT reproduced*.

Results on `synthetic_demo_NOT_paper` validate engineering only. The paper's
41 datasets (including MOA drift cases) must be obtained and their evaluation
settings matched for Lab 1. The paper studies 11 multi-label metrics plus time;
this starter implements just 3 multi-label metrics and does not measure time.

**Model parameter caveat:** Paper Section 4.1 broadly reports tree defaults
`delta=1e-5`, `grace_period=200`; its tuned MLHAT Table 5 reports `delta_spl=1e-7`.
The starter uses **generic-baseline** `1e-5`, NOT final tuned MLHAT config.
This distinction must be maintained in all results. The paragraph beneath
Table 5 also conflicts with the table on eta and alternate-tree size
(620/450 versus 750/200); log this uncertainty rather than silently choosing
a number. Full spec in docs/paper_notes.md.

## Suggested roadmap

1. Stage 1 (done): stream protocol, entropy/split equation tests, binary HT,
   majority baseline, prequential metrics, public experiment tracking integration.
2. Stage 2: numeric/categorical Gaussian observers; independent LP-kNN leaf;
   independent BR+ADWIN-Bagging+LR leaf; threshold selection (eta=750).
3. Stage 3: ADWIN Hamming loss per internal node, alternate trees, confidence
   interval replacement, Poisson(1) resampling and probabilistic blending.
4. Stage 4: real/MOA benchmark loaders; full metrics; verification of accuracy,
   time, memory, group baselines; multi-seed runs; comparison with Table 6+.
5. Stage 5: theoretical critique and stress-test ablations; deck/poster.

**Do not change the seed or tune on test results without recording the new run.**
Record dataset version, source link, feature transformations, seed, computer
specification, code commit SHA, parameters and metrics for every experiment.
