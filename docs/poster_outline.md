# Poster outline — M3 + Lab2 (50% together)

- **Top left:** Paper, goal and independent reconstruction summary.
- **Main claim:** Eq. (5) marginal entropy cannot on its own distinguish
  different label dependence patterns with identical per-label frequencies.
- **Theory panel:** two 2-label joint probability tables, both H_eq5=2 bits,
  but H_joint=1 vs 2 bits. Derivation in own words.
- **Methods:** matched streaming generator, ablation (Eq.5 vs joint entropy),
  fixed seeds and split bounds, drift scenarios; MLHAT model variants.
- **Results:** line plots for prequential example-F1, grouped Hamming loss,
  macro-F1, memory and drift adaptation; confidence intervals across seeds.
- **Interpretation:** confirm/refute claim, discuss alternative explanations,
  represent small/non-significant results faithfully.
- **Reproducibility footer:** commit SHA, dataset URL/version, public W&B links,
  experiment commands and exact parameters.

**Current status:** mathematical counterexample exists; full ablations and
poster graphics are not yet complete. Only experimental results may populate
these panels, no invented paper reproduction scores.
