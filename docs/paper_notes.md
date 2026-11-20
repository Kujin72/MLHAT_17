# MLHAT — read-along notes / independent implementation specification

**Paper:** Esteban et al., arXiv:2410.20242v1, 31-page uploaded PDF.
This is original paraphrase and interpretation, not paper text.

## Five foundational concepts (Section 2)

1. **Data stream:** infinite/large chronological sequence `(x_t, y_t)`;
   train/observe instance-by-instance, not random-split train/test.
2. **Multi-label:** each `y_t` is a length-L binary vector, not one class ID.
3. **Incremental tree:** learn split decisions using sample statistics without
   retaining whole past data; Hoeffding inequality gives a confidence margin.
4. **Concept drift:** conditional relationship between X and Y can change over
   time; static branches may become obsolete.
5. **Imbalance/correlations:** rare labels and co-occurring labels challenge
   prediction, splitting criteria, and inference.

## Paper's key components

- Section 3.2 / Eq. (5): sum of marginal Bernoulli entropies across labels.
- Section 3.3: dynamic leaf learner selected using entropy and sample count.
  Majority labelset if pure; LP-kNN if few samples; BR variant of ADWIN
  Bagging with logistic regressions if many samples.
- Section 3.4 / Eq. (7): `epsilon = sqrt(log2(|Lsets|)^2 * ln(1/delta_spl)/(2*W))`.
  Eq. (8)-(9): info gain and margin between first two split candidates.
  Numeric feature threshold probabilities from per-label Gaussian moments;
  categorical features use counts.
- Section 3.5: per-node ADWIN on multi-label Hamming loss. Early warning
  produces alternate background subtree; confirm replacement with a Hoeffding
  comparison using main and alternate errors (Eq. 16).
- Section 3.6 / Algorithm 3: predict using main leaf and sufficiently mature
  alternate leaves; combine error-weighted probabilities.
- Algorithm 1: online training including Poisson bootstrap; Algorithm 2:
  node statistics, candidate feature thresholds and leaf training.

## Experiment facts to replicate (paper Section 4)

- RQ1: comparison with prior online multi-label decision trees/classifiers.
- RQ2: native multi-label split vs transformed single-label criterion.
- RQ3: robustness to distribution drift.
- The paper reports 41 datasets, including synthetic drift test streams.
- Prequential (predict before learn) Section 4.1.3.
- Forgetting factor `alpha=0.995`, evaluation interval `50` instances.
- 11 multi-label metrics + computation time (12 total measures).
- Paper's general comparison uses grace period 200 and delta 1e-5
  for Hoeffding-based trees (Section 4.1.1).
- **Critical distinction:** tuned final MLHAT **Table 5**: delta_spl = 1e-7,
  grace period = 200, delta_alt = 0.05, kappa_alt=200, eta=750, lambda=1,
  drift metric = Hamming loss. **Inconsistency:** the paragraph immediately
  below Table 5 instead refers to thresholds **620** (leaf), **450**
  (alternate tree). Record both, use explicitly labelled Table 5 values
  for the initial faithful configuration, and verify against independent
  reference experiments without copying source code. Leaf low complexity LP-kNN, high complexity
  BR+Bagging+LR. No automatic assumption that baseline delta=1e-5 is correct
  for the tuned MLHAT configuration.
- Paper Table 6 lists example-F1 by dataset and model; reproduce selected
  rows with matched dataset version and report discrepancies.
- Dataset origins: paper points to multi-label dataset repository and original
  project's synthetic drift dataset information. Reference only. Do not copy
  researcher's implementation.

## Technical non-equivalences to close before Lab 1

Stage-1 baseline currently supports binary features only; all examples from
`run_stream.py` are generated locally, not paper data. For numerical data,
replace simplistic histograms with online Gaussian moments (Eq. 14-15);
for true MLHAT use adaptive leaves, Poisson(1), ADWIN alternate trees and
probabilistic prediction mixing. Memory/time instrumentation still missing.

## Implementation and scientific integrity

1. Transcribe mathematical definitions into independent tests.
2. Implement self-contained components without opening/copying original source.
3. Compare behavior against analytical examples; only then use the original
   repository to diagnose discrepancies, not to transplant code.
4. Keep source references with PDF section and equation numbers for every
   component, and be prepared to defend equations in the oral exam.
