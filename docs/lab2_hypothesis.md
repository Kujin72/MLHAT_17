# M3 + Lab2: concrete critique candidate — Eq. (5) and label dependence

## Precise observation

Section 3.2 discusses multi-variate Bernoulli distributions and modelling
co-occurrence. However, the entropy explicitly written in paper **Eq. (5)** is

    H_eq5(Y1,...,YL) = sum_j H(Y_j)

not the joint entropy

    H_joint(Y1,...,YL) = -sum_{y-vector} P(y) log2(P(y)).

Therefore, in general

    H_eq5 >= H_joint,

with equality for mutually independent labels (the difference is multivariate
total correlation). It is **not** valid to infer from Eq. (5) alone that all
label dependences affect the entropy/splitting criterion.

## Exact counterexample, verified with our independent test

- Dataset A: 50 samples labelset (0,0), 50 samples labelset (1,1).
- Dataset B: 25 samples of each (0,0), (0,1), (1,0), (1,1).
- Both datasets have P(Y1=1) = P(Y2=1)=0.5, so Eq. (5) = 2 bits in both.
- A has H_joint=1 bit and B has H_joint=2 bits.
- This proves Eq. (5) by itself does not distinguish perfectly correlated
  labels from independent labels with the same marginals.

Command:

```powershell
python .\experiments\dependence_counterexample.py
```

## Bounded claim

This **does not yet prove that the complete MLHAT model fails to detect label
correlations**. Other parts of the model (e.g. majority labelsets, LP-kNN
or joint labelset frequencies) can encode co-occurrence. The valid critique is
narrow: the entropy expression alone is insensitive to dependence given fixed
marginals, despite the interpretation of the split criterion.

## Empirical stress test TO DO (Lab 2)

1. Generate paired streaming datasets with exactly matched feature marginals,
   per-label frequencies, drift locations and seeds. Vary dependence between
   output labels, including cases with XOR/synergy not visible individually.
2. Compare fully implemented MLHAT's Eq. (5) split vs an **ablation** using
   joint-labelset entropy at otherwise identical parameters (note that the
   Hoeffding range then must be recalibrated); also compare a BR baseline.
3. Record prequential example-F1, Hamming loss, subset accuracy and macro F1,
   as well as split count, detection delay, time, peak memory, run-to-run spread.
4. Study incremental drift vs abrupt drift separately. Avoid leakage from
   seeding experiments or using future observations to set drift thresholds.
5. Publish all seeds (suggested 5+) and run links. Use paired comparison and
   error bars. If no performance difference is found, report that honestly.

## Competing independent critique

The paper itself describes limitations on incremental drift and nontrivial
hyperparameter tuning (Sections 4.5 and 5). An alternative stress test is to
hold all else fixed while varying drift duration, label imbalance and number
of labels; compare fixed vs adaptive threshold eta and per-node detectors.

Do not claim this hypothesis is a demonstrated model weakness before the
controlled ablation with full implementation and matched data is completed.
