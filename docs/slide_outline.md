# Slide deck outline — M2 + Lab1 (45% together)

Every figure must be CREATED by team rather than copied from article.

1. Title, paper citation, team, public repo link, W&B project link.
2. Problem: what is streaming multi-label classification? Running example.
3. Why a static classifier and standard Hoeffding Tree are insufficient.
4. Mathematical setup, labelsets, co-occurrence and drift scenarios.
5. Own architecture diagram for MLHAT (tree, leaves, ADWIN, alternatives).
6. Eq. (5): example entropy computation with 2 labels.
7. Eq. (7)–(9): a numeric toy Hoeffding split example (show both gains).
8. Dynamic low/high-complexity leaf classifiers and their state switching.
9. Per-node ADWIN, background tree, replacement with Eq. (16).
10. Prediction mixing, bootstrap and overall Algorithm 1 sequence.
11. Reproduction protocol (datasets, paper parameters, hardware, metric, W&B).
12. Paper vs independently reproduced results: paired comparison table.
13. Divergences: delta/time, reproducibility causes and failed cases.
14. Conclusions, limitations, repository/run links and division of work.

**Current status:** outline only. Slides 11–13 must await full independent
implementation and matching paper benchmark runs. Do not paste demo results.
