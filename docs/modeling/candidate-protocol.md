# Phase 3 candidate modelling protocol

**Protocol:** `candidate_v1`
**Status:** Frozen before candidate metric execution; compute amendment recorded
**Frozen:** 2026-08-26; compute amendment: 2026-08-27

## Purpose and evidence boundary

This protocol governs the Week 4 CatBoost candidate experiment. It was fixed
from the reviewed Phase 2 development evidence before any Phase 3 candidate
result was produced. Its purpose is to test whether bounded model complexity
adds defensible value over the L2 logistic baseline without model shopping.

Only the 24,000-row development partition and its sealed 5-fold × 3-repeat
assignments may be loaded. The 6,000-row test partition remains inaccessible to
candidate fitting, parameter selection, ablation, scoring, and evaluation.

The machine-readable contract is `configs/modeling/candidate_v1.json`. Its
complete amended SHA-256 is
`4bd9a404064d410e0339e0638464aaf6c1ac0bca632156a47af14a822d7cb5f3` and is
pinned by the candidate-protocol integrity test.
It is bound to the reviewed baseline summary at SHA-256
`11e0332fc9df6f7abf36080a8d09304b3e975f34ad060f70f8611f4fc0ad69d6`.

## Reference baseline

The advancement gate is relative to the reviewed `logistic_l2` repeated-CV
means, not to a test result:

| Measure | Logistic reference |
| --- | ---: |
| Average precision | 0.5412935563 |
| Brier score | 0.1363615900 |
| Lift at 10% | 3.1569033716 |
| Average-precision repeat standard deviation | 0.0005794410 |

These figures describe development-fold evidence for the published 2005 Taiwan
sample. They are not promotion thresholds, causal evidence, or evidence of
performance for an Indian lender.

## Feature policy and ablations

`account_id`, `default_next_month`, and all four demographic audit fields are
prohibited predictors. The candidate uses only the operational fields approved
by `feature_contract_v1`.

| Feature view | Predictors | Role |
| --- | ---: | --- |
| `repayment_status_only` | 6 | Diagnostic ablation; cannot advance |
| `monetary_only` | 13 | Diagnostic ablation; cannot advance |
| `operational_full` | 19 | Hyperparameter search and candidate view |

The two reduced views answer whether the added feature family provides material
ranking value. They cannot become the selected candidate or satisfy the advancement
gate, and they are not the Week 6 demographic ablation or fairness review.

The six validated repayment-status codes are passed to CatBoost's native
categorical interface as strings. The 13 monetary fields remain raw numeric
values. No imputation, clipping, scaling, resampling, or target encoding is
permitted; any such change requires a new reviewed protocol version.

## Candidate and bounded search

CatBoost `1.2.5` is the only new model family. A second challenger is excluded
so the ten-hour slice is spent on governed comparison, diagnostics, and
reproducibility rather than a model leaderboard.

Every fit uses CPU `Logloss`, Bayesian bootstrap, seed 42, four threads, no class
weights, no CatBoost filesystem output, no verbosity, and no evaluation-fold
early stopping. Setting `use_best_model=false` prevents the scoring fold from
implicitly choosing the iteration count.

Scikit-learn `1.4.2` `ParameterSampler` draws exactly eight distinct configurations
with seed 42 from this finite space:

| Parameter | Values |
| --- | --- |
| Iterations | 300, 600 |
| Depth | 4, 6 |
| Learning rate | 0.03, 0.05, 0.10 |
| L2 leaf regularisation | 3, 7, 12 |
| Random strength | 0, 1 |
| Bagging temperature | 0, 1 |

The resulting ordered configurations are materialized as `cb_cfg_001` through
`cb_cfg_008` in the machine-readable contract. Execution must reproduce that exact
list from the locked sampler before the first candidate fit.

All eight configurations use all 15 reviewed folds on `operational_full`. The
selected hyperparameters are then reused unchanged for the two reduced feature
views. This fixes the maximum at 150 CatBoost fold fits: 120 search fits plus 30
additional ablation fits. The full-view selected configuration reuses its search
predictions and is not fitted a second time for the ablation comparison.

### Compute amendment

The original 12-configuration space included depth-8, 900-tree fits. Before any
candidate metric was calculated or inspected, local runtime-only benchmarks found
one such fold required approximately 238 seconds. Capping the space at depth 6 and
600 trees reduced the representative ceiling to approximately 67 seconds. The
amendment preserves every reviewed fold, gate, feature boundary, and tie-break while
removing impractical compute rather than reacting to model quality.

## Evaluation and advancement

Average precision is the primary measure. Brier score is the probability
guardrail, and lift at 10% is the primary capacity measure. ROC-AUC, KS, Gini,
log loss, and precision, recall, and lift at 5%, 10%, and 20% remain reported.
Capacity metrics retain the reviewed fractional-expected tie policy.

A CatBoost variant is eligible only if every balanced-gate condition passes:

| Condition | Relative rule | Derived absolute boundary |
| --- | --- | ---: |
| Average precision | Improve by at least 0.010 | At least 0.5512935563 |
| Brier score | Degrade by no more than 0.005 | At most 0.1413615900 |
| Lift at 10% | Regress by no more than 0.10 | At least 3.0569033716 |
| AP repeat stability | Standard deviation at most 0.01 | At most 0.01 |

Search selection chooses the highest-average-precision eligible full-view
configuration. If none is eligible, the highest-AP configuration may still be
used for diagnostic ablations, but CatBoost cannot advance.

Eligible full-view configurations within 0.002 average precision of the best are
treated as practically equivalent. The deterministic tie-break order is lower
depth, fewer iterations, higher L2 regularisation, lower learning rate, lower
random strength, lower bagging temperature, and then configuration ID. Reduced
views remain diagnostic regardless of their observed metrics. If no full-view
configuration passes every gate, `logistic_l2` remains the candidate for Week 5.

## Required implementation evidence

The future candidate command must record the candidate-config and baseline
evidence hashes, data and split lineage, code and dependency identity, the eight
sampled configurations, fold and repeat metrics, complete OOF coverage, feature
view, fit budget, and the advancement decision. It must reject incomplete folds,
non-finite scores, probability-bound violations, unapproved features, changed
hashes, dirty reviewed outputs, and any attempt to load test assignments.

Runtime MLflow identifiers, row-level predictions, and fitted estimators remain
ignored. Any deterministic aggregate report must exclude timestamps and
machine-local paths and must be published only after successful tracking and
validation.

## Deferred decisions and exit gate

Calibration selection, bootstrap confidence intervals, simulated economics,
operating-policy selection, and the one-time holdout evaluation belong to Week
5. Demographic ablation, subgroup analysis, explanations, and the final feature
use decision belong to Week 6.

Week 4 passes only when the implementation reproduces the frozen sampled
configurations, all variants use the common folds, the fit budget is respected,
lineage and OOF coverage are complete, the advancement rule is applied exactly,
the holdout remains untouched, and static, test, and container compatibility
checks pass. Passing Week 4 does not close G2.
