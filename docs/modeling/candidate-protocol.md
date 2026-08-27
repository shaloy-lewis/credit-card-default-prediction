# Phase 3 candidate modelling protocol

**Protocol:** `candidate_v1`
**Status:** Completed with independently reproduced development-only evidence
**Frozen:** 2026-08-26; compute amendment: 2026-08-27; evidence: 2026-08-27

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

The reviewed aggregate evidence is published as
`reports/modeling/candidate_v1/summary.json` and `candidate-report.md`. Their
complete SHA-256 digests are
`55aaa971417bddbcad00b8bdf388f74baa13f6ad96304dd108227e48de23ea83` and
`156967cfda68ddf6c49e4f1e1666266d69261c58628820df3a2a821e560b17c2`,
respectively.

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

### Resumption and independent evidence

Every completed fold is written atomically as a non-pickle NumPy checkpoint
below that execution's ignored tracking root. Its task hash binds the candidate
configuration, governed data and split lineage, Git commit or dirty-diff hash,
feature view, sampled parameters, exact train/validation populations, labels,
repeat, and fold. Reuse requires exact account, label, probability, diagnostic,
and tree-count validation. Invalid and interrupted checkpoints are moved to a
content-addressed quarantine location and the fold is refitted.

One fit runs at a time with four CatBoost threads. Runtime checkpoints do not
change protocol order, selection, or deterministic evidence, and resume counts
are console-only operational information.

Official evidence is published by `credit-risk model candidate-evidence`. It
runs or resumes a primary execution and a second execution under independent
tracking roots, requires byte-identical summary, report, OOF, and diagnostic
artifacts, and atomically promotes the primary aggregate files. A third 150-fit
pass is prohibited. Comparison or publication failure preserves prior official
evidence and marks both completed MLflow parent runs failed.
The single-run `candidate` command defaults to an ignored provisional directory
and is prohibited from writing the official Phase 3 report destination.

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

The candidate command records the candidate-config and baseline
evidence hashes, data and split lineage, code and dependency identity, the eight
sampled configurations, fold and repeat metrics, complete OOF coverage, feature
view, fit budget, and the advancement decision. It must reject incomplete folds,
non-finite scores, probability-bound violations, unapproved features, changed
hashes, dirty reviewed outputs, and any attempt to load test assignments.

Runtime MLflow identifiers, row-level predictions, and fitted estimators remain
ignored. Any deterministic aggregate report must exclude timestamps and
machine-local paths and must be published only after successful tracking and
validation.

## Reviewed outcome and Phase 4 handoff

Two independent executions from clean implementation commit `2b46d4c` produced
byte-identical summary, report, OOF, and fold-diagnostic artifacts. Across the
two executions, the 720,000 OOF rows have SHA-256
`94ee8a56b731008722a63a9913f696dbe1bc827f64e1e903208bf98a0c44fd46`
and the 150 fold diagnostics have SHA-256
`a3b5a2e64c7a145ce408d2e6a77ffc2034fe71e76387271a9ff9bdf19e5a2174`.

Six configurations fell within the `0.002` equivalence band. The frozen
tie-break selected `cb_cfg_006`: depth 4, 300 iterations, learning rate `0.03`,
L2 leaf regularisation `12`, random strength `0`, and bagging temperature `0`.
It passed all four gates with mean average precision `0.556419`, AP repeat
standard deviation `0.000821`, Brier score `0.134101`, and lift at 10%
`3.202110`. CatBoost therefore advances as the Phase 4 candidate.

Phase 4 reuses only `cb_cfg_006`; the eight-configuration search is not repeated.
This decision remains development-CV model-selection evidence. The selected
estimator is not serialized or connected to `/predict`, and the holdout remains
sealed.

## Deferred decisions and exit gate

Calibration selection, bootstrap confidence intervals, simulated economics,
operating-policy selection, and the one-time holdout evaluation belong to Week
5. Demographic ablation, subgroup analysis, explanations, and the final feature
use decision belong to Week 6.

Week 4 passed after the implementation reproduced the frozen sampled
configurations, all variants used the common folds, the fit budget was respected,
lineage and OOF coverage were complete, the advancement rule was applied exactly,
and the holdout remained untouched. Passing Week 4 does not close G2.
