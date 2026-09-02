# One-pass model selection and release protocol

**Protocol:** `selection_v1`  
**Status:** frozen before validation comparison  
**Authoritative training budget:** four fits

## Decision

The portfolio now optimises for lifecycle governance rather than model-search
volume. Historical Phase 2 repeated-CV baselines and Phase 3 CatBoost search
remain immutable audit evidence, but their public commands are retired. New
release work uses one fixed fit for each of four classifiers and never repeats
cross-validation or parameter search.

The reviewed development partition is divided using the existing sealed split
assignment: `cv_fold_r0 != 0` supplies 19,200 training rows and
`cv_fold_r0 == 0` supplies 4,800 validation rows. The 6,000-row test partition
is not returned by the modelling data interface and cannot participate in model
selection.

## Fixed comparison

| Model | Fixed design | Feature treatment |
| --- | --- | --- |
| `logistic_l2` | L2, C=1, lbfgs, 2,000 max iterations | Fixed one-hot status encoder and training-only monetary scaling |
| `random_forest` | 100 trees, Gini, sqrt features, seed 42 | Validated numeric values without scaling |
| `hist_gradient_boosting` | 100 iterations, learning rate 0.1, 31 leaves, seed 42 | Validated numeric values without scaling |
| `catboost_fixed` | Historical `cb_cfg_006`: depth 4, 300 trees, learning rate 0.03, L2 12 | Status strings as native categorical values; raw monetary values |

All models use the same 19 operational predictors. Demographics, account ID,
and target are prohibited. Imputation, clipping, resampling, target encoding,
feature selection, validation-based early stopping, and final refitting are
prohibited.

## Selection rule

Average precision is primary. A model is eligible only when its Brier score is
no more than 0.005 worse than validation logistic regression and its lift at
10% is no more than 0.10 lower. Among eligible models within 0.002 average
precision of the best, the simpler model wins in this fixed order:
`logistic_l2`, histogram gradient boosting, random forest, then CatBoost.

The exact fitted validation winner is serialized without refitting. Calibration
is identity. Calibration diagnostics, 500 seed-42 stratified bootstrap samples,
and 80%, 90%, and 95% risk thresholds are derived from stored validation
predictions and do not fit another estimator.

## Release evidence

`credit-risk model select` requires a clean commit and verified offline data. It
records one MLflow parent with four child runs, then atomically publishes the
aggregate summary/report, ignored row-level validation evidence, and a
checksum-protected winner bundle. A sklearn winner uses joblib, which has pickle
semantics and must be loaded only as a trusted local artifact after manifest and
digest validation. A CatBoost winner uses its native CBM format.

Historical `baseline`, `candidate`, `candidate-evidence`, and legacy `train`
commands fail immediately with migration guidance. Their code, configurations,
and reviewed evidence remain available for audit.

## One-time test boundary

After selection evidence and the bundle are reviewed and committed,
`credit-risk model freeze-test` freezes absolute test gates from validation
without loading data or the estimator. `final-test` remains disabled in this
delivery. A separate explicit request is required to implement and execute the
single 6,000-account test evaluation. G2 remains open until that evaluation
passes; no force, retraining, retuning, refitting, or reevaluation path is
permitted.

## Reviewed selection outcome

The official run executed from clean implementation commit `f7c99f2` and
completed exactly four fits. `catboost_fixed` was the sole model within 0.002
average precision of the best eligible result and was selected without refit.
Its validation metrics are average precision `0.556510`, Brier score `0.133539`,
and lift at 10% `3.210923`. Histogram gradient boosting was eligible with average
precision `0.554306`; random forest failed the lift guardrail.

The selected native CBM has SHA-256
`844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c`.
The aggregate summary, report, manifest, and model binary are protected by
complete-file integrity tests. These are validation results only; the test
partition remains sealed and G2 remains open.

The reviewed `final_test_v1.json` authorization was generated from evidence
commit `d334b88` without loading data or the model. It freezes minimum average
precision `0.526510`, maximum Brier score `0.153539`, minimum lift at 10%
`2.910923`, identity calibration, the three validation risk thresholds, and
exactly 6,000 unique test accounts. Its execution flag remains false.
