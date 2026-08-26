# Baseline experiment protocol v1

## Purpose and decision boundary

This protocol governs the Week 3 scientific-baseline experiment for the UCI
credit-default snapshot. It compares deliberately simple references on the
sealed development folds before candidate models, calibration, or an operating
policy are selected. It is not a model-promotion decision and it does not
authorize use of the 6,000-row test partition.

The experiment is valid only after the Phase 1 offline verification succeeds
against the reviewed split lock. The workflow consumes the recorded fold
assignments; it never regenerates or substitutes cross-validation folds.

## Modelling views

The predictor view contains exactly 19 operational columns, in governed order:

- current credit limit;
- repayment status for lags 0 through 5;
- bill amount for lags 0 through 5; and
- payment amount for lags 0 through 5.

`account_id`, `default_next_month`, sex, education, marital status, and age are
forbidden predictors. The latter six fields form a separate development-only
audit view. Demographics are retained for the later ablation and fairness gate,
not for baseline fitting.

## Common validation protocol

- Population: the 24,000 development accounts only.
- Validation: the reviewed five-fold assignments repeated three times.
- Positive class: `default_next_month = 1`.
- Transformations: learned independently inside each training fold.
- Parallelism: folds execute serially to reduce nondeterministic numerical
  variation and make failures attributable.
- Output: one out-of-fold score per account, model, and repeat. Each model must
  therefore produce 72,000 rows; all three models produce 216,000 rows.
- Holdout: Phase 1 may re-read the canonical snapshot to verify its governed
  hashes and schema, but the modelling interface must not expose test IDs or
  outcomes to fitting or evaluation. Test scores and metrics must not appear in
  runtime artifacts, tracking records, or reports.

## Locked baselines

1. **Fold prevalence.** Every validation row receives the corresponding
   training fold's event rate.
2. **Recency-weighted arrears score.** For repayment lag `i`, negative and zero
   status codes contribute zero and positive codes contribute
   `(6 - i) * status`. The six contributions are summed. This is an unfitted
   ranking score, not a probability.
3. **L2 logistic regression.** The six repayment-status columns use fixed
   one-hot categories from -2 through 9 with the first category dropped. Credit
   limit, bill amounts, and payment amounts are standardized within each fold.
   The model uses `C=1`, `lbfgs`, no class weighting, 2,000 maximum iterations,
   tolerance `1e-8`, and seed 42. There is no imputation, clipping, feature
   engineering, resampling, or hyperparameter search.

The optional scorecard is deferred. A more complex baseline would consume the
time reserved for reproducibility and would not change the Week 3 decision.

## Metrics and interpretation

Non-interpolated average precision is the primary ranking measure, Brier score
is the probability-quality guardrail, and lift at 10% is the primary capacity
measure. The report also
includes ROC-AUC, KS, Gini, log loss, and precision, recall, and lift at 5%, 10%,
and 20% capacity. Brier score and log loss are not applicable to the arrears
ranking score.

When a capacity boundary crosses tied scores, metrics use fractional expected
allocation across the complete tie group. `account_id` is never used as a
predictive tie-breaker. Metrics are calculated for every fold and for each
repeat's complete out-of-fold predictions. Mean, standard deviation, minimum,
and maximum across the three repeats are descriptive variation, not confidence
intervals. Formal bootstrap uncertainty and calibration belong to Week 5.

## Tracking and evidence

The command logs directly to an ignored local SQLite MLflow store with one
parent execution and one nested run per baseline. Tracking evidence includes
the source, canonical, assignment, split-lock, feature-contract, experiment
configuration, Python/SciPy/dependency-lock, and code-state identities; fold definitions;
model parameters; metrics; and runtime artifacts. It excludes raw or canonical
data, test records, and fitted pickle files. A non-executable JSON artifact
records fold-level logistic convergence and coefficient diagnostics.

An exploratory run may use `--allow-dirty`; its Git dirty state and diff digest
remain visible in tracking and it cannot be accepted as reviewed evidence. The
official run must execute from a clean implementation commit. Deterministic
aggregate evidence excludes timestamps, machine-local paths, and MLflow run
identifiers so the evidence commit does not create circular lineage.

## Failure policy and exit gate

Missing or incompatible governed inputs, changed hashes, forbidden features,
fold leakage, incomplete out-of-fold coverage, non-finite scores, probability
bounds violations, logistic non-convergence, tracking failures, and partial
evidence publication fail the command. A failure must not replace the last
successful report. Caught validation, tracking, and filesystem failures are
rolled back; each report file uses atomic replacement. A host crash between the
two file replacements is detected by the summary hash embedded in the Markdown
report and repaired by a deterministic rerun; a single-pointer versioned report
store is deferred to the registry phase.

Week 3 passes only when all three models use the same reviewed development
folds, the required out-of-fold coverage is exact, lineage is complete, no test
evidence exists, the deterministic summary is reproducible, and all static,
test, and container compatibility gates pass. Candidate acceptance thresholds
are fixed in the [Phase 3 candidate protocol](candidate-protocol.md) from this
development evidence before Week 4 candidate fitting. G2 remains open.
