# Feature availability and leakage review

- **Applies to:** `uci_credit_default_v1`
- **Decision point:** monthly scoring after the latest billing snapshot closes
- **Status:** Phase 1 / G1 evidence

## Availability convention

The UCI file contains no row-level `as_of_date` or event timestamps. The product
decision point is therefore a documented simulation boundary, not a value
reconstructed from the source. Only published attributes that describe the
current or prior six billing periods may be candidates at that boundary. The
next-month default label is future information and is never available to
prediction.

## Feature-use matrix

| Source fields | Meaning and relative period | Availability at the simulated cutoff | Current policy | Leakage / governance note |
| --- | --- | --- | --- | --- |
| `ID` | Published row/account identifier | Present in the snapshot | Lineage and split joins only | Never a predictor; sequential values could encode row order without business meaning |
| `LIMIT_BAL` | Published credit limit | Current snapshot | Candidate predictor | A limit is not EAD and must not be presented as realised exposure |
| `SEX` | Published binary sex code | Current snapshot | Audit and ablation only | Excluded from promoted predictive candidates; categories are coarse |
| `EDUCATION` | Published education code | Current snapshot | Audit and ablation only | Excluded; undocumented codes 0, 5, and 6 are retained and reported |
| `MARRIAGE` | Published marital-status code | Current snapshot | Audit and ablation only | Excluded; undocumented code 0 is retained and reported |
| `AGE` | Published age in years | Current snapshot | Audit and ablation only | Excluded under the current demographic-feature policy |
| `PAY_0` | September 2005 repayment status | Most recent published period | Candidate predictor | Name is intentional; no `PAY_1`. Codes -2 and 0 are undocumented by UCI |
| `PAY_2`–`PAY_6` | August through April 2005 repayment status | Prior five published periods | Candidate predictors | Suffix is not months-ago uniformly with `PAY_0`; mapping must remain explicit |
| `BILL_AMT1`–`BILL_AMT6` | September through April statement balances | Current/prior six periods | Candidate predictors | Negative amounts are valid source values, not automatic quality failures |
| `PAY_AMT1`–`PAY_AMT6` | September through April previous-payment amounts | Current/prior six periods | Candidate predictors | Values describe payments already made in the published history |
| `Y` / `default_next_month` | Next-month default indicator | Future outcome | Label only | Must be absent from transformations, feature matrices, thresholds, and scoring inputs |

Engineered features may use only the permitted behavioural candidates above.
Examples include balance trends, payment-to-bill ratios, utilization proxies,
and delinquency summaries. Every learned transformation, imputation rule,
outlier threshold, feature selection step, calibrator, and policy threshold must
be fit within the relevant training fold.

## Demographic boundary

The canonical dataset retains demographics because deleting them would prevent
data-quality review, subgroup analysis, and the Week 6 ablation study. The
future Phase 2 modelling interface must derive separate views from the canonical
dataset:

- a predictor view that excludes `ID`, the target, and all four demographic
  fields; and
- an audit view keyed by `ID` that contains the target and demographics.

Joining the audit view into a feature matrix is a contract violation. A future
change to demographic-feature use requires a recorded governance decision and
new versioned feature contract; exploratory performance improvement alone is
not sufficient.

The compatibility CatBoost model is exempt only so its frozen endpoint can be
regression-tested. Its use of legacy inputs does not override the Phase 1 policy
and its output is not promotion evidence.

## Split and transformation controls

The dataset has no defensible chronological ordering. Phase 1 therefore uses a
fixed 80/20 deterministic stratified holdout with seed 42 and five-fold
stratified cross-validation repeated three times on the development partition.
The split configuration is `configs/data/split_v1.json`; account membership is
stored in the ignored `split_assignments.csv`. Its digest and aggregate counts
are recorded in the runtime split manifest. The committed
`configs/data/split_v1.lock.json` is the reviewed reference for that manifest.

The following controls apply:

1. Assign partitions from stable `ID` values and the target using the pinned
   random seed; never depend on physical row order.
2. Seal the holdout before baseline or candidate comparison. Until the model,
   calibrator, and selection policy are fixed, code may validate its schema and
   counts but may not use its outcomes for selection.
3. Fit transformations and thresholds on development/training rows only and
   refit inside each cross-validation fold.
4. Never create an `as_of_date`, chronological split, or temporal-performance
   claim from the row number or feature suffixes.
5. Keep account membership in `split_assignments.csv`, and verify its digest
   against the runtime manifest and reviewed lock before every experiment.
6. Report the 35 duplicate non-ID records. Do not drop or group them as presumed
   duplicate customers without evidence; use a sensitivity check in modelling
   if their influence is material.

## Automated leakage assertions

Before a Phase 2 feature matrix is accepted, tests or build-time assertions
must establish that:

- the predictor view contains neither `ID`, the target, nor demographic fields;
- each source `ID` appears exactly once and in exactly one holdout partition;
- every development row has the expected cross-validation assignments and no
  holdout row has a training-fold assignment;
- repeated builds with the same source and split configuration produce identical
  assignments and hashes;
- any split-seed drift from the reviewed value 42 is rejected; and
- schema failures stop the build before any trusted processed artifact is
  published.

This review establishes the feature boundary for G1. It does not establish that
all permitted features are stable, fair, causal, or suitable for promotion;
those questions remain part of modelling, stress testing, and governance gates.
