# Phase 5 validation-only governance protocol

**Protocol:** `phase5_v1`

**Status:** completed against the corrected contract; G3 closed with conditions

**Model:** reviewed `selected_v1` CatBoost bundle
**Training budget:** zero fits

## Review boundary

Phase 1 integrity verification parses the complete canonical snapshot and split
assignments. After that verification boundary, the modelling loader must return
exactly the 24,000 development accounts and no test account. The review then uses
only the 4,800 development-validation accounts assigned to `cv_fold_r0 == 0`.
Test accounts may not be selected, returned, scored, explained, or included in
subgroup analysis. Aggregate metrics from the already-completed final test may
be cited, but its row-level predictions and loader are prohibited.

The selected bundle is scored once on validation. Its average precision, Brier
score, and lift at 10% must reproduce the reviewed selection evidence within
`1e-12`. Demographics remain in a separate audit frame and are never estimator
inputs.

## Explanation contract

Exactly 1,000 validation accounts are selected with seed-42
`StratifiedShuffleSplit`, stratified jointly by target and frozen risk band.
CatBoost native SHAP values must return 19 canonical feature contributions plus
one base value. Their sum must reproduce the raw model score, and applying the
logistic function must reproduce the selected-model probability, within
`1e-10`.

Feature contributions are aggregated into credit capacity, repayment status,
billing balance, and payment behaviour. Positive values are labelled
`risk_increasing`; negative values are `risk_mitigating`. These labels describe
model behaviour only and are not causal explanations or adverse-action reasons.

## Subgroup review

The audit reports source-coded sex, education, marital-status, and fixed age-band
groups. A group requires at least 100 rows, 25 defaults, and 25 non-defaults for
performance metrics. Smaller groups disclose counts and `insufficient_support`
only. No intersectional conclusion is made.

At the frozen validation `q90` threshold, the review reports prevalence, mean
probability, calibration-in-the-large, Brier score, selection rate, TPR, and FPR.
Prevalence receives a two-sided 95% Wilson score interval. The other measures use
500 seed-42 within-group stratified bootstrap resamples and percentile 95%
intervals. The stratified bootstrap never represents its fixed class proportion
as prevalence uncertainty.

Human review is triggered by a selection-rate ratio outside `[0.80, 1.25]`, an
absolute TPR/FPR gap above `0.10`, Brier degradation above `0.02`, or absolute
calibration-in-the-large above `0.05`. Thresholds were selected before a
non-published planning preview showed expected selection-rate triggers for
education codes 1 and 3; the thresholds were not changed afterward.

## Governance outcome

The reviewed G3 result is `closed_with_conditions`. The dispositions are to keep
all demographics audit-only, restrict use to human-owned outreach prioritisation,
prohibit adverse action and India/compliance claims, and require representative
data plus monitoring before any real use. This outcome is not a fairness or
production certification.

## Reviewed corrected outcome

The initial Phase 5 evidence was withdrawn after review identified an ambiguous
test-access claim and degenerate prevalence intervals. It remains available only
in Git history. The clean corrected build from implementation commit `9b156c5`
reproduced validation AP `0.556510`, Brier score `0.133539`, and lift at 10%
`3.210923`. Complete-file integrity and semantic tests protect all seven aggregate
artifacts, and offline verification reproduced their manifest.

Native SHAP passed with maximum raw additivity error below `3.6e-15` and maximum
sigmoid/probability error below `1.2e-16`. The supported-group prevalence intervals
are non-degenerate Wilson intervals. The predeclared selection-rate triggers fired
for education code 1 (`0.696014`) and code 3 (`1.256410`); both require human
review and neither automatically rejects the model.
