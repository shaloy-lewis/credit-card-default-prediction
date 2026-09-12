# Release A — defensible model

Status: **complete**

Summary SHA-256: `a8cfdd1f4e4655235082430fdc0cd76e0034ec1797b3cdf9b0f08f5ff8719acb`

Release A consolidates the reviewed model evidence without fitting, scoring, or
loading row-level predictions. The selected artifact is the exact four-fit
validation winner and the one authorized final-test evaluation is permanently
consumed.

## Reproducible data and baselines

The checksum-pinned UCI snapshot contains 30,000 rows with a sealed 24,000-row
development partition and 6,000-row test partition. The source, canonical data,
and assignment hashes are bound in the machine-readable summary.

| Baseline | Average precision | Brier | Lift@10% |
| --- | ---: | ---: | ---: |
| fold_prevalence | 0.221175 | 0.172275 | 0.999247 |
| repayment_burden_rule | 0.473102 | n/a | 2.909619 |
| logistic_l2 | 0.541294 | 0.136362 | 3.156903 |

## Fixed model comparison

| Model | Validation AP | Validation Brier | Validation lift@10% | Eligible |
| --- | ---: | ---: | ---: | --- |
| logistic_l2 | 0.542468 | 0.136089 | 3.182674 | true |
| random_forest | 0.533161 | 0.138335 | 3.045198 | false |
| hist_gradient_boosting | 0.554306 | 0.134004 | 3.210923 | true |
| catboost_fixed | 0.556510 | 0.133539 | 3.210923 | true |

Selected model: **catboost_fixed**. Exactly four
fits were performed, with no tuning, cross-validation iteration, or winner refit.

## Identity calibration

No calibrator was fitted. Identity calibration retained mean probability
`0.221929` against observed prevalence
`0.221250`. Ten-bin ECE is
`0.013558`.

| Bin | Rows | Mean probability | Observed event rate |
| ---: | ---: | ---: | ---: |
| 1 | 480 | 0.061843 | 0.037500 |
| 2 | 480 | 0.081577 | 0.062500 |
| 3 | 480 | 0.094945 | 0.106250 |
| 4 | 480 | 0.109657 | 0.114583 |
| 5 | 480 | 0.131231 | 0.129167 |
| 6 | 480 | 0.155657 | 0.131250 |
| 7 | 480 | 0.193670 | 0.214583 |
| 8 | 480 | 0.258979 | 0.285417 |
| 9 | 480 | 0.422123 | 0.420833 |
| 10 | 480 | 0.709606 | 0.710417 |

## Validation-only uncertainty

Intervals use 500 seed-42 stratified prediction-only percentile-bootstrap
resamples. They are validation intervals, not final-test intervals.

| Metric | Point | 95% interval |
| --- | ---: | ---: |
| average_precision | 0.556510 | [0.525431, 0.587755] |
| brier_score | 0.133539 | [0.128826, 0.137924] |
| lift_at_0_1 | 3.210923 | [3.027072, 3.375942] |

## Capacity-aware evaluation

### Validation

| Capacity | Selected | Precision | Recall | Lift | Expected true positives |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 240 | 0.725000 | 0.163842 | 3.276836 | 174.000000 |
| 10% | 480 | 0.710417 | 0.321092 | 3.210923 | 341.000000 |
| 20% | 960 | 0.565625 | 0.511299 | 2.556497 | 543.000000 |

### One-time final test

| Capacity | Selected | Precision | Recall | Lift | Expected true positives |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 300 | 0.733333 | 0.165787 | 3.315750 | 220.000000 |
| 10% | 600 | 0.683333 | 0.308968 | 3.089676 | 410.000000 |
| 20% | 1200 | 0.563333 | 0.509420 | 2.547099 | 676.000000 |

Final-test average precision is
`0.542867`, Brier score is
`0.136304`, and lift at 10% is
`3.089676`. All three frozen gates
passed, closing G2. The evaluation cannot be rerun.

## Evidence boundary and limitations

This dossier performed complete-snapshot integrity verification only. It did not
deserialize a model, generate predictions, regenerate bootstrap evidence, select
test accounts, or load final-test predictions. Robustness and population-shift
stress evidence is explicitly deferred to G4/Release B.

Results describe a single 2005 Taiwan dataset. They do not establish causal
impact, India-specific validity, regulatory compliance, fairness certification,
production suitability, or realised business value.
