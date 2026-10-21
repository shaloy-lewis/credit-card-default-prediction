# Scientific decisions and evidence limits

**Reading guide:** existing data contracts and authenticated aggregate results,
interpreted without fitting, historical scoring, row-level EDA or new resampling.
Historical reports describe the state at their own checkpoint; use the
[current governance status](../governance/model-governance-status.md) for gates.

## Data, population and the prediction question

The [data card](../data/data-card.md) records UCI dataset 350, its attribution,
CC BY 4.0 license, source size/hash and reviewed raw-to-split lineage. The
published observation is one record for a Taiwanese credit-card client in a
2005 snapshot; the product simulates account-level monthly scoring. A real
customer/account mapping is unavailable. April–September history predicts the
published following-month binary default label. This is not a reconstructed
event ledger or an independently verified regulatory definition of default.

There are 30,000 records and 6,636 defaults (22.12%) in the documented source.
These are published quality aggregates, not newly computed EDA. They explain why
a majority-class accuracy can look favorable while finding no defaults; that
source-wide prevalence is not a new validation or final-test accuracy result.

### Five boundaries that must not be conflated

| Boundary | Existing rule and rationale | Authority |
| --- | --- | --- |
| Data quality | Pinned source fingerprint, exact ordered schema, complete finite integral values, unique IDs and expected class counts; reject incompatible source rather than silently repair it. | [Validation policy](../data/validation-policy.md), [schema](../../src/credit_risk/data/schema.py) |
| Statistical assumptions | Stratified random splits describe within-snapshot generalization. Resampling treats records as sampling units; unknown dependence and duplicate profiles limit that interpretation. | [Selection protocol](../modeling/selection-protocol.md), [data card](../data/data-card.md) |
| Operational eligibility | Active accounts, required history, contact permission and exclusions need external operational records and human policy. Passing a schema does not establish eligibility. | [Accepted brief](../product-brief.md#intended-population-and-exclusions) |
| Model inputs | Exactly 19 predictors: credit limit and six lags each of status, bill and payment amounts. No new engineered predictors are introduced. | [Feature contract](../../configs/modeling/feature_contract_v1.json) |
| Audit attributes | Sex, education, marital status and age remain separately available for subgroup diagnostics; ID joins lineage and splits, target supplies historical outcomes. None enters prediction. | [Availability review](../data/feature-availability.md), [dataset interface](../../src/credit_risk/modeling/dataset.py) |

### Cleaning and feature reasoning

The canonical schema accepts a positive credit limit, age 18–100, sex codes 1/2,
education 0–6, marital status 0–3, integral repayment codes -2 through 9,
signed bill amounts, nonnegative payments and a binary target. These are
**contract domains**, not claims that every boundary occurred in the source.
The separate [wire contract](../../src/credit_risk/inference/contracts.py)
requires strict integer values for all 19 live predictors too; it does not accept
fractional monetary values or coerce arbitrary strings into API numbers. Monetary
amounts follow the source's NTD units. Batch CSV parsing validates its integer
tokens before constructing the same operational feature contract.

| Existing choice | Scientific and operational reasoning | Residual limitation |
| --- | --- | --- |
| Reject missing/non-finite inputs; no imputation | The pinned source reports no missingness; inventing fitted replacement values is unnecessary and would hide operational failures. | Coverage for incomplete future records is unknown; reject/reconcile instead of assuming missing-at-random. |
| Preserve undocumented status -2/0 and audit-category codes | Preserve source meaning/uncertainty rather than invent ordinal or social interpretations. Logistic status encoding is fixed one-hot; CatBoost uses status strings as categories. | A category's association is not a causal explanation; unknown future semantics need review. |
| Preserve negative bills and large amounts; no clipping | Negative balances are documented accepted anomalies. Schema validity differs from an outlier judgment; clipping would alter the scientific contract. Logistic monetary scaling is learned on training rows only. | Valid extremes can still be rare or unstable. No claim of robust extrapolation beyond the historical population. |
| Retain 35 duplicate non-ID records, reject duplicate IDs | Identical published profiles do not prove identical customers. Deleting them would require an unsupported entity-resolution decision. | Duplicate profiles may overlap random partitions; no duplicate-sensitivity performance result is claimed. Future sensitivity analysis needs approval. |
| Use raw operational predictors, exclude demographics/ID/target | Matches the scoring decision and no-training governance boundary; prevents direct demographic use and obvious target/identifier leakage. | Credit limit, payment behavior and balances can encode demographic or socioeconomic proxies. Exclusion/invariance is not proof of fairness. |
| No ratios, trend features, resampling or target encoding in release | Existing fixed models already use the approved history. Added transformations would change the selected artifact and require new authorized evaluation. | Availability of an interesting feature does not establish incremental value; proposals remain future work. |

The [source feature mapping](../data/feature-availability.md) explains `PAY_0`
versus `PAY_2`–`PAY_6` and relative monthly suffixes. The simulated cutoff assumes
all predictors are known before the next-month label. No row-level timestamps
exist to verify real ingestion latency; new-population validation must test
point-in-time availability using actual event and ingestion timestamps.

## What the existing exploration establishes

| Existing aggregate finding | Interpretation and consequence | Limit |
| --- | --- | --- |
| Source imbalance and accepted anomalies | Use prevalence-aware comparisons and explicit data contracts; do not “clean” away undocumented values without a policy. | [Data card](../data/data-card.md) is descriptive, not causal or current-population evidence. |
| Historical repayment-rule baseline AP 0.473102, lift 2.909619 | The defined repayment-burden heuristic carried useful ranking signal relative to the prevalence baseline; compare the richer model with a meaningful simple workflow. | [Release A baseline table](../../reports/releases/release_a_v1/release-a-report.md) summarizes development CV; not a feature-level causal effect or same-split test comparison. |
| Validation mean score 0.221929 versus prevalence 0.221250; ten-bin ECE 0.013558 | Overall probability scale and reliability bins support inspecting the retained identity mapping. | Global agreement can hide local errors; binning, sampling and population changes matter. |
| Queue precision decreases as published capacity increases | Historical marginal reviews are less concentrated; staffing and missed-risk trade-offs must be explicit. | [Capacity analysis](decision-analysis.md) changes no operating policy and estimates no treatment benefit. |
| Two education selection-rate review triggers | Excluded demographics can still be associated with allocation differences. | Validation diagnostics prompt human review; they do not identify the causal mechanism or certify discrimination/fairness. |

No new pairwise correlations, significance claims, feature-effect plots or
independent feature-importance discoveries are asserted. Existing SHAP checks
establish attribution behavior and numerical consistency, not causal drivers.
An exhaustive EDA or dependence sensitivity study is not retrospectively supplied
by these aggregates.

## Why these algorithms and why this winner

Historical development work included repeated-CV baselines and a bounded CatBoost
search. The [candidate protocol](../modeling/candidate-protocol.md) and its
[report](../../reports/modeling/candidate_v1/candidate-report.md) retain that
history. The final release comparison reused a fixed development split:
19,200 training and 4,800 validation rows, with 6,000 test rows kept outside
selection. All candidates had identical predictor access.

| Alternative | Purpose in the comparison | Reviewed outcome / trade-off |
| --- | --- | --- |
| Fold prevalence and repayment rule | Expose whether complexity beats a prior-only predictor and a simple behavior-based ranking. | Historical CV baselines, not competitors newly scored on the test. |
| L2 logistic regression | Regularized additive reference with explicit status encoding and training-only monetary scaling. | Validation AP 0.542468; retained as the simplest eligible reference. |
| Random forest | Fixed bagged-tree nonlinear/interactions alternative without scaling. | AP 0.533161; lift 3.045198 fails the logistic-minus-0.10 guardrail (3.082674). |
| Histogram gradient boosting | Fixed efficient nonlinear boosting alternative without native status-string treatment. | AP 0.554306, Brier 0.134004 and lift 3.210923; eligible and a close competitor. |
| Fixed CatBoost | Shallow regularized boosting, using the historically chosen configuration and native categorical statuses. | AP 0.556510, Brier 0.133539 and lift 3.210923; retained exact fitted winner. Native category handling is a design rationale, not an isolated measured cause of its advantage. |

The [frozen rule](../modeling/selection-protocol.md) first applied Brier/lift
guardrails, then AP and a 0.002 simplicity-equivalence margin. CatBoost's AP
advantage over histogram boosting was about 0.002204, just outside that margin;
their reported lift was identical. The rule selected CatBoost, but this is not
evidence of statistically significant superiority or a large operational gain.
No paired superiority test is claimed. A smaller model remains a credible
alternative in a future separately approved study.

The four fits describe this final comparison, not the entire project history.
Because earlier CV/search used the development population, the final validation
slice is not a pristine, independent confirmation of all prior research choices.
The one consumed test is the separate historical evaluation; it cannot become
another development set. Protocols and holdout controls limit selection freedom,
but do not erase earlier experimentation or establish temporal generalization.

## What each metric answers

| Concept | Meaning here | What it does not establish |
| --- | --- | --- |
| Discrimination / average precision | Precision summarized across recall thresholds; emphasizes finding the positive class and depends on prevalence. Repository AP is not asserted to equal trapezoidal PR area. | A deployed threshold, equal subgroup behavior or direct financial value. |
| Probability quality / Brier | Mean squared probability error, rewarding probabilistic accuracy; useful alongside reliability bins and base-rate references. | Calibration alone: Brier reflects both calibration and resolution and depends on event prevalence. |
| Capacity ranking / lift at 10% | Queue event rate divided by cohort event rate at the frozen review fraction. Final-test lift 3.089676 describes historical concentration. | Prevented defaults, optimal allocation or probability calibration. |
| Accuracy | Requires a threshold and can reward predicting the majority class when capacity is limited. | A sufficient objective for this ranking workflow; no new accuracy result is calculated. |
| Model/metric uncertainty | Existing validation resampling describes variation conditional on fixed fitted predictions and the historical sampling setup. | Fitting uncertainty across new training samples, model-selection uncertainty, causal uncertainty or new-population risk. |
| Transportability | Requires external/temporal/current-population evidence with compatible definitions and availability. | Something demonstrated by a random snapshot split or by synthetic stress tests. |
| Decision effectiveness | Requires mature outcomes and a causal comparison of human intervention policies. | A consequence that follows from good AP, calibration or SHAP. |

## Calibration, uncertainty and the no-refit decision

Identity calibration was fixed in the selection protocol; it was not selected
after comparing newly fitted calibrators. The
[Release A diagnostics](../../reports/releases/release_a_v1/release-a-report.md)
report reliability bins and mean-score/prevalence agreement without fitting
another estimator. Low aggregate ECE and good Brier support describing those
historical probabilities, not claiming perfect calibration or transferability.
Platt/isotonic recalibration remains future approved work, requiring appropriate
independent data and a changed versioned probability contract.

The preserved 500-resample seed-42 stratified prediction-only bootstrap produced
validation AP interval [0.525431, 0.587755], Brier [0.128826, 0.137924] and lift
[3.027072, 3.375942]. These are the existing percentile intervals, not regenerated
estimates. Stratification conditions on event composition; they do not capture
unrestricted prevalence shifts or the full selection/fitting process. There is
no newly estimated final-test interval and no interval for real business impact.

Refitting the winner on all development rows could change the scored model,
probabilities, bands, explanation behavior and evidence chain. Preserving the
exact fitted winner sacrifices additional training rows to keep model identity
and the reviewed validation/test contract explicit. Digest checks and the
[replay tombstone](../../src/credit_risk/modeling/final_test_workflow.py)
preserve that boundary. Verification reads stored aggregates and hashes; it
does not grant permission to repeat fitting, calibration or final-test scoring.

## Responsible interpretation and remaining failure modes

The [subgroup review](../../reports/governance/phase5_v1/fairness-report.md)
uses validation only. Supported groups require at least 100 records, 25 defaults
and 25 non-defaults; smaller groups publish counts only. Prevalence uses Wilson
intervals; other measures use the existing within-group stratified bootstrap.
These are descriptive diagnostics with support/uncertainty limits, not a
multiple-comparison-adjusted certification or an intersectional harm assessment.
Group selection is measured at the fixed q90 score threshold; operational batch
selection instead enforces an exact count with deterministic ties.

The [G3 decision](../../reports/governance/phase5_v1/g3-review.md) retains
education-code 1/3 selection-rate triggers and required human review. Financial
behavior can proxy demographic circumstances even when demographic inputs are
excluded. Native SHAP decomposes raw log odds into model contributions; additivity
and probability reconstruction validate the decomposition, not causal actions,
recourse, or legally sufficient adverse-action explanations.

Unverified eligibility, source-label ambiguity, duplicate dependence, changes in
population/prevalence, incomplete history, changed collection policies and
intervention feedback remain failure modes. Drift detects distribution change;
it does not prove degraded AP or reveal outcomes before labels mature. The
[runbooks](../operations/release-b-runbooks.md) require investigation, evidence
and human disposition. No alert authorizes refitting, promotion or policy change.

Future adoption needs representative timestamped data, approved cohort/label
definitions and missingness handling, external/temporal evaluation, supported
subgroup review, security/privacy operations and an approved impact study.
Those are future proposals under G5, not work executed in this release.
