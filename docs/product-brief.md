# Product and decision brief

**Working title:** Credit Risk Early-Warning and Intervention Prioritisation Platform
**Status:** Accepted Phase 0 baseline
**Owner:** Shaloy Lewis
**Last updated:** 2026-08-07

**Approval:** Project owner approved the Phase 0 baseline on 2026-08-07.

## Executive summary

The system produces a monthly probability of next-month default for existing
credit-card customers who have the required six-month account history. It then
applies an explicit capacity or cost policy to create a prioritised queue for a
hypothetical customer-support or collections team.

The model does not make an autonomous credit decision. It does not approve or
decline new applicants, change credit limits, trigger collections activity, or
provide a legally sufficient adverse-action reason. A human-owned operating
policy remains responsible for any downstream action.

The portfolio project is intended to demonstrate senior data-science, MLOps,
and model-governance judgment. It is not evidence that the model is suitable for
use by an Indian lender or in a regulated production environment.

## Decision contract

| Item | Proposed definition |
| --- | --- |
| Decision unit | One active credit-card account with the required six-month history |
| Scoring time | Once per billing cycle, after the monthly data snapshot closes |
| Prediction horizon | Default during the following month, following the source dataset label |
| Primary user | Hypothetical credit-risk or collections operations manager |
| Decision owner | Hypothetical credit-risk policy owner; never the model itself |
| Model output | Calibrated probability of default, risk band, model version, and reviewed reason categories |
| Primary action | Rank accounts for a capacity-constrained human intervention queue |
| Primary policy | Demonstration assumption: review the top 10% of eligible accounts; report sensitivity at 5%, 10%, and 20% and keep `K` configurable |
| Alternative policy | Rank by simulated expected loss using `PD x proxy EAD x assumed LGD - intervention cost` |
| Primary interface | Idempotent monthly batch scoring |
| Secondary interface | Versioned FastAPI endpoint for integration tests and demonstrations |
| Human control | Review policy, exceptions, promotion decisions, and retraining decisions |

## Users and responsibilities

The roles below are a portfolio simulation used to make ownership explicit.

| Role | Responsibility |
| --- | --- |
| Credit-risk policy owner | Defines eligibility, capacity, actions, risk appetite, and guardrails |
| Collections operations manager | Consumes the prioritised queue and owns operational outcomes |
| Data scientist / model owner | Designs validation, modelling, calibration, and model documentation |
| ML engineer / service owner | Owns reproducibility, registry, inference, observability, and rollback |
| Model-risk reviewer | Reviews evidence, limitations, subgroup results, and promotion gates |

## Goals

1. Rank eligible existing accounts so a limited intervention capacity captures
   a useful share of next-month defaults.
2. Produce probabilities that are sufficiently calibrated for risk bands and
   transparent sensitivity analysis.
3. Make the complete path from source data to registered model and scored batch
   reproducible and auditable.
4. Demonstrate controlled promotion, monitoring, incident response, and
   rollback in a zero-cost local environment.
5. Evaluate subgroup behaviour and document the permitted use of demographic
   attributes.
6. Separate measured model performance, simulated economics, and unknown real
   business impact.

## Product hypothesis

At a fixed review capacity, ranking eligible accounts by calibrated next-month
default risk will capture more future defaults than random selection or a simple
rule. The dataset can test that ranking hypothesis. It cannot establish that an
intervention reduces default; that requires a controlled impact study.

## Intended population and exclusions

The production-like contract assumes active existing cardholders with sufficient
history and permitted features available at the scoring cutoff. A real policy
would exclude closed, written-off, fraudulent, deceased, disputed, or legally
restricted accounts; accounts already in default or late-stage collections;
customers in an active hardship process; and records that fail data-quality or
history requirements.

The source dataset lacks the operational fields needed to implement and verify
all of these exclusions. Offline results therefore describe the published sample,
while production eligibility remains a documented design assumption.

## Non-goals

- New-customer underwriting or credit approval.
- Automated denial, limit reduction, collections action, or customer messaging.
- Claims of Indian-population validity, production scale, realised financial
  benefit, or compliance with RBI, Basel, DPDP, or another regulation.
- A genuine out-of-time validation using the current dataset.
- Blind retraining or promotion in response to drift.
- Streaming inference, Kubernetes, or a feature store without a demonstrated
  product requirement.
- Using a GenAI system to make or modify the risk decision.

## Success measures

No single metric is the objective. Model selection must consider discrimination,
calibration, decision utility, stability, and governance evidence.

| Layer | Measures | How it will be used |
| --- | --- | --- |
| Baseline | Target prevalence and simple policy performance | Establish the minimum credible comparison |
| Discrimination | PR-AUC, ROC-AUC, KS/Gini with bootstrap confidence intervals | Compare candidates on common folds and the sealed holdout |
| Calibration | Brier score, log loss, reliability curve, calibration slope/intercept | Determine whether probabilities support policy and risk bands |
| Capacity decision | Recall, precision, and lift at 5%, 10%, and 20% capacity; default rate by risk band | Evaluate the 10% demonstration policy and its sensitivity |
| Simulated economics | Expected loss captured, intervention cost, sensitivity ranges | Explore decisions without claiming realised impact |
| Robustness | Missingness, category, range, and plausible population-shift stress tests | Identify failure modes and monitoring requirements |
| Fairness audit | Group calibration, TPR/FPR, contact-selection rates, uncertainty, sample sizes | Detect material subgroup differences and inform use restrictions |
| Operations | Batch completeness, idempotency, latency, failures, lineage, rollback time | Demonstrate service ownership rather than only model development |

Quantitative acceptance thresholds will be set after the baseline is measured.
They must not be reverse-engineered from the final test result. If a complex
model does not provide a defensible improvement over the interpretable baseline,
the baseline remains eligible for selection.

## Guardrails and foreseeable harms

| Risk | Proposed control |
| --- | --- |
| Unnecessary or harmful customer contact | Capacity limit, risk bands, human-owned action policy, and complaint-rate guardrail in the experiment design |
| Demographic discrimination | Separate predictive and audit datasets; demographic ablation; subgroup metrics with uncertainty; documented feature-use policy |
| Misleading explanations | Validate transformed feature names and SHAP additivity; aggregate to reviewed reason categories; do not call raw SHAP values adverse-action reasons |
| Data leakage | Feature-availability table, `as_of_date` contract, sealed test set, and leakage tests |
| Silent input failure | Versioned schema, range/category checks, quarantine path, and batch manifest |
| Model deterioration | Data and prediction drift checks plus delayed-label performance and calibration monitoring |
| Unsafe automation | Manual promotion, investigation before retraining, model versioning, and tested rollback |
| Privacy leakage | Synthetic identifiers in demos, data minimisation, no raw personal attributes in ordinary logs, and documented retention assumptions |

## Data and evidence boundary

The first release uses the public UCI Default of Credit Card Clients dataset.
It contains 30,000 Taiwanese customer records and lagged billing and repayment
attributes. It is a useful learning dataset but has important constraints:

- It is not representative evidence for an Indian population.
- It provides one modelling snapshot rather than repeated, dated account-month
  observations suitable for genuine out-of-time validation.
- It does not contain intervention assignment, contact cost, complaint outcomes,
  EAD, LGD, or treatment effectiveness.
- Lagged columns must not be presented as a true production event history.

Model-performance claims will use only the source dataset. Synthetic data may
be used to exercise batch volume, schema failures, drift detection, monitoring,
and incident-response paths; those results will be labelled as operational tests.

## Proposed data and feature policy

- Preserve an immutable source snapshot and checksum.
- Record dataset hash, code commit, configuration, folds, dependencies, and
  environment with every training run.
- Use a fixed stratified holdout and repeated stratified cross-validation; do
  not create a chronological split from row order.
- Keep `SEX`, `AGE`, and `MARRIAGE` available for audit and ablation analysis.
- Default proposal: exclude sensitive demographic attributes from the promoted
  predictive candidate unless evidence and a reviewed use policy justify them.
- Fit every transformation, outlier rule, calibrator, and policy threshold using
  training or validation data only.

## Governance stage gates

| Gate | Evidence required before progression |
| --- | --- |
| G0: scope approval | Decision contract, non-goals, evidence boundary, and accepted ADR |
| G1: data readiness | Data card, schema results, lineage, feature-availability matrix, and leakage review |
| G2: model candidate | Baselines, common validation protocol, experiment lineage, uncertainty, calibration, and stress tests |
| G3: promotion review | Model card, subgroup evaluation, reason-code tests, policy evaluation, risk register, and sign-off checklist |
| G4: release readiness | Batch/API parity, contract tests, container scan, monitoring, rollback evidence, and runbooks |
| G5: ongoing review | Drift/performance report, incident log, retraining rationale, and updated approvals |

## Impact measurement plan

The predictive model can identify risk; it cannot prove that an intervention
prevents default. A production owner would need a randomised experiment or an
appropriate quasi-experiment. The portfolio deliverable will therefore specify:

- randomisation unit and eligibility;
- control and intervention arms;
- primary, secondary, and customer-harm guardrail metrics;
- minimum detectable effect and sample-size assumptions;
- intention-to-treat analysis and stopping rules; and
- explicit separation of observational lift from causal intervention impact.

## Open decisions for later phases

1. Whether a future stakeholder context should replace the 10% demonstration
   capacity assumption.
2. The hypothetical intervention catalogue and cost assumptions.
3. The final sensitive-feature policy after demographic ablation.
4. The model-promotion thresholds after the baseline is measured.
5. The local service-level objectives after an initial benchmark.

These decisions do not block Phase 1. They must be resolved before model
promotion or claims about the operating policy.
