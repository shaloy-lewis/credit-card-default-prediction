# ADR 0002: no-training governance review

- **Status:** accepted
- **Applies from:** Phase 5 / Week 6
- **Decision owner:** portfolio owner

## Context

The selected `catboost_fixed` model was trained, reviewed, serialized without
refitting, and evaluated once on the sealed test partition. Phase 4 permanently
retired that evaluator. The roadmap originally requested a demographic model
ablation during Week 6, but the promoted model already excludes demographics
and a literal ablation would require fitting another model after release.

The local API demonstrates technical integration. It is not evidence that an
external model-risk function approved production use. G3 therefore remains a
separate governance review even though the bundle can already be served locally.

## Decision

Phase 5 performs no training, refitting, tuning, cross-validation, calibration
fitting, or test evaluation. Phase 1 integrity verification necessarily parses
the complete canonical snapshot, so Phase 5 does not claim that test bytes or
rows are never inspected for integrity. The modelling boundary guarantees that
test accounts are never selected, returned, scored, explained, or included in
subgroup analysis. It replaces demographic model ablation with:

1. contract proof that demographics cannot enter the estimator;
2. input-invariance tests showing that audit-only values do not change the
   projected predictor frame;
3. validation-only subgroup metrics and predeclared human-review triggers; and
4. a documented feature-use decision retaining demographics for audit only.

Explanations use CatBoost's native SHAP values on a deterministic validation
sample. They are model attributions in raw-score space, not causal findings,
customer-facing reasons, or legally sufficient adverse-action reasons.

Only aggregate committed final-test metrics may be cited. Phase 5 may not select
or return test accounts, load final-test predictions, generate new test scores,
or include test rows in explanations, subgroup analysis, or published evidence.

## Consequences

- G3 may close with documented conditions; it does not certify fairness,
  regulatory compliance, Indian-population validity, or production suitability.
- The two predeclared education selection-rate triggers require explicit
  disposition rather than automatic model rejection.
- The local API remains a portfolio demonstration until later registry,
  rollback, monitoring, and operational gates are complete.
- A future demographic-inclusive model requires a new feature contract,
  training authorization, scientific protocol, and independent review.
