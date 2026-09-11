# G3 Review Decision

Decision: **closed_with_conditions**.

The gate confirms demographic exclusion, predictor invariance to audit-field changes, validation
subgroup evidence, native-SHAP numerical checks, and explicit prohibited uses. It does not certify
fairness, regulatory compliance, or production suitability.

Predeclared triggers:

- `education_code=1`: selection_rate_ratio=0.696014 triggered below_lower_bound.
- `education_code=3`: selection_rate_ratio=1.256410 triggered above_upper_bound.

Required conditions are human review of these triggers, audit-only demographics, human-owned
outreach prioritisation, no adverse action, no India/compliance claim, and representative data plus
monitoring before real use.
