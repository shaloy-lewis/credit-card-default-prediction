# Model Card — selected_v1

## Intended use

Prioritise a human-owned retention or support outreach queue in a local portfolio demonstration.
The model must not make adverse-action, lending, pricing, or eligibility decisions.

## Model and inputs

`catboost_fixed` uses 19 operational credit-limit, repayment-status, billing, and payment fields.
Sex, education, marital status, age, account ID, and target are excluded from prediction.
Identity calibration and validation-derived risk thresholds are unchanged.

## Evidence

- Validation: AP 0.556510; Brier 0.133539; lift@10%
  3.210923.
- Final test (aggregate committed evidence only): AP
  0.542867; Brier
  0.136304; lift@10%
  3.089676.
- Explanations: native SHAP in raw-log-odds space, labelled risk-increasing or
  risk-mitigating and never represented as causal.
- G3: closed with conditions following two predeclared education selection-rate triggers.

## Limitations and controls

The 2005 Taiwan dataset is not representative production data for India. Fairness, regulatory
compliance, monitoring, registry promotion, and rollback readiness are not claimed. Retraining is
a separately governed process. Representative production data and monitoring are required before
real-world use.
