# One-pass governed model-selection report

**Deterministic summary SHA-256:** `8c11b1d443c782a8ef14aa3e708e3fffa064ecb4c9fe58d3e51a6effa46efbd7`

Each fixed classifier was fitted exactly once on 19,200 development rows and scored on the same 4,800 validation accounts. No cross-validation loop, tuning, calibration fit, winner refit, or sealed-test access occurred.

| Model | Average precision | Brier | Lift@10% | Eligible |
| --- | ---: | ---: | ---: | --- |
| logistic_l2 | 0.542468 | 0.136089 | 3.182674 | true |
| random_forest | 0.533161 | 0.138335 | 3.045198 | false |
| hist_gradient_boosting | 0.554306 | 0.134004 | 3.210923 | true |
| catboost_fixed | 0.556510 | 0.133539 | 3.210923 | true |

Selected model: **catboost_fixed**.
Identity calibration and validation-derived risk bands are frozen in the bundle.
The joblib format uses pickle semantics and must be loaded only from a trusted, digest-verified local bundle.

## Governance boundary

The 6,000-row test partition remains sealed. G2 stays open until a separately authorized, one-time test evaluation passes gates frozen from this validation result.
Historical Phase 2/3 evidence is retained for audit but is not an executable workflow.
