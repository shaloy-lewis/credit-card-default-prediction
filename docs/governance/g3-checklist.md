# G3 promotion-review checklist

**Decision:** `closed_with_conditions`

- [x] The exact `selected_v1` manifest and model digests were verified before scoring.
- [x] Exactly 4,800 reviewed development-validation accounts were used.
- [x] The 19 operational predictors were isolated from ID, target, and demographics.
- [x] Mutating audit fields left the projected predictor frame unchanged.
- [x] Validation AP, Brier score, and lift at 10% reproduced selection evidence within `1e-12`.
- [x] Native CatBoost SHAP returned 19 contributions and one base value.
- [x] Raw-score additivity and sigmoid parity passed the frozen `1e-10` tolerance.
- [x] The 1,000 explanation rows were selected deterministically by target and risk band.
- [x] Supported subgroups received frozen metrics and 500-resample intervals.
- [x] Unsupported groups disclose counts only.
- [x] Education codes 1 and 3 received the predeclared human-review disposition.
- [x] No training, refitting, tuning, cross-validation, or calibration fitting occurred.
- [x] No sealed-test row, prediction, or loader was accessed.
- [x] Row-level predictions, SHAP values, and bootstrap distributions remain ignored.
- [x] The model card, fairness report, risk register, and aggregate evidence were reviewed.

## Conditions carried forward

Demographics remain audit-only. Use is limited to human-owned outreach prioritisation;
adverse action and India-specific or compliance claims are prohibited. Representative
production data and monitoring are required before real use. G3 closure is not fairness,
regulatory, or production certification.
