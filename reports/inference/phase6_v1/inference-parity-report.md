# Phase 6 inference parity report

Status: **complete**

## Reviewed result

One shared, dependency-validated `selected_v1` engine scored 20 synthetic accounts through offline, monthly-batch, and `/v1/predict` paths. The batch selected 2 accounts under the fixed 10% review capacity. No model fitting, refitting, tuning, calibration fitting, or sealed-test access occurred.

The full-precision batch probabilities matched the shared engine exactly. The largest absolute API rounding difference was `4.2811987377433525e-07`, within the frozen `5e-7` tolerance. Risk bands and both reason categories and directions matched exactly. The identical batch rerun reused verified files without rewriting them.

## Explanation boundary

Native SHAP used `raw_log_odds` contributions across the four reviewed categories. These are model attributions, not causal or adverse-action reasons.

## Interface and lifecycle boundary

`POST /predict` is retired; `POST /v1/predict` is the only prediction endpoint. Streamlit calls that endpoint and does not load the model. G4 remains open for stress testing, registry promotion, scanning, rollback, monitoring, and runbooks.
