# Phase 5 Governance Review

Status: **closed with conditions**

Summary SHA-256: `9431910060c38c5c50fe58508871dd85c5fbf118f3c1af00ffaa8807420ee123`

The reviewed `selected_v1` bundle was scored once on the 4,800-row development-validation
slice. No model fitting, calibration fitting, cross-validation, or sealed-test access occurred.

Validation AP was 0.556510, Brier score was
0.133539, and lift at 10% was 3.210923.

Native CatBoost SHAP values were checked in raw-log-odds space for 1,000 deterministic
validation rows. Contributions are model attributions, not causal or adverse-action reasons.

| Attribution category | Mean absolute contribution | Mean signed contribution | Mean direction |
| --- | ---: | ---: | --- |
| billing balance | 0.097517 | -0.005149 | risk_mitigating |
| credit capacity | 0.194510 | -0.004809 | risk_mitigating |
| payment behaviour | 0.264627 | 0.006922 | risk_increasing |
| repayment status | 0.631751 | 0.000955 | risk_increasing |

## Conditions

- Retain all demographics as audit-only fields.
- Restrict use to human-owned outreach prioritisation.
- Prohibit adverse action.
- Prohibit India-specific or compliance claims.
- Require representative data and monitoring before real use.
