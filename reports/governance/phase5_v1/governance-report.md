# Phase 5 Governance Review

Status: **closed with conditions**

Summary SHA-256: `9173ce26d9821aea6fc1744dc07b3d504888e074532703f9dd80a3e706e97c73`

The reviewed `selected_v1` bundle was scored once on the 4,800-row development-validation
slice. Full canonical-file verification was performed for integrity, but no test account was
selected, returned, scored, explained, or included in subgroup analysis. No model fitting,
calibration fitting, or cross-validation occurred.

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
