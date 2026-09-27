# Release B owner review candidate

Dossier SHA-256: `f5fa342b89e06c016ea7b632c8a252c472502186ab52ac68aba6992d652c8c88`

Implementation/report commit: `179f893b9675041fd7b1be6e1a48cf03d7a81a79`.
[Exact-commit CI](https://github.com/shaloy-lewis/credit-card-default-prediction/actions/runs/36296458441).

Decision is pending. This proposal grants no approval. Approving it accepts every disposition below for the local portfolio demonstration, retains G3 conditions, and leaves G5 production review and Release C communication open.

## Measured controls

| Measure | Acceptance | Frozen target |
| --- | ---: | ---: |
| api_p95_ms | 7.797950 | 32.365462 |
| batch_seconds | 0.465012 | 1.456218 |
| recovery_seconds | 2.729980 | 6.950855 |

Valid-request failures: 0; prediction parity: 0.190382. Targets were retained from the original three rehearsals and were not relaxed after recovery failure.

The unchanged model, 19 predictors, identity calibration, risk bands and 10% review policy remain fixed. No training, tuning, calibration fitting, bootstrap regeneration or sealed-test scoring is authorized.

## Material sensitivity

| Scenario | Mean absolute probability change | Queue turnover | Risk-band movement |
| --- | ---: | ---: | ---: |
| repayment_plus_1 | 0.226695 | 38.54% | 65.04% |
| repayment_plus_2 | 0.352767 | 91.04% | 68.42% |

All 14 scenarios and both historical subsets completed. Only unchanged historical rows have outcome metrics. The two repayment stress findings require the operating restrictions below.

## All required dispositions

| Item | Proposed disposition | Operating restriction |
| --- | --- | --- |
| incident:artifact_integrity | resolved | The isolated synthetic drill passed detection, containment, recovery and verification. Continue the corresponding release runbook and retain failed receipts; never alter historical artifacts or silently select duplicate accounts. |
| incident:duplicate_ids | resolved | The isolated synthetic drill passed detection, containment, recovery and verification. Continue the corresponding release runbook and retain failed receipts; never alter historical artifacts or silently select duplicate accounts. |
| incident:invalid_values | resolved | The isolated synthetic drill passed detection, containment, recovery and verification. Continue the corresponding release runbook and retain failed receipts; never alter historical artifacts or silently select duplicate accounts. |
| incident:missing_columns | resolved | The isolated synthetic drill passed detection, containment, recovery and verification. Continue the corresponding release runbook and retain failed receipts; never alter historical artifacts or silently select duplicate accounts. |
| incident:phase7_sqlite_rollback | resolved | Demonstrated recovery applies to the isolated Phase 7 SQLite workflow. Phase 8 remains fixed-state bootstrap and restart recovery; no PostgreSQL promotion/rollback claim. |
| incident:population_shift | accepted_with_restriction | Synthetic drift detection and restored controls passed. Hold queue use during real investigation; only a human owner may restore use, without automatic model changes. |
| incident:prediction_shift | accepted_with_restriction | Synthetic drift detection and restored controls passed. Hold queue use during real investigation; only a human owner may restore use, without automatic model changes. |
| incident:service_interruption | resolved | Use isolated API-container restart and the original frozen recovery target. Preserve volumes and authenticated state; suspend dependent requests while unavailable. |
| monitoring:batch_disposition | accepted_with_restriction | The deterministic synthetic batch intentionally differs from historical validation and triggers investigation. Accept for the local demonstration only; preserve completed scores and require human review without model, calibration, threshold or promotion changes. |
| risk:artifact_integrity | accepted_with_restriction | Authenticate model and deployment digests before use; reject missing or corrupt artifacts. |
| risk:calibration_drift | accepted_with_restriction | Identity calibration remains fixed; investigate drift manually, with no automatic recalibration. |
| risk:consumed_test_protection | accepted_with_restriction | The sealed test remains retired; no fits, tuning, bootstrap regeneration or new test scoring. |
| risk:education_disparities | accepted_with_restriction | Retain both G3 education disparity conditions; audit-only demographic fields and human review. |
| risk:explanation_language | accepted_with_restriction | Explanations describe model attribution, never causality or adverse-action reasons. |
| risk:geographic_temporal_transportability | accepted_with_restriction | Local demonstration only; no claims for current or other populations without representative evaluation. |
| risk:human_owned_use | accepted_with_restriction | Human-owned outreach demonstration only; prohibit lending, adverse action and automated customer decisions. |
| risk:object_store_upstream_availability | accepted_with_restriction | Use the source-pinned legacy MinIO build only in isolated local resources with private S3 networking; maintained object storage and production supportability remain G5 work. |
| risk:phase7_phase8_boundary | accepted_with_restriction | Phase 7 SQLite demonstrates promotion/rollback; Phase 8 demonstrates fixed bootstrap/persistence only. |
| risk:production_monitoring | accepted_with_restriction | CLI monitoring is a local demonstration; longitudinal evaluation and production operations remain G5 work. |
| risk:production_privacy | accepted_with_restriction | Use synthetic operational fixtures; keep validation scores and account mappings in ignored local storage. |
| robustness:billing_balance_0.75 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:billing_balance_0.9 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:billing_balance_1.1 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:billing_balance_1.25 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:credit_capacity_0.75 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:credit_capacity_0.9 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:credit_capacity_1.1 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:credit_capacity_1.25 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:low_credit_limit | accepted_with_restriction | Historical subgroup diagnostic only; preserve the exact cohort and denominators. No causal, population-transfer or prospective performance claim. |
| robustness:payment_behaviour_0.75 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:payment_behaviour_0.9 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:payment_behaviour_1.1 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:payment_behaviour_1.25 | accepted_with_restriction | No material finding under the frozen thresholds. Restrict interpretation to this validation cohort and perturbation; no causal, prospective or production robustness claim. |
| robustness:repayment_at_least_two | accepted_with_restriction | Historical subgroup diagnostic only; preserve the exact cohort and denominators. No causal, population-transfer or prospective performance claim. |
| robustness:repayment_plus_1 | accepted_with_restriction | Sensitivity demonstration only. Do not interpret retained labels as stressed outcomes or act on perturbed queues. Comparable real drift requires investigation and suspended human use until disposition. |
| robustness:repayment_plus_2 | accepted_with_restriction | Sensitivity demonstration only. Do not interpret retained labels as stressed outcomes or act on perturbed queues. Comparable real drift requires investigation and suspended human use until disposition. |

All eight incident drills passed detection, containment, recovery and verification. Outage recovery measured 1.754615 seconds. Monitoring recorded the intentionally injected failed batch and health failure; these are drill detections, not unexpected valid-request failures.

MinIO uses the same source-pinned legacy release after its original distribution became unavailable. Private object-store networking and local isolation remain required; maintained storage and production supportability remain G5 work.

Historical evidence and model bytes remain unchanged. The owner decision must bind the exact dossier digest above.
