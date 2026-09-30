# Release C claim-to-evidence inventory

**Status:** author-prepared inventory for owner review; not owner acceptance.
No new historical scoring, uncertainty resampling or impact evaluation is performed.

| Claim | Class and population | Source / qualification |
| --- | --- | --- |
| AP 0.542867; Brier 0.136304; lift at 10% 3.089676 | Measured historical: one authorised 6,000-account final test | [Release A report](../../reports/releases/release_a_v1/release-a-report.md), [summary](../../reports/releases/release_a_v1/summary.json). Not prospective performance; final test cannot be rerun. |
| 600 reviewed accounts contained 410 defaults | Measured historical: same final-test cohort and 10% policy | Release A report's final-test capacity table. Lift measures concentration, not prevented defaults. |
| Four fixed classifier fits; 19 predictors; identity calibration | Implemented scientific release contract | [Selection protocol](../modeling/selection-protocol.md), [selected manifest](../../models/selected_v1/manifest.json), Release A report. Earlier search is retained historical context; no new fits occur. |
| Top 10% review queue, floor rounding and deterministic ties | Implemented local policy | [Frozen inference configuration](../../configs/inference/phase6_v1.json), [batch code](../../src/credit_risk/inference/batch.py). Capacity is a demonstration assumption. |
| Prediction 0.190382 with explanation checks | Measured synthetic: reviewed request fixture | [Request fixture](../../tests/fixtures/prediction_request.json), [parity report](../../reports/inference/phase6_v1/inference-parity-report.md), live check recorded by the demo helper. Not a real customer's result. |
| Promotion/rollback works on SQLite; persistent bootstrap and restart recovery work on PostgreSQL/MinIO | Measured synthetic operational controls | [Phase 7 report](../../reports/registry/phase7_v1/registry-release-report.md), [Phase 8 manifest](../../reports/platform/phase8_v1/evidence-manifest.json), [verified baseline CI](https://github.com/shaloy-lewis/credit-card-default-prediction/actions/runs/36594629955). No PostgreSQL transition or deployed Azure claim. |
| Releases A/B complete; G4 closed locally, G3 conditional, G5 open | Recorded governance decisions | [Governance status](../governance/model-governance-status.md), [detached owner approval](../../configs/releases/release_b_owner_approval_v1.json). Historical dossier candidate wording remains immutable; approval is a separate authenticated record. |
| 30% versus 27%; three percentage points / 10% relative reduction | Hypothetical planning assumptions | [Study protocol](intervention-study.md), [calculator method](planning-calculator.md). Not inferred from historical prevalence or the high-risk queue. |
| 3,554 analysable / 3,949 recruited per arm; 7,898 total at alpha .05, power .80 and attrition .10 | Computed planning estimates | [Standalone calculator](../../src/credit_risk/portfolio/planning.py), [reference and regression tests](../../tests/unit/portfolio/test_planning.py). Normal approximation; not an assurance of power under an unknown operating population. |
| Seven days before due date; one attempt within three business days; maturity plus 14 days; maximum 12 cohorts | Hypothetical protocol rules | Study protocol. Chosen design assumptions, not dataset facts or approved real operating policy. |
| 400 synthetic rows and 40 selected accounts in the walkthrough | Synthetic demonstration target, measured only when the helper passes | [Demo script](demo-script.md) and its ignored run receipt. Repeats known fixture patterns with unique synthetic IDs; not 400 independently sampled customers. |
| Monitoring thresholds 0.10 / 0.20; fewer than 200 valid rows is insufficient | Demonstration policy, not a validated production tolerance | [Implemented monitoring thresholds](../../src/credit_risk/monitoring/drift.py), [monitoring workflow](../../src/credit_risk/monitoring/workflow.py). Alerts require human investigation and do not change scoring. |
| Five-minute script; final MP4 acceptance requires 4-6 minutes | Planned communication deliverable | [Acceptance plan](release-c-acceptance-plan.md). No video exists or is accepted merely because this script is written. |

## Week 12 verification claims

The [Week 12 record](week12-review.md) and [aggregate inventory](week12-evidence.json)
bind the fresh-checkout synthetic counts, parity, traces, monitoring result and
hashes to `6c3e8a56cf0551dc22a67f91d8e204a1b8de4984`. UI results were checked with
AppTest through the live API; visual browser review is still open. Hardware and
tool versions describe the existing Windows machine, not a new machine or local
Docker execution. The 21-cue, 300-second caption draft is preparation, not a
measured video duration. No final MP4 or owner acceptance is claimed.

## Claim review checklist

- Historical metrics above are transcribed from authenticated aggregates, without recomputation.
- No validation bootstrap interval is described as a final-test interval.
- No causal, monetary, prospective, geographic-transfer or compliance benefit is asserted.
- Cloud mappings are conceptual; registry phase boundaries remain explicit.
- Review and retrieval use the fixed historical digest anchors in the [setup guide](../../README.md#release-b-sign-off).
- Runtime checks must record their own implementation commit and receipt; verified baseline CI is not evidence of a later commit's result.
- Owner review of these claims and the eventual full recording remains open.
