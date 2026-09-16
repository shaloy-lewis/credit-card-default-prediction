# Phase 6 inference architecture

## Outcome

Phase 6 implements one prediction-only boundary around the unchanged
`selected_v1` CatBoost bundle. Monthly batch scoring, the versioned HTTP API,
and the Streamlit demonstration do not contain separate model logic.

```text
strict CSV snapshot ──> contract validation ──> shared inference engine ──> ranked batch files
                                                   │
strict JSON request ───────────────────────────────┼───────────────> /v1 response
                                                   │
Streamlit API client ──────────────────────────────┘

selected_v1 manifest + model ──> digest/dependency validation ──> one loaded estimator
```

The engine validates the reviewed manifest/model digests and exact serving
dependency versions before deserializing the estimator. It accepts the 19
ordered operational predictors, performs vectorised prediction and CatBoost
native SHAP calculation, verifies class order, probability bounds, output
cardinality, raw-log-odds additivity, and sigmoid parity, and then returns one
canonical result structure.

## Batch boundary

The batch command requires `account_id` plus the exact operational columns. It
rejects demographics, target, extra fields, invalid identifiers, nulls,
non-finite or fractional values, invalid repayment codes, nonpositive limits,
and negative payments. All occurrences of a duplicate account ID are rejected.
Valid rows can still complete when other rows are rejected.

Ranking is deterministic: probability descending, then account ID ascending.
The human-review queue contains `floor(valid_rows × 0.10)` rows and is separate
from the validation-frozen risk bands. A batch identity binds the input bytes,
scoring date, snapshot ID, Phase 6 configuration, bundle manifest, and model.
Publication is atomic. An identical verified identity reuses its files without
rewriting; changed or corrupt evidence fails closed.

## Online boundary

`POST /v1/predict` is the only prediction route. It accepts the same operational
feature contract but no account ID because online ranking is not meaningful
without a portfolio. An allowlisted request ID becomes the trace ID; otherwise
the service creates one. Contract errors return `422`, unavailable readiness
returns `503`, and unexpected inference failures return a generic traceable
`500` without implementation details.

`GET /ping` remains process liveness. `GET /ready` proves the exact reviewed
bundle and active dependencies loaded successfully. Streamlit calls the v1 HTTP
interface and never deserializes the model.

## Explanations and logs

Native SHAP values are grouped into `credit_capacity`, `repayment_status`,
`billing_balance`, and `payment_behaviour`. The two largest absolute category
contributions are returned with deterministic category-name tie-breaking and a
`risk_increasing`, `risk_mitigating`, or `neutral` direction. They are
non-causal model attributions, not customer-facing or adverse-action reasons.

JSON logs use an explicit metadata allowlist. They may contain the operation,
status, trace or batch ID, model/bundle/policy identifiers, aggregate counts,
and duration. They never contain features, account IDs, probabilities, SHAP
values, targets, demographics, local paths, or exception text in client-visible
responses.

## Evidence and remaining boundary

The authenticated [parity report](../../reports/inference/phase6_v1/inference-parity-report.md)
uses 20 committed synthetic records. It proves exact offline/batch probability
parity, API agreement within `5e-7`, exact risk-band and reason-category parity,
deterministic capacity selection, trace propagation, endpoint retirement, and
no-rewrite rerun behavior. Row-level outputs remain ignored.

No Phase 6 path fits, tunes, calibrates, loads the sealed test partition, or
changes the model or policy. G4 remains open for robustness stress tests,
registry promotion, image scanning, rollback, monitoring, and runbooks.
