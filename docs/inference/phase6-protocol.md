# Phase 6 inference protocol

**Protocol:** `phase6_v1`

**Status:** frozen before implementation

**Model:** unchanged reviewed `selected_v1` CatBoost bundle
**Training budget:** zero fits

## Interfaces

The primary interface is an idempotent monthly CSV scorer. Input contains one
opaque `account_id` followed by exactly the 19 operational model features.
Demographics, target, unknown columns, non-finite values, and invalid operational
codes are rejected. Valid rows are still scored when other rows fail; partial
completion returns exit code 3 and requires operational review.

The secondary interface is `POST /v1/predict`. The former `/predict` route is
removed. `/ping` and `/ready` retain their current liveness and selected-bundle
readiness meanings.

## Policy and explanations

Valid batch rows are ranked by probability descending and account ID ascending.
Exactly `floor(valid_rows * 0.10)` rows enter the human-owned outreach queue.
Frozen validation-derived risk bands remain independent of the capacity decision.

Native CatBoost SHAP values are aggregated into credit capacity, repayment
status, billing balance, and payment behaviour. The two categories with greatest
absolute raw-log-odds contributions are reported with directional labels. They
describe model behavior only and are neither causal explanations nor adverse-
action reasons. Additivity and sigmoid parity must remain within `1e-10`.

## Idempotency and privacy

The batch identity binds the input, scoring date, snapshot ID, contract, and
model bundle. Identical completed runs are verified and reused byte-for-byte.
Conflicting or corrupt existing destinations fail without overwrite. Runtime
logs use an explicit metadata allowlist and exclude features, account IDs,
probabilities, explanations, demographics, targets, and local paths.

No Phase 6 path may train, refit, tune, calibrate, generate bootstrap evidence,
load the final-test workflow, or score the sealed test partition.
