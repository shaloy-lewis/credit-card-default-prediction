# ADR 0004: Use one governed engine for batch and versioned API inference

**Status:** Accepted  
**Date:** 2026-09-16  
**Decision owner:** Project owner

## Context

ADR 0001 made monthly batch scoring the primary product interface. Release A
produced a reviewed `selected_v1` bundle, while the existing API remained an
unversioned single-record demonstration that loaded the same model but did not
provide batch completeness, capacity ranking, trace metadata, or reviewed reason
categories.

## Decision

Phase 6 introduces one prediction-only engine for both monthly batch and online
scoring. The batch interface scores valid rows, quarantines invalid rows, ranks
the valid portfolio, and selects no more than the frozen 10% human-review
capacity. Partial completion publishes evidence and returns exit code 3 so an
orchestrator cannot treat rejected rows as an unnoticed success.

The online interface moves directly to `POST /v1/predict`; the unversioned
prediction endpoint is removed. Liveness and readiness endpoints remain stable.
Both paths return the same probability policy and the same two reviewed native
SHAP reason categories. These are non-causal model attributions, not adverse-
action reasons.

Every batch is keyed by the input digest, scoring date, snapshot identifier,
protocol digest, and reviewed bundle digests. An identical rerun verifies and
reuses prior outputs without rewriting them. A changed or corrupt prior run is
never overwritten silently.

## Consequences

- Batch and API behavior can be contract-tested for numerical and policy parity.
- Row rejection, completeness, lineage, and idempotency become observable.
- The API change is intentionally breaking and advances the package to `0.2.0`.
- Native SHAP adds prediction latency but no fitting or model mutation.
- Registry promotion, rollback, stress testing, and production monitoring remain
  later G4/G5 work.
