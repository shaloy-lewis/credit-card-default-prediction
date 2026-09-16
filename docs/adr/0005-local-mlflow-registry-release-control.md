# ADR 0005: Use a local MLflow registry for controlled release transitions

**Status:** Accepted

**Date:** 2026-09-16
**Decision owner:** Project owner

## Context

Release A produced one reviewed model bundle, and Phase 6 proved batch/API
parity. The next portfolio milestone must demonstrate registration, manual
promotion, local deployment, and rollback without introducing another model or
repeating training. The full PostgreSQL, MinIO, and long-running MLflow service
belongs to the following platform phase.

## Decision

Phase 7 uses MLflow Model Registry 3.15.0 with a local SQLite backend and a
content-addressed filesystem artifact store. The runtime registry is ignored;
deterministic aggregate evidence is committed separately.

Two MLflow model versions represent `phase7_rev_001` and `phase7_rev_002`.
Both revisions contain the exact reviewed `selected_v1` manifest and model
bytes. They demonstrate release-control mechanics only and must never be
described as different model candidates or as evidence of improved quality.

Promotion is manual and approval-bound. The candidate becomes champion only
after the pinned implementation commit has passed the quality and scanned
container checks. The previous champion becomes the rollback target. Rollback
requires a separate approval and changes the registry aliases and active local
deployment pointer through one controlled command.

The API image remains independent of MLflow. A read-only deployment root can
be mounted for the local drill; otherwise the existing committed bundle remains
the startup default.

## Consequences

- Registry state, approvals, deployment state, and rollback are independently
  testable without model fitting or sealed-test access.
- The drill proves lifecycle controls but not behaviour differences between
  model versions.
- PostgreSQL, MinIO, a persistent MLflow server, monitoring, and incident
  automation remain later work.
- G4 and Release B remain open after this slice because robustness and
  monitoring controls are not part of Phase 7.
