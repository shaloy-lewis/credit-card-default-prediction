# ADR 0006: Bootstrap a separate persistent local MLOps platform

**Status:** Accepted

**Date:** 2026-09-17

## Context

Phase 7 proved release transitions with an ignored SQLite registry and local
content-addressed artifacts. Week 9 requires a reproducible, zero-cost stack
with durable metadata and object storage without weakening the reviewed model
or rewriting the Phase 7 evidence chain.

## Decision

Phase 8 uses Docker Compose with PostgreSQL 16, MinIO, MLflow 3.15, the existing
FastAPI image, and a separate Streamlit image. External images are immutable
digest references. Named volumes hold database metadata, object bytes, and the
active deployment. The API keeps no MLflow dependency.

The platform is bootstrapped into a new namespace. It uploads the exact two-file
`selected_v1` bundle, creates two deployment revisions that share those bytes,
restores Phase 7's final `champion=1` and `rollback=2` aliases, and materialises
the authenticated revision-1 deployment pointer. This is deterministic
re-registration, not migration of the Phase 7 SQLite database.

All bootstrap mutations are protected by the fail-fast PostgreSQL advisory lock
with signed key `-4653285090134190835`. One session holds the lock from before
the first persistent write through final verification; closing that session
releases it on both success and failure. Read-only verification takes no lock.
The registered-model tags are part of the exact lineage contract, not merely
informational metadata.

Credentials live in an ignored `.env`; the committed `.env.example` contains
local placeholders only. PostgreSQL and the MinIO object API are not published
to the host. The MLflow UI, API, Streamlit UI, and MinIO console are local demo
interfaces, not externally secured production services.

The internal MLflow launcher percent-encodes PostgreSQL database and credential
components and supplies the URI via `MLFLOW_BACKEND_STORE_URI`, never as a
process argument. Workflow dependency injection is exact: `environment=None`
uses the process environment, while an explicit empty or partial mapping never
falls back to ambient values. External MLflow operations and active-deployment
resolution fail through one controlled platform-domain error boundary. Both
publication and cleanup reject a symlink in any existing deployment ancestor or
nested release component.

## Consequences

- Phase 7 runtime and evidence remain immutable.
- Startup is idempotent only after the complete registry, object, and deployment
  state verifies successfully.
- Concurrent bootstrap attempts fail before persistent writes rather than wait,
  retry, or create conflicting registry versions.
- Service restarts must preserve registry aliases and artifact bytes.
- Phase 8 proves local platform architecture, not availability, disaster
  recovery, production authentication, monitoring, or compliance.
- No training, refitting, tuning, or sealed-test access is permitted.
