# Phase 8 persistent local platform protocol

Phase 8 is governed by `configs/platform/phase8_v1.json`. It starts from merged
Phase 7 commit `447813d` and keeps the reviewed model and all prior evidence
byte-identical.

## Service boundary

- PostgreSQL stores MLflow metadata on a named volume.
- MinIO stores immutable bundle objects on a separate named volume.
- MLflow is the control plane and proxies artifact access.
- The bootstrap job verifies every digest, rebuilds the approved final alias
  state, and writes the active deployment to a dedicated volume.
- Bootstrap mutations are serialized by the frozen PostgreSQL advisory-lock key
  `-4653285090134190835`. Lock contention fails immediately, before MinIO,
  MLflow, or deployment writes; verification remains read-only and lock-free.
- FastAPI mounts only that deployment volume and contains no MLflow package.
- Streamlit calls `/v1/predict`; it never deserializes a model.

PostgreSQL and the MinIO object API remain internal to the Compose network.
MLflow, the MinIO console, API, and UI are bound to localhost-facing ports for
the portfolio demonstration.

## Persistence and failure policy

An identical restart is a verified no-op. Missing, partial, foreign, symlinked,
or digest-mismatched state fails closed. The routine workflow never deletes
volumes and the documentation provides no destructive reset command. Phase 7's
SQLite registry is neither read as a migration source nor modified.

The registered-model object and both registered versions have distinct exact
tag contracts. Verification rejects missing, altered, or additional model-level
lineage tags as well as version, source, alias, object, and deployment drift.
Missing or unsafe paths are normalized into controlled CLI failures without a
traceback. Direct workflow callers receive the same platform-domain error for
MLflow client, model, version, alias, and deployment failures. Deployment paths
are checked lexically before mutation, and every existing ancestor or nested
release component must be free of symlinks; cleanup applies the same boundary.

Only `environment=None` opts into process-environment discovery. An explicitly
supplied empty or partial mapping is authoritative and cannot inherit ambient
credentials. The MLflow launcher percent-encodes the PostgreSQL database name,
user, and password with the standard library, passes the resulting backend URI
only through `MLFLOW_BACKEND_STORE_URI`, and keeps database credentials out of
the process argument vector and application output.

The prerequisite gate requires configuration validation, dependency locking,
unit tests for bootstrap contracts, Compose validation, and a running Docker
engine. The local prerequisite rehearsal passed clean startup, repeated-bootstrap
idempotency, restart persistence, API/UI health, exact object identity, and zero
fitting or sealed-test access. CI repeats those runtime checks, scans the new
MLflow and UI images, and publishes their SBOMs.

The blocking image scan discovered fixable vulnerabilities after the frozen
base-image digest was published. The MLflow and Streamlit runtime images
therefore install Debian's fixed `libpcre2-8-0` package
`10.42-1+deb12u1`, and the locked platform extra
requires GitPython `3.1.59` and cryptography `50.0.0`. These are security-only
updates: the base-image digest, platform contract, model, registry, and serving
policy remain unchanged, and the scan threshold is not relaxed.
MLflow's upstream fix changed only its cryptography upper-bound metadata; the
lock applies that same compatibility relaxation while retaining frozen MLflow
`3.15.0` and verifies the complete platform behavior in tests and containers.

The deployment named volume is created root-owned by Docker. A read-only-root
one-shot bootstrap container therefore runs as UID 0 solely to initialise that
single writable volume. The long-running MLflow, API, and UI processes run as
their image users; the API receives the deployment volume read-only. The later
official Phase 8 evidence run must authenticate these results in a deterministic
aggregate package before Week 9 is marked complete.
