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
base-image digest was published. The runtime images originally installed Debian's
fixed `libpcre2-8-0` package `10.42-1+deb12u1`. During recording preparation on
2026-10-21, both container CI jobs failed because that revision was no longer
available from the live Bookworm repositories. The API, MLflow and Streamlit
runtime images now pin `10.42-1+deb12u2`, the replacement listed by
[Debian's security tracker](https://security-tracker.debian.org/tracker/source-package/pcre2).
The historical Release B evidence retains its original source and image hashes;
the updated images require their own passing CI checks. The locked platform extra
now requires GitPython `3.1.60` and retains cryptography `50.0.0`. The subsequent
image scan also identified fixed vulnerabilities in Perl and Python image/network
dependencies. Runtime images pin Perl `5.36.0-7+deb12u4`, and the lock updates
Pillow to `13.0.0`, GitPython to `3.1.60`, aiohttp to `3.14.4`, certifi to
`2026.10.19` and urllib3 to `2.8.0`. aiohttp also requires yarl `1.25.1`.
All other locked package versions are retained. These are security-only
updates: the base-image digest, platform contract, model, registry, and serving
policy remain unchanged, and the scan threshold is not relaxed.
MLflow's upstream fix changed only its cryptography upper-bound metadata; the
lock applies that same compatibility relaxation while retaining frozen MLflow
`3.15.0` and verifies the complete platform behavior in tests and containers.

Streamlit `1.64.0` declares Pillow `<13`. The explicit uv override selects
Pillow `13.0.0` to address the blocking findings while retaining the existing UI.
The [compatibility tests](../../tests/integration/test_demo_ui.py) exercise PNG
and JPEG conversion/resizing plus the Predictor against the real API with the
reviewed synthetic fixture. This verifies those local demo paths; it is not a
claim of general upstream compatibility. Remove the override when the selected
Streamlit release supports Pillow 13. The model's exact dependency contract and
all authenticated historical evidence remain unchanged.

The deployment named volume is created root-owned by Docker. A read-only-root
one-shot bootstrap container therefore runs as UID 0 solely to initialise that
single writable volume. The long-running MLflow, API, and UI processes run as
their image users; the API receives the deployment volume read-only. The later
official Phase 8 evidence run must authenticate these results in a deterministic
aggregate package before Week 9 is marked complete.
