# Delivery progress

This log records completed checkpoints, verification evidence, and consciously
deferred risks. It is not a substitute for commit history or CI results.

## Phase 0 — approved

**Completed:** 2026-08-07

- Product and decision contract accepted.
- Batch-first architecture accepted.
- Twelve-week, zero-cost local delivery plan accepted.
- MLOps and model governance selected as the principal seniority signals.

## Week 1 — engineering foundation complete

**Completed:** 2026-08-08

### Delivered

- Migrated the import package from the ambiguous `src` name to a true
  `src/credit_risk` layout.
- Converted training from an import-time side effect into an explicit workflow.
- Added a cross-platform `credit-risk` CLI with `version`, `doctor`, and `train`
  commands.
- Replaced empty/duplicate packaging files and hand-maintained dependency lists
  with PEP 621 metadata and a committed `uv.lock`.
- Preserved the legacy binary stack by pinning CatBoost, scikit-learn, NumPy,
  pandas, Pydantic, Typer, and Click compatibility versions.
- Added Ruff, mypy, pytest, coverage, pre-commit, and GitHub Actions gates.
- Added unit and integration tests for preprocessing, CLI behaviour, API health,
  artifact loading, input-column order, and the frozen legacy prediction.
- Added one shared trusted-artifact readiness contract for the model,
  preprocessor, threshold schema, binary classes, transformed feature names,
  and feature-importance dimensions.
- Made FastAPI load one validated pipeline during startup, expose `/ping` for
  process liveness and `/ready` for inference readiness, and reuse that pipeline
  for every prediction.
- Derived the package version from installed distribution metadata and restricted
  the runtime image to the three approved inference artifacts.
- Rebuilt the API image from the lockfile using a multi-stage Dockerfile and a
  non-root runtime user.
- Reframed the README and Streamlit problem statement to match the approved
  early-warning scope and evidence limitations.

### Verification evidence

| Check | Result |
| --- | --- |
| Ruff lint | Passed |
| Ruff formatting | 28 maintained Python files formatted |
| Mypy | Passed for artifact validation, CLI, FastAPI, and Streamlit boundaries |
| Pytest | 26 passed |
| Legacy probability | Preserved at `0.44088` for the documented request |
| Coverage baseline | 52% overall; 88% for artifact validation and 87% for legacy inference; no vanity gate set |
| CLI doctor | Loaded the committed trusted artifacts and passed the full readiness contract |
| Docker build | Baseline image previously passed from the cross-platform frozen lockfile |
| Container contract | CI now verifies non-root execution, the three-file allowlist, `/ping`, `/ready`, and `/predict`; the remediation image awaits a responsive local Docker daemon or the next CI run |
| Docker Compose | Configuration validated; health check targets `/ready` |

### Deferred risks

- The committed pickle artifacts remain a temporary compatibility mechanism.
  Model registry packaging and promotion replace manual artifact loading later.
- Instance SHAP values still use the legacy raw/transformed feature mapping.
  The API marks this compatibility behaviour explicitly; governance Week 6 adds
  dimensional, naming, additivity, aggregation, and reason-category tests.
- The current API input model still has weak domain validation. Versioned schemas
  are introduced with the inference contract.
- The existing modelling components remain lightly typed and under-tested. Tests
  expand when the reproducible data and scientific pipelines replace them.
- The local API image is approximately 986 MB. Image composition and dependency
  reduction will be revisited after the runtime boundary stabilises.
- Starlette currently emits an upstream `python-multipart` pending-deprecation
  warning during test import; it does not affect the tested endpoints.

## Next checkpoint — Week 2 reproducible data

The next slice will add the official UCI acquisition manifest, checksum,
immutable raw-data convention, schema contract, validation failure policy, data
card, feature-availability matrix, and deterministic split manifest. It will not
begin model comparison until the data gate passes.
