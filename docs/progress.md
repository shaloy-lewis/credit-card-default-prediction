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
| Container contract | Passed in CI and in the local regression: non-root execution, the three-file allowlist, `/ping`, `/ready`, and `/predict` |
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

## Phase 1 / Week 2 — reproducible data and G1 complete

**Completed:** 2026-08-18

### Delivered

- Pinned UCI dataset 350 to the official normalized CSV by URL, byte size,
  SHA-256, ordered source schema, row count, and target distribution.
- Added retry-bounded streaming acquisition, immutable content-addressed raw
  storage, atomic no-overwrite publication, offline reuse, and hash-addressed
  quarantine for corrupt or conflicting bytes.
- Added semantic canonical names and a strict Pandera-backed data contract. The
  pipeline rejects structural, type, identifier, domain, and class-count drift
  while preserving and reporting documented source anomalies.
- Added deterministic canonical CSV and quality-report generation with
  transactional promotion and stable, sampled failure evidence.
- Sealed an 80/20 stratified development/test holdout and 5-fold × 3-repeat
  development-only cross-validation protocol at seed 42.
- Added deterministic per-account split assignments plus a committed reviewed
  lock tying the source, canonical table, split configuration, scikit-learn
  version, counts, and assignment digest together without timestamps.
- Added `credit-risk data fetch`, `build`, and strictly offline `verify`
  interfaces. The legacy `credit-risk train` path remains compatibility-only.
- Added the dataset card, feature-availability/leakage review, validation and
  quarantine policy, and clean-checkout reproduction instructions.
- Kept Pandera in the optional `data` dependency boundary; the inference image
  includes the data CLI package but not Pandera or generated data.

### Verification evidence

| Check | Result |
| --- | --- |
| Toolchain | Python 3.12 with `uv 0.11.28`; frozen lock passes `uv lock --check` |
| Official source | 2,897,080 bytes; SHA-256 `45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0` |
| Canonical data | 30,000 validated rows; SHA-256 `75b2a746781a584b0456f843f1f269190b51e90983cba44c4ed6c4a8685e6c1c` |
| Split assignments | 24,000 development and 6,000 test rows; SHA-256 `2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e` |
| Offline verification | Passed against the reviewed split lock with no network access |
| Full tests | 183 passed; one documented upstream Starlette warning |
| Phase 1 branch coverage | 157 tests passed; 97.19% for `credit_risk.data` against a 90% CI gate |
| Static gates | Ruff format/lint, mypy, pre-commit, and whitespace checks passed |
| Inference compatibility | Artifact doctor passed and documented prediction probability remained `0.44088` |
| Container contract | Compose, non-root user, source-package inclusion, optional-extra isolation, three-artifact allowlist, liveness, readiness, and prediction checks passed |

### Accepted limitations and deferred work

- The source is a static 2005 Taiwan sample with no event timestamps, India
  validation, or defensible out-of-time split. It supports engineering and
  governance demonstrations, not contemporary portfolio-performance claims.
- Demographic columns are retained for audit and ablation but excluded from the
  default predictive policy until the fairness decision is completed.
- Generated data remains intentionally Git-ignored; a clean checkout must fetch
  the exact pinned public bytes before building, then can verify fully offline.
- Legacy training and committed pickle artifacts are not connected to the new
  canonical data boundary. Replacing that compatibility path belongs to the
  modelling and registry phases.

## Next checkpoint — Week 3 scientific baselines

Build leakage-safe baseline experiments on the sealed development assignments,
define decision-relevant evaluation metrics and uncertainty, and keep the test
partition untouched until the model and policy selection protocol is frozen.

### Implementation status — code complete; clean evidence pending

**Started:** 2026-08-19

- Phase 2 is limited to the Week 3 baseline-and-tracking checkpoint; G2 remains
  open through candidate modelling, calibration, uncertainty, and stress tests.
- The governed feature boundary, three-baseline protocol, metric hierarchy, and
  local SQLite MLflow design are locked before the first experiment.
- The development-only workflow, exact OOF coverage checks, machine-readable
  metric priorities, content-validated MLflow artifacts, and fold-level
  logistic diagnostics are implemented.
- Two consecutive provisional executions produced identical OOF and diagnostic
  bytes: 216,000 OOF rows at SHA-256
  `c8ec30bec3c323ed0cfbe050aa3313ac356eb5d717ab305dee1b4365a0e51abe`
  and 15 logistic fold records at SHA-256
  `9a6c0ebe027fe00eda305d319bdf4dd1c7dfc84e470f3a8a7e00cf387ffda425`.
- Development-only repeat means were: logistic average precision `0.541294`,
  ROC-AUC `0.767968`, lift at 10% `3.156903`, and Brier score `0.136362`;
  the fixed repayment rule achieved average precision `0.473102` and lift at
  10% `2.909619`. These are baseline evidence, not holdout results or promotion
  thresholds.
- Implementation evidence remains provisional until it is rerun from a clean,
  reviewed commit. No test-partition metric is permitted in this phase.
