# Credit Risk Early-Warning Platform

A portfolio project for monthly credit-risk early warning and
capacity-constrained intervention prioritisation for existing cardholders.

> **Current status:** Phase 1 / G1 and the reviewed Phase 2/3 evidence are
> complete. A simpler authoritative release protocol compared four fixed
> classifiers, one fit each, one shared validation split, and no tuning, repeated
> CV, calibration fit, or winner refit. It selected the exact fitted
> `catboost_fixed` model. The test remains sealed and G2 remains
> open. The committed compatibility model continues to serve the existing API
> until a later, explicitly governed inference migration.

## Product intent

At the end of a monthly billing cycle, the proposed system ranks eligible
existing accounts by calibrated probability of next-month default. A
hypothetical human-owned policy can then allocate a limited review or proactive
support capacity.

This is not a new-customer underwriting system. It does not autonomously approve
or deny credit, alter limits or pricing, initiate collections, or provide legally
sufficient adverse-action reasons.

The approved scope and delivery evidence are documented in:

- [Product and decision brief](docs/product-brief.md)
- [Twelve-week roadmap](docs/roadmap.md)
- [Batch-first architecture decision](docs/adr/0001-batch-first-scoring.md)
- [Dataset card and evidence limits](docs/data/data-card.md)
- [Feature availability and leakage review](docs/data/feature-availability.md)
- [Data validation and quarantine policy](docs/data/validation-policy.md)
- [Baseline experiment protocol](docs/modeling/experiment-protocol.md)
- [Reviewed baseline report](reports/modeling/baseline_v1/baseline-report.md)
- [Frozen candidate modelling protocol](docs/modeling/candidate-protocol.md)
- [Reviewed candidate report](reports/modeling/candidate_v1/candidate-report.md)
- [One-pass model selection protocol](docs/modeling/selection-protocol.md)
- [Reviewed one-pass selection report](reports/modeling/selection_v1/selection-report.md)

## Current capabilities

- A legacy CatBoost default-probability model and preprocessing artifacts.
- FastAPI inference plus separate liveness and inference-readiness endpoints.
- A local Streamlit demonstration.
- A reproducible Python 3.12 environment managed through `pyproject.toml` and
  `uv.lock`.
- Checksum-pinned acquisition of the official UCI CSV, strict canonical schema
  validation, deterministic quality evidence, and content-addressed quarantine.
- A sealed 80/20 development/test holdout plus 5-fold × 3-repeat development
  cross-validation assignments tied to a reviewed lineage lock.
- Unit and integration tests that protect the legacy transformation, artifact,
  prediction, API-health, and CLI contracts.
- Ruff, mypy, pytest, pre-commit, and GitHub Actions quality gates.
- A non-root, locked-dependency Docker API image.
- A versioned Week 3 experiment protocol that keeps baseline fitting and
  evaluation on the reviewed development folds only.
- A governed 19-feature modelling view, three deterministic baselines,
  repeated-CV capacity metrics, SQLite MLflow lineage, and non-executable
  fold-level logistic diagnostics.
- A clean, digest-protected Phase 2 baseline report tied to reviewed commit
  `c695c60`, with the sealed holdout explicitly unevaluated.
- A versioned Phase 3 CatBoost contract with eight bounded search variants,
  150 reviewed fold fits, deterministic advancement and fallback rules,
  content-bound NumPy checkpoints, and two-run evidence verification.
- Digest-protected Phase 3 aggregate evidence selecting the lightweight
  `cb_cfg_006` configuration from development folds only; this remains historical evidence.
- A frozen one-pass comparison of logistic regression, histogram gradient
  boosting, random forest, and fixed CatBoost, with an exact four-fit budget,
  validation guardrails, deterministic simplicity tie-break, and no winner refit.
- A checksum-protected native CatBoost winner bundle tied to clean implementation
  commit `f7c99f2` and reviewed validation evidence; it is not yet connected to the API.

Planned releases add calibration, capacity-based policies, the model registry,
model-risk gates, batch/API parity, monitoring, and incident exercises.

## Dataset and evidence limits

The first release uses the public UCI Default of Credit Card Clients dataset: a
historical sample of 30,000 Taiwanese customers. It provides one modelling
snapshot, not repeated account-month observations.

Consequently, this repository does not claim:

- validity for Indian customers or a current lender portfolio;
- genuine out-of-time or longitudinal performance;
- realised financial or causal intervention impact; or
- compliance with RBI, Basel, DPDP, or another regulation.

Synthetic data may later test operational failures, batch volume, and drift. It
will not be used as evidence of real model performance.

## Local development

### Prerequisites

- Python 3.12
- [uv 0.11.28](https://docs.astral.sh/uv/), matching CI and the container build
- Docker Desktop for the container demonstration

### Install the locked environment

```bash
uv sync --locked --all-extras --dev
```

### Reproduce the governed data snapshot

```bash
uv run credit-risk data fetch
uv run credit-risk data build
uv run credit-risk data verify
```

`fetch` makes acquisition of the checksum-pinned UCI CSV explicit. `build` is
idempotent and can perform the same acquisition when the verified raw snapshot
is absent; it then validates the canonical schema and creates the sealed
holdout and cross-validation assignments. `verify` is strictly offline and
checks the complete raw-to-split lineage against the reviewed lock.

Downloaded raw data, processed outputs, quality reports, and split assignments
remain under the Git-ignored root `data/` directory. Source and split manifests,
governance evidence, and the reviewed lock are version controlled. The legacy
`credit-risk train` command and the Phase 2/3 experiment commands are retired;
their reviewed evidence and source remain available for audit.

### Run the governed one-pass selection

Install the `data` and `modeling` extras, verify the sealed lineage, and commit
the reviewed implementation before the official run:

```bash
uv sync --locked --extra data --extra modeling --dev
uv run credit-risk data verify
uv run credit-risk model select
```

The reviewed run fitted exactly four fixed models on 19,200 development rows, evaluated
the same 4,800 validation accounts, and atomically publishes the aggregate
evidence plus the exact winner bundle without refitting. Row-level predictions,
bootstrap evidence, and MLflow state remain ignored under `experiment/`.
`catboost_fixed` won with validation average precision `0.556510`, Brier score
`0.133539`, and lift at 10% `3.210923`. The native bundle is protected by its
manifest and model digests. If a future selection produces joblib, it has pickle
semantics and must be loaded only as a trusted local input after digest verification.

After the evidence and bundle are reviewed and committed, freeze—but do not
execute—the one-time test authorization:

```bash
uv run credit-risk model freeze-test
```

This command loads neither data nor the estimator. `credit-risk model final-test`
is intentionally disabled until a separate explicit request implements and
authorizes the one-time sealed-test evaluation. The reviewed authorization is
now committed at `configs/modeling/final_test_v1.json`; it is explicitly marked
not executed and not authorized. The historical baseline and
candidate reports remain available, but their public fitting commands fail fast
with guidance to use `model select`.

### Check the local inference artifacts

```bash
uv run credit-risk doctor
```

`doctor` loads the model and preprocessor and validates their shared feature
contract plus the outlier-threshold schema. Because pickle deserialization can
execute code, use this command only with trusted project artifacts.

### Run quality checks

```bash
uv run ruff format --check api.py app.py src/credit_risk tests
uv run ruff check api.py app.py src/credit_risk tests
uv run mypy src/credit_risk/artifacts.py src/credit_risk/data src/credit_risk/modeling src/credit_risk/cli.py api.py app.py
uv run pytest --cov --cov-report=term-missing
uv run pytest tests/unit/data tests/unit/test_data_cli.py tests/integration/test_data_workflow.py --cov=credit_risk.data --cov-branch --cov-fail-under=90
uv run pytest tests/unit/modeling tests/unit/test_modeling_cli.py tests/integration/test_baseline_experiment.py tests/integration/test_candidate_model.py --cov=credit_risk.modeling --cov-branch --cov-fail-under=90
```

### Run the API

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080/docs` for the generated API documentation. `GET /ping`
reports process liveness; `GET /ready` reports that the inference bundle
loaded and passed its compatibility checks. Invalid artifacts fail application
startup instead of leaving a non-functional service marked ready.

### Run the Streamlit demo

```bash
uv run streamlit run app.py
```

### Run with Docker

```bash
docker compose up --build
```

The API is exposed at `http://localhost:8080`. The runtime image contains only
`model.pkl`, `preprocessor.pkl`, and `outlier_threshold.json`; generated raw,
train, and test datasets are excluded from the image.

## Legacy prediction request

The compatibility endpoint remains `POST /predict` during the engineering
foundation. It accepts six months of bill, payment, and repayment-status history
plus the existing account attributes. Its current output should be treated as a
legacy contract: the endpoint, schemas, calibrated policy output, and reviewed
reason categories will be versioned in later phases.

The tests freeze the documented sample probability at `0.44088` so that package,
dependency, and container refactors cannot silently change model behaviour.

## Repository structure

```text
.
├── api.py                     # Legacy-compatible FastAPI entrypoint
├── app.py                     # Local Streamlit demonstration
├── artifacts/                 # Legacy compatibility artifacts
├── configs/data/              # Source manifest, split policy, and reviewed lock
├── configs/modeling/          # Feature and scientific-baseline contracts
├── data/                      # Ignored reproducible raw/processed/split products
├── docs/                      # Product, roadmap, governance, and ADR evidence
├── experiment/                # Ignored MLflow, OOF, and exploratory evidence
├── reports/modeling/          # Reviewed aggregate experiment evidence
├── src/credit_risk/           # Installable application package
├── tests/                     # Unit, integration, and compatibility tests
├── pyproject.toml             # Direct dependencies and tool configuration
├── uv.lock                    # Exact cross-platform dependency resolution
├── Dockerfile
└── docker-compose.yml
```

Generated data, logs, environments, caches, and experiment outputs are excluded
from version control.

## Delivery milestones

- **Release A — defensible model:** reproducible data, baselines, calibration,
  uncertainty, and capacity-aware evaluation.
- **Release B — governed ML product:** model/data cards, subgroup analysis,
  reason-code tests, registry promotion gates, and rollback.
- **Release C — local platform:** batch/API parity, Docker Compose services,
  monitoring, incident drills, and recorded portfolio demo.

See the [roadmap](docs/roadmap.md) for weekly acceptance gates and the honest
mapping from the local implementation to Azure Databricks production concepts.
