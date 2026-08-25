# Credit Risk Early-Warning Platform

A portfolio project for monthly credit-risk early warning and
capacity-constrained intervention prioritisation for existing cardholders.

> **Current status:** Phase 1 / G1 is complete and Phase 2's governed scientific
> baseline workflow is in progress. The committed CatBoost model remains
> available for compatibility testing while modelling, registry, serving, and
> monitoring workflows are rebuilt in staged releases.

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

Planned releases add candidate modelling, calibration, capacity-based policies,
the model registry, model-risk gates, batch/API parity, monitoring, and incident
exercises.

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
governance evidence, and the reviewed lock are version controlled. The existing
`credit-risk train` command is compatibility-only until the modelling workflow
is migrated to these governed inputs.

### Run the governed scientific baselines

Install both the `data` and `modeling` extras, then verify the sealed data
lineage before starting an experiment:

```bash
uv sync --locked --extra data --extra modeling --dev
uv run credit-risk data verify
uv run credit-risk model baseline --allow-dirty \
  --output-root experiment/provisional/baseline_v1
```

`--allow-dirty` is for exploratory development only. A reviewed evidence run
must execute from a clean implementation commit and writes its deterministic
aggregate report under `reports/modeling/baseline_v1`. Runtime MLflow SQLite
state, artifacts, and row-level out-of-fold predictions remain under the
Git-ignored `experiment/` root. The command never evaluates the test partition.
The same runtime boundary stores schema-validated logistic convergence and
coefficient diagnostics as JSON; no fitted model pickle is logged.

The [baseline experiment protocol](docs/modeling/experiment-protocol.md)
defines the exact feature boundary, models, metrics, tie handling, lineage, and
failure policy.

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
uv run pytest tests/unit/modeling tests/unit/test_modeling_cli.py tests/integration/test_baseline_experiment.py --cov=credit_risk.modeling --cov-branch --cov-fail-under=90
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
