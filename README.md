# Credit Risk Early-Warning Platform

A portfolio project for monthly credit-risk early warning and
capacity-constrained intervention prioritisation for existing cardholders.

> **Current status:** Portfolio v2 engineering foundation. The committed
> CatBoost model remains available for compatibility testing while the data,
> modelling, governance, registry, and monitoring workflows are rebuilt in
> staged releases.

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

## Current capabilities

- A legacy CatBoost default-probability model and preprocessing artifacts.
- FastAPI inference plus separate liveness and inference-readiness endpoints.
- A local Streamlit demonstration.
- A reproducible Python 3.12 environment managed through `pyproject.toml` and
  `uv.lock`.
- Unit and integration tests that protect the legacy transformation, artifact,
  prediction, API-health, and CLI contracts.
- Ruff, mypy, pytest, pre-commit, and GitHub Actions quality gates.
- A non-root, locked-dependency Docker API image.

Planned releases add reproducible data acquisition, scientific baselines,
calibration, capacity-based policies, MLflow tracking and registry, model-risk
gates, batch/API parity, monitoring, and incident exercises.

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
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop for the container demonstration

### Install the locked environment

```bash
uv sync --locked --all-extras --dev
```

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
uv run mypy src/credit_risk/artifacts.py src/credit_risk/cli.py api.py app.py
uv run pytest --cov --cov-report=term-missing
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
├── docs/                      # Product, roadmap, governance, and ADR evidence
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
