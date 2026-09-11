# Credit Risk Early-Warning Platform

A portfolio project for monthly credit-risk early warning and
capacity-constrained intervention prioritisation for existing cardholders.

> **Current status:** Phase 1 / G1 and the reviewed Phase 2/3 evidence are
> complete. A simpler authoritative release protocol compared four fixed
> classifiers, one fit each, one shared validation split, and no tuning, repeated
> CV, calibration fit, or winner refit. It selected the exact fitted
> `catboost_fixed` model. One separately authorized, prediction-only evaluation
> then passed every frozen test gate, closing G2 without training or refitting.
> Phase 4 release hardening has permanently retired that consumed evaluator and
> requires exact serving-library compatibility before loading the model. The API
> and Streamlit demo now serve that exact digest-verified winner. Phase 5 work has
> not started.

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
- [Reviewed one-time final-test report](reports/modeling/final_test_v1/final-test-report.md)

## Current capabilities

- A checksum-protected native CatBoost release bundle selected without refitting.
- FastAPI inference plus separate liveness and selected-bundle readiness endpoints.
- A local Streamlit demonstration.
- A reproducible Python 3.12 environment managed through `pyproject.toml` and
  `uv.lock`.
- Checksum-pinned acquisition of the official UCI CSV, strict canonical schema
  validation, deterministic quality evidence, and content-addressed quarantine.
- A sealed 80/20 development/test holdout plus 5-fold × 3-repeat development
  cross-validation assignments tied to a reviewed lineage lock.
- Unit and integration tests that protect the selected-bundle, prediction,
  API-health, CLI, and retained historical-artifact contracts.
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
  commit `f7c99f2` and reviewed validation evidence.
- One immutable prediction-only evaluation of exactly 6,000 test accounts. All
  frozen gates passed, G2 closed, and the reviewed release bundle now serves the API.
- A byte-preserved, non-importable archive of the executed evaluator plus a
  no-option tombstone that rejects every final-test replay before data or model access.
- Exact startup checks for the six inference dependencies recorded by the reviewed
  bundle; intentionally absent modelling and data-validation extras are excluded.

No Phase 5 or later lifecycle implementation has started. Planned releases add
governed explanations, subgroup analysis, the model registry, batch/API parity,
monitoring, rollback, and incident exercises.

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

The one-time test gates were frozen independently from model selection:

```bash
uv run credit-risk model freeze-test
```

That command loads neither data nor the estimator. A separate approval record
then authorized exactly one `credit-risk model final-test` execution. It scored
6,000 unique test accounts with the unchanged selected bundle, performed zero
fits, and passed the frozen average-precision, Brier, and lift gates. Its durable
receipts and the active no-option tombstone permanently prevent reevaluation; the
reviewed evidence is under `reports/modeling/final_test_v1/`. The exact executed
source is preserved as a non-importable text artifact under
`docs/modeling/evidence/`. Historical baseline and candidate reports remain
available, while their public fitting commands fail fast.

### Check the retired compatibility artifacts

```bash
uv run credit-risk doctor
```

`doctor` loads the retained legacy model and preprocessor and validates their shared feature
contract plus the outlier-threshold schema. Because pickle deserialization can
execute code, use this command only with trusted project artifacts.

### Run quality checks

```bash
uv run ruff format --check api.py app.py src/credit_risk tests
uv run ruff check api.py app.py src/credit_risk tests
uv run mypy src/credit_risk/artifacts.py src/credit_risk/data src/credit_risk/modeling src/credit_risk/cli.py api.py app.py
uv run pytest -m "not training" --cov --cov-report=term-missing
uv run pytest tests/unit/data tests/unit/test_data_cli.py tests/integration/test_data_workflow.py --cov=credit_risk.data --cov-branch --cov-fail-under=90
uv run pytest tests/unit/modeling tests/unit/test_modeling_cli.py tests/integration/test_baseline_experiment.py tests/integration/test_candidate_model.py --cov=credit_risk.modeling --cov-branch --cov-fail-under=90
```

### Run the API

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080/docs` for the generated API documentation. `GET /ping`
reports process liveness; `GET /ready` reports that the inference bundle
loaded and passed its compatibility checks. Readiness requires exact agreement
between the bundle manifest and installed `catboost`, `joblib`, `numpy`, `pandas`,
`pydantic`, and `scikit-learn` versions. `mlflow` and `pandera` are intentionally
not runtime requirements. Invalid artifacts or dependency drift fail application
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
`models/selected_v1/manifest.json` and `model.cbm`; legacy artifacts, generated
data, reports, training dependencies, and experiment state are excluded.

## Governed prediction request

`POST /predict` accepts exactly the 19 operational features in the selected-model
contract: credit limit plus six months each of repayment status, signed bill
amount, and non-negative payment amount. Demographics, account ID, target, nulls,
non-finite values, and unknown fields are rejected. The response returns the
default probability, validation-frozen risk band, model ID, and bundle ID.

The synthetic request in `tests/fixtures/prediction_request.json` returns
probability `0.190382` and risk band `standard`. Tests freeze this non-holdout
example so package, dependency, and container changes cannot silently alter the
released model contract.

## Repository structure

```text
.
├── api.py                     # Governed selected-model FastAPI entrypoint
├── app.py                     # Local Streamlit demonstration
├── artifacts/                 # Legacy compatibility artifacts
├── configs/data/              # Source manifest, split policy, and reviewed lock
├── configs/modeling/          # Feature and scientific-baseline contracts
├── data/                      # Ignored reproducible raw/processed/split products
├── docs/                      # Product, roadmap, governance, and ADR evidence
│   └── modeling/evidence/     # Non-importable archive of the consumed evaluator
├── experiment/                # Ignored MLflow, OOF, and exploratory evidence
├── models/selected_v1/        # Digest-protected released model bundle
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
