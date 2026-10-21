# Credit Risk Early-Warning Platform

Rank existing credit-card accounts for a capacity-constrained human review queue.
This independently developed portfolio connects statistical model selection with
reproducible inference, operational controls and an honest plan to measure impact.

**The decision:** after a monthly snapshot closes, a hypothetical credit-risk
operations manager reviews the highest-risk 10% of valid accounts. The model
predicts the published next-month default label; people establish operational
eligibility, choose appropriate support and own every consequential action.

**Measured historical evidence:** one authorized 6,000-account evaluation of the
frozen CatBoost model recorded **AP 0.542867**, **Brier 0.136304** and
**lift at 10% 3.089676**. The 600-account queue contained **410 defaults** and
190 non-defaults; 917 defaults remained outside it. These results establish
historical ranking and workload trade-offs. They do not measure prevented
defaults or savings. [Authenticated results](reports/releases/release_a_v1/release-a-report.md)
and [decision implications](docs/portfolio/decision-analysis.md) supply the evidence.

**My ownership — Shaloy Lewis:** problem formulation, data/feature contracts,
historical experiments, the final four-classifier selection, shared batch/API
inference, artifact verification, monitoring and release evidence. The selected
model retains 19 operational predictors, identity calibration and its exact fitted
bytes. Earlier development experiments are retained; the final test is consumed.
This is personal portfolio work, not employment or a live bank deployment.

**Implemented locally:** validated, deterministic and idempotent batch scoring;
FastAPI and Streamlit; MLflow release controls; human-reviewed monitoring and
incident drills. SQLite demonstrates promotion/rollback. PostgreSQL/MinIO
demonstrates fixed-state persistence/recovery. Cloud mappings are conceptual.

**Limits and status:** historical 2005 Taiwan snapshot; no longitudinal or
current-population validation, demonstrated intervention benefit, fairness or
regulatory certification. G3 conditions remain, G4 is closed for approved local
Release B scope, G5 is open. Release C remains planned; video and owner acceptance
are outstanding in the [Week 12 record](docs/portfolio/week12-review.md).

## Choose a review path

| Audience / time | Suggested route |
| --- | --- |
| Recruiter or hiring manager / 60–90 seconds | Overview above → [executive case study](docs/portfolio/case-study.md) → [three evidence-backed resume bullets](docs/portfolio/interview-packet.md#three-resume-bullets). |
| Senior Data Scientist or Applied Scientist / 5–10 minutes | [Scientific decisions and EDA reasoning](docs/portfolio/scientific-decisions.md) → [authenticated evaluation](reports/releases/release_a_v1/release-a-report.md) → [decision economics](docs/portfolio/decision-analysis.md) → [proposed causal study](docs/portfolio/intervention-study.md). |
| Senior ML Engineer or technical interviewer / 10–20 minutes | [Architecture and trade-offs](docs/portfolio/architecture.md) → [local quickstart](#local-development) → [repeatable synthetic demonstration](docs/portfolio/demo-script.md) → [interview packet](docs/portfolio/interview-packet.md). |

The [senior portfolio audit](docs/portfolio/senior-portfolio-audit.md) distinguishes
existing capabilities from presentation gaps and blocked future evidence. The
[claims inventory](docs/portfolio/claims-inventory.md) maps measurements,
synthetic checks, assumptions and proposals to their sources.

**Stack:** Python 3.12, uv, pandas/Pandera, scikit-learn, CatBoost, FastAPI,
Streamlit, MLflow, SQLite, PostgreSQL, MinIO, Docker and GitHub Actions;
Ruff, mypy, pytest and pre-commit for maintained code.

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
- [Release A audit-closure decision](docs/adr/0003-release-a-audit-closure.md)
- [Local registry and release-control decision](docs/adr/0005-local-mlflow-registry-release-control.md)
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
- [Corrected Phase 5 governance protocol](docs/governance/phase5-protocol.md)
- [Reviewed Phase 5 governance report](reports/governance/phase5_v1/governance-report.md)
- [Phase 5 model card](reports/governance/phase5_v1/model-card.md)
- [G3 review decision](reports/governance/phase5_v1/g3-review.md)
- [Authenticated Release A dossier](reports/releases/release_a_v1/release-a-report.md)
- [Release A evidence manifest](reports/releases/release_a_v1/evidence-manifest.json)
- [Phase 6 inference contract](docs/inference/phase6-protocol.md)
- [Phase 6 inference architecture](docs/inference/architecture.md)
- [Reviewed Phase 6 parity report](reports/inference/phase6_v1/inference-parity-report.md)
- [Phase 6 evidence manifest](reports/inference/phase6_v1/evidence-manifest.json)
- [Phase 7 registry protocol](docs/registry/phase7-protocol.md)
- [Phase 7 registry architecture](docs/registry/architecture.md)
- [Reviewed Phase 7 release-control report](reports/registry/phase7_v1/registry-release-report.md)
- [Phase 7 evidence manifest](reports/registry/phase7_v1/evidence-manifest.json)
- [Phase 8 persistent-platform decision](docs/adr/0006-persistent-local-mlops-platform.md)
- [Phase 8 platform protocol](docs/platform/phase8-protocol.md)
- [External artifact-distribution decision](docs/adr/0007-external-artifact-distribution.md)
- [Artifact storage and recovery guide](docs/artifacts/storage-architecture.md)

## Current capabilities

| Capability | Implemented evidence |
| --- | --- |
| Reproducible data | Checksum-pinned UCI acquisition, strict schema, content-addressed quarantine, deterministic canonical data and reviewed split lineage. [Data controls](docs/data/validation-policy.md). |
| Frozen science | Historical baselines/search, final four-classifier comparison, exact fitted winner, identity calibration, validation-only uncertainty and one consumed test. [Scientific reasoning](docs/portfolio/scientific-decisions.md). |
| Inference contracts | Strict 19-predictor requests, digest/dependency verification before model loading, shared batch/API engine, native-SHAP attribution categories and separate health/readiness. [Inference architecture](docs/inference/architecture.md). |
| Batch operation | Duplicate rejection, partial-row evidence, deterministic 10% ranking, atomic publication, verified no-rewrite reuse and allowlisted traceable events. [Batch contract](docs/inference/phase6-protocol.md). |
| Release control | Immutable revisions and manual SQLite promotion/rollback; separately PostgreSQL/MinIO fixed-state bootstrap and persistent recovery. [Architecture and limits](docs/portfolio/architecture.md). |
| Assurance | Subgroup diagnostics with retained G3 conditions, drift/service monitoring, robustness and incident drills, locked environments, Linux image tests, scans and SBOMs. [Approved local Release B](docs/reviews/release-b-owner-review.md). |
| Impact study design | Hypothetical customer-randomized outreach protocol and tested standalone sample-size/MDE calculator. [Study](docs/portfolio/intervention-study.md); no observed intervention benefit. |

All current model, feature, risk-band and capacity contracts remain frozen.
Historical evidence is inspected through its fixed trust anchors; synthetic
fixtures exercise the current implementation without reopening model selection.

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

### Synthetic-only quickstart

The API, UI and batch demonstration need the reviewed model and synthetic
fixtures; historical customer data is unnecessary. Install Python/uv as above,
then run from the repository root:

```bash
uv sync --locked --all-extras --dev
uv run credit-risk artifacts pull
uv run credit-risk artifacts verify
uv run uvicorn api:app --host 127.0.0.1 --port 8080
```

Leave the API running. In a second terminal at the same root:

```bash
uv run streamlit run app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501` and use the exact
[synthetic fixture and demo instructions](docs/portfolio/demo-script.md), rather
than arbitrary UI defaults, to reproduce probability `0.190382` (UI `0.1904`).
`http://127.0.0.1:8080/ready` should return `{"status":"ready"}`.
For a short batch check in a third terminal:

```bash
uv run credit-risk inference batch --input tests/fixtures/inference_batch_v1.csv --as-of-date 2026-09-30 --snapshot-id monthly-demo
uv run credit-risk inference verify --run-root experiment/inference/batches/2026-09-30/monthly-demo
```

The committed fixture contains 20 synthetic rows; the fixed policy selects two.
Identical reruns verify/reuse existing files; conflicting output fails without
overwrite. A clean batch exits 0, a published partial batch exits 3, and a failed
batch exits 1. The [full walkthrough](docs/portfolio/demo-script.md) uses new run
IDs for its separate 400-row monitoring demonstration. These counts are technical
checks, not new customer-performance evidence. Stop foreground services with Ctrl+C.
Detailed restoration and historical verification instructions follow.

### Materialise reviewed model artifacts

Fresh clones intentionally contain no production model binaries. Retrieve and
verify the selected model from the full immutable Hugging Face revision before
running inference:

```bash
uv sync --locked --all-extras --dev
uv run credit-risk artifacts pull
uv run credit-risk artifacts verify
```

The destination remains `models/selected_v1/model.cbm`, so the reviewed loader,
governance, release, registry, and platform contracts are unchanged. A valid
existing file is reused without network access or rewriting. `--offline` uses
only an already populated revision-aware cache. Pulling never deserializes a
model.

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
governance evidence, and the reviewed lock are version controlled. Retired
Phase 2/3 experiment executors are no longer shipped; their reviewed
configurations, reports, protocols, and integrity checks remain available for
audit.

### Inspect historical selection without rerunning it

The completed selection fitted four fixed classifiers on 19,200 development rows
and compared the same 4,800 validation accounts. It preserved the exact winner
without refitting: `catboost_fixed`, validation AP `0.556510`, Brier `0.133539`
and lift at 10% `3.210923`. Earlier development CV/search remains historical
context; four fits describes the final comparison, not all prior research.

Read the [selection protocol](docs/modeling/selection-protocol.md) and
[scientific decision narrative](docs/portfolio/scientific-decisions.md) for the
guardrails, close competitor, preprocessing and uncertainty limits. New fitting,
calibration, uncertainty regeneration or historical scoring is outside the current
scope. Reproduction here means restoring reviewed bytes, verifying lineage and
repeating synthetic technical checks.

The separately authorized 6,000-account final evaluation passed its frozen gates
once. Its [aggregate evidence and receipts](reports/modeling/final_test_v1/final-test-report.md)
remain immutable. A no-option tombstone prevents replay before data/model access;
the executed evaluator survives only as a non-importable text artifact. Training,
selection, gate freezing and final-test execution are not quickstart steps.

### Verify Phase 5 governance evidence

```bash
uv run credit-risk governance verify \
  --expected-manifest-sha256 6df8745f6deefcd138d7d1c821e6ad38ca7762aa7581fa0fae28042cf7f2b853 \
  --aggregate-only
```

The historical `governance build` is a controlled prediction-only interface; it is
not authorized by this quickstart. It scores the existing bundle on validation,
computes subgroup and native-SHAP
evidence, and refuses dirty worktrees, existing destinations, or paths outside
the approved governance report/runtime subtrees. Phase 1 may parse the complete
snapshot for integrity; test accounts are never selected, returned, scored,
explained, or audited by subgroup. Row-level evidence remains ignored under
`experiment/`. The authenticated official build from commit `226b7d7` reproduced the
reviewed validation metrics, generated non-degenerate Wilson prevalence intervals,
and retained the two predeclared education-group review triggers.

Verification requires the externally reviewed manifest digest, preventing a
modified artifact and self-updated manifest from authenticating each other. A
normal verification also requires and hashes the three ignored runtime artifacts.
The explicit `--aggregate-only` mode is intended for a clean checkout where those
row-level files are deliberately absent; it reports that narrower scope.

### Verify the authenticated Release A dossier

```bash
uv run credit-risk release verify \
  --expected-manifest-sha256 7e65c7b854de15742f05c4b8c2de891f50512518f8eb2339241f87f98754edf7
```

The [Release A report](reports/releases/release_a_v1/release-a-report.md)
consolidates reproducible data, reviewed baselines, the fixed four-model
selection, identity-calibration diagnostics, validation-only uncertainty,
capacity evidence, and the permanently consumed final test. Verification works
from a clean checkout: it authenticates the manifest against the external digest
and then verifies the complete source and output allowlists. It neither loads a
model nor accesses row-level prediction evidence.

### Run and verify Phase 6 inference

Score a strict operational snapshot with the unchanged `selected_v1` bundle:

```bash
uv run credit-risk inference batch \
  --input tests/fixtures/inference_batch_v1.csv \
  --as-of-date 2026-09-30 \
  --snapshot-id monthly-demo

uv run credit-risk inference verify \
  --run-root experiment/inference/batches/2026-09-30/monthly-demo
```

The CSV contract is `account_id` followed by the 19 ordered operational fields.
Valid rows are scored even when other rows fail. A clean batch exits `0`; a
published partial batch exits `3`; a file-level or all-invalid failure exits `1`.
Wrong-width rows receive `invalid_column_count`; `.` and `..` are prohibited as
snapshot path components. Every duplicate account ID occurrence is rejected.
Valid rows are ranked by
full-precision probability descending and account ID ascending, and exactly
`floor(valid_rows × 0.10)` are selected for human review. An identical verified
rerun reuses its files without rewriting them; a conflicting or corrupt run is
never overwritten. Verification reconciles the strict manifest with both CSVs,
including identity, ranking, selection, bands, traces, counts, rules, and lineage.

Each batch CLI attempt reports an opaque invocation `trace_id` in its terminal
log event and CLI result, including rejected arguments and failures before input
identification. Once the input bytes have been read, failures also carry the
same deterministic `batch_id` used for successful scoring. An idempotent rerun
gets a new invocation trace while preserving its batch ID and published files.
Log events contain allowlisted operational metadata only; customer IDs and
feature values remain outside service logs.

Authenticate the committed aggregate parity evidence without runtime row data:

```bash
uv run credit-risk inference verify-evidence \
  --expected-manifest-sha256 919087229d20fe83c1846da65d5901ea103ac3182c9c2975c3490424a49f4df8
```

The [Phase 6 report](reports/inference/phase6_v1/inference-parity-report.md)
binds the unchanged model/config digests and clean corrected implementation
lineage. It publishes no account IDs, feature values, probabilities, local
paths, or wall-clock timestamps. Its native-SHAP categories are non-causal model
attributions and are not adverse-action reasons.

### Verify the Phase 7 release-control evidence

```bash
uv run credit-risk registry verify-evidence \
  --expected-manifest-sha256 ce36f33da60fe6470d28a76b8053d102e74731115d069c4d476d0c2abbc47da9
```

The [Phase 7 report](reports/registry/phase7_v1/registry-release-report.md)
authenticates two transparent release revisions of the same reviewed bundle.
The drill registered both revisions, promoted revision 2, activated it through
an immutable deployment plus atomic pointer, and then restored revision 1 using
the approved rollback path. The evidence publisher independently loaded both
revisions and obtained the same full-precision output digest, probability
`0.190382`, risk band, and reason attributions from the pinned synthetic fixture.
This is deployment-control evidence, not a comparison of different models.

Runtime SQLite, MLflow, deployment, and receipt state remains ignored. The API
image contains no MLflow dependency. Its CI job pins build inputs, blocks
fixable HIGH/CRITICAL findings without an in-repository waiver, uploads a
CycloneDX SBOM, and exercises the deployment-pointer contract. Phase 8 adds the
persistent PostgreSQL/MinIO stack with authenticated startup, bootstrap, restart
and parity evidence. Release B also completes robustness, monitoring and incident
controls for the local portfolio scope. Promotion and rollback remain the
separate Phase 7 SQLite demonstration; Phase 8 provides fixed-state bootstrap
and restart recovery.

### Run the Phase 8 persistent local platform

Copy `.env.example` to `.env`, replace both `change-me` secrets, and start the
digest-pinned stack:

```bash
docker compose --env-file .env -f docker-compose.platform.yml up --build --wait
```

The stack exposes only the local MLflow UI (`5000`), MinIO console (`9001`), API
(`8080`), and Streamlit UI (`8501`). PostgreSQL and the MinIO object API are
internal-only. Verify the complete registry, object, alias, and deployment state:

```bash
docker compose --env-file .env -f docker-compose.platform.yml run --rm --no-deps \
  bootstrap credit-risk platform verify --deployment-root /deployment
```

Stopping and restarting the services preserves all three named volumes. The
one-shot bootstrap is the only container with write access to the deployment
volume; the non-root API mounts it read-only. No training or sealed-test workflow
is present in this stack. Bootstrap uses a frozen fail-fast PostgreSQL advisory
lock, so a concurrent writer exits before changing MinIO, MLflow, or deployment
state. The verification command is read-only and also checks the exact
registered-model lineage tags. The MLflow launcher percent-encodes PostgreSQL
credential components and passes its backend URI through the process environment
rather than command arguments. Platform workflows reject symlinked deployment
ancestors, normalize expected service failures, and treat an explicitly supplied
environment mapping as authoritative; only `None` enables ambient discovery.

### Run quality checks

```bash
uv run ruff format --check api.py app.py src/credit_risk tests
uv run ruff check api.py app.py src/credit_risk tests
uv run mypy src/credit_risk/portfolio src/credit_risk/assurance src/credit_risk/robustness src/credit_risk/monitoring src/credit_risk/incidents src/credit_risk/artifact_distribution src/credit_risk/data src/credit_risk/modeling src/credit_risk/governance src/credit_risk/release src/credit_risk/inference src/credit_risk/registry src/credit_risk/platform src/credit_risk/cli.py api.py app.py
uv run pytest -m "not training and not artifact" --cov --cov-report=term-missing
uv run pytest tests/unit/artifact_distribution --cov=credit_risk.artifact_distribution --cov-branch --cov-fail-under=90
uv run pytest tests/unit/data tests/unit/test_data_cli.py tests/integration/test_data_workflow.py --cov=credit_risk.data --cov-branch --cov-fail-under=90
uv run pytest \
  tests/unit/modeling/test_selection_workflow.py \
  tests/unit/modeling/test_final_test.py \
  tests/unit/modeling/test_final_test_workflow.py \
  tests/unit/modeling/test_selection_contracts.py \
  tests/unit/modeling/test_selection_analysis.py \
  tests/unit/modeling/test_selection_models.py \
  tests/unit/modeling/test_selected_bundle.py \
  tests/integration/test_selection_protocol.py \
  tests/integration/test_selection_split.py \
  -m "not training" \
  --cov=credit_risk.modeling.selection_contracts \
  --cov=credit_risk.modeling.selection_analysis \
  --cov=credit_risk.modeling.selection_models \
  --cov=credit_risk.modeling.selected_bundle \
  --cov=credit_risk.modeling.selection_workflow \
  --cov=credit_risk.modeling.final_test \
  --cov=credit_risk.modeling.final_test_workflow \
  --cov=credit_risk.modeling.risk_policy \
  --cov-branch --cov-fail-under=90
uv run pytest tests/unit/governance tests/unit/test_governance_cli.py tests/integration/test_phase5_protocol.py tests/integration/test_phase5_evidence.py tests/integration/test_governance_explanation_smoke.py --cov=credit_risk.governance --cov-branch --cov-fail-under=90
uv run pytest tests/unit/release tests/unit/test_release_cli.py tests/integration/test_release_a_protocol.py tests/integration/test_release_a_evidence.py --cov=credit_risk.release --cov-branch --cov-fail-under=90
uv run pytest tests/unit/inference tests/unit/test_inference_cli.py tests/integration/test_api_health.py tests/integration/test_phase6_protocol.py tests/integration/test_phase6_evidence.py tests/integration/test_inference_parity.py --cov=credit_risk.inference --cov-branch --cov-fail-under=90
uv run pytest tests/unit/registry tests/unit/test_registry_cli.py tests/integration/test_phase7_protocol.py tests/integration/test_registry_workflow.py tests/integration/test_phase7_evidence.py --cov=credit_risk.registry --cov-branch --cov-fail-under=90
uv run pytest tests/unit/platform tests/unit/test_platform_cli.py tests/integration/test_phase8_protocol.py --cov=credit_risk.platform --cov-branch --cov-fail-under=90
```

### Run the API

```bash
uv run credit-risk artifacts pull
uv run uvicorn api:app --host 127.0.0.1 --port 8080
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

The API is exposed at `http://localhost:8080`. A dedicated build stage retrieves
and verifies the selected model anonymously from the pinned Hugging Face commit.
The final offline runtime contains only `models/selected_v1/manifest.json` and
`model.cbm`; it contains no Hugging Face client, cache, token, legacy artifacts,
generated data, reports, training dependencies, or experiment state.

To run the API against a previously materialised Phase 7 deployment pointer,
use the registry override; the deployment directory is mounted read-only and
the image still has no MLflow dependency:

```bash
docker compose -f docker-compose.yml -f docker-compose.registry.yml up --build
```

## Governed prediction request

`POST /v1/predict` accepts exactly the 19 operational features in the selected-model
contract: credit limit plus six months each of repayment status, signed bill
amount, and non-negative payment amount. Demographics, account ID, target, nulls,
non-finite values, and unknown fields are rejected. `POST /predict` is removed
and returns `404`. The v1 response returns a trace ID, six-decimal default
probability, validation-frozen risk band, two reviewed reason-category objects,
and model, bundle, manifest, and policy identifiers. Queue rank and selection
status remain batch-only because they require portfolio context.

The synthetic request in `tests/fixtures/prediction_request.json` returns
probability `0.190382` and risk band `standard`. Tests freeze this non-holdout
example so package, dependency, and container changes cannot silently alter the
released model contract.

## Repository structure

```text
.
├── api.py                     # Governed selected-model FastAPI entrypoint
├── app.py                     # Local Streamlit demonstration
├── configs/artifacts/         # Selected-model-only HF revision lock
├── configs/data/              # Source manifest, split policy, and reviewed lock
├── configs/modeling/          # Feature and scientific-baseline contracts
├── configs/governance/        # Frozen validation-only governance contract
├── configs/inference/         # Frozen batch/API parity contract
├── configs/registry/          # Frozen registry contract and reviewed approvals
├── configs/platform/          # Frozen persistent local-platform contract
├── configs/releases/          # Frozen release-level audit contract
├── data/                      # Ignored reproducible raw/processed/split products
├── docs/                      # Product, roadmap, governance, and ADR evidence
│   └── modeling/evidence/     # Non-importable archive of the consumed evaluator
├── experiment/                # Ignored MLflow, OOF, and exploratory evidence
├── models/selected_v1/        # Tracked manifest plus ignored materialised CBM
├── reports/modeling/          # Reviewed aggregate experiment evidence
├── reports/governance/        # Reviewed aggregate governance evidence
├── reports/inference/         # Authenticated aggregate inference evidence
├── reports/registry/          # Authenticated aggregate release-control evidence
├── reports/releases/          # Authenticated release dossiers
├── src/credit_risk/           # Installable application package
├── tests/                     # Unit, integration, and evidence-integrity tests
├── pyproject.toml             # Direct dependencies and tool configuration
├── uv.lock                    # Exact cross-platform dependency resolution
├── Dockerfile
├── docker-compose.yml
└── docker-compose.registry.yml
```

Generated data, logs, environments, caches, experiment outputs, and the
externally distributed selected-model binary are excluded from version control.

## Release B sign-off

The additive platform-evidence, robustness, monitoring, benchmark, incident and
Release B dossier commands are implemented. Official Linux evidence and the
[dossier](reports/releases/release_b_v1/release-b-report.md) verify from the pinned
approved checkout described below. The [owner review and proposed restrictions](docs/reviews/release-b-owner-review.md)
were explicitly approved by the project owner for local portfolio use. G4 is
closed for that scope. The [owner decision](configs/releases/release_b_owner_approval_v1.json)
binds the immutable dossier and every required disposition. Verification combines
the original review candidate with this separate decision to report approved status.
Follow the
[sign-off procedure](docs/operations/release-b-signoff.md); the separate
Release B evidence workflow authenticates the pinned historical release without
scoring. The four standard CI jobs test the current implementation, including
Linux container checks, persistence and synthetic prediction parity. Historical
approval does not approve later implementation changes.

## Delivery milestones

- **Release A — defensible model (complete):** reproducible data, baselines,
  identity calibration, validation-only uncertainty, capacity-aware evaluation,
  and one authenticated evidence chain.
- **Release B — governed ML product (complete for local portfolio scope):**
  model/data cards, subgroup analysis, reason-category tests, batch/API parity, registry promotion gates,
  scanning, rollback, robustness, monitoring, and incident controls have measured
  evidence, a verified dossier and an explicit owner decision. G4 is closed for
  this local scope; G3 restrictions and G5 production review remain.
- **Release C — portfolio communication (open):** executive narrative, demo video,
  intervention experiment design and interview materials.

See the [roadmap](docs/roadmap.md) for weekly acceptance gates and the honest
mapping from the local implementation to Azure Databricks production concepts.

Release B verification anchors:

- Dossier: `f5fa342b89e06c016ea7b632c8a252c472502186ab52ac68aba6992d652c8c88`.
- Owner approval: `9a40d17f038056e3e475762810ead7887cdf764be836dc6387bc19638e4f8f1f`.

The benchmark manifest binds the original source bytes. Verify that historical
release in its own checkout and locked environment; verification in a modified
implementation correctly rejects changed source hashes. From the current
repository root, create a new ignored checkout (an existing destination is never
reset or overwritten):

```bash
git clone --no-hardlinks . .cache/release-b-approved
git -C .cache/release-b-approved switch --detach 7e571fc4fbb4e6cb99b0d66f8e7d72b24feccff4
cd .cache/release-b-approved
uv sync --locked --all-extras --dev
uv run credit-risk artifacts pull
uv run credit-risk artifacts verify
uv run credit-risk data build
uv run credit-risk release verify-b \
  --expected-manifest-sha256 f5fa342b89e06c016ea7b632c8a252c472502186ab52ac68aba6992d652c8c88 \
  --approval configs/releases/release_b_owner_approval_v1.json \
  --approval-sha256 9a40d17f038056e3e475762810ead7887cdf764be836dc6387bc19638e4f8f1f
```

The command must return `approved_local_portfolio_release` and `g4_status: closed`.
This restores reviewed model bytes and Phase 1 data solely for lineage checks;
no training or scoring is performed. Return to the current checkout before
running its synthetic demonstration or tests. Record both the historical
release commit and the current implementation commit in the rehearsal.

CI also compares committed reports, approvals, reviewed configurations and the
model manifest with that snapshot. It does not regenerate evidence or change
its digest anchors. Release C remains planned and G5 remains open.
