# Twelve-week portfolio delivery roadmap

**Capacity:** 10 hours per week, approximately 120 hours total
**Target profile:** Senior Data Scientist and Senior AIML/ML Engineer
**Emphasis:** MLOps and model governance
**Deployment:** Zero-cost local Docker demonstration plus recorded video
**Cloud context:** Azure Databricks experience documented through an honest architecture mapping; no claim of a deployed Azure system

## Delivery principles

1. Every week ends with reviewable evidence, not only code.
2. The model test set remains sealed until the candidate, calibrator, and policy
   selection procedure have been fixed.
3. Batch scoring is the product path; the API and UI are integration and demo paths.
4. Measured results, simulated assumptions, and untested production hypotheses
   are labelled separately.
5. A simpler system wins when additional infrastructure does not solve a stated
   requirement.
6. Drift triggers investigation. Retraining and promotion remain controlled decisions.

## Weekly plan

| Week | Approx. effort | Outcome and deliverables | Exit gate | Seniority evidence |
| --- | ---: | --- | --- | --- |
| 1: Scope and foundation | 10 h | Approve product brief and batch-first ADR; repository inventory; target package layout; dependency and test strategy | Decision owner, action, horizon, metrics, limitations, and non-goals are explicit | Product framing and technical direction |
| 2: Reproducible data | 10 h | Official download/checksum; immutable raw layer; schema and quality checks; data card; feature-availability matrix; deterministic folds | A clean checkout reproduces the expected data and rejects corrupted fixtures | Lineage, contracts, and leakage awareness |
| 3: Baseline and tracking | 10 h | Prevalence/rule baseline; regularised logistic model; optional scorecard; MLflow experiment tracking; sealed holdout | Runs record data hash, code commit, config, folds, metrics, and artifacts | Scientific discipline and auditability |
| 4: Candidate modelling | 10 h | CatBoost plus at most one justified challenger; common CV; ablations; bounded hyperparameter search | Candidate selection procedure is fixed without consulting the test result | Trade-offs rather than model shopping |
| 5: Calibration and policy | 10 h | Calibration comparison; confidence intervals; lift/gains; recall and precision at capacity; risk bands; simulated economics | Candidate, calibrator, and operating-policy rules are documented before one-time test evaluation | Decision science and uncertainty |
| 6: Governance and explanation | 10 h | Correct SHAP feature mapping; additivity tests; reason categories; demographic ablation; fairness report; model card; risk register | Explanations are dimensionally correct and feature use is governed | Responsible AI and model-risk ownership |
| 7: Batch and API inference | 10 h | Versioned model bundle; idempotent monthly scorer; `/v1` API; validated contracts; model/trace metadata; structured safe logs | Offline, batch, and API probabilities and policies agree within tolerance | Production inference and parity |
| 8: Registry, CI, and rollback | 10 h | MLflow registry; candidate/champion workflow; promotion checklist; unit/integration/contract/model tests; GitHub Actions; image scan | A model can be registered, promoted, deployed locally, and rolled back without replacing files by hand | Controlled software and model delivery |
| 9: Local MLOps platform | 10 h | Docker Compose stack for API, UI, MLflow, PostgreSQL, and MinIO; persistent volumes; health checks; one-command startup | A fresh machine can start the stack and reproduce the demo using documented commands | Platform architecture without paid infrastructure |
| 10: Monitoring and incidents | 10 h | Batch manifest; data and prediction drift report; delayed-label evaluation design; service metrics; runbooks; schema and drift incident drills | Injected failures are detected and lead to an actionable investigation or rollback path | Operability and failure management |
| 11: Impact and communication | 10 h | Intervention experiment design; MDE/sample-size workbook or script; architecture diagram; executive case study; README rewrite; demo script | Model lift is not described as causal impact, and a reviewer can understand the system quickly | Product experimentation and stakeholder communication |
| 12: Hardening and interview packet | 10 h | Clean-machine rehearsal; reproducibility audit; security/privacy checklist; 4–6 minute video; system-design walkthrough; resume bullets using measured results | CI is green, docs match behaviour, limitations are prominent, and the complete demo is repeatable | End-to-end ownership and technical leadership |

## Release milestones

### Release A: defensible model — end of Week 5

- Reproducible data and split protocol.
- Interpretable baseline and CatBoost candidate.
- Calibration, uncertainty, capacity metrics, and a documented selection rule.
- No unsupported temporal, causal, India-specific, or compliance claim.

This release is the minimum scientifically credible senior-data-science story.

### Release B: governed ML product — end of Week 8

- Model and data cards, risk register, subgroup evaluation, and explanation tests.
- Versioned batch/API inference with parity.
- Registry, promotion gates, CI, container checks, and rollback.

This release creates the core hybrid DS/MLE story.

### Release C: flagship portfolio case study — end of Week 12

- Reproducible local MLOps stack.
- Monitoring and incident demonstrations.
- Experiment design for measuring intervention impact.
- Executive narrative, technical documentation, demo video, and interview packet.

## Zero-cost local target architecture

| Component | Local implementation | Purpose |
| --- | --- | --- |
| Source and fixtures | Versioned download manifest, checksums, local immutable directory | Reproducible source and test data |
| Pipeline | Python CLI and configuration files | Data preparation, training, evaluation, registration, and scoring |
| Tracking and registry | Self-hosted MLflow | Experiment lineage, artifacts, candidate/champion state, and promotion evidence |
| Metadata store | PostgreSQL container | Durable MLflow backend metadata |
| Artifact store | MinIO container | S3-compatible local model and report storage |
| Batch inference | Idempotent CLI/container task | Monthly portfolio scoring and manifest generation |
| Online inference | FastAPI container | Versioned integration and demonstration endpoint |
| Demo UI | Streamlit calling the API | Human-readable local demonstration |
| Model/data monitoring | Scheduled report job using explicit statistical checks | Drift, quality, prediction, calibration, and subgroup reports |
| Service monitoring | Structured logs and lightweight metrics; add Prometheus/Grafana only if justified by Week 10 | Latency, failures, health, and batch completion |
| Delivery control | GitHub Actions, container scanning, manual promotion gate | Reproducible quality checks and controlled release |

PostgreSQL and MinIO are introduced in Week 9, after the scientific pipeline is
stable. Earlier weeks may use a local MLflow file or SQLite backend so platform
work does not block modelling.

## Azure and Databricks interview mapping

The local project will document this mapping without claiming that the Azure
components were deployed.

| Local portfolio component | Azure/Databricks analogue | Transferable design concern |
| --- | --- | --- |
| Python pipeline and batch CLI | Databricks Jobs / Workflows | Parameters, retries, idempotency, lineage, and job outputs |
| Local tabular files and MinIO | ADLS Gen2 and Delta tables | Immutable inputs, versioning, schemas, and access control |
| MLflow tracking server | Databricks-managed MLflow | Experiment lineage and reproducibility |
| MLflow registry aliases and gates | Unity Catalog registered models | Ownership, promotion, approvals, and rollback |
| Pandera-style schema checks | Delta expectations or governed quality rules | Contract failures and quarantine behaviour |
| FastAPI Docker container | Azure Container Apps or managed serving | Versioned contracts, health, scaling, and secrets |
| GitHub Actions | GitHub Actions or Azure DevOps | Test, build, scan, promote, and environment controls |
| Local monitoring reports | Databricks Lakehouse Monitoring / Azure Monitor patterns | Drift, delayed labels, alerts, and investigation |
| Model/data cards and risk register | Unity Catalog metadata plus organisational model-risk process | Evidence, accountability, audit trail, and review gates |

## Governance evidence set

The final repository should contain:

- Product and decision brief.
- Data card and feature-availability matrix.
- Experiment protocol and result report.
- Model card and demographic-ablation report.
- Risk register and responsible-use statement.
- Model-inventory entry and version manifest.
- Promotion checklist and approval record template.
- Monitoring specification and periodic report.
- Retraining, rollback, and incident runbooks.
- Architecture decision records for significant trade-offs.

## Scope controls

The following are explicitly deferred unless a later requirement justifies them:

- Paid Azure deployment.
- Spark for the 30,000-row model-development dataset.
- Kubernetes, a feature store, and streaming inference.
- Automated retraining or automatic champion promotion.
- A GenAI decision-maker or customer-level assistant.

If Spark needs to be demonstrated for a target role, it will be a separate,
clearly labelled batch-volume exercise using generated operational data. It will
not be used to imply that the source dataset requires distributed compute.

## Definition of done

The project is portfolio-ready when:

- the business decision and ownership are explicit;
- a clean checkout can reproduce data, experiments, model registration, and scoring;
- baselines, uncertainty, calibration, capacity metrics, and limitations are reported;
- every released model has data/code/config lineage and governance evidence;
- batch and online paths are contract-tested and prediction-equivalent;
- promotion and rollback are demonstrated through the registry;
- drift and schema incidents are detected and handled through a runbook;
- measured, simulated, and hypothetical claims are visually distinguishable; and
- the README and video support both a two-minute recruiter scan and a detailed
  senior-level system-design discussion.

## Weekly working agreement

At each weekly checkpoint:

1. Review the previous exit gate and evidence.
2. Decide any unresolved product or architecture question.
3. Implement only the next bounded slice.
4. Run the relevant tests and capture reproducible outputs.
5. Update the decision log and portfolio narrative before expanding scope.

The next implementation slice after Phase 0 approval is Week 1's engineering
foundation. It will not change the statistical model.
