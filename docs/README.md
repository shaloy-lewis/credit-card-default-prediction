# Portfolio v2 documentation

This directory contains the decision, delivery, and governance evidence for the
portfolio upgrade. The documents are written before the implementation so that
the system is evaluated against an explicit product contract rather than only
against model accuracy.

## Phase 0 documents

- [Product and decision brief](product-brief.md)
- [Twelve-week delivery roadmap](roadmap.md)
- [ADR 0001: batch-first scoring architecture](adr/0001-batch-first-scoring.md)
- [ADR 0002: no-training governance review](adr/0002-no-training-governance-review.md)
- [ADR 0003: Release A audit closure](adr/0003-release-a-audit-closure.md)
- [ADR 0004: versioned inference parity](adr/0004-versioned-inference-parity.md)
- [ADR 0005: local MLflow registry release control](adr/0005-local-mlflow-registry-release-control.md)
- [Delivery progress and verification evidence](progress.md)

The Phase 0 product brief and batch-first architecture decision were accepted on
2026-08-07. Later material changes should be recorded in an ADR or decision log.

## Phase 1 data-readiness evidence

- [Dataset card and evidence limits](data/data-card.md)
- [Feature availability and leakage review](data/feature-availability.md)
- [Validation and quarantine policy](data/validation-policy.md)

The executable source and split contracts live under `configs/data/`. Generated
data and runtime lineage remain under the Git-ignored root `data/` directory;
the reviewed split lock is version controlled with the configuration.

## Phase 2 scientific-baseline evidence

- [Baseline experiment protocol v1](modeling/experiment-protocol.md)
- [Reviewed aggregate baseline report](../reports/modeling/baseline_v1/baseline-report.md)
- [Reviewed machine-readable baseline summary](../reports/modeling/baseline_v1/summary.json)

Phase 2 completed the Week 3 baseline and tracking slice from clean reviewed commit
`c695c60`. Runtime MLflow state and row-level predictions remain ignored; the
deterministic aggregate result is version-controlled and protected by complete
file digests. Candidate modelling, calibration, and the sealed holdout remained
outside that historical checkpoint; G2 was closed later by the governed release
workflow documented below.

## Phase 3 candidate-modelling evidence

- [Frozen CatBoost candidate protocol](modeling/candidate-protocol.md)
- Machine-readable contract: `../configs/modeling/candidate_v1.json`
- [Reviewed aggregate candidate report](../reports/modeling/candidate_v1/candidate-report.md)
- [Reviewed machine-readable candidate summary](../reports/modeling/candidate_v1/summary.json)

The amended protocol fixes the development-only feature views, deterministic
eight-trial search, 150-fold-fit ceiling, balanced advancement gate, and
logistic fallback before candidate results exist. The compute amendment uses
four CatBoost threads and was based only on runtime benchmarks; no candidate
metric was observed before it was frozen. Two independent executions from clean
commit `2b46d4c` produced byte-identical aggregate and runtime evidence.
Configuration `cb_cfg_006` passed every development-CV advancement condition and
was the historical Phase 4 candidate. That expensive workflow is now superseded
as an executable release process, while its evidence remains immutable.

## One-pass release selection

- [Frozen one-pass selection protocol](modeling/selection-protocol.md)
- Machine-readable contract: `../configs/modeling/selection_v1.json`
- [Reviewed aggregate selection report](../reports/modeling/selection_v1/selection-report.md)
- [Reviewed machine-readable selection summary](../reports/modeling/selection_v1/summary.json)
- [Reviewed one-time final-test report](../reports/modeling/final_test_v1/final-test-report.md)
- [Reviewed one-time final-test summary](../reports/modeling/final_test_v1/summary.json)
- [Archived executed final-test source](modeling/evidence/final_test_workflow_v1_executed.py.txt)

The authoritative workflow fits four fixed binary classifiers exactly once on
the frozen training slice, selects on one shared validation slice, and bundles
the exact winner without refitting. Calibration and bootstrap diagnostics reuse
stored predictions. The clean four-fit run selected `catboost_fixed`; its
digest-protected manifest remains committed under `../models/selected_v1/`, and
the unchanged native CBM is explicitly materialised there from its immutable
public distribution revision. A separately reviewed approval authorized one prediction-only test
evaluation. The unchanged bundle scored exactly 6,000 test accounts, passed all
three frozen gates, and closed G2 with zero fitting, refitting, or retuning.
Durable receipts prevent reevaluation, and row-level test predictions remain
ignored. Phase 4 release hardening additionally replaced the active evaluator
with a no-option permanent tombstone while preserving its approved source
byte-for-byte as a non-importable evidence artifact. Serving readiness now
requires exact versions for the six runtime dependencies recorded in the bundle
manifest. G2 remains closed and Phase 4 is complete. Corrected Phase 5 evidence
is reviewed below without beginning later lifecycle work.

## Phase 5 governance protocol

- [Frozen validation-only governance protocol](governance/phase5-protocol.md)
- Machine-readable contract: `../configs/governance/phase5_v1.json`
- [Reviewed governance report](../reports/governance/phase5_v1/governance-report.md)
- [Validation subgroup report](../reports/governance/phase5_v1/fairness-report.md)
- [Model card](../reports/governance/phase5_v1/model-card.md)
- [Risk register](../reports/governance/phase5_v1/risk-register.md)
- [G3 review decision](../reports/governance/phase5_v1/g3-review.md)
- [G3 checklist](governance/g3-checklist.md)
- [Model-governance status](governance/model-governance-status.md)

The protocol uses the existing selected bundle for validation inference and
native CatBoost explanations only. It permits complete-snapshot integrity
verification but prohibits selecting, returning, scoring, explaining, or auditing
test accounts. It also prohibits further fitting, replaces demographic ablation
with exclusion/invariance evidence, and freezes subgroup review triggers before
corrected official evidence is published. The clean authenticated build from commit
`226b7d7` was verified against an external manifest trust anchor and full-file digests, uses Wilson intervals for
prevalence and stratified-percentile intervals for performance measures, and
closes G3 as `closed_with_conditions`. Row-level runtime evidence remains ignored.
The verifier requires an externally reviewed manifest digest. By default it also
hashes the ignored prediction, SHAP, and bootstrap artifacts; an explicit
aggregate-only mode supports clean checkouts without overstating its scope.

## Release A audit dossier

- Machine-readable contract: `../configs/releases/release_a_v1.json`
- [Consolidated Release A report](../reports/releases/release_a_v1/release-a-report.md)
- [Machine-readable Release A summary](../reports/releases/release_a_v1/summary.json)
- [Externally authenticated evidence manifest](../reports/releases/release_a_v1/evidence-manifest.json)
- [Published validation-only uncertainty](../reports/releases/release_a_v1/validation-uncertainty.json)

The dossier was assembled from clean implementation commit `20186ad` without
model loading, prediction, fitting, bootstrap generation, test-row selection,
or final-test reevaluation. The uncertainty file is byte-identical to the
reviewed 500-resample selection-runtime artifact. The external manifest digest
is `7e65c7b854de15742f05c4b8c2de891f50512518f8eb2339241f87f98754edf7`.
Release A is complete; robustness and population-shift stress evidence remains
explicitly deferred to G4/Release B.

## Phase 6 inference parity

- [Frozen inference protocol](inference/phase6-protocol.md)
- [Inference architecture](inference/architecture.md)
- Machine-readable contract: `../configs/inference/phase6_v1.json`
- [Reviewed parity report](../reports/inference/phase6_v1/inference-parity-report.md)
- [Machine-readable parity summary](../reports/inference/phase6_v1/summary.json)
- [Externally authenticated evidence manifest](../reports/inference/phase6_v1/evidence-manifest.json)

Phase 6 uses the unchanged `selected_v1` bundle through one shared vectorised
engine. It adds strict partial-row batch handling, deterministic 10% ranking,
atomic idempotent publication, native-SHAP reason categories, safe JSON logs,
and a breaking `/v1/predict` contract. The Streamlit demonstration is now an API
client. The clean official evidence run from corrected implementation commit
`f6b37af` proved exact offline/batch probabilities, API agreement within `5e-7`,
exact band/reason parity, and verified no-rewrite reuse on 20 synthetic rows.
Its external manifest digest is
`919087229d20fe83c1846da65d5901ea103ac3182c9c2975c3490424a49f4df8`.
No model fitting, model change, sealed-test access, or row-level evidence
publication occurred. G4 remains open for the remaining release-readiness work.

## Phase 7 registry protocol

- [Frozen registry and rollback protocol](registry/phase7-protocol.md)
- [Registry and deployment architecture](registry/architecture.md)
- Machine-readable contract: `../configs/registry/phase7_v1.json`
- [Reviewed registry release report](../reports/registry/phase7_v1/registry-release-report.md)
- [Promotion checklist](../reports/registry/phase7_v1/promotion-checklist.md)
- [Rollback runbook](../reports/registry/phase7_v1/rollback-runbook.md)
- [Externally authenticated evidence manifest](../reports/registry/phase7_v1/evidence-manifest.json)

Phase 7 is complete for its bounded release-control slice. A local MLflow SQLite
registry demonstrated manual candidate/champion promotion, immutable deployment
revisions, an atomic active pointer, exact synthetic smoke parity, and approved
rollback around the unchanged reviewed model. GitHub Actions blocks fixable
HIGH/CRITICAL image findings and publishes a CycloneDX SBOM. The evidence trust
anchor is `ce36f33d...7da9`. PostgreSQL, MinIO, a persistent registry service,
robustness testing, monitoring, and incident controls remain deferred.

## Phase 8 persistent local platform — prerequisites complete

- [Persistent-platform architecture decision](adr/0006-persistent-local-mlops-platform.md)
- [Frozen prerequisite protocol](platform/phase8-protocol.md)
- Machine-readable contract: `../configs/platform/phase8_v1.json`
- Compose topology: `../docker-compose.platform.yml`

The prerequisite layer re-registers the unchanged Phase 7 bundle into MLflow
3.15 backed by PostgreSQL and MinIO, restores the reviewed champion/rollback
aliases, and materialises the approved deployment pointer into a named volume.
The stack passed local bootstrap, idempotency, restart persistence, API/UI health,
and prediction-parity checks with zero fitting or sealed-test access. Phase 8 is
still in progress until its deterministic aggregate evidence is published and
reviewed.

## External artifact distribution

- [ADR 0007: external binary-artifact distribution](adr/0007-external-artifact-distribution.md)
- [Storage, retrieval, publishing, and recovery guide](artifacts/storage-architecture.md)
- [Public Hugging Face repository card source](artifacts/hugging-face-repository-card.md)
- Machine-readable distribution lock: `../configs/artifacts/hf_distribution_v1.lock.json`
- Legacy trust manifest: `../configs/artifacts/legacy_v1.json`

GitHub retains source, manifests, aggregate evidence, and checksums; UCI remains
the sole data source; Hugging Face distributes exact reviewed binary bytes; and
MLflow/MinIO retain release-control duties. Normal imports and application
startup remain offline. The selected model is acquired only through an explicit
pull or an isolated Docker build stage, while legacy pickles require a separate
opt-in pull and pre-deserialization digest authentication.
