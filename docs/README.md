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
stored predictions. The clean four-fit run selected `catboost_fixed`; its exact
native CBM is committed under `../models/selected_v1/` with digest-protected
lineage. A separately reviewed approval authorized one prediction-only test
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
corrected official evidence is published. The clean corrected build from commit
`9b156c5` was verified against full-file digests, uses Wilson intervals for
prevalence and stratified-percentile intervals for performance measures, and
closes G3 as `closed_with_conditions`. Row-level runtime evidence remains ignored.
The verifier requires an externally reviewed manifest digest. By default it also
hashes the ignored prediction, SHAP, and bootstrap artifacts; an explicit
aggregate-only mode supports clean checkouts without overstating its scope.
