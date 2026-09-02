# Portfolio v2 documentation

This directory contains the decision, delivery, and governance evidence for the
portfolio upgrade. The documents are written before the implementation so that
the system is evaluated against an explicit product contract rather than only
against model accuracy.

## Phase 0 documents

- [Product and decision brief](product-brief.md)
- [Twelve-week delivery roadmap](roadmap.md)
- [ADR 0001: batch-first scoring architecture](adr/0001-batch-first-scoring.md)
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
file digests. Candidate modelling, calibration, and the sealed holdout remain
outside this checkpoint, so G2 is still open.

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

The authoritative workflow fits four fixed binary classifiers exactly once on
the frozen training slice, selects on one shared validation slice, and bundles
the exact winner without refitting. Calibration and bootstrap diagnostics reuse
stored predictions. The clean four-fit run selected `catboost_fixed`; its exact
native CBM is committed under `../models/selected_v1/` with digest-protected
lineage. The holdout remains unevaluated and G2 stays open until a
separately authorized one-time test passes gates frozen from validation.
