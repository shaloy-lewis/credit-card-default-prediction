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
