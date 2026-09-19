# ADR 0008: Pre-Release B repository cleanup

**Status:** Accepted

## Context

After Release A and the Phase 5-8 implementation work, the repository still
contained generated CatBoost telemetry, a 2024 exploratory notebook, the
original tutorial-style training pipeline, retired Phase 2/3 experiment
executors, and an opt-in pickle compatibility path that no current product or
planned Release B workflow used.

At reviewed commit `47168f873eeb37b030d338892b9763d302521d9e`, Git tracked
286 files totalling approximately 5.94 MiB. The obsolete notebook and generated
CatBoost outputs accounted for approximately 3.37 MiB by themselves. The packed
Git object database was approximately 2.27 MiB, so rewriting history would save
little while invalidating commit-based lineage and review references.

## Decision

- Remove tracked `catboost_info` output and the obsolete exploratory notebook.
- Remove the unreachable tutorial ingestion, transformation, GridSearchCV,
  evaluation, training, and prediction packages.
- Remove retired Phase 2/3 experiment executors and their execution tests. Keep
  their reviewed configurations, protocols, aggregate evidence, and integrity
  tests because later selection and Release A evidence authenticate them.
- Keep the governed four-fit selection implementation, selected-bundle loader,
  metrics, data contracts, risk policy, and tracking utilities.
- Retire the legacy pickle loader, `doctor` command, legacy request contract,
  and `0.44088` compatibility test. The old public Hugging Face bytes remain
  inert historical objects and are not deleted or represented as supported.
- Replace the mixed artifact distribution contract with a selected-model-only
  v2 contract. It reuses the already-reviewed immutable Hugging Face revision
  and authenticates the model through the unchanged selected manifest.
- Remove obsolete command tombstones for legacy training and Phase 2/3
  execution. Keep the permanently consumed final-test tombstone.
- Preserve ordinary Git history. No history rewrite or force-push is permitted.

## Preserved evidence boundary

The selected manifest, Phase 1 data lineage, baseline and candidate evidence,
selection evidence, final-test authorization and receipts, Release A dossier,
Phase 5-8 contracts and reports, model/data cards, risk register, governance
checklists, and architecture decisions remain byte-identical. The local
`codex/archive-phase4-no-training` recovery branch also remains untouched.

The cleanup performs no model fitting, tuning, calibration, bootstrap
generation, final-test evaluation, or sealed-test access.

## Consequences

- A fresh checkout supports one production artifact: `selected_v1`.
- Historical scientific results remain auditable but their expensive superseded
  experiment runners are no longer executable from the current source tree.
- Public artifact-management commands materialise and verify only the selected
  CatBoost model. Normal imports and runtime startup remain network-free.
- Deleting files from the current tree reduces checkout and build-context size;
  historical blobs remain reachable through normal Git history by design.
- Removing public compatibility commands is a pre-1.0 breaking change, so the
  package version advances from `0.5.0` to `0.6.0`.
