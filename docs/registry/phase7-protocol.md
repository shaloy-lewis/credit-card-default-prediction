# Phase 7 registry and rollback protocol

## Objective

Demonstrate a fail-closed release workflow around the unchanged `selected_v1`
bundle: register, approve, promote, deploy, and roll back without copying model
files by hand.

## Fixed boundaries

- MLflow 3.15.0 uses ignored SQLite metadata and content-addressed local
  artifacts.
- `credit-risk-default` is the only registered model.
- `phase7_rev_001` and `phase7_rev_002` contain identical reviewed bundle
  bytes and differ only as deployment revisions.
- The only aliases are `candidate`, `champion`, and `rollback`.
- Promotion and rollback require digest-authenticated, role-labelled approval
  records tied to a CI-green implementation commit.
- The container scan blocks fixable HIGH and CRITICAL vulnerabilities and has
  no repository waiver.
- No training, refitting, tuning, calibration fitting, bootstrap generation,
  final-test loading, or sealed-test scoring is permitted.

## State transitions

Initial registration sets revision 1 as champion and revision 2 as candidate.
Promotion moves revision 2 to champion and revision 1 to rollback. The rollback
drill returns revision 1 to champion and retains revision 2 as the displaced
rollback target. Every transition is protected by a single-writer lock,
validated before mutation, and compensated if an operation fails.

## Deployment

Registry artifacts are materialised as immutable releases under the ignored
deployment root. `active.json` is an atomic pointer to the selected release.
The Compose override mounts that root read-only and the API resolves it during
startup. A restart is required to activate a changed pointer.

## Evidence policy

Only aggregate reports and an authenticated manifest are committed. Runtime
SQLite files, MLflow artifact paths, deployment files, timestamps, account
identifiers, predictions, and local paths remain uncommitted. The published
report must state that identical model bytes were used in both revisions and
that the exercise is not a model-quality comparison.

## Pre-evidence smoke-parity amendment

Before official evidence publication, review found that exact artifact identity
and alias receipts did not alone record the planned deployment smoke parity.
The digest-protected contract now pins the existing non-holdout synthetic fixture,
expected six-decimal probability `0.190382`, standard risk band, both revisions,
and a `1e-6` public-probability tolerance. Evidence publication must load each
immutable release independently, perform prediction and native-SHAP checks, and
require an identical full-precision output digest. This adds no model, policy,
training, calibration, or sealed-test computation.

The earlier runtime drill is superseded and will not be published. Refreshed
approvals must bind the amended clean implementation before the official drill.
See the [registry architecture](architecture.md) for the control-plane/runtime
boundary.
