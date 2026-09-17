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

The earlier runtime drill was superseded and is not published. Refreshed
approvals bind amended clean implementation commit `cb63b39`.

## Reviewed outcome

The official drill registered both reviewed revisions, promoted and deployed
`phase7_rev_002`, and then restored `phase7_rev_001` through the approved
rollback workflow. The final aliases are `champion=1` and `rollback=2`; both
versions retain model digest `844ec1c3...4d88c` and bundle-manifest digest
`df5ce6ce...7cd88`.

The amended publisher independently loaded both immutable bundles. Their
full-precision prediction-and-explanation digest matched exactly, and both
returned probability `0.190382` and risk band `standard` for the pinned
synthetic fixture. The committed evidence is externally authenticated by
manifest SHA-256 `ce36f33da60fe6470d28a76b8053d102e74731115d069c4d476d0c2abbc47da9`.

The container quality gate passed its blocking fixable HIGH/CRITICAL scan and
produced a CycloneDX SBOM. No waiver, model fitting, model change, sealed-test
access, row-level publication, or runtime MLflow dependency was introduced.
See the [registry architecture](architecture.md) for the control-plane/runtime
boundary. G4 and Release B remain open for robustness testing, monitoring, and
incident controls.
