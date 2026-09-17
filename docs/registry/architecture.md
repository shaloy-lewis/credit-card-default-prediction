# Phase 7 registry and deployment architecture

## Control plane and serving boundary

Phase 7 separates release control from inference. The optional MLflow environment
owns local registration and alias state; the API loads only a verified two-file
bundle from an immutable deployment directory.

```text
selected_v1 bundle ──> digest validation ──> content-addressed registry artifact
                                               │
reviewed approvals ──> promote / rollback ─────┤
                                               v
                              immutable release revision
                                               │
                                               v
                                    atomic active.json pointer
                                               │ read-only mount
                                               v
                                    FastAPI selected-bundle loader
```

MLflow 3.15 uses ignored local SQLite metadata for this bounded demonstration.
PostgreSQL, MinIO, and a persistent registry service are intentionally deferred.
The registry database, runtime receipts, locks, and deployment directories are
not release artifacts and are never copied into the API image.

## Identity and state transitions

`phase7_rev_001` and `phase7_rev_002` are deployment revisions of the same
reviewed model—not alternative candidates. Their bundle manifest and model
SHA-256 digests are identical. Initial registration assigns revision 1 to
`champion` and revision 2 to `candidate`; promotion assigns revision 2 to
`champion` and revision 1 to `rollback`; the approved drill restores revision 1.

Promotion and rollback accept only digest-authenticated approval files bound to
the frozen configuration, implementation commit, exact model, and successful
quality/container checks. A single-writer lock serialises mutations. Every
transition validates the full existing state first and restores prior aliases
and deployment state if a later step fails.

## Deployment activation

Each materialised release contains exactly `manifest.json` and `model.cbm`.
The release directory is immutable; `active.json` is atomically replaced and
binds registry version, release revision, approval, configuration, manifest,
and model digests. Application startup resolves bundle roots in this order:

1. an explicit `create_app(bundle_root=...)` test or integration override;
2. the authenticated pointer beneath `CREDIT_RISK_DEPLOYMENT_ROOT`;
3. the committed `models/selected_v1` default.

Changing the pointer requires an application restart. Startup fails readiness
if the pointer, release allowlist, bundle digests, dependency versions, or model
contract differs from the reviewed state. Prediction routes and response
schemas are unchanged.

## Supply-chain controls and evidence

GitHub Actions pins third-party actions and the Python base image. The container
job runs runtime contract tests, blocks fixable HIGH or CRITICAL Trivy findings
without a repository waiver, and uploads a CycloneDX SBOM. The authenticated
[release-control report](../../reports/registry/phase7_v1/registry-release-report.md)
records successful registration, promotion, deployment, synthetic smoke parity,
and rollback without publishing runtime paths, timestamps, account identifiers,
or row-level predictions.

This architecture demonstrates local release mechanics. It does not establish
external production operation, compare model quality, close G4, or replace the
remaining robustness, monitoring, and incident-control work.
