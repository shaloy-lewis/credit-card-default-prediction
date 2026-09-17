# Phase 7 governed registry release report

**Deterministic summary SHA-256:** `f88324b6258e014bf399ef6bb5da246e927edd614a14cd2cf2770d1824d61865`

## Outcome

The reviewed `selected_v1` bundle was registered as two immutable deployment
revisions. Revision 2 was manually promoted and deployed, then the governed
rollback restored revision 1. Both revisions contain the same manifest and
model bytes; this is a release-control exercise, not a model comparison.

## Controls demonstrated

- MLflow SQLite registry with content-addressed artifacts.
- Digest-authenticated manual promotion and rollback approvals.
- Candidate, champion, and rollback alias transitions.
- Atomic active-deployment pointer with no hand replacement of model files.
- CI quality, container contract, fixable HIGH/CRITICAL vulnerability gate,
  and CycloneDX SBOM generation.
- Prediction-only smoke parity across both revisions using the reviewed synthetic
  fixture: probability `0.190382`
  with identical risk band, reasons, and full-precision output digest.
- Zero fitting and no sealed-test access.

## Remaining boundary

G4 and Release B remain open for robustness stress testing, monitoring, and
incident controls. This local portfolio drill is not a production-readiness,
fairness, compliance, or model-quality claim.
