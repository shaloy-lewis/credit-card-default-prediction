# Phase 7 rollback runbook

## Trigger

Use rollback when a promoted local release fails readiness, contract, security,
or operational validation. Do not retrain or alter the registered artifact as
part of incident response.

## Procedure

1. Stop further promotion activity and obtain the reviewed rollback approval.
2. Run `credit-risk registry rollback` with the approval's external digest.
3. Restart the API through the registry Compose override.
4. Require `/ready` and the committed synthetic `/v1/predict` smoke test to pass.
5. Run `credit-risk registry status` and confirm revision 1 is champion and active.
6. Preserve the displaced revision and receipts for investigation.

The command compensates registry aliases and the active pointer if activation
fails. Phase 10 will extend this local procedure with monitoring alerts and
incident ownership.
