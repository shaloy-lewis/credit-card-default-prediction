# Release B operating and incident runbooks

Scope: local portfolio demonstration. Operator and release decision owner: project
owner. Each drill receipt records detection, containment, recovery and owner
disposition. Do not treat pending dispositions as approval.

| Runbook ID | Detection and containment | Recovery and verification |
| --- | --- | --- |
| missing_columns | Stop the invalid file; report failed batch attempt. | Correct the producer schema, use a new snapshot identity, verify the batch. |
| invalid_values | Quarantine invalid rows; report rejected counts and rules. | Correct source values; publish under a new identity, preserving the partial run. |
| duplicate_ids | Reject every occurrence; investigate producer identity handling. | Correct the source; never silently choose one duplicate. |
| population_shift | Investigate reference-fixed feature drift; hold human use of the queue. | Confirm lineage, source change and representativeness; owner accepts a restriction or keeps use suspended. |
| prediction_shift | Investigate score/band drift; preserve the scored batch. | Verify artifact and feature lineage; owner reviews sensitivity before restoring human use. |
| artifact_integrity | Startup/verification refuses missing or altered bytes; stop promotion. | Restore only digest-authenticated bytes in the isolated rehearsal; require readiness and smoke parity. |
| service_interruption | Health probe fails; suspend dependent demonstration requests. | Restart the isolated API; verify frozen recovery target, all persistent state and smoke prediction. |
| phase7_sqlite_rollback | Stop further promotion on failed operational validation. | Use the reviewed Phase 7 approval/digest in an isolated SQLite registry; verify champion revision 1, API readiness and smoke parity. |

The drill runner uses new ignored runtime paths and the separate
credit-risk-release-b Compose project. It must refuse existing rehearsal
resources rather than deleting volumes or reusing historical deployments.
Retain failed receipts for diagnosis; remediate the cause before creating a new
reviewed evidence version. Never weaken a scan or timing target after seeing a
failed official run.

The Phase 8 PostgreSQL/MinIO stack has fixed-state bootstrap and verification.
Its recovery procedure is restart plus exact-state verification. Its bootstrap
does not implement promotion/rollback. The Phase 7 rollback demonstration is
separately evidenced and must never be represented as a PostgreSQL transition.

Monthly operation: score and verify the batch, run monitor batch against the
authenticated reference, review status/rejections, preserve the report and
record any investigation disposition. Run monitor service on allowlisted JSON
events. Fewer than 200 valid rows is insufficient data, not a clean result.
Drift never authorizes automatic training, promotion or policy changes.

Service targets are twice the worst of three measured rehearsals on the recorded
machine. A separate acceptance run and outage drill must meet those frozen targets.
Retain G3 restrictions: demographics are audit-only; outreach is human-owned;
no adverse action, fairness/compliance certification, or geographic transfer claim.
Representative data, privacy design and ongoing monitoring are required before real use.


For the isolated API outage drill, resolve the API container with the dedicated
Compose project and restart that container by its verified ID. Do not invoke
project dependency startup during recovery. Preserve the existing frozen target
and the failed recovery receipt; never increase the target after a timing failure.
