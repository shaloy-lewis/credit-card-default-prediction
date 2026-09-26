# ADR 0009: Release B operational assurance

Status: accepted implementation boundary; official evidence and owner sign-off pending.

Release B adds authenticated platform, validation-only robustness, monitoring,
incident and release evidence. All new commands preserve the selected model,
19 predictors, identity calibration, risk bands and 10% review policy. No fit,
tuning, calibration, bootstrap generation or sealed-test evaluation is permitted.
Full-file Phase 1 integrity checks may parse the source; only the reviewed 4,800
development-validation accounts may enter model analysis.

Monetary feature groups are independently scaled by 0.9, 1.1, 0.75 and 1.25 with
round-to-nearest-even integer conversion. Nonnegative repayment codes increase
by one or two, capped at nine; negative categorical codes remain unchanged.
Population diagnostics use credit limit <= the validation 25th percentile and
any repayment code >= 2. Perturbed features have no observed counterfactual
labels: report prediction/queue/explanation sensitivity, not predictive quality.
Historical subset metrics disclose support and are not out-of-time validation.

Monitoring uses fixed reference deciles with overflow bins, exact repayment
categories, and total variation distance. Distances >= 0.10 warn; >= 0.20 require
investigation. Fewer than 200 valid rows is insufficient data. These are local
demonstration thresholds, not statistically calibrated production guarantees.
Alerts do not retrain, promote, alter scoring, or discard completed batches.

Benchmark three rehearsals on one documented machine, each with 20 warmups,
200 serial requests, a 10,000-row synthetic batch and restart recovery. Freeze
targets at twice the worst rehearsal before a separate acceptance run.

Phase 7 SQLite remains the promotion/rollback demonstration. Phase 8 remains
fixed-state persistent bootstrap/verification and restart recovery. Do not
represent bootstrap as PostgreSQL promotion or rollback.

Existing scientific and operational evidence is immutable. New publishers
require clean committed source, refuse overwrites and symlinked paths, and bind
aggregate outputs to source/configuration digests. Offline verification requires
an external manifest digest. Runtime row data and detailed receipts stay ignored.
The Release B builder cannot approve itself: project-owner approval must refer
to the completed dossier digest. G4 stays open until every required control
passes and every finding has an explicit compatible disposition. G3 conditions,
G5 production review and Release C communication work remain in force/open.


Sequential CI collection permits complete new, untracked packages only beneath
the four approved platform/robustness/monitoring/incident report subtrees. Every
such package must verify before another publisher runs. Any changed tracked
file or untracked source/configuration/test file blocks publication. This permits
one clean implementation commit to produce several packages without inventing
intermediate code commits; review and commit the aggregate artifacts afterward.

Exact-commit CI is supplied as an externally hashed runtime receipt to build-b,
not stored as a hash inside the implementation commit it describes. This avoids
a circular commit/CI/configuration dependency. Owner approval is likewise a
separate record bound to the finalized dossier digest.


The original MinIO image at Quay returned unauthorized in both Linux CI runs on
2026-09-26; its corresponding official binary archive returned HTTP 410. Phase 8
now builds the same signed upstream release commit
`7ced9663e6a791fef9dc6be798ff24cda9c730ac` from a checksum-pinned source archive,
with pinned Go and runtime image digests. This is a packaging recovery, not an
object-store version upgrade. The MinIO source license and credits are included.
The existing blocking scan scope remains API, MLflow and UI; the old MinIO
application/dependency risk and upstream distribution availability need an
explicit local-demo restriction in the current Release B disposition. Production
supportability and a maintained object-store replacement remain G5 work.
