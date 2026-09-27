# Release B implementation and sign-off procedure

The implementation adds platform evidence, prediction-only robustness,
reference-fixed monitoring, measured service targets, isolated incident exercises,
and a zero-scoring release dossier. G4 and Release B remain open until measured
evidence passes review and a project-owner decision binds the final dossier.

## Execution order

Use Python 3.12 and the repository's locked uv environment. Materialize and verify
the reviewed model. Build Phase 1 data only to reproduce the approved validation
cohort. Do not run model select, historical training, calibration or final-test
commands.

The Release B evidence GitHub workflow runs the complete Linux rehearsal on the
implementation branch. It uses a separate Compose project, new named volumes,
ephemeral ignored credentials, Trivy 0.70.0, blocking fixable HIGH/CRITICAL scans
and CycloneDX inventories. It uploads aggregate evidence and scan diagnostics;
source rows, scores, credentials and private runtime logs are not uploaded.

For a local Linux rehearsal, create an ignored .env from .env.example with fresh
local secrets, install the same scanner version, and run:

    uv run python -m credit_risk.assurance.collect

The individual commands are also available:

    credit-risk platform rehearse
    credit-risk platform publish-evidence
    credit-risk platform verify-evidence --expected-manifest-sha256 DIGEST
    credit-risk robustness build
    credit-risk robustness verify --expected-manifest-sha256 DIGEST
    credit-risk monitor reference
    credit-risk monitor benchmark
    credit-risk monitor acceptance --expected-benchmark-sha256 DIGEST
    credit-risk monitor drill --expected-benchmark-sha256 DIGEST

Monthly monitoring consumes a verified batch and the exact original input:

    credit-risk monitor batch --input INPUT --run-root BATCH --expected-reference-sha256 DIGEST --output reports/monitoring/NEW_REPORT
    credit-risk monitor service --log-path experiment/EVENTS.jsonl --output reports/monitoring/NEW_SERVICE_REPORT
    credit-risk monitor verify --kind batch --evidence-root REPORT --expected-manifest-sha256 DIGEST

Reports never approve changes to the model or scoring policy. An investigation
state requires human disposition. Insufficient sample size must remain explicit.

## Review and release

Download and verify the aggregate artifact package from the exact implementation
run. Populate the frozen release contract using its proposed contract, review and
commit the reports/configuration, then require every normal CI job to pass for
that exact commit. Capture CI separately to avoid circular commit hashes:

    credit-risk release capture-ci
    credit-risk release build-b --expected-ci-sha256 CI_RECEIPT_DIGEST
    credit-risk release verify-b --expected-manifest-sha256 DOSSIER_DIGEST

The result remains pending_owner_signoff. The dossier includes an approval
template with every stress finding, incident and carried-forward risk. The
project owner must supply an explicit decision, identity and operating restriction
for every disposition, retaining the G3 conditions. The decision must name the
exact finalized dossier digest. Then authenticate that separate approval:

    credit-risk release verify-b --expected-manifest-sha256 DOSSIER_DIGEST --approval APPROVAL_JSON --approval-sha256 APPROVAL_DIGEST

Only after that succeeds may the roadmap, governance status, progress log and
README mark G4 closed for the local portfolio scope. Production approval, G5
ongoing review and Release C communication remain separate work.

Existing evidence is immutable. A failed official run requires diagnosis and a
new reviewed evidence version; never overwrite reports, delete historical
volumes, relax a scan threshold, or adjust a timing target to make a run pass.


The 2026-09-27 outage rehearsal exceeded its 6.950855-second recovery target.
The targets from the original three rehearsals in implementation `f365481`
are now preserved in `configs/monitoring/release_b_service_targets_v1.json`,
with the original manifest and summary bytes. The collector reuses these
externally authenticated targets; it does not estimate new targets after this
failure. The recovery fix starts only the identified isolated API container,
without Compose dependency orchestration. A new separate acceptance run and
outage drill must pass the unchanged limits. Failed-run archives remain ignored.
