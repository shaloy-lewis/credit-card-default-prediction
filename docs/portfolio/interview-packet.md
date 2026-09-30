# Release C interview packet

**Status:** author-prepared review material for a local portfolio project.
Release C remains planned. Shaloy Lewis owns the project and the eventual
acceptance decision; authorship and acceptance are separate records.

## Ten-minute system-design walkthrough

### 0:00-1:00 - Start with the decision

The product supports a human-owned monthly review queue, with capacity fixed at
the top 10% of valid accounts. It does not approve loans or automatically contact
customers. I owned the scientific choices, inference implementation and evidence
trail. The useful question is whether ranking concentrates cases for review and
whether the implementation can preserve its agreed contract. That is different
from proving that an intervention prevents missed payments.

Use the [case study](case-study.md) for the executive story and the
[product brief](../product-brief.md) for operational exclusions. Identify this as
local portfolio work before discussing any results.

### 1:00-2:00 - Establish the data and leakage boundary

The historical source is a fixed UCI snapshot, not a current operating population.
Reviewed lineage fixes account membership and train/validation/test boundaries.
Feature availability is anchored to the scoring date; future outcomes are not
predictors. The selected model uses 19 operational predictors. Demographic
exclusion from scoring does not establish absence of proxy effects or fairness.

Explain the [feature availability review](../data/feature-availability.md) and
[selection protocol](../modeling/selection-protocol.md). A single authorised final
test was consumed. Current verification authenticates stored evidence; neither
interview preparation nor reproduction authorises another test evaluation.

### 2:00-3:00 - Explain scientific judgement

Four fixed classifier fits were compared under the reviewed selection contract.
The released bundle retains identity calibration and the frozen risk bands.
Discuss ranking, probability accuracy and queue capacity separately: the
historical 6,000-account final test recorded AP 0.542867, Brier 0.136304 and lift at
10% of 3.089676. The 600-account queue contained 410 observed defaults. These are
historical measurements, not future performance or avoided defaults. Read them
from the authenticated [Release A report](../../reports/releases/release_a_v1/release-a-report.md);
do not recompute them.

### 3:00-4:00 - Walk through inference contracts

One [inference engine](../../src/credit_risk/inference/engine.py) serves offline,
batch and versioned API paths; Streamlit calls the API. The API returns probability,
band, two explanation categories and lineage. Explanations are model attribution,
not causal advice. The synthetic fixture returns 0.190382; the UI displays 0.1904.

Batch input validation distinguishes invalid rows from invalid batch identities.
Outputs bind original input bytes, model/configuration identity and the batch
identity in a manifest. Deterministic ranking and tie handling produce the review
queue. Reusing an identical completed batch verifies existing outputs without
rewriting them. Each invocation has a distinct opaque trace even when the batch
ID stays the same. See [batch implementation](../../src/credit_risk/inference/batch.py)
and the [repeatable walkthrough](demo-script.md).

### 4:00-5:00 - Explain artifact lineage and release transitions

The model is explicitly retrieved from a revision-pinned distribution and verified
against reviewed hashes. Startup does not download, train or repair a model.
Invalid artifacts fail verification. An external manifest digest supplies trust;
a file agreeing with its own editable checksum is insufficient.

[Phase 7](../registry/architecture.md) demonstrates manual promotion and rollback
in a SQLite-backed MLflow registry. Its immutable deployment revisions and atomic
pointer transitions use identical reviewed model bytes. This proves the release
mechanism, not superiority of a newly trained model.

### 5:00-6:00 - Separate persistent infrastructure from transitions

Show the [architecture diagram](architecture.md). Phase 8 provides PostgreSQL
metadata, MinIO artifacts and deployment volumes, with fixed-state bootstrap,
verification and restart recovery. It does not demonstrate PostgreSQL promotion
or rollback. Repeated bootstrap must verify existing state; corrupt existing
state is preserved for diagnosis rather than removed by failed creation cleanup.

The local Windows rehearsal runs native API/UI processes. Linux CI separately
proves container startup, persisted state, restart checks, blocking vulnerability
scans and SBOM production. Azure/Databricks analogues are conceptual interview
mappings only. No deployed cloud system is claimed.

### 6:00-7:00 - Monitor without silently changing the decision

The authenticated reference fixes numeric bins and repayment categories.
Monitoring reconciles the original batch input hash and account coverage before
reporting feature and prediction total-variation distance. Demonstration thresholds
are 0.10 for warning and 0.20 for investigation; fewer than 200 valid rows means
insufficient data. These thresholds are operational demonstration choices, not
proof of acceptable risk. See the [monitoring workflow](../../src/credit_risk/monitoring/workflow.py) and
[delayed-label contract](../operations/delayed-label-contract.md).

The 400-row synthetic walkthrough deliberately repeats fixture patterns and is
expected to investigate. Retain its completed scores, record a human disposition
and avoid any automatic fit, promotion or policy change. Drift is a prompt to
investigate; it does not by itself measure prediction quality.

### 7:00-8:00 - Make failures diagnosable and recovery bounded

Allowlisted events retain invocation traces and batch identities without customer
fields. Health probes, request failures and latency support service diagnosis.
Runbooks distinguish validation rejection, population shift, corrupt artifacts,
service interruption and SQLite rollback. A repair must restore authenticated
state and prediction parity before resuming use.

The [incident runbooks](../operations/release-b-runbooks.md) and authenticated
[Release B report](../../reports/releases/release_b_v1/release-b-report.md) supply
historical rehearsal evidence. Frozen targets were evaluated separately from the
rehearsals that set them. Never describe a synthetic drill as a production SLO
history or imply that detection alone resolves an incident.

### 8:00-9:00 - Connect ranking to a future causal study

The [hypothetical protocol](intervention-study.md) proposes voluntary human support
outreach versus usual support after eligibility review. Randomise customers once,
using the highest-ranked qualifying account as the index account. The proposed
primary endpoint is next-cycle minimum payment unpaid at its contractual due date;
it is different from the historical dataset's default label.

Analyse original assignment using intention to treat. Reconcile outcomes through
due date plus 14 days; unresolved outcomes remain missing. Report completeness and
missing-outcome bounds before making any definitive effectiveness claim. The
[calculator](planning-calculator.md) gives approximate planning estimates under
explicit assumptions. No real recruitment, contact or causal evaluation occurred.

### 9:00-10:00 - Close with trade-offs and remaining work

Batch-first design suits a fixed monthly capacity and makes reconciliation and
idempotency visible. Local files and digest-bound evidence favour auditability
over distributed throughput. The API demonstrates an integration contract, not
high-scale infrastructure. Calibration, transportability and fairness would need
new approved evidence in a real operating population.

G3 education-related conditions require human review and prohibit a fairness or
compliance claim. G4 is closed only for approved Release B local scope; G5 remains
open. Release C needs a reviewed video, reproducibility record and explicit owner
acceptance. Production use needs a separate governance decision, current data,
security and privacy controls, outcomes, staffing and ongoing review.

## Discussion prompts

| Prompt | Evidence-led answer to develop |
| --- | --- |
| Where could leakage arise? | Distinguish scoring-time availability, split membership, outcome timing and preprocessing ownership; use the feature availability review. The consumed holdout is not available for iterative improvement. |
| Why identity calibration? | Explain the reviewed selection decision and frozen probability contract; discuss reliability and Brier evidence without inventing a new fit or extrapolating calibration to a new population. |
| How would delayed labels change monitoring? | Drift can be measured before outcomes. Performance evaluation requires fixed cohort identity, maturity, completeness and duplicate handling under the delayed-label contract. The source dataset supplies no longitudinal follow-up. |
| What if outreach outcomes are missing? | Retain assignment, report missingness by arm, calculate bounds and withhold definitive effectiveness claims while unresolved; do not treat missing as successful payment. |
| What is portable to cloud infrastructure? | Explain interfaces, artifacts and tests that could transfer, then identify unimplemented identity, secrets, networking, orchestration and database transition work. The diagram's cloud mapping is conceptual. |
| Does lift show outreach effectiveness? | No. It describes concentration in a historical ranked queue. Customer-level randomisation, mature outcomes and a harm-aware analysis would be required for causal claims. |
| Is rollback safe in every deployment mode? | The demonstrated transition workflow is SQLite-specific. Phase 8 authenticates a fixed PostgreSQL/MinIO state and restart recovery; extending transitions needs its own design and evidence. |
| Why not retrain when drift exceeds 0.20? | An alert is not an approved diagnosis or intervention. Investigate data integrity, population and operational changes, retain completed scores, and require a separate authorised model change. |

## Three resume bullets

These describe individual local portfolio work, not employment or production
impact. Keep the linked population and scope qualifications when reusing them.

- Built and governed a local credit-risk portfolio model using 19 operational predictors; the single authorised 6,000-account historical test recorded AP 0.542867 and lift at 10% of 3.089676, with explicit limits on calibration, transferability and causal interpretation ([Release A evidence](../../reports/releases/release_a_v1/release-a-report.md), [selection protocol](../modeling/selection-protocol.md)).
- Engineered shared batch/API inference with digest-verified artifacts, deterministic review queues and idempotent output reuse; demonstrated SQLite release transitions and separately verified PostgreSQL/MinIO fixed-state persistence and recovery for local portfolio scope ([inference parity](../../reports/inference/phase6_v1/inference-parity-report.md), [registry evidence](../../reports/registry/phase7_v1/registry-release-report.md), [platform evidence](../../reports/platform/phase8_v1/evidence-manifest.json)).
- Delivered a digest-bound local Release B assurance package covering robustness, monitoring and incident exercises, with explicit owner approval, retained G3 conditions and an open production-review gate; designed a hypothetical outreach study without claiming prevented defaults ([Release B](../../reports/releases/release_b_v1/release-b-report.md), [owner decision](../reviews/release-b-owner-review.md), [study design](intervention-study.md)).

The [claims inventory](claims-inventory.md) supplies the classification and
limitations for quantitative statements. The Week 12 review record must separately
record author checks and the owner's eventual package decision.
