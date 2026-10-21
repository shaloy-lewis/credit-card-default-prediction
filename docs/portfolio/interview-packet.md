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

## Twenty evidence-backed interview answers

The short answer is suitable for a stakeholder. The reasoning column provides
the technical defense and its limits; use the linked evidence for follow-up.

| Question | Stakeholder answer | Technical reasoning and evidence |
| --- | --- | --- |
| 1. Why this business problem? | Limited review capacity makes prioritization a concrete decision with measurable workload. | A monthly existing-account queue aligns historical billing features with a bounded human action. It avoids inventing a real-time lending use case for a snapshot dataset. [Accepted decision brief](../product-brief.md), [batch-first ADR](../adr/0001-batch-first-scoring.md). |
| 2. How did you formulate the prediction task? | Estimate the published next-month default risk after the current billing snapshot closes. | One historical client record is the observation; account-level operation is simulated. Six history periods do not create longitudinal observations. The source label is not the proposed trial's missed-minimum-payment endpoint. [Data/time semantics](../data/data-card.md#time-and-label-semantics), [study](intervention-study.md). |
| 3. How did you prevent leakage? | Keep future outcomes, demographics and identifiers out of prediction; protect the final evaluation. | Stable reviewed split membership, training-only logistic scaling, fixed status encoding, development-only selection and an immutable consumed-test tombstone restrict information use. Real ingestion-time availability cannot be verified from this timestamp-free source; duplicate-profile dependence remains a limitation. [Availability review](../data/feature-availability.md), [split tests](../../tests/integration/test_selection_split.py), [tombstone](../../src/credit_risk/modeling/final_test_workflow.py). |
| 4. Why these features? | Use only the available credit-limit and six-month repayment, bill and payment history needed for the stated decision. | The 19-feature contract excludes ID, target and four audit demographics. CatBoost treats integral status codes as strings/categories and retains raw money values; no new ratios, imputation or clipping enter the release. Proxy risk remains. [Feature contract](../../configs/modeling/feature_contract_v1.json), [cleaning/encoding rationale](scientific-decisions.md#cleaning-and-feature-reasoning). |
| 5. Why CatBoost? | It won the predeclared comparison, although the closest simpler model was competitive. | Four fixed fits shared validation; Brier/lift guardrails preceded AP and a 0.002 simplicity band. AP 0.556510 versus histogram boosting 0.554306 just exceeds that band, with identical reported lift. This does not prove significant superiority; the four fits followed earlier CV/search. [Selection rule/outcome](../modeling/selection-protocol.md), [decision narrative](scientific-decisions.md#why-these-algorithms-and-why-this-winner). |
| 6. What do the metrics demonstrate? | Historical ranking and probability quality, with explicit capacity and uncertainty limits. | Final-test AP 0.542867 summarizes precision/recall ranking; Brier 0.136304 measures probability error; lift 3.089676 describes concentration in the 10% queue. Brier is not calibration alone. Preserved bootstrap intervals are validation-only and conditional on fixed predictions. [Authenticated dossier](../../reports/releases/release_a_v1/release-a-report.md), [metric distinctions](scientific-decisions.md#what-each-metric-answers). |
| 7. Why not accuracy? | A model can favor the majority class and still fail to find useful cases for a limited queue. | Accuracy needs a decision threshold and ignores the intended capacity trade-off. Compare with prevalence and repayment-rule baselines; use AP, probability quality and precision/recall at capacity. No new accuracy result is claimed. [Baseline evidence](../../reports/modeling/baseline_v1/baseline-report.md), [metrics](scientific-decisions.md#what-each-metric-answers). |
| 8. How does the top-10% policy work? | Review exactly the highest-ranked tenth of valid accounts, rounded down. | Batch ranks full-precision probability descending then account ID ascending and selects `floor(valid_rows * 0.10)`. Every duplicate ID occurrence is rejected. Fixed bands are separate from portfolio selection; human eligibility must still be established. [Contract](../../configs/inference/phase6_v1.json), [batch](../../src/credit_risk/inference/batch.py). |
| 9. How would you show financial impact? | First test whether a voluntary support offer helps, then value measured benefit and incremental cost. | Customer-randomized ITT analysis needs mature outcomes, missingness bounds and harm guardrails. The proposed endpoint differs from historical default. The trial tests the offer in the queued pool, not the superiority of ranking; independent assumed break-even scenarios are not realized savings. [Protocol](intervention-study.md), [decision economics](decision-analysis.md). |
| 10. How would you validate a new population? | Treat transfer as a new evidence and approval problem. | First verify label semantics, eligible population and point-in-time feature availability on representative timestamped data; predeclare external/temporal evaluation, probability/capacity/subgroup checks and label completeness. Any fitting/recalibration requires separate approval and new independent evidence. The consumed test is unavailable. [Scientific limits and future adoption](scientific-decisions.md#responsible-interpretation-and-remaining-failure-modes), [delayed-label design](../operations/delayed-label-contract.md). |
| 11. What fairness concerns remain? | Excluding demographics does not remove proxies or ensure equitable access to support. | Validation q90 diagnostics retain education-code 1/3 selection-rate triggers; small unsupported groups report counts only. Wilson prevalence and within-group bootstrap intervals do not certify fairness or causality. Exact-count batch queues differ from fixed-threshold audit selection. [Subgroup report](../../reports/governance/phase5_v1/fairness-report.md), [G3 conditions](../../reports/governance/phase5_v1/g3-review.md). |
| 12. What happens when drift is detected? | Investigate and record a human decision; an alert does not retrain the model. | Authenticate cohort/reference, check ingestion, population and eligibility changes, distinguish distribution shift from performance deterioration, and wait for mature labels before quality claims. Retain completed scores and apply the relevant runbook. [Monitoring](../../src/credit_risk/monitoring/workflow.py), [runbooks](../operations/release-b-runbooks.md). |
| 13. How do you respond to an inference incident? | Contain unsafe use, preserve evidence, restore verified state and check recovery before resuming. | Use trace/batch IDs, rejections and readiness to distinguish input, artifact and service failures. Never overwrite a conflicting published batch. Verify hashes and synthetic parity after repair; use approved SQLite rollback only in its demonstrated mode. [Runbooks](../operations/release-b-runbooks.md), [batch trace tests](../../tests/unit/inference/test_batch_traceability.py), [registry evidence](../../reports/registry/phase7_v1/registry-release-report.md). |
| 14. How would you scale? | Measure the deadline and workload first, then remove the demonstrated bottleneck. | Profile in-memory batch/ranking and SHAP, API concurrency, storage and completion times with authorized synthetic workloads. Distributed execution would need globally deterministic ranking, coordinated activation, idempotency and tested failure semantics. No throughput guarantee or deployed cloud is claimed. [Architecture limits](architecture.md#what-production-or-cloud-deployment-would-require). |
| 15. What would you change and why? | Improve independent population evidence before adding model complexity. | In a separately approved future study, use timestamped cohorts, dependence/duplicate sensitivity and paired comparisons of simpler contenders. Evaluate whether the small AP advantage merits operational complexity; fit or recalibrate only under a new approved protocol. Today I would preserve the artifact and improve evidence usability. [Audit](senior-portfolio-audit.md), [close comparison](scientific-decisions.md#why-these-algorithms-and-why-this-winner). |
| 16. What were the main trade-offs? | Favor a reviewable decision and reproducible evidence over extra algorithms or infrastructure. | Preserving the exact winner gives up a full-development refit; fixed capacity gives up recall; excluding demographics retains proxy risk; local files favor auditability over distributed availability; identity calibration is protocol-fixed, not a newly optimized calibrator. [Scientific decisions](scientific-decisions.md), [architecture decisions](architecture.md#engineering-decisions-and-practical-limits). |
| 17. What would you prioritize in six more weeks? | Finish the portfolio evidence and prepare the next real validation decision. | Proposed sequence: weeks 1–2 complete actual media, visual/privacy review and final-revision checks, then seek owner acceptance; weeks 3–4 design representative cohort, label/availability and access contracts with hypothetical stakeholders; weeks 5–6 predeclare external evaluation, eligibility/cost feasibility and pilot guardrails. Without data/approval, deliver reviewed designs rather than fabricated performance. No refitting or trial execution is assumed. [Open criteria](release-c-acceptance-plan.md), [future study](intervention-study.md). |
| 18. What did you personally own? | I independently developed the scientific, engineering and governance portfolio. | My work spans problem framing, data contracts, development experiments, fixed release selection, artifact/inference implementation, tests, monitoring, incident exercises and documentation. Git history and linked source/protocols are inspectable; hypothetical team roles are not employment or external reviewers. Author checks and owner acceptance remain separate even when the same person fills both roles. [Case study](case-study.md), [source map](architecture.md#component-to-implementation-map). |
| 19. What supports reliability? | Contract tests, artifact checks, parity, safe retries and bounded recovery drills. | Authenticated evidence binds historical implementation and model identities; Linux CI separately exercises images, scans and persistence. Synthetic checks and the September fresh-checkout rehearsal show local behavior, not production SLO history or approval of later commits. [Parity](../../reports/inference/phase6_v1/inference-parity-report.md), [CI definition](../../.github/workflows/ci.yml), [rehearsal identity/results](week12-review.md). |
| 20. What does this project not prove? | Real-world benefit, current-population validity, fairness/compliance certification or production readiness. | No longitudinal validation, actual outreach effect, financial saving, PostgreSQL promotion/rollback, cloud deployment or accepted Release C video is established. G3 conditions and G5 remain. SHAP explains model attribution, not causal recourse or adverse-action reasons. [Claims inventory](claims-inventory.md), [governance status](../governance/model-governance-status.md), [Week 12 open media](week12-review.md). |

## Three resume bullets

These describe independently developed local portfolio work, not employment or production
impact. Keep the linked population and scope qualifications when reusing them.

- Built and governed a local credit-risk portfolio model using 19 operational predictors; the single authorised 6,000-account historical test recorded AP 0.542867 and lift at 10% of 3.089676, with explicit limits on calibration, transferability and causal interpretation ([Release A evidence](../../reports/releases/release_a_v1/release-a-report.md), [selection protocol](../modeling/selection-protocol.md)).
- Engineered shared batch/API inference with digest-verified artifacts, deterministic review queues and idempotent output reuse; demonstrated SQLite release transitions and separately verified PostgreSQL/MinIO fixed-state persistence and recovery for local portfolio scope ([inference parity](../../reports/inference/phase6_v1/inference-parity-report.md), [registry evidence](../../reports/registry/phase7_v1/registry-release-report.md), [platform evidence](../../reports/platform/phase8_v1/evidence-manifest.json)).
- Delivered a digest-bound local Release B assurance package covering robustness, monitoring and incident exercises, with explicit owner approval, retained G3 conditions and an open production-review gate; designed a hypothetical outreach study without claiming prevented defaults ([Release B](../../reports/releases/release_b_v1/release-b-report.md), [owner decision](../reviews/release-b-owner-review.md), [study design](intervention-study.md)).

The [claims inventory](claims-inventory.md) supplies the classification and
limitations for quantitative statements. The Week 12 review record must separately
record author checks and the owner's eventual package decision.
