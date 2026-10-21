# Executive case study: credit-risk early warning

**Audience:** Senior Data Scientist and Senior ML Engineer reviewers.
**Status:** Week 11 review candidate; local portfolio scope only.

## Decision and individual ownership

A hypothetical operations manager has limited capacity to review existing
credit-card accounts. The product ranks a monthly snapshot into a top-10% queue,
with probability estimates and model-attribution categories that help a human
understand a score. It does not decide whom to lend to or automatically contact
customers. Shaloy Lewis owns the portfolio's scientific design, engineering,
governance evidence and explicit release decisions. Git history and the linked
protocols provide the implementation record; operating roles are a simulation,
not evidence of a deployed team or real customer adoption.

## Scientific judgement

I kept the decision, prediction horizon and review capacity explicit before
presenting model results. After historical baseline and search work, the
authoritative release protocol compared four fixed classifiers using one shared
validation split, one fit per classifier and no winner refit. It selected the
exact fitted CatBoost artifact, using 19 operational predictors and identity
calibration. Demographic fields remain audit-only. Earlier experiments remain
immutable context rather than a routine rerun path.

**Measured historical results:** the authorised 6,000-account test produced
average precision 0.542867, Brier score 0.136304 and lift at 10% of 3.089676.
The 600-account queue contained 410 historical defaults. The
[Release A dossier](../../reports/releases/release_a_v1/release-a-report.md)
contains the exact denominators, selection history and validation-only uncertainty.
The test was consumed once and cannot be rerun. Those metrics support a historical
ranking claim, not temporal transportability or intervention effectiveness.

The queue also contained 190 non-defaults, while 917 defaults remained outside
it: capacity creates both review burden and missed risk. The published 20%
sensitivity doubles review workload to 1,200 accounts and contains 676 defaults;
it is not an authorized policy change. The [decision analysis](decision-analysis.md)
explains these aggregate trade-offs and independent, explicitly assumed economic
break-even scenarios. The [scientific narrative](scientific-decisions.md) explains
the close model comparison, source anomalies, encoding choices, calibration and
conditional uncertainty without claiming a new experiment.

## What a business stakeholder should do next

```mermaid
flowchart LR
  Snapshot["Closed monthly snapshot"] --> Validate["Validate and reconcile inputs"]
  Validate --> Rank["Frozen scoring and top-10% account queue"]
  Rank --> Review["Human eligibility and support review"]
  Review -. "future approvals required" .-> Pilot["Voluntary randomized outreach pilot"]
  Pilot --> Outcomes["Mature outcomes, harms and staff effort"]
  Outcomes --> Decision["Human evidence review and adoption decision"]
```

Solid arrows before human review describe the local technical workflow;
operational eligibility requires records absent from UCI. The pilot, outcome
collection and adoption decision are proposals, not implemented customer flows.

| Recommendation | Evidence boundary and success criterion |
| --- | --- |
| Treat the queue as a human-review aid | Historical concentration supports investigation; humans verify eligibility and never use the score to impose lending or adverse actions. |
| Measure operational usefulness before expanding capacity | Reconcile valid/rejected records, complete reviews, preserve idempotency and record staff effort, exclusions, complaints and opt-outs. No achieved customer outcome is claimed. |
| Investigate shifts, subgroup triggers and inference failures | Follow the existing human-owned monitoring/incident runbooks, retain evidence and require explicit model/policy change approval. |
| Test actual benefit in an approved representative pilot | Use original randomized assignment, mature endpoints, missingness accounting and harm review; rank quality alone cannot establish effectiveness. |

## Engineering ownership

The operational path uses strict batch contracts, row-level rejection receipts,
deterministic selection, atomic publication and verified idempotent reuse.
One inference engine serves batch and the versioned API; Streamlit is an API
client. The reviewed synthetic request returns 0.190382, with explanation
checks and preserved policy metadata. Artifact hashes and serving-library
compatibility are checked before loading.

Release control is bounded honestly: Phase 7 demonstrates SQLite registry
promotion and rollback around identical model bytes. Phase 8 demonstrates a
persistent PostgreSQL/MinIO stack with fixed-state bootstrap and restart
recovery. It does not claim PostgreSQL promotion/rollback. Linux CI tests the
images, readiness, parity, persistence and blocking vulnerability scans, and
produces software bills of materials. The
[architecture](architecture.md) maps each claim to implementation.

Release B adds reference-based drift monitoring, service targets, robustness
scenarios and incident evidence. Alerts prompt a named human investigation,
not model changes. Subsequent review fixes preserved pre-existing deployments
on bootstrap failure, retained literal account IDs during monitoring and made
batch failures traceable through real CLI events. Approved historical evidence
is verified in its pinned checkout; patched implementation checks run separately.

## Limitations and proposed impact evaluation

The data is a historical Taiwanese snapshot, with no real longitudinal outcomes,
intervention assignments, current-population validation or realised financial
benefit. Education-related G3 conditions and all local-use restrictions remain.
Model attribution is not causality or a legally sufficient adverse-action reason.
The [claim inventory](claims-inventory.md) separates scientific results, synthetic
checks and assumptions.

**Hypothetical proposal:** a customer-randomised trial would compare voluntary
human support outreach with usual support, measuring next-cycle missed minimum
payments. That endpoint is explicitly different from the model's historical
default label. The [protocol](intervention-study.md) covers eligibility, harm,
label maturity, missingness and intention-to-treat analysis; the
[calculator](planning-calculator.md) provides planning estimates only. No customer
is enrolled and no default reduction is claimed.

## Delivery state

Releases A and B are complete for local portfolio use. G4 is closed for that
scope; G3 conditions persist and G5 production review is open. Week 11 prepares
this communication package for review. The [Week 12 record](week12-review.md)
already documents a functional fresh-checkout rehearsal and prepared interview
material at its recorded revision. Release C still requires actual media, visual
review, full-video privacy/claims review, final-package verification and explicit
owner acceptance under the [acceptance plan](release-c-acceptance-plan.md).
