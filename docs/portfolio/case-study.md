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
this communication package for review. Release C still requires the fresh-checkout
rehearsal, reviewed recording, privacy/claims review, interview material and
explicit owner acceptance defined in the [acceptance plan](release-c-acceptance-plan.md).
