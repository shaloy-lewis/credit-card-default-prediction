# Hypothetical voluntary support outreach study

**Status:** Week 11 review candidate; design only. No customers are enrolled or
contacted, and no effect has been measured. Shaloy Lewis owns this portfolio
proposal; the operating roles below are hypothetical future roles. This is not
permission to run a lending or customer intervention trial.

## Question, population and assignment

Would offering one voluntary human support conversation, in addition to usual
support, reduce next-cycle missed minimum payments among eligible customers in
a human-reviewed risk queue? The model identifies a proposed recruitment pool;
it is not an estimate of treatment benefit and need not identify the customers
who benefit most from support. This study compares outreach with usual support
within that pool; it does not establish the benefit of model ranking itself.

Use the unchanged monthly top-10% **account** queue. Human review then checks an
active account, six months of required history, accurate next due date and
minimum payment, contact permission, and at least seven calendar days between
randomisation and the next contractual due date. Apply the
[product brief's exclusions](../product-brief.md#intended-population-and-exclusions):
closed, written-off, fraudulent, deceased, disputed or restricted accounts;
existing default or late-stage collections; active hardship cases; and failed
quality/history checks. The historical dataset cannot establish this eligibility.

The unit of randomisation and analysis is a **customer**, with one index account
and one enrolment during the study. If several accounts qualify, use the highest
ranked account, retaining the existing account-ID tie break. An operational
customer-to-account mapping and prior-enrolment register are required future
inputs, not inferred from the source dataset. Deduplication occurs after queue
review and does not change the scorer or fill vacated queue places.

An allocation service independent of outreach staff assigns customers 1:1 by a
concealed random permutation of equal-sized arms, generated once before
recruitment. Allocate only after eligibility is locked; record sequence version,
assignment and cohort before revealing the arm. Staff cannot reassign customers.
The planned number of customers is even; stop at the fixed target. If recruitment
ends early, retain the actual arm sizes and report the shortfall. Analysts receive
masked arm labels until the outcome and analysis specification are locked.

## Arms and delivery fidelity

| Arm | Hypothetical delivery |
| --- | --- |
| Treatment | Usual support plus one attempt at a voluntary, neutral support conversation within three business days of assignment. Explain available existing support channels; participation is optional. |
| Control | Usual support remains available on identical terms. No study-initiated outreach attempt. |

An opt-out ends the attempt immediately; do not retry. The intervention must not
change account terms, fees, limits, credit decisions or collections actions.
Routine customer-initiated support remains available in both arms. Log assigned
arm separately from attempted, reached and declined contact. Record cross-arm
contact and other support as contamination, retaining original assignment.
Neither the randomiser nor contact delivery is implemented in this repository.

## Outcomes and label maturity

**Primary binary outcome:** whether the index account's next contractual minimum
payment remains unpaid in full at its contractual due date, using payment
effective dates. Freeze the due date and amount at assignment. This proposed
endpoint is **not** the historical dataset's next-month default label, a
regulatory default definition or a claim about realised default reduction.

At due date plus 14 calendar days, reconcile the payment ledger and authoritative
corrections to establish what was paid by the due date. Money received after the
due date does not erase the primary missed-payment event. Missing ledgers,
ambiguous effective dates, disputed amounts or unresolved changes to the frozen
obligation remain unknown, never silently coded as payment success or failure.
Retain label version, maturity date, corrections and the previous version.

Secondary measures are reached-contact and support-engagement rates, completion
of the originally due payment by maturity, complaints, opt-outs and staff minutes
per assigned customer. Report arm denominators and observation windows. Compare
staff effort descriptively; monetisation would require separately declared cost
assumptions. Guardrails include serious complaints, coercion, unauthorised
contact, privacy incidents and unequal access to usual support.

Required future records: immutable eligibility and assignment snapshots;
customer/index-account keys held in restricted storage; due date/minimum amount;
consent and exclusions; delivery/contamination events; effective-dated payments;
label status/version; complaint and opt-out receipts; staff effort. Keep audit
attributes separate from predictors. Apply the existing
[delayed-label principles](../operations/delayed-label-contract.md) to cohort
identity, duplicate rejection, corrections and completeness, using this study's
explicit endpoint and maturity window.

## Analysis, missingness and stopping

Estimate the intention-to-treat risk difference **treatment minus control** by
original assignment, regardless of contact, uptake or cross-over. Negative
values favour outreach. With complete mature outcomes, report each arm's event
count/rate, the difference, a 95% Newcombe interval based on Wilson binomial
intervals, and a two-sided pooled two-proportion test at alpha 0.05 without
continuity correction. Secondary outcomes are exploratory; no multiplicity-free
confirmatory claim is made for them. Per-contact comparisons cannot replace ITT.

Keep every randomised customer in the cohort inventory. For an arm with `e`
known events, `m` unknown outcomes and `N` assigned customers, its event-rate
bounds are `[e/N, (e+m)/N]`. Subtract the opposite endpoints to bound the ITT risk
difference. Report completeness by arm and cohort, and reasons for unknowns.
Do not claim an observed-case rate is a full-cohort ITT estimate. Withhold a
definitive effectiveness conclusion while outcomes remain unresolved; assumed
attrition in the power calculation does not remove missingness bias.

The [planning calculator](planning-calculator.md) fixes the recruitment target
before enrolment. The illustrative 30% versus 27% planning scenario requires
3,554 analysable customers per arm and 3,949 recruited per arm with 10% assumed
attrition: 7,898 total. These are hypothetical planning estimates, not estimates
from the historical queue. Representative future baseline evidence and recruitment
feasibility must replace the assumptions before any real trial is approved.

Stop recruitment at that target or after 12 monthly recruitment cohorts,
whichever occurs first. Report an unreached target as a feasibility limitation;
do not extend recruitment after looking at effects. Analyse only after the last
outcome window matures and the data are locked. There is no early efficacy
stopping or repeated significance testing. A serious complaint, coercion,
privacy breach or unauthorised contact immediately pauses study outreach and
new recruitment; usual support continues. An independent safety reviewer and
policy owner document containment and a resume/terminate decision. Do not
resume automatically. Retain assignments and safely obtainable outcomes after
an early stop; report its reason and limitations.

## Ownership and acceptance

The hypothetical policy owner approves eligibility and any future operating
protocol; the operations owner controls delivery and opt-outs; the data owner
certifies mature labels; the analyst freezes analysis before unmasking; an
independent reviewer assesses harm and claims. The project owner separately
reviews this portfolio document. None of those real-world approvals is supplied
by Week 11 completion.

[Release C acceptance](release-c-acceptance-plan.md) requires agreement between
this protocol, calculator assumptions, narrative and claims inventory. G3
conditions and local-use restrictions remain active; G5 stays open. There is no
training, cohort rescoring, actual outreach or causal-effect estimation here.
