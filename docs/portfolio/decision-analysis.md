# Review capacity, decision economics and impact

**Scope:** historical aggregate interpretation plus explicitly hypothetical
planning. This document loads no model or customer rows, creates no new
performance result, and changes neither the top-10% policy nor the study design.

## What the operations owner can decide

A hypothetical credit-risk operations manager has a limited monthly review
budget. The system ranks valid existing-account records after snapshot close,
estimating the source label's next-month default. An operations team would still
need to establish eligibility, contact permission, hardship status and suitable
support before taking any action. The source cannot establish those conditions.

The implementation selects `floor(valid_rows * 0.10)` by full-precision risk
descending and account ID ascending. Operational eligibility is a human-owned
upstream responsibility, not a property established by schema validation. A
single API score has no queue position; risk bands are fixed validation-derived
thresholds, independent of the portfolio's exact-count selection. See the
[inference contract](../../configs/inference/phase6_v1.json) and
[product exclusions](../product-brief.md#intended-population-and-exclusions).

Ten percent is a capacity assumption for this demonstration, not an estimated
economic optimum. The published alternative capacities below describe existing
historical sensitivity evidence; they do not authorize changing the policy.

## Capacity and operational burden

**Measured historical evidence:** the one authorized test contained 6,000
accounts and 1,327 defaults. Counts, precision and recall below come from the
[Release A capacity table](../../reports/releases/release_a_v1/release-a-report.md)
and [final-test summary](../../reports/modeling/final_test_v1/summary.json).
Non-default and outside-queue counts are subtraction of those aggregates only.

| Published capacity | Accounts reviewed | Defaults in queue | Non-defaults in queue | Defaults outside queue | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% sensitivity | 300 | 220 | 80 | 1,107 | 73.33% | 16.58% |
| **10% frozen policy** | **600** | **410** | **190** | **917** | **68.33%** | **30.90%** |
| 20% sensitivity | 1,200 | 676 | 524 | 651 | 56.33% | 50.94% |

The 10% queue concentrated historical events but left most defaults outside it.
Moving from the published 10% to 20% capacity would double workload and add 266
historical defaults and 334 non-defaults to review. This is a retrospective
description of the existing aggregate table, not an approved capacity change or
a forecast. Higher recall comes with lower queue precision and more work.

The 190 non-defaults in the frozen queue illustrate potential unnecessary review
or contact burden; a non-default is not proof that support would be useless.
Likewise the 917 defaults outside the queue illustrate missed risk, not 917
preventable events. Human review, opt-outs and customer-level deduplication can
reduce the downstream study pool. The [study](intervention-study.md) does not
backfill vacancies or rerank after exclusions.

Opportunity cost includes staff time displaced from other support, missed
customers who might benefit, and potential harm from unwanted contact. Neither
credit limit nor a score measures exposure at default, loss given default,
customer benefit, or responsiveness to outreach. The accepted brief's proposed
expected-loss alternative is not the implemented release policy; no expected-loss
optimization or monetary impact is claimed here.

## Standalone hypothetical break-even analysis

Use a future, clearly defined endpoint and comparator. Here the endpoint is the
study's **next-cycle missed minimum payment**, not the historical default label.
All monetary quantities below are illustrative **monetary units (MU)**; none is
an observed bank cost, saving, or currency-specific estimate.

Define:

- `K`: account reviews, including reviews that do not lead to an eligible customer.
- `M`: eligible unique customers offered the intervention after review; `M <= K`.
- `delta`: assumed absolute reduction in endpoint probability per customer
  assigned an offer, relative to usual support. This is an intention-to-treat
  assumption, incorporating non-contact and non-uptake rather than multiplying
  them in again. It may be zero or negative.
- `B`: assumed incremental value per avoided endpoint event, with an explicit
  valuation method required before real use. A missed payment is not a full
  credit loss. Avoid double counting recoveries, fees or downstream benefits.
- `c_review`: incremental staff cost per account review.
- `c_offer`: incremental intervention cost per assigned customer, including
  unsuccessful attempts; `h`: assumed harm/opportunity cost per assigned customer.
- `F`: fixed incremental delivery cost allocated to this cohort. Research costs
  and implementation investment require separate accounting if relevant.

Relative to usual support with no additional queue review, the assumed net value is:

```text
net = M * delta * B - K * c_review - M * (c_offer + h) - F
delta_break_even = (K * c_review + M * (c_offer + h) + F) / (M * B)
```

The break-even expression requires `M > 0` and `B > 0`. With no eligible customers
there is only cost. If the threshold exceeds the feasible baseline event rate,
even eliminating every event cannot break even under those assumptions. A
negative effect increases harm/cost; a favorable scenario is not evidence that
the intervention works. Non-monetizable safety guardrails can prohibit an
intervention even when assumed net value is positive.

**Illustrative scenario, independent of all historical cohorts:** assume
`K = M = 1,000`, `c_review = 2 MU`, `c_offer = 3 MU`, `h = F = 0`, `B = 250 MU`.
Break-even is an absolute reduction of `5/250 = 0.02` (two percentage points).
At the study's illustrative `0.30` versus `0.27` planning rates, `delta = 0.03`
and assumed net is `1,000 * 0.03 * 250 - 1,000 * 5 = 2,500 MU`.
The zero harm/fixed-cost assumptions are optimistic simplifications, not findings.
The historical queue's event rate is not substituted for the trial baseline.

### Sensitivity to effect, cost and value

For that independent 1,000-review/1,000-customer scenario with `B = 250 MU`,
zero fixed/harm costs and total variable cost per customer as shown:

| Assumed absolute reduction | Net at 5 MU total variable cost | Net at 10 MU total variable cost |
| --- | ---: | ---: |
| 0 percentage points | -5,000 MU | -10,000 MU |
| 1 percentage point | -2,500 MU | -7,500 MU |
| 2 percentage points | 0 MU | -5,000 MU |
| 3 percentage points | 2,500 MU | -2,500 MU |
| 5 percentage points | 7,500 MU | 2,500 MU |

Break-even also depends strongly on the assumed value of avoiding an event:

| Total variable cost (`K = M`, no fixed/harm cost) | B = 100 MU | B = 250 MU | B = 500 MU |
| --- | ---: | ---: | ---: |
| 5 MU per customer | 5 percentage points | 2 percentage points | 1 percentage point |
| 10 MU per customer | 10 percentage points | 4 percentage points | 2 percentage points |

Capacity does not simply scale profit: at `K = 1,000`, `M = 500` and the same
2 MU review / 3 MU offer costs, effective cost per offered customer becomes
`(1,000*2 + 500*3)/500 = 7 MU`; break-even at `B = 250 MU` becomes 2.8 percentage
points. Expanding review may also change eligibility, baseline risk, treatment
response, staffing costs and harms. Those relationships are unknown. None of
these scenarios justifies changing capacity or approving customer contact.

## Findings to action

| Evidence or observation | Qualified recommendation | Additional evidence / owner |
| --- | --- | --- |
| Historical ranking concentrates defaults at fixed capacity | In a future approved pilot, use the queue as a starting point for human review, with usual support preserved and no adverse action. | Current-population and eligibility validation; policy owner. |
| Material non-default workload and missed risk remain | Budget reviews, examine exclusions and opt-outs, and measure staff effort rather than promising prevented defaults. | Prospective review times, customer harms and coverage; operations owner. |
| Education selection-rate triggers and unsupported small subgroups | Retain G3 restrictions and investigate access, calibration and proxy pathways before real use. | Representative supported subgroup evidence and human risk review. |
| Feature/prediction drift or unexplained review-volume change | Investigate data lineage, population and eligibility; retain completed scores and record disposition. | [Monitoring/runbooks](../operations/release-b-runbooks.md); service and model owners. |
| Missing or immature outcomes | Report coverage and unresolved labels; withhold definitive performance/effectiveness claims. | [Delayed-label contract](../operations/delayed-label-contract.md), data owner. |
| Schema failure, corrupt artifact or inference incident | Reject unsafe input/state and use the relevant containment/recovery runbook. | Authenticated state, synthetic parity and documented resume decision. |

## How actual impact could be established

The existing [randomized protocol](intervention-study.md) compares voluntary
outreach plus usual support with usual support among eligible queued customers.
Customer-level assignment, one enrollment, concealed allocation, original-arm
analysis, outcome maturity, missingness bounds and harm stopping rules address
causal interpretation. The [calculator](planning-calculator.md) already provides
approximate sample-size/MDE planning; no second calculator is needed.

That trial could estimate the causal effect of the **offer within its recruitment
pool**. It would not establish that risk ranking is the best allocation strategy:
high risk need not mean high treatment benefit. A separately approved comparison
of allocation policies would be needed for that question. Real economic impact
also requires measured incremental resource use, a defensible event valuation,
uncertainty and safety review. Adoption would depend on those results together
with external validation and G5 review, not on model lift alone.
