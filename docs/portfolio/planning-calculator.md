# Standalone sample-size and MDE calculator

**Claim class: hypothetical planning estimates.** This tool supports the
[proposed outreach study](intervention-study.md); it does not evaluate the model,
load data, contact customers or estimate an observed treatment effect.

## Method and scope

Two independent binary arms have equal allocation. Set baseline event rate `p0`,
absolute reduction `d`, treatment rate `p1 = p0 - d`, and `pbar = (p0 + p1) / 2`.
The continuous analysable sample per arm is:

```text
n = [ z(1 - alpha/2) * sqrt(2*pbar*(1-pbar))
      + z(power) * sqrt(p0*(1-p0) + p1*(1-p1)) ]^2 / d^2
```

This is the normal approximation corresponding to the two-sided
[R stats power.prop.test](https://stat.ethz.ch/R-manual/R-patched/library/stats/html/power.prop.test.html)
with `strict=FALSE`, without continuity correction. It counts rejection in the
anticipated direction using a two-sided critical value, omitting the opposite
tail from its power approximation. It is not an exact finite-sample test.

Round analysable sample size upward (minimum two per arm), then divide by
`1 - attrition` and round recruitment upward. For MDE, `n-per-arm` means recruited
customers: use `floor(n-per-arm * (1-attrition))` as the analysable count and
invert the same equation by bounded bisection over reductions `(0, p0]`.
Refuse a request if even a zero treatment event rate cannot achieve target power.

All inputs are fractions, not percentages: `0.03` means three percentage points.
Alpha defaults to 0.05, power to 0.80, attrition to zero. Baseline/alpha must lie
strictly inside `(0, 1)`, power inside `(0.5, 1)`, attrition inside `[0, 1)`, and
reduction inside `(0, baseline]`. Participant counts must be integers from 2
through `2**53 - 1`. Non-finite values and numerically unrepresentable requests
are rejected. Expected event or non-event cells below 10 trigger an approximation
warning. Zero treatment rate is a boundary case and always carries that warning.

Independent customer outcomes and equal allocation are assumptions. Clustered
or unequal-arm designs, repeated measures, multiple testing, adaptive stopping,
non-adherence adjustments and economic benefit calculations are unsupported.
Attrition inflation is a recruitment allowance, not a missing-data correction.
The study's primary endpoint differs from the historical model label.

## Run it

Only Python's standard library is required; Python 3.12 is the repository runtime.
From the repository root:

```sh
python src/credit_risk/portfolio/planning.py sample-size --baseline-rate 0.30 --absolute-reduction 0.03 --alpha 0.05 --power 0.80 --attrition 0.10
python src/credit_risk/portfolio/planning.py mde --baseline-rate 0.30 --n-per-arm 3949 --attrition 0.10 --json
```

The installed module also works:

```sh
uv run python -m credit_risk.portfolio.planning sample-size --baseline-rate 0.30 --absolute-reduction 0.03 --attrition 0.10 --json
```

Both subcommands emit assumptions and effects, continuous required sample size,
analysable/recruited counts per arm, total recruitment, method and warnings.
`--json` emits one deterministic JSON object with
`label: planning_estimate_not_observed_effect`; normal output starts with the
same limitation in plain language. Valid requests exit 0; invalid CLI/numerical
requests exit 2 with an explanation. There are no output files or network calls.

## Illustrative result and independent checks

The **assumed** reduction from 30% to 27% is three percentage points, or a 10%
relative reduction. At alpha 0.05 and power 0.80, the calculation gives
`3553.055208` continuous analysable customers per arm, rounded to **3,554**.
With 10% assumed attrition, recruit **3,949 per arm; 7,898 total**. These rates
are not taken from the dataset or the model's top-10% queue. The selected queue
and exclusions can substantially limit recruitment; no feasible duration or
real-world efficacy is claimed. With those recruited counts, MDE is approximately
three percentage points.

Regression references come from the R documentation, independently of this
implementation. Complementing both event definitions preserves the variance and
sample-size calculation while expressing its examples as reductions:

| Published R example | Equivalent reduction input | Expected result |
| --- | --- | --- |
| `p1=.50, p2=.75, power=.90` | baseline .50, reduction .25 | continuous n approximately 76.7 per arm; recruit 77 without attrition |
| `n=50, p1=.5, power=.90` gives `p2=.8026` | baseline .50, recruited n 50, power .90 | treatment approximately .1974; MDE approximately .3026 |
| `.5` versus `.501`, alpha .001, power .90 | baseline .50, reduction .001 | continuous n approximately 10,451,937 per arm |

Tests also exercise rounding, invalid and unattainable inputs, expected
sensitivity, sample-size/MDE consistency, and isolated execution without site
packages, repository data/model reads or network access.
