# Governed baseline experiment report

- **Experiment:** `baseline_v1`
- **Evaluation boundary:** sealed development folds only; holdout rows were not exposed to model fitting, scoring, or evaluation
- **Primary metric:** `average_precision`
- **Probability guardrail:** `brier_score`
- **Primary capacity metric:** `lift@0.1`
- **Deterministic summary SHA-256:** `11e0332fc9df6f7abf36080a8d09304b3e975f34ad060f70f8611f4fc0ad69d6`

## Protocol result across complete repeats

The headline values are means of the three complete repeated-CV evaluations. `average_precision` is non-interpolated average precision, not a trapezoidal PR-curve area.

| Baseline | Average precision | ROC-AUC | KS | Gini | Brier score | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fold_prevalence` | 0.221175 | 0.499903 | 0.000000 | -0.000193 | 0.172275 | 0.528433 |
| `repayment_burden_rule` | 0.473102 | 0.735775 | 0.410800 | 0.471551 | n/a | n/a |
| `logistic_l2` | 0.541294 | 0.767968 | 0.409742 | 0.535936 | 0.136362 | 0.437036 |

## Repeat-level variation

These are descriptive results across the three fixed CV repeats; they are not an independence-based confidence interval.

| Baseline | Average precision mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `fold_prevalence` | 0.221175 | 0.000000 | 0.221175 | 0.221175 |
| `repayment_burden_rule` | 0.473102 | 0.000000 | 0.473102 | 0.473102 |
| `logistic_l2` | 0.541294 | 0.000579 | 0.540775 | 0.542102 |

## Capacity evidence

| Baseline | Capacity | Precision | Recall | Lift |
| --- | ---: | ---: | ---: | ---: |
| `fold_prevalence` | 5% | 0.221042 | 0.049962 | 0.999247 |
| `fold_prevalence` | 10% | 0.221042 | 0.099925 | 0.999247 |
| `fold_prevalence` | 20% | 0.221042 | 0.199849 | 0.999247 |
| `repayment_burden_rule` | 5% | 0.705886 | 0.159552 | 3.191046 |
| `repayment_burden_rule` | 10% | 0.643632 | 0.290962 | 2.909619 |
| `repayment_burden_rule` | 20% | 0.557211 | 0.503789 | 2.518944 |
| `logistic_l2` | 5% | 0.750278 | 0.169586 | 3.391725 |
| `logistic_l2` | 10% | 0.698333 | 0.315690 | 3.156903 |
| `logistic_l2` | 20% | 0.560139 | 0.506436 | 2.532178 |

## Reproducibility and governance

- Canonical data SHA-256: `75b2a746781a584b0456f843f1f269190b51e90983cba44c4ed6c4a8685e6c1c`
- Split assignment SHA-256: `2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e`
- Baseline config SHA-256: `1666691fffea7d10debd233ed26114af74737bb8f66e0d442a7f4233d68762e0`
- Feature contract SHA-256: `8978277ae1c92b6f0b8daed94cccf3cd51d8e6cae0aa9c0620d8cfb813384a4b`
- Git commit: `c695c600b5d48263b40c56b81be7b66f1edb9f2f`
- Dirty worktree recorded: `false`
- Git diff SHA-256: `9a241090acae12338c583ec34ecbd94caa9a0e2ca9a4ca9338e275b822a065b2`
- Python version: `3.12.10`
- Dependency lock SHA-256: `e04f7992165bf996ff3d0a600878e573034a7d8f3a6ca64ccad5bce6e9792156`
- No fitted estimator, pickle, raw source, or holdout row is stored in this report or the MLflow evidence artifacts.
- Fold-level logistic convergence and coefficient diagnostics are stored as non-executable JSON runtime evidence.
- The repayment rule is a ranking score, so Brier score and log loss are intentionally not reported for it.
- The machine-readable summary retains pooled OOF metrics as descriptive evidence only; each development account appears once in every repeat, so pooled values are not the protocol-level headline.
- Results describe this published 2005 Taiwan dataset and do not establish causal impact, India-specific performance, or production suitability.
