# Validation Subgroup Review

This is a validation-only diagnostic, not a fairness certification, India compliance review,
or proof of production suitability. Demographics were excluded from the estimator and retained
only for audit.

The q90 threshold represents 10% review capacity. Supported groups required at least 100 rows,
25 defaults, and 25 non-defaults; smaller groups report counts only. Prevalence uses a two-sided
95% Wilson score interval. The remaining measures use 500 seed-42 within-group stratified
bootstrap resamples and percentile 95% intervals.

## Predeclared review triggers

- `education_code=1`: selection_rate_ratio=0.696014 triggered below_lower_bound.
- `education_code=3`: selection_rate_ratio=1.256410 triggered above_upper_bound.

The thresholds were frozen before the planning preview exposed these results and were not changed
after the two triggers became visible.

## Group evidence

| Axis | Group | Support | Rows | Prevalence | Mean probability | Calibration gap | Brier | Selection | TPR | FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| age_band | 18_29 | reviewed | 1565 | 0.2358 | 0.2414 | 0.0056 | 0.1377 | 0.1144 | 0.3442 | 0.0435 |
| age_band | 30_39 | reviewed | 1792 | 0.2015 | 0.2029 | 0.0015 | 0.1292 | 0.0843 | 0.2964 | 0.0307 |
| age_band | 40_49 | reviewed | 1014 | 0.2288 | 0.2185 | -0.0103 | 0.1276 | 0.1016 | 0.3448 | 0.0294 |
| age_band | 50_59 | reviewed | 373 | 0.2413 | 0.2383 | -0.0030 | 0.1485 | 0.1099 | 0.2889 | 0.0530 |
| age_band | 60_100 | insufficient | 56 | — | — | — | — | — | — | — |
| education_code | 1 | reviewed | 1681 | 0.1957 | 0.1965 | 0.0008 | 0.1238 | 0.0696 | 0.2553 | 0.0244 |
| education_code | 2 | reviewed | 2253 | 0.2437 | 0.2352 | -0.0084 | 0.1405 | 0.1167 | 0.3534 | 0.0405 |
| education_code | 3 | reviewed | 780 | 0.2321 | 0.2467 | 0.0147 | 0.1424 | 0.1256 | 0.3481 | 0.0584 |
| education_code | 4 | insufficient | 17 | — | — | — | — | — | — | — |
| education_code | undocumented_0_5_6 | insufficient | 69 | — | — | — | — | — | — | — |
| marital_status_code | 1 | reviewed | 2193 | 0.2280 | 0.2206 | -0.0074 | 0.1341 | 0.0976 | 0.3200 | 0.0319 |
| marital_status_code | 2 | reviewed | 2550 | 0.2141 | 0.2224 | 0.0083 | 0.1326 | 0.1016 | 0.3223 | 0.0414 |
| marital_status_code | 3 | insufficient | 46 | — | — | — | — | — | — | — |
| marital_status_code | undocumented_0 | insufficient | 11 | — | — | — | — | — | — | — |
| sex_code | 1 | reviewed | 1902 | 0.2387 | 0.2341 | -0.0045 | 0.1397 | 0.1130 | 0.3370 | 0.0428 |
| sex_code | 2 | reviewed | 2898 | 0.2098 | 0.2139 | 0.0041 | 0.1295 | 0.0914 | 0.3092 | 0.0336 |

Both triggers require human review. Education remains audit-only, and the model is restricted to
human-owned outreach prioritisation. No automatic rejection follows from a trigger.
