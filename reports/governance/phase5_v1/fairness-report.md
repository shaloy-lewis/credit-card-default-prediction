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

| Axis | Group | Support | Rows | Prevalence [95% CI] | Mean probability [95% CI] | Calibration gap [95% CI] | Brier [95% CI] | Selection [95% CI] | TPR [95% CI] | FPR [95% CI] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| age_band | 18_29 | reviewed | 1565 | 0.2358 [0.2154, 0.2574] | 0.2414 [0.2328, 0.2514] | 0.0056 [-0.0030, 0.0157] | 0.1377 [0.1290, 0.1456] | 0.1144 [0.1003, 0.1304] | 0.3442 [0.2994, 0.3930] | 0.0435 [0.0318, 0.0573] |
| age_band | 30_39 | reviewed | 1792 | 0.2015 [0.1835, 0.2207] | 0.2029 [0.1955, 0.2107] | 0.0015 [-0.0060, 0.0092] | 0.1292 [0.1217, 0.1356] | 0.0843 [0.0725, 0.0965] | 0.2964 [0.2521, 0.3422] | 0.0307 [0.0224, 0.0391] |
| age_band | 40_49 | reviewed | 1014 | 0.2288 [0.2040, 0.2556] | 0.2185 [0.2079, 0.2288] | -0.0103 [-0.0209, -0.0000] | 0.1276 [0.1176, 0.1387] | 0.1016 [0.0833, 0.1183] | 0.3448 [0.2779, 0.4095] | 0.0294 [0.0179, 0.0429] |
| age_band | 50_59 | reviewed | 373 | 0.2413 [0.2006, 0.2872] | 0.2383 [0.2222, 0.2565] | -0.0030 [-0.0191, 0.0152] | 0.1485 [0.1302, 0.1682] | 0.1099 [0.0817, 0.1394] | 0.2889 [0.2000, 0.3778] | 0.0530 [0.0283, 0.0777] |
| age_band | 60_100 | insufficient | 56 | — | — | — | — | — | — | — |
| education_code | 1 | reviewed | 1681 | 0.1957 [0.1775, 0.2154] | 0.1965 [0.1895, 0.2044] | 0.0008 [-0.0063, 0.0087] | 0.1238 [0.1166, 0.1307] | 0.0696 [0.0589, 0.0803] | 0.2553 [0.2112, 0.3070] | 0.0244 [0.0170, 0.0325] |
| education_code | 2 | reviewed | 2253 | 0.2437 [0.2264, 0.2618] | 0.2352 [0.2285, 0.2422] | -0.0084 [-0.0152, -0.0014] | 0.1405 [0.1337, 0.1476] | 0.1167 [0.1050, 0.1287] | 0.3534 [0.3142, 0.3908] | 0.0405 [0.0311, 0.0499] |
| education_code | 3 | reviewed | 780 | 0.2321 [0.2038, 0.2629] | 0.2467 [0.2337, 0.2605] | 0.0147 [0.0017, 0.0284] | 0.1424 [0.1305, 0.1553] | 0.1256 [0.1038, 0.1500] | 0.3481 [0.2762, 0.4254] | 0.0584 [0.0409, 0.0785] |
| education_code | 4 | insufficient | 17 | — | — | — | — | — | — | — |
| education_code | undocumented_0_5_6 | insufficient | 69 | — | — | — | — | — | — | — |
| marital_status_code | 1 | reviewed | 2193 | 0.2280 [0.2109, 0.2460] | 0.2206 [0.2135, 0.2279] | -0.0074 [-0.0145, -0.0001] | 0.1341 [0.1273, 0.1404] | 0.0976 [0.0873, 0.1094] | 0.3200 [0.2840, 0.3640] | 0.0319 [0.0242, 0.0402] |
| marital_status_code | 2 | reviewed | 2550 | 0.2141 [0.1986, 0.2305] | 0.2224 [0.2149, 0.2289] | 0.0083 [0.0008, 0.0148] | 0.1326 [0.1267, 0.1386] | 0.1016 [0.0904, 0.1120] | 0.3223 [0.2838, 0.3608] | 0.0414 [0.0324, 0.0499] |
| marital_status_code | 3 | insufficient | 46 | — | — | — | — | — | — | — |
| marital_status_code | undocumented_0 | insufficient | 11 | — | — | — | — | — | — | — |
| sex_code | 1 | reviewed | 1902 | 0.2387 [0.2201, 0.2584] | 0.2341 [0.2262, 0.2430] | -0.0045 [-0.0125, 0.0043] | 0.1397 [0.1328, 0.1477] | 0.1130 [0.1017, 0.1257] | 0.3370 [0.2974, 0.3811] | 0.0428 [0.0331, 0.0535] |
| sex_code | 2 | reviewed | 2898 | 0.2098 [0.1954, 0.2250] | 0.2139 [0.2086, 0.2196] | 0.0041 [-0.0012, 0.0098] | 0.1295 [0.1234, 0.1350] | 0.0914 [0.0825, 0.1004] | 0.3092 [0.2738, 0.3463] | 0.0336 [0.0273, 0.0406] |

Both triggers require human review. Education remains audit-only, and the model is restricted to
human-owned outreach prioritisation. No automatic rejection follows from a trigger.
