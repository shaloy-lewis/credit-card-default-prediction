# One-time final-test report

- **G2 status:** closed
- **Model:** `catboost_fixed` from bundle `selected_v1`
- **Evaluation boundary:** exactly 6,000 sealed test accounts, scored once
- **Training, refitting, retuning, and cross-validation:** not performed
- **Average precision:** 0.542867
- **Brier score:** 0.136304
- **Lift at 10%:** 3.089676

| Gate | Observed | Frozen requirement | Result |
| --- | ---: | ---: | --- |
| average_precision | 0.542867 | >= 0.526510 | pass |
| brier_score | 0.136304 | <= 0.153539 | pass |
| lift_at_0_1 | 3.089676 | >= 2.910923 | pass |

Row-level test predictions remain ignored; their checksum is retained in the summary.
Summary SHA-256: `8b5e018f5e29a5128285afb877e0adaeca35f4b450061cac21e08ea3a51bda56`
