# Governed CatBoost candidate report

- **Protocol:** `candidate_v1`
- **Evaluation boundary:** 24,000 development rows using the reviewed 5-fold × 3-repeat assignments; the holdout was not fitted, scored, or evaluated
- **Deterministic summary SHA-256:** `55aaa971417bddbcad00b8bdf388f74baa13f6ad96304dd108227e48de23ea83`
- **Completed fold fits:** `150`
- **Official-publication gate:** two independently checkpointed executions must produce byte-identical summary, report, OOF, and diagnostic artifacts; no third fit pass is permitted

## Advancement decision

- **Selected configuration:** `cb_cfg_006`
- **Phase 4 candidate:** `catboost_v1`
- **CatBoost advances:** `true`
- Reduced feature views are diagnostic only and cannot advance.

## Full-view bounded search

| Configuration | AP | AP std | Brier | Lift@10% | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| `cb_cfg_001` | 0.558128 | 0.000915 | 0.133847 | 3.192064 | true |
| `cb_cfg_002` | 0.557956 | 0.000515 | 0.133858 | 3.183902 | true |
| `cb_cfg_003` | 0.558122 | 0.000436 | 0.133949 | 3.180762 | true |
| `cb_cfg_004` | 0.557492 | 0.000735 | 0.133875 | 3.187041 | true |
| `cb_cfg_005` | 0.554165 | 0.000705 | 0.134817 | 3.176995 | true |
| `cb_cfg_006` | 0.556419 | 0.000821 | 0.134101 | 3.202110 | true |
| `cb_cfg_007` | 0.558371 | 0.000660 | 0.133844 | 3.195831 | true |
| `cb_cfg_008` | 0.545612 | 0.001565 | 0.136514 | 3.108558 | false |

## Diagnostic feature-family ablations

| Feature view | Predictors | AP | AP delta vs logistic | Brier | Lift@10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| `repayment_status_only` | 6 | 0.536989 | -0.004305 | 0.136560 | 3.188327 |
| `monetary_only` | 13 | 0.454548 | -0.086745 | 0.149837 | 2.535317 |

## Reference and governance

- Logistic AP reference: `0.541294`
- Logistic Brier reference: `0.136362`
- Logistic lift@10% reference: `3.156903`
- Candidate config SHA-256: `4bd9a404064d410e0339e0638464aaf6c1ac0bca632156a47af14a822d7cb5f3`
- Git commit: `2b46d4c3d0e2c37c7b8ef056244c5870d7b098b6`
- Dirty worktree recorded: `false`
- No fitted estimator, executable model, raw row, MLflow identifier, or holdout result is committed.
- These development-CV results are model-selection evidence, not an unbiased final performance estimate.
- Results describe the published 2005 Taiwan sample and do not establish causal impact, India-specific performance, or production suitability.
- Calibration, uncertainty, policy selection, and one-time holdout evaluation remain Phase 4 work.
