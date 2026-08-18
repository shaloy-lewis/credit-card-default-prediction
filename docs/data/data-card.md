# Data card — UCI Default of Credit Card Clients

- **Dataset version:** `uci_credit_default_v1`
- **Governance status:** Phase 1 / G1 evidence
- **Dataset owner:** I-Cheng Yeh; distributed by the UCI Machine Learning Repository
- **Portfolio data steward:** Shaloy Lewis

## Purpose and permitted use

This dataset supports an offline portfolio study of next-month default-risk
ranking for existing credit-card accounts. It may be used to compare models,
calibration, capacity-based ranking policies, and governed ML workflows.

The source contains four demographic attributes: `SEX`, `EDUCATION`,
`MARRIAGE`, and `AGE`. They are retained for data-quality review, subgroup audit,
and demographic ablation only. They are not permitted inputs to a promoted
predictive candidate under the current feature policy. `ID` is used for lineage
and split assignment only and is never a predictor.

The committed CatBoost artifacts and `credit-risk train` path predate this data
contract. They remain compatibility demonstrations, not Phase 1 scientific
evidence and not candidates for promotion.

## Source, licence, and lineage

| Item | Pinned value |
| --- | --- |
| UCI dataset | ID 350, [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Dataset DOI | [10.24432/C55S3H](https://doi.org/10.24432/C55S3H) |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Citation | Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H |
| Phase 1 source | `https://archive.ics.uci.edu/static/public/350/data.csv` |
| Source size | 2,897,080 bytes |
| Source SHA-256 | `45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0` |
| Source layout | 30,001 lines including one header; columns `ID`, `X1`–`X23`, `Y` |

UCI's metadata API advertises the normalized CSV above as its `data_url`; it is
the only acquisition source for Phase 1. No mirror or archive is an automatic
fallback.

For original-publication lineage, the dataset page also serves
`https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip`.
It is 5,539,494 bytes with SHA-256
`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`
and contains one uncompressed member, `default of credit card clients.xls`
(5,539,328 bytes; SHA-256
`30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`).
That workbook has one `Data` sheet with two header rows. It is recorded for
provenance only and is not read by the Phase 1 pipeline.

The source contract is versioned in
`configs/data/uci_credit_default_v1.json`. A valid CSV is stored at
`data/raw/uci_credit_default/v1/<source-sha256>/data.csv`. The
content-addressed raw snapshot is Git-ignored, never committed, and never
silently replaced.

### Reviewed Phase 1 fingerprint

The official build was reproduced with the locked Python environment and
verified offline on 2026-08-18. The reviewed machine-readable source of truth is
`configs/data/split_v1.lock.json`; the values are repeated here as portfolio
evidence for this data-card version.

| Evidence | SHA-256 |
| --- | --- |
| Official source CSV | `45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0` |
| Source manifest | `4e6463e3acce879e00435f21b64cc905b11dc5fd894b20a32cddd0f775be6979` |
| Canonical CSV | `75b2a746781a584b0456f843f1f269190b51e90983cba44c4ed6c4a8685e6c1c` |
| Quality report | `99fc5364806561c77740017ea31d4f83b21f718482797fbf222511296097278e` |
| Split configuration | `36d8bb9ae6221a60dafa389a0bdf65796bd726d4a96413f810c002de43842398` |
| Split assignments | `2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e` |
| Reviewed split lock | `b2312380fa46924ca414acbcfef63b0435d1321083e87e4df5ec04f18736093d` |

An ordinary `credit-risk data build` writes runtime lineage to
`data/splits/uci_credit_default/v1/split_manifest.json`. The committed lock is
its byte-exact reviewed copy and changes only through a deliberate lock review.

The reproducible build outputs are:

- `data/processed/uci_credit_default/v1/canonical.csv`;
- `data/processed/uci_credit_default/v1/quality_report.json`;
- `data/splits/uci_credit_default/v1/split_assignments.csv`; and
- `data/splits/uci_credit_default/v1/split_manifest.json`.

## Composition

The published snapshot contains 30,000 rows, 23 candidate source features, one
identifier, and one binary target. UCI reports no missing values.

| Property | Observed value in the pinned source |
| --- | ---: |
| Rows | 30,000 |
| Columns | 25 (`ID`, 23 features, `Y`) |
| Default (`Y=1`) | 6,636 (22.12%) |
| No default (`Y=0`) | 23,364 (77.88%) |
| Unique IDs | 30,000, sequential from 1 to 30,000 |
| Fully duplicated rows | 0 |
| Duplicates after excluding `ID` | 35 |

The feature groups are:

- credit limit (`LIMIT_BAL`);
- demographic attributes (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`);
- six repayment-status fields (`PAY_0`, `PAY_2`–`PAY_6`);
- six bill-statement amounts (`BILL_AMT1`–`BILL_AMT6`); and
- six previous-payment amounts (`PAY_AMT1`–`PAY_AMT6`).

The normalized CSV uses UCI's generic `ID`, `X1`–`X23`, and `Y` header. The
canonical build maps these names explicitly to UCI's descriptive variable
metadata. `PAY_0` is the published name for the most recent repayment status;
there is no `PAY_1`.

## Time and label semantics

UCI describes repayment, bill, and payment history for April through September
2005 and defines `Y=1` as default payment in the following month. The file does
not contain a row-level event timestamp, snapshot date, label timestamp, or
`as_of_date`. Phase 1 therefore does not manufacture one.

The suffixes encode relative recency: repayment status `PAY_0` is September and
`PAY_2`–`PAY_6` run from August back to April; bill and payment suffix `1` is
September and suffix `6` is April. This supports a fixed-snapshot modelling
exercise, not a chronological or genuine out-of-time evaluation.

## Known source anomalies

These values occur in the checksum-pinned source and are preserved and reported,
not silently corrected:

| Field | Published description | Observed anomaly / treatment |
| --- | --- | --- |
| `EDUCATION` | Codes 1–4 | Codes 0 (14 rows), 5 (280), and 6 (51) also occur; retain for audit and require an explicit later recoding policy |
| `MARRIAGE` | Codes 1–3 | Code 0 occurs in 54 rows; retain as undocumented/unknown for audit |
| `PAY_*` | `-1`, then delay codes 1–9 | Undocumented `-2` and `0` occur; observed range is -2 through 8 |
| `BILL_AMT*` | Statement amounts | Negative balances occur and are not treated as corruption |
| Non-ID records | One row per published record | 35 rows duplicate all non-ID values; do not infer that they are the same customer or delete them automatically |

Any later category consolidation is a versioned modelling transformation fitted
or defined from development data. Acquisition and canonicalization preserve the
published values.

## Limitations and foreseeable bias

- The sample represents Taiwanese card clients in 2005. It is not evidence for
  Indian customers, a current lender portfolio, or another geography or period.
- The file is one published snapshot, not repeated account-month data. Row order
  is not time and cannot support a temporal split.
- The label is a published binary outcome; the repository cannot independently
  audit its operational definition, observation process, or selection process.
- The source lacks intervention assignment and outcomes, hardship, fraud,
  dispute, closure, collections-stage, complaint, EAD, LGD, and cost fields.
- Demographic categories are coarse, include undocumented codes, and do not
  establish legally or socially complete protected-class coverage.
- A random holdout estimates performance only within this historical sample.
  It does not establish transportability, production stability, or causal impact.

## Prohibited claims

Results derived from this dataset must not be described as:

- validated for India, a named lender, or a current population;
- genuine out-of-time, longitudinal, or production performance;
- evidence that an intervention prevents default or creates financial benefit;
- proof of compliance with RBI, Basel, DPDP, fair-lending, or another regime;
- a sufficient basis for underwriting, adverse action, limit changes, or
  autonomous collections; or
- evidence of production scale or deployment merely because the workflow runs
  in containers or maps conceptually to Azure Databricks.

## Reproduction and review

From the locked development environment:

```bash
uv run credit-risk data fetch
uv run credit-risk data build
uv run credit-risk data verify
```

`fetch` makes acquisition explicit; `build` also performs the same idempotent
fetch/no-op before validating and canonicalizing the source and producing the
deterministic holdout/fold assignments. `verify` is strictly offline and checks
the raw CSV, canonical outputs, split assignments, runtime manifest, and
reviewed lock. See
[Validation and quarantine policy](validation-policy.md) for failure handling
and [Feature availability and leakage review](feature-availability.md) for the
permitted predictor boundary.

G1 data readiness passes only when a clean checkout can reproduce the validated
canonical data and split assignments from the pinned source, tests reject
corrupt and incompatible inputs, and the runtime manifest plus reviewed lock
record the resulting lineage. Passing G1 authorizes baseline modelling; it does
not approve a model.
