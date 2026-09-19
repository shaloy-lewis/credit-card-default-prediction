# Delivery progress

This log records completed checkpoints, verification evidence, and consciously
deferred risks. It is not a substitute for commit history or CI results.

## Phase 0 — approved

**Completed:** 2026-08-07

- Product and decision contract accepted.
- Batch-first architecture accepted.
- Twelve-week, zero-cost local delivery plan accepted.
- MLOps and model governance selected as the principal seniority signals.

## Week 1 — engineering foundation complete

**Completed:** 2026-08-08

### Delivered

- Migrated the import package from the ambiguous `src` name to a true
  `src/credit_risk` layout.
- Converted training from an import-time side effect into an explicit workflow.
- Added a cross-platform `credit-risk` CLI with `version`, `doctor`, and `train`
  commands.
- Replaced empty/duplicate packaging files and hand-maintained dependency lists
  with PEP 621 metadata and a committed `uv.lock`.
- Preserved the legacy binary stack by pinning CatBoost, scikit-learn, NumPy,
  pandas, Pydantic, Typer, and Click compatibility versions.
- Added Ruff, mypy, pytest, coverage, pre-commit, and GitHub Actions gates.
- Added unit and integration tests for preprocessing, CLI behaviour, API health,
  artifact loading, input-column order, and the frozen legacy prediction.
- Added one shared trusted-artifact readiness contract for the model,
  preprocessor, threshold schema, binary classes, transformed feature names,
  and feature-importance dimensions.
- Made FastAPI load one validated pipeline during startup, expose `/ping` for
  process liveness and `/ready` for inference readiness, and reuse that pipeline
  for every prediction.
- Derived the package version from installed distribution metadata and restricted
  the runtime image to the three approved inference artifacts.
- Rebuilt the API image from the lockfile using a multi-stage Dockerfile and a
  non-root runtime user.
- Reframed the README and Streamlit problem statement to match the approved
  early-warning scope and evidence limitations.

### Verification evidence

| Check | Result |
| --- | --- |
| Ruff lint | Passed |
| Ruff formatting | 28 maintained Python files formatted |
| Mypy | Passed for artifact validation, CLI, FastAPI, and Streamlit boundaries |
| Pytest | 26 passed |
| Legacy probability | Preserved at `0.44088` for the documented request |
| Coverage baseline | 52% overall; 88% for artifact validation and 87% for legacy inference; no vanity gate set |
| CLI doctor | Loaded the committed trusted artifacts and passed the full readiness contract |
| Docker build | Baseline image previously passed from the cross-platform frozen lockfile |
| Container contract | Passed in CI and in the local regression: non-root execution, the three-file allowlist, `/ping`, `/ready`, and `/predict` |
| Docker Compose | Configuration validated; health check targets `/ready` |

### Deferred risks

- The committed pickle artifacts remain a temporary compatibility mechanism.
  Model registry packaging and promotion replace manual artifact loading later.
- Instance SHAP values still use the legacy raw/transformed feature mapping.
  The API marks this compatibility behaviour explicitly; governance Week 6 adds
  dimensional, naming, additivity, aggregation, and reason-category tests.
- The current API input model still has weak domain validation. Versioned schemas
  are introduced with the inference contract.
- The existing modelling components remain lightly typed and under-tested. Tests
  expand when the reproducible data and scientific pipelines replace them.
- The local API image is approximately 986 MB. Image composition and dependency
  reduction will be revisited after the runtime boundary stabilises.
- Starlette currently emits an upstream `python-multipart` pending-deprecation
  warning during test import; it does not affect the tested endpoints.

## Phase 1 / Week 2 — reproducible data and G1 complete

**Completed:** 2026-08-18

### Delivered

- Pinned UCI dataset 350 to the official normalized CSV by URL, byte size,
  SHA-256, ordered source schema, row count, and target distribution.
- Added retry-bounded streaming acquisition, immutable content-addressed raw
  storage, atomic no-overwrite publication, offline reuse, and hash-addressed
  quarantine for corrupt or conflicting bytes.
- Added semantic canonical names and a strict Pandera-backed data contract. The
  pipeline rejects structural, type, identifier, domain, and class-count drift
  while preserving and reporting documented source anomalies.
- Added deterministic canonical CSV and quality-report generation with
  transactional promotion and stable, sampled failure evidence.
- Sealed an 80/20 stratified development/test holdout and 5-fold × 3-repeat
  development-only cross-validation protocol at seed 42.
- Added deterministic per-account split assignments plus a committed reviewed
  lock tying the source, canonical table, split configuration, scikit-learn
  version, counts, and assignment digest together without timestamps.
- Added `credit-risk data fetch`, `build`, and strictly offline `verify`
  interfaces. The legacy `credit-risk train` path remains compatibility-only.
- Added the dataset card, feature-availability/leakage review, validation and
  quarantine policy, and clean-checkout reproduction instructions.
- Kept Pandera in the optional `data` dependency boundary; the inference image
  includes the data CLI package but not Pandera or generated data.

### Verification evidence

| Check | Result |
| --- | --- |
| Toolchain | Python 3.12 with `uv 0.11.28`; frozen lock passes `uv lock --check` |
| Official source | 2,897,080 bytes; SHA-256 `45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0` |
| Canonical data | 30,000 validated rows; SHA-256 `75b2a746781a584b0456f843f1f269190b51e90983cba44c4ed6c4a8685e6c1c` |
| Split assignments | 24,000 development and 6,000 test rows; SHA-256 `2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e` |
| Offline verification | Passed against the reviewed split lock with no network access |
| Full tests | 183 passed; one documented upstream Starlette warning |
| Phase 1 branch coverage | 157 tests passed; 97.19% for `credit_risk.data` against a 90% CI gate |
| Static gates | Ruff format/lint, mypy, pre-commit, and whitespace checks passed |
| Inference compatibility | Artifact doctor passed and documented prediction probability remained `0.44088` |
| Container contract | Compose, non-root user, source-package inclusion, optional-extra isolation, three-artifact allowlist, liveness, readiness, and prediction checks passed |

### Accepted limitations and deferred work

- The source is a static 2005 Taiwan sample with no event timestamps, India
  validation, or defensible out-of-time split. It supports engineering and
  governance demonstrations, not contemporary portfolio-performance claims.
- Demographic columns are retained for audit only and excluded from the released
  model. ADR 0002 superseded the earlier ablation plan with no-training
  exclusion/invariance evidence and subgroup review.
- Generated data remains intentionally Git-ignored; a clean checkout must fetch
  the exact pinned public bytes before building, then can verify fully offline.
- Legacy training and committed pickle artifacts are not connected to the new
  canonical data boundary. Replacing that compatibility path belongs to the
  modelling and registry phases.

## Phase 2 / Week 3 — scientific baselines complete

**Completed:** 2026-08-26

### Delivered

- Phase 2 is limited to the Week 3 baseline-and-tracking checkpoint. G2 remains
  open through candidate modelling, calibration, uncertainty, and stress tests.
- The governed feature boundary, three-baseline protocol, metric hierarchy, and
  local SQLite MLflow design are locked before the first experiment.
- The development-only workflow, exact OOF coverage checks, machine-readable
  metric priorities, content-validated MLflow artifacts, and fold-level
  logistic diagnostics are implemented.
- Three clean executions from reviewed commit `c695c60` produced byte-identical
  summaries, reports, OOF predictions, and diagnostics. The 216,000 OOF rows have SHA-256
  `c8ec30bec3c323ed0cfbe050aa3313ac356eb5d717ab305dee1b4365a0e51abe`
  and 15 logistic fold records at SHA-256
  `9a6c0ebe027fe00eda305d319bdf4dd1c7dfc84e470f3a8a7e00cf387ffda425`.
- Development-only repeat means were: logistic average precision `0.541294`,
  ROC-AUC `0.767968`, lift at 10% `3.156903`, and Brier score `0.136362`;
  the fixed repayment rule achieved average precision `0.473102` and lift at
  10% `2.909619`. These are development-only baseline evidence, not holdout
  results or promotion thresholds.

### Verification evidence

| Check | Result |
| --- | --- |
| Clean lineage | `git_dirty=false` at reviewed commit `c695c60` |
| Summary | SHA-256 `11e0332fc9df6f7abf36080a8d09304b3e975f34ad060f70f8611f4fc0ad69d6` |
| Markdown report | SHA-256 `2830b4080f954e773dfdf0c37ed6eaabeaa31917f32071c783ec36abafb63a10` |
| Evaluation boundary | 24,000 development rows; 5 folds × 3 repeats; holdout unevaluated |
| Runtime artifacts | OOF and diagnostic hashes matched the two earlier provisional executions |
| Integrity protection | Complete summary and report digests plus readable semantic assertions |

### Accepted limitations and deferred work

- Repeated-CV variation is descriptive and is not an independence-based confidence interval.
- No candidate, calibrator, operating policy, or test result is approved by this checkpoint.
- Runtime MLflow state, OOF rows, and diagnostics remain ignored; only deterministic aggregate
  evidence is committed.
- No test-partition metric is permitted until the candidate, calibrator, and policy rules are
  frozen.

## Phase 3 / Week 4 — governed candidate modelling complete

**Completed:** 2026-08-27

### Delivered

- The frozen CatBoost protocol evaluated exactly eight full operational-view
  configurations and two diagnostic feature-family ablations on the reviewed
  5-fold × 3-repeat development assignments.
- The compute amendment was made from runtime-only evidence before candidate
  metrics were observed. Atomic, content-bound checkpoints made both independent
  executions safely resumable without changing the protocol order or evidence.
- Both executions completed the 150-fit ceiling and produced byte-identical
  summary, report, 720,000 OOF rows, and 150 fold diagnostics. No third fit pass
  was performed for publication.
- Six full-view configurations were within the frozen equivalence band.
  Deterministic tie-breaking selected `cb_cfg_006`, the least-complex eligible
  option: depth 4, 300 iterations, learning rate `0.03`, L2 regularisation `12`,
  random strength `0`, and bagging temperature `0`.
- `cb_cfg_006` passed all four advancement conditions with average precision
  `0.556419`, AP repeat standard deviation `0.000821`, Brier score `0.134101`,
  and lift at 10% `3.202110`. CatBoost advances to Phase 4.

### Verification evidence

| Check | Result |
| --- | --- |
| Clean lineage | `git_dirty=false` at implementation commit `2b46d4c` |
| Candidate contract | SHA-256 `4bd9a404064d410e0339e0638464aaf6c1ac0bca632156a47af14a822d7cb5f3` |
| Summary | SHA-256 `55aaa971417bddbcad00b8bdf388f74baa13f6ad96304dd108227e48de23ea83` |
| Markdown report | SHA-256 `156967cfda68ddf6c49e4f1e1666266d69261c58628820df3a2a821e560b17c2` |
| Runtime artifacts | OOF SHA-256 `94ee8a56...fd46`; diagnostics SHA-256 `a3b5a2e6...2174`, identical across independent roots |
| Evaluation boundary | 24,000 development rows; holdout unfitted, unscored, and unevaluated |
| Published artifacts | Aggregate JSON and Markdown only; no estimator or row-level evidence committed |

### Historical Phase 3 handoff (superseded)

- Phase 4 must reuse only `cb_cfg_006`; it must not repeat the eight-variant
  search or reinterpret the diagnostic feature views as advancement candidates.
- At this historical checkpoint, calibration, bootstrap uncertainty,
  capacity-based operating-policy selection, and sealed-holdout evaluation were
  pending and G2 was open. The later one-pass release workflow below superseded
  this handoff and closed G2.
- Demographic exclusion/invariance, subgroup analysis, explanations, and the
  final feature-use decision were subsequently completed in Week 6.
- The selected estimator is not connected to the compatibility `/predict`
  endpoint and no fitted CatBoost artifact is committed by this checkpoint.

## Simplified release workflow — selection and final test complete

- Selection completed: 2026-09-02
- Final test and serving migration completed: 2026-09-10

- Historical Phase 2/3 evidence remains immutable, but the expensive baseline,
  candidate, candidate-evidence, and legacy train commands are retired.
- The authoritative selection budget is four fits: one fixed logistic, random
  forest, histogram gradient boosting, and historical `cb_cfg_006` CatBoost.
- Existing `cv_fold_r0` assignments created 19,200 training and 4,800 validation
  rows. Test data remained isolated throughout selection.
- Selection uses validation average precision, Brier/lift guardrails, a fixed
  0.002 equivalence band, and a simplicity tie-break. The winner is never refit.
- Identity-calibration diagnostics, bootstrap intervals, and risk bands operate
  only on stored validation predictions. The selected bundle is digest protected.
- Final-test gates were frozen separately from clean evidence commit `d334b88`
  without loading data or the model. Approval commit `d001d21` then pinned the
  workflow and selected-bundle digests and authorized exactly one evaluation.
- The official clean run at implementation commit `f7c99f2` completed four fits
  and selected `catboost_fixed` without refit. Validation average precision was
  `0.556510`, Brier score `0.133539`, and lift at 10% `3.210923`.
- Reviewed file digests are: summary `8c11b1d4...efbd7`, report
  `16c8748e...cee1`, bundle manifest `df5ce6ce...cd88`, and native CBM
  `844ec1c3...d88c`. Runtime validation predictions and bootstrap evidence remain ignored.
- The one prediction-only evaluation scored exactly 6,000 unique test accounts
  with zero fits. Average precision `0.542867`, Brier score `0.136304`, and lift
  at 10% `3.089676` passed all frozen gates, so G2 is closed.
- Durable started/completed receipts prevent reevaluation. The aggregate summary
  and report are committed; row-level test predictions remain ignored.
- The API and Streamlit demo now use the unchanged `selected_v1` native CatBoost
  bundle and the 19-feature operational schema. The pinned synthetic API example
  returns probability `0.190382` and risk band `standard`.

## Phase 4 release hardening — complete

- Release hardening completed: 2026-09-11
- The approved evaluator source is preserved byte-for-byte as a non-importable
  text artifact with SHA-256 `13ff9d3b...ab2aaa`, matching the immutable approval.
- The active workflow and no-option `credit-risk model final-test` command now
  reject every invocation before data, model, prediction, or caller-path access.
  Fresh output or runtime roots can no longer create a reevaluation path.
- Serving startup requires exact manifest agreement for `catboost`, `joblib`,
  `numpy`, `pandas`, `pydantic`, and `scikit-learn` before model deserialization.
  `mlflow` and `pandera` remain intentionally absent from the runtime readiness set.
- Authorization, approval, final-test evidence, model bundle, and started/completed
  receipts remain immutable. The one evaluation is permanently consumed and G2
  remains closed.
- At the Phase 4 checkpoint, explanations, subgroup review, registry, monitoring,
  rollback, and incident work had not started. Phase 5 subsequently completed
  explanations and subgroup review; the later lifecycle controls remain open.

## Phase 5 governance and explanation review — complete with conditions

- ADR 0002 replaces demographic model ablation with demographic exclusion,
  input-invariance tests, and validation-only subgroup analysis. No further
  model or calibrator fitting is authorized.
- Phase 1 full-file verification may parse the complete canonical snapshot for
  integrity. The modelling boundary returns exactly 24,000 development accounts;
  test accounts cannot be selected, returned, scored, explained, or audited by subgroup.
- Native CatBoost SHAP additivity, reviewed reason categories, group support
  rules, Wilson prevalence intervals, 500-resample performance uncertainty, and
  human-review triggers are fixed before corrected evidence publication.
- A non-published planning preview occurred only after the trigger thresholds
  were selected. The thresholds remain unchanged and the expected education
  code 1/code 3 selection-rate triggers require documented disposition.
- The superseded aggregate evidence was withdrawn after review identified an
  ambiguous test-access claim and degenerate prevalence intervals. It remains
  available in Git history and is not presented as active evidence.
- The clean prediction-only authenticated build from implementation commit `226b7d7`
  reproduced validation AP `0.556510`, Brier score `0.133539`, and lift at 10%
  `3.210923` without fitting or final-test scoring.
- Exactly 1,000 deterministic explanation rows passed native-SHAP raw additivity
  and sigmoid/probability parity at the frozen tolerance.
- Supported-group prevalence uses non-degenerate Wilson intervals; the other
  measures retain the frozen 500-resample within-group stratified bootstrap.
- The two predeclared education selection-rate triggers received documented
  human-review conditions. G3 is `closed_with_conditions`, not a fairness,
  regulatory, production, or India-validity certification.
- The local API is technical portfolio integration, not evidence of external
  governance approval. Phase 6 later completed inference parity; registry,
  monitoring, and rollback remain subsequent work.

## Release A audit closure — complete

**Completed:** 2026-09-12

- ADR 0003 fixes the Release A boundary to the milestone's five existing
  defensible-model criteria and assigns robustness/population-shift stress
  evidence to G4/Release B without claiming it was already completed.
- A frozen release contract binds the reviewed source manifest, split lock,
  feature contract, baseline and selection evidence, selected bundle, final-test
  authorization, approval, durable receipts, executed evaluator, and final evidence.
- The clean build from implementation commit `20186ad` copied the reviewed
  500-resample validation uncertainty byte-for-byte and assembled the dossier
  without model loading, prediction, fitting, bootstrap generation, test-row
  selection, or final-test reevaluation.
- The dossier reports validation AP `0.556510` with 95% interval
  `[0.525431, 0.587755]`, Brier `0.133539` with interval
  `[0.128826, 0.137924]`, and lift at 10% `3.210923` with interval
  `[3.027072, 3.375942]`.
- It also binds ten reliability bins, validation and final-test capacities at
  5%, 10%, and 20%, the four-fit/no-refit selection rule, and the one permanently
  consumed final-test evaluation that closed G2.
- Complete artifact digests are summary `a8cfdd1f...19acb`, report
  `7d5873bf...73a86`, uncertainty `187004bd...0ffa2`, and manifest
  `7e65c7b8...4edf7`. The last digest is the external verification trust anchor.
- Release A is `complete`; G1 and G2 remain closed and G3 remains
  `closed_with_conditions`. Phase 6 later completed parity and Phase 7 completed
  the registry/rollback slice, while G4 stress, monitoring, and incident work remains open.

## Phase 6 idempotent batch and versioned API parity — complete

**Completed:** 2026-09-16

- ADR 0004 and the digest-protected `phase6_v1` configuration froze the
  unchanged `selected_v1` bundle, ordered 19-feature contract, risk thresholds,
  four explanation categories, 10% capacity policy, idempotency keys, v1 API,
  and safe-log allowlist before implementation.
- Package version `0.2.0` marks the breaking replacement of `POST /predict`
  with `POST /v1/predict`; liveness and readiness contracts remain stable.
- One vectorised inference engine performs digest/dependency validation,
  prediction, native-SHAP category aggregation, additivity checks, sigmoid
  parity, risk bands, and deterministic reason ordering for both batch and API.
- Monthly batch inference supports deterministic ranking, exact floor-based
  capacity, safe partial-row rejection evidence, atomic publication, conflict
  refusal, and verified identical-run reuse without rewriting files. Pre-push
  review hardened wrong-width row handling, reserved snapshot paths, strict
  manifest parsing, complete CSV reconciliation, and output allowlisting.
- Streamlit is an API client and no longer loads the selected bundle. Structured
  JSON logs expose allowlisted operational metadata without features, account
  IDs, probabilities, contributions, demographics, targets, local paths, or
  client-visible exception details.
- Final pre-push review added failure trace headers, pre-model snapshot preflight,
  deterministic reason-order verification, and strict API-client response validation.
- The official clean prediction-only evidence build from corrected implementation
  commit `f6b37af` used 20 synthetic records. Offline and batch probabilities matched
  exactly; the maximum API rounding difference was
  `4.2811987377433525e-07`, below `5e-7`; risk bands and both reason categories
  and directions matched exactly; two rows were selected under the 10% policy.
- Published digests are summary `3f5e9744...f6afa`, report
  `6132a645...9ee50`, and external manifest trust anchor
  `91908722...f4df8`. Row-level batch files and logs remain ignored.
- No model fitting, tuning, calibration fitting, model/policy change, final-test
  loading, or sealed-test scoring occurred. Phase 7 later completed registry,
  scanning, and rollback; G4 remains open for robustness, monitoring, and incidents.

## Phase 7 governed registry, deployment, and rollback — complete

**Completed:** 2026-09-17

- ADR 0005 and digest-protected `phase7_v1` froze the local SQLite registry,
  content-addressed artifacts, manual approvals, alias transitions, deployment
  layout, scan policy, and no-training/no-test boundaries before implementation.
- Pre-publication review identified that artifact identity and alias receipts did
  not explicitly capture the planned deployment smoke parity. The amended clean
  implementation `cb63b39` pins the existing synthetic fixture and requires both
  immutable revisions to produce the same full-precision output digest.
- Package version `0.3.0` marks the release-control layer. The API runtime remains
  independent of MLflow and its `/ping`, `/ready`, and `/v1/predict` contracts are unchanged.
- Two MLflow versions transparently represent deployment revisions of identical
  `selected_v1` bytes. Registration, promotion, deployment, and rollback validate
  full state, use a single-writer lock, publish deterministic receipts, and
  compensate prior aliases and pointers when a transition fails.
- The official drill promoted `phase7_rev_002`, restored `phase7_rev_001` through
  the approved rollback, and independently loaded both bundles. Each returned
  probability `0.190382`, risk band `standard`, and the same prediction-and-reason digest.
- GitHub Actions passed quality and container jobs for the implementation and
  approval commits. The container job pins build inputs, blocks fixable
  HIGH/CRITICAL Trivy findings without a repository waiver, and uploads a CycloneDX SBOM.
- Published digests are summary `f88324b6...1865`, report `d918d773...bb26`,
  promotion checklist `8af86710...e407`, rollback runbook `37ae777c...0f2c`,
  and external evidence-manifest trust anchor `ce36f33d...7da9`.
- The drill performed zero fits, did not change model bytes, did not access the
  sealed test, and published no runtime paths, timestamps, row-level data, or
  MLflow state. It is release-control evidence, not a model-quality comparison
  or external production-readiness claim.
- Phase 7's registry/rollback slice is complete. G4 and Release B remain open for
  robustness/population-shift stress tests, monitoring, and incident controls;
  PostgreSQL, MinIO, and persistent platform services are now available as the
  Phase 8 prerequisite layer described below.

## Phase 8 persistent local platform — prerequisites complete, evidence pending

**Prepared:** 2026-09-17

- ADR 0006 and digest-protected `phase8_v1` bind the exact Phase 7 evidence,
  selected bundle, immutable service images, persistent volumes, approved aliases,
  security boundaries, and explicit zero-training/no-test constraints.
- Package version `0.4.0` adds a separately locked `platform` extra for MLflow,
  PostgreSQL, and S3-compatible object-store clients; the API image still contains
  no MLflow dependency.
- The Compose stack runs PostgreSQL, MinIO, MLflow, the existing API, and an
  API-only Streamlit UI. PostgreSQL and the MinIO object API are not published to
  the host; the deployment volume is read-only in the non-root API container.
- The one-shot bootstrap copied the exact two-file selected bundle to a
  content-addressed MinIO prefix, created two transparent MLflow release versions,
  restored `champion=1` and `rollback=2`, and materialised the approved revision-1
  deployment pointer. Re-running bootstrap verified the existing state without
  creating conflicting versions.
- A live local restart preserved PostgreSQL metadata, MinIO bytes, registry
  aliases, the deployment pointer, and prediction `0.190382`. All five services
  returned healthy status after recovery.
- Unit and static integration tests enforce the frozen config, source digests,
  path and object allowlists, failure behaviour, internal-only ports, read-only
  deployment mount, ignored runtime secrets, and a `>=90%` platform branch gate.
- Pre-commit review hardened this prerequisite with controlled missing-path
  failures, exact registered-model tag verification, and a fail-fast PostgreSQL
  advisory lock spanning every bootstrap mutation and final verification.
- Final pre-push review normalized every MLflow and deployment-service failure,
  made explicit environment mappings authoritative, rejected symlinks across
  the complete deployment path (including cleanup), and moved the percent-encoded
  PostgreSQL URI from process arguments into the MLflow process environment.
- Remote blocking scans found the same fixable Debian PCRE2 findings in both
  platform runtime images. The MLflow and Streamlit images now install the exact
  fixed Debian package while retaining the pinned base image and scan policy.
- After inspecting their Compose labels, the three disposable rehearsal volumes
  were irreversibly removed and recreated under the corrected contract. The
  rebuilt state reproduced two versions, `champion=1`, `rollback=2`, exact object
  hashes, restart persistence, and probability `0.190382`; no historical artifact
  or non-Phase-8 volume was altered.
- GitHub Actions now validates the Compose contract, builds and restarts the
  stack, verifies the persistent state, checks prediction parity, blocks fixable
  HIGH/CRITICAL findings for the new platform/UI images, and publishes their
  CycloneDX SBOMs.
- The first remote platform scan correctly blocked three fixable PCRE2 findings
  plus fixed-version GitPython and cryptography findings. The remediation retains
  the frozen base digest and blocking policy while installing Debian's fixed
  PCRE2 package and locking the two Python packages at their published fixes.
- This is prerequisite and runtime verification, not the reviewed official Phase
  8 evidence package. No model fitting, refitting, tuning, final-test access, or
  sealed-test scoring occurred. Phase 8 remains in progress until aggregate
  evidence is published and reviewed.

## External artifact distribution migration

**Prepared:** 2026-09-18

- Package version `0.5.0` adds an optional, pinned artifact client and explicit
  `artifacts pull`, `verify`, and maintainer-only `publish` commands.
- The exact selected CatBoost model and two legacy compatibility pickles were
  published to public repository `ShaloyL/credit-card-default-prediction` at
  immutable revision `f73ca4ee7a2c2d2ea51741e75fccf66ae7a4a640`.
- Anonymous downloads matched the existing selected manifest and the new legacy
  trust manifest before Git stopped tracking the three binary paths.
- Selected-model loaders, API, batch, governance, Release A, registry, deployment,
  and platform paths remain local-only and continue using the historical path and
  unchanged SHA-256. Legacy loading authenticates all bytes before `pickle.load`.
- Docker retrieves and verifies only the selected model in an isolated build
  stage. The final API and platform images remain offline and contain neither
  Hugging Face dependencies, cache state, credentials, nor legacy files.
- No model fitting, refitting, calibration, final-test execution, sealed-test
  access, or historical evidence regeneration occurred.
