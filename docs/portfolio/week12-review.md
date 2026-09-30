# Week 12 rehearsal and review record

**Status: partial implementation; Release C remains planned.**
The fresh-checkout functional rehearsal and interview material are prepared.
Window-capture preflight failed; no final MP4, visual UI review, complete video
privacy audit or owner acceptance is recorded. This is not a completed Release C
package and must not be presented for final owner sign-off yet.

## Baseline, source and scope

Week 11 merged through [PR #19](https://github.com/shaloy-lewis/credit-card-default-prediction/pull/19)
at `f8a5bb98a6443af4714c84183729adfac63c9057`. Its
[four main CI jobs](https://github.com/shaloy-lewis/credit-card-default-prediction/actions/runs/36693228867)
and [historical verification](https://github.com/shaloy-lewis/credit-card-default-prediction/actions/runs/36693229025)
passed before local main was fast-forwarded and `codex/release-c-week12` created.
Existing branches and local work were preserved.

Preparation commit **R = `6c3e8a56cf0551dc22a67f91d8e204a1b8de4984`** contains the
[recording guide](week12-recording-guide.md), [timed captions](week12-captions.srt)
and [interview packet](interview-packet.md). R is the prepared recording source,
not a claim that a video was recorded. Its [four Linux CI jobs](https://github.com/shaloy-lewis/credit-card-default-prediction/actions/runs/36723348265)
and [historical verifier](https://github.com/shaloy-lewis/credit-card-default-prediction/actions/runs/36723348384)
all passed. The later checkpoint commit containing this record has its own CI
results in the Week 12 PR. A final completed package P remains to be identified.

No prediction API, CLI, model, scoring policy or dependency lock changed. No
training, tuning, calibration fitting, bootstrap regeneration or sealed-test
evaluation was performed. G3 conditions remain active and G5 stays open.

## Fresh-checkout rehearsal on the existing Windows machine

This is a fresh remote clone on the existing Windows machine, not a new-machine
or Docker rehearsal. Both checkouts were cloned from GitHub with `--no-hardlinks`,
with separate environments, dependency caches and independent model retrieval.
No environment, generated data or model was copied from the working checkout.

| Item | Recorded value |
| --- | --- |
| Date | 2026-09-30 |
| Windows | Windows 11 Home Single Language, 10.0.26200, build 26200 |
| CPU | Intel Core i5-1155G7 @ 2.50 GHz; 4 physical cores, 8 logical processors |
| Memory | 16,917,716,992 bytes of physical RAM |
| Current checkout | `.cache/release-c-week12/current-20260930`, detached at R |
| Historical checkout | `.cache/release-c-week12/historical-20260930`, detached at `7e571fc4fbb4e6cb99b0d66f8e7d72b24feccff4` |
| Python | Current: 3.12.13; historical: 3.12.3; separate managed runtimes and virtual environments |
| uv | 0.11.28; locked all-extras/dev installation in each checkout |
| Dependency lock SHA-256 | `60249471e9c6687a730d2c21c02e8d6cdd436eb05cb49b96ac0c0edfde85007b` |
| Model distribution revision | `f73ca4ee7a2c2d2ea51741e75fccf66ae7a4a640` |
| Native services | Uvicorn at `127.0.0.1:8080`; Streamlit at `127.0.0.1:8501` |

The [aggregate evidence inventory](week12-evidence.json) records exact artifact,
receipt, model, reference and tool hashes. Detailed runtime files remain ignored
under `D:/projects/credit-card-default-prediction/.cache/release-c-week12/`.
Use the recording guide to repeat commands with new destinations and run IDs.

| Command / check | Outcome at R |
| --- | --- |
| `uv sync --python 3.12.13 --locked --all-extras --dev --cache-dir .cache/uv` in current clone | Passed; independent environment. |
| `credit-risk artifacts pull --cache-dir .cache/huggingface`, then `artifacts verify` in each clone | Passed; independently retrieved reviewed bytes. |
| Historical `credit-risk data build` | Passed; restored governed source and split lineage solely for evidence verification. No historical scoring. |
| Historical `python -m credit_risk.assurance.collect` with tracked Phase 8 evidence already present | `existing_evidence_verified`, `scoring_performed: false`. |
| Historical `credit-risk release verify-b` with unchanged dossier and approval anchors | `approved_local_portfolio_release`, `g4_status: closed`. |
| Native `/ready` and Streamlit `/_stcore/health` | `ready` and `ok`. |
| `python -m credit_risk.portfolio.demo --run-id week12-rehearsal-001 --api-url http://127.0.0.1:8080` | Passed with actual batch CLI and monitoring publisher/verifier. |
| Streamlit AppTest through the live native API | `0.1904`, standard band, two non-causal attribution categories; no UI exception/error. This does not constitute a visual browser check. |
| Protected files compared with approved historical snapshot | No changes to tracked `reports/`, `configs/` or selected model manifest. |

The fixed dossier anchor remains
`f5fa342b89e06c016ea7b632c8a252c472502186ab52ac68aba6992d652c8c88`, and the approval
anchor remains `9a40d17f038056e3e475762810ead7887cdf764be836dc6387bc19638e4f8f1f`.
Historical approval authenticates that snapshot, not a new Week 12 release.

## Synthetic evidence and disposition

| Observation | Verified result |
| --- | --- |
| API probability / UI display | `0.190382` / `0.1904`; synthetic request only |
| Valid / rejected / selected accounts | 400 / 0 / 40 |
| Batch ID | `cbb6e64d311d32bbfcdfff59179917f13f8e00f5a48436332ab654934f8f71bf` |
| Initial invocation trace | `4747a34734d1442291c1a87991e360e2` |
| Reuse invocation trace | `3b011959605d4bda8d56c6029bf640b2` |
| Reuse | Same batch identity; all published file hashes and modification times unchanged |
| Monitoring | Reference and new report authenticated; `investigate`; `automatic_model_change: false` |
| Human review disposition | Expected concentration of repeated synthetic fixture patterns; retain scores and allow demonstration only. No model, calibration or policy change. |

The helper's existing `week11` runtime namespace is intentionally unchanged:
`current-20260930/experiment/portfolio/week11/week12-rehearsal-001/receipt.json`.
The monitoring package is at
`current-20260930/reports/monitoring/release_c_demo/week12-rehearsal-001/`.
Both are new ignored destinations. These 400 IDs repeat 20 fixture patterns; they
are not independent customer observations or prospective effectiveness evidence.

## Recording and media inventory

FFmpeg and ffprobe **9.0.2-essentials_build-www.gyan.dev** were obtained from the
provider linked by the [official download page](https://ffmpeg.org/download.html).
The archive matches its published SHA-256:
`60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba`.
The [inventory](week12-evidence.json) records each executable hash. Tooling lives
under `.cache/tools/ffmpeg-9.0.2/`; it does not change project dependencies.

The committed caption draft contains 21 cues over 300 seconds, with maximum
15.8 characters per second. This validates timing structure only, not readability
on footage. The six-scene checklist and window-only, no-audio capture/editing
recipes are in the recording guide.

A ten-second `gdigrab` preflight targeted only
`title=Release C capture preflight`. It exited with an I/O error because the window
could not be found. No fallback desktop capture occurred and no preflight MP4 was
produced. The attempted command, dedicated terminal script and error log remain
in ignored storage. The task-owned preflight terminal was stopped afterward.

| Required media property | Current evidence |
| --- | --- |
| Final MP4 / raw clips | Not produced; recording remains open |
| Intended local directory | `D:/projects/credit-card-default-prediction/.cache/release-c-week12/media-20260930/` |
| Duration / size / video SHA-256 | Unset; no video exists to measure or hash |
| Recording commit | Unset; R is prepared source only |
| Decode / H.264 / no-audio check | Pending actual recording |
| Scene completeness / caption readability | Pending full playback review |
| Linux scenes | Must visibly identify recorded Linux CI, its run and source SHA; Docker did not run locally on Windows |

## Privacy and claims audit

These are implementer checks, separate from owner acceptance. The full video
cannot be audited before it exists.

| Check | Result / disposition |
| --- | --- |
| Written quantitative claims | Reconciled with the [claims inventory](claims-inventory.md), existing authenticated aggregates and synthetic receipts; no new historical scoring. |
| Three resume bullets | Exactly three, with scientific, engineering and governance scope; local portfolio work is explicit and claims link to evidence. |
| Written causal / prospective claims | No prevented-default, intervention-effectiveness or current-population claim. Planning assumptions and proposed outcomes remain labelled. |
| Architecture and operating boundaries | SQLite transitions, PostgreSQL fixed-state recovery, conceptual cloud mapping, education-related G3 conditions and open G5 remain explicit. |
| Committed material privacy | Aggregate evidence and synthetic traces only; no credentials, customer records or raw video committed. |
| Browser/video privacy | Pending; must inspect every scene for personal data, unrelated content, historical rows and credentials. |
| Caption readability / covered evidence | Pending full recording and playback, including both explanations and the queue/alert states. |
| Owner viewing and claims decision | Pending; no owner acceptance inferred from author checks or CI. |

## Findings and required follow-up

| Finding | Disposition |
| --- | --- |
| Native UI automation reports missing Computer Use pipe; browser inventory is empty | Open for visual demonstration. Owner-operated dedicated window requested, as agreed in the plan. AppTest is functional support only. |
| FFmpeg cannot find the task-created capture window | Blocking media criterion. Repeat short preflight in a visible owner-operated window before full recording; retain failed receipt. Do not fabricate substitute screens or claim completed media review. |
| Sandbox could not write/read some fresh-checkout runtime receipts | Resolved by normal authorised execution of the same checks; runtime receipts were retained and hashed. No access controls were disabled. |
| Windows symlink creation is unavailable | Local tests explicitly skip affected cases; Linux CI exercises those paths. No platform-wide settings changed. |

The owner-operated browser request is outstanding. Once a visible window is
available: repeat preflight; record real API/UI and batch scenes with fresh run
IDs; assemble the silent MP4; review the full video and privacy/claims checklist;
record exact location, duration, size, checksum, recording source and cut list.
Then update this record and inventory in a separate media-review commit and rerun
all required checks on the completed package revision.

## Verification and handoff

At R, Ruff format/lint and mypy pass (82 maintained files). The ordinary suite
passes 887 tests, with five explicit Windows symlink skips. Portfolio tests pass
70 tests with 98.14% branch coverage; the artifact-marked walkthrough is checked
separately by the live helper and artifact suite (119 passed, one Windows symlink
skip). All four Linux jobs and the
historical Release B verifier pass, including existing package coverage gates,
image scans and persistent-platform restart checks.

The preparation documents render, relative destinations resolve, the existing
Mermaid diagram renders, and all 13 owner-acceptance items remain unchecked.
The PR records checks for the checkpoint introducing this record. Later completed
media/package commits require their own final-revision CI results.

Do not create the final owner-review request yet. Once the completed package P is
concrete and verified, `docs/reviews/release-c-owner-review.md` must identify P,
the evidence inventory and final video checksum. Only explicit owner acceptance
permits completion updates to roadmap, governance status, progress log, README
and the acceptance checklist. Those completion changes need CI again. No merge,
tag, public upload or production approval is authorised by this review record.
