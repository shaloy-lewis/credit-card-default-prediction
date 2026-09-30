# Week 12 rehearsal and silent recording guide

**Status:** preparation recipe; executing commands is not owner acceptance.
Use the [five-minute script](demo-script.md), [timed captions](week12-captions.srt)
and [acceptance contract](release-c-acceptance-plan.md). Keep Release C planned
until every mandatory criterion and an explicit owner decision are recorded.

## Freeze and identify the recording source

Commit the preparation material on `codex/release-c-week12`. Record that full
SHA as the recording source R before capture. Clone from the remote repository
into a new ignored `.cache/release-c-week12/current-<run>` directory and detach
at R. Do not copy the working checkout's environment, model, outputs or data.
Reserve fresh destinations; never overwrite a prior run. A later package commit
P adds the review record, inventory and findings. Record both R and P in the PR;
P does not pretend to be the source shown in a recording made at R.

Use uv 0.11.28 and a separately installed Python 3.12 runtime. Record the actual
patch version, Windows edition/build, CPU, RAM, command outcomes and lockfile hash.
For a new checkout, run these from its root, with uv's cache and managed Python
install directory set to fresh ignored paths:

```powershell
uv sync --python 3.12 --locked --all-extras --dev --cache-dir .cache/uv
.venv/Scripts/credit-risk.exe artifacts pull --cache-dir .cache/huggingface
.venv/Scripts/credit-risk.exe artifacts verify
git status --porcelain
git rev-parse HEAD
Get-FileHash models/selected_v1/model.cbm -Algorithm SHA256
```

The expected model SHA-256 is
`844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c`.
Existing uv tooling may be reused; environments, artifact downloads and generated
outputs must belong to the new checkouts. Do not run any training, tuning,
calibration fitting, bootstrap regeneration or sealed-test evaluation commands.

## Authenticate the historical release separately

Clone again into a distinct ignored directory, detach at
`7e571fc4fbb4e6cb99b0d66f8e7d72b24feccff4` and restore that checkout's own locked
environment with Python 3.12.3, matching the historical verification workflow.
Follow [Release B setup](../../README.md#release-b-sign-off): pull and verify the
reviewed model and run the governed `credit-risk data build` only to restore
source/split lineage. Require the tracked Phase 8 evidence directory to exist
before calling the collector; its existing-evidence branch verifies without
scoring. Run the fixed command from the acceptance plan and retain its result:

```powershell
.venv/Scripts/python.exe -m credit_risk.assurance.collect
.venv/Scripts/credit-risk.exe release verify-b --expected-manifest-sha256 f5fa342b89e06c016ea7b632c8a252c472502186ab52ac68aba6992d652c8c88 --approval configs/releases/release_b_owner_approval_v1.json --approval-sha256 9a40d17f038056e3e475762810ead7887cdf764be836dc6387bc19638e4f8f1f
```

Require `existing_evidence_verified`, `scoring_performed: false`,
`approved_local_portfolio_release` and closed local G4. This authenticates the
historical snapshot, not a new implementation. Retain the fixed dossier and
approval hashes. Compare `reports/`, `configs/` and the selected model manifest
with that snapshot; no Release C record belongs in those protected directories.

## Rehearse current native behavior

From the clean checkout at R, start native API and Streamlit as documented in the
[demo script](demo-script.md). Keep both bound to localhost and record the PIDs of
processes started for this run. Use a dedicated browser profile/window without
accounts, unrelated tabs or notifications. The owner operates the Predictor when
native/browser automation is unavailable; command preparation and verification
remain the implementer's work.

Use the exact fixture and execute the existing helper with a fresh run ID:

```powershell
.venv/Scripts/python.exe -m credit_risk.portfolio.demo --run-id week12-rehearsal-001 --api-url http://127.0.0.1:8080
```

The unchanged helper uses `experiment/portfolio/week11/<run-id>` even for Week 12.
That namespace is retained for compatibility, not evidence of an earlier run.
Monitoring output goes to the existing narrowly ignored
`reports/monitoring/release_c_demo/<run-id>`. Both destinations must be new.
The synthetic snapshot date `2026-09-30` is a frozen batch identifier.

Require API 0.190382, UI 0.1904, standard band and both explanation categories;
400 valid synthetic rows, zero rejections and 40 selected accounts; distinct
invocation traces but the same batch ID; identical file bytes and modification
times on reuse. Authenticate the monitoring reference with SHA-256
`5f2a43675cbd9f6ed44bf4a421df6647d3e780cf5b2e9eae6ba782e94849c4ac` and the new report
with its returned manifest digest. Require `investigate` and
`automatic_model_change: false`. Disposition: known synthetic fixture concentration;
retain outputs and permit demonstration only, with no model/policy change.

A Streamlit application test against the live API can support functional checks,
but cannot replace browser viewing or actual-window recording. Record failures
and resolutions, including platform-specific restrictions, in `week12-review.md`.
Describe this as a fresh-checkout rehearsal on the existing Windows machine.

## Pin recording tools and preflight window capture

Use the Gyan Windows essentials build linked by the
[official FFmpeg download page](https://ffmpeg.org/download.html).
Pin **9.0.2**, archive `ffmpeg-9.0.2-essentials_build.zip`, with SHA-256
`60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba` from the
[provider's checksum](https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-essentials_build.zip.sha256).
Retain archive, download URL, ffmpeg/ffprobe versions and executable hashes under
ignored tooling storage. Verify the archive before extraction or execution.
A later version requires a separately recorded pin and review, not a silent swap.

Set `$ffmpeg`, `$ffprobe` and `$media` to those absolute executable paths and a
new ignored media directory. Record the exact title of a dedicated demonstration
window. FFmpeg's [gdigrab device](https://ffmpeg.org/ffmpeg-devices.html#gdigrab)
supports `title=<window title>` or `hwnd=<window handle>`. Do not select `desktop`.
Keep the window visible, unminimised and unobscured. A title/handle mismatch,
black frames, frozen display or unrelated content fails preflight.

```powershell
# Run from the new media directory; -n prevents overwriting existing recordings.
& $ffmpeg -n -f gdigrab -framerate 30 -i "title=$captureTitle" -t 10 -an -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p preflight.mp4
& $ffprobe -v error -show_streams -show_format -of json preflight.mp4
& $ffmpeg -v error -i preflight.mp4 -f null -
& $ffmpeg -n -ss 5 -i preflight.mp4 -frames:v 1 preflight-frame.png
```

Inspect the frame and play the complete preflight before recording full scenes.
No microphone or system-audio device is selected. If capture is unavailable, keep
video acceptance open, prepare the remaining package and use owner-operated
controls or a later recording session. Do not generate replacement application
screens or label application-test output as live UI footage.

## Scene checklist and caption timing

The captions assume a 300-second edit. Trim actual clips to these segments, record
source filenames/in-out times in `edit-list.csv`, and retain every raw clip. Do not
stretch execution footage or hide failed commands. Corrections require a fresh
run ID and a recorded disposition. Captions may be re-timed to match actual
footage, preserving every required claim and a final duration of 240-360 seconds.

| Timeline | Actual material to capture | Required check |
| --- | --- | --- |
| 0:00-0:35 | README/case study in dedicated document window | Human decision, ownership and local portfolio boundary are legible. |
| 0:35-1:35 | Live API request/result and native Streamlit Predictor | Exact fixture, 0.190382 / 0.1904, standard band and both attribution categories. |
| 1:35-2:45 | Actual helper execution and resulting synthetic queue in dedicated terminal | 400 valid, 40 selected, distinct traces, verified reuse; no historical rows. |
| 2:45-3:35 | Actual monitoring summary and verified receipt | Investigation, known synthetic shift, human disposition, no automatic change. |
| 3:35-4:30 | Rendered architecture and authenticated Linux CI records | Prominent recorded-Linux-evidence caption; SQLite transitions distinct from PostgreSQL bootstrap/recovery. |
| 4:30-5:00 | Study protocol, calculator assumptions and limitations | Hypothetical outcome, planning estimates, G3 conditions, G5 open and Release C pending acceptance. |

For each scene, use the same window-specific capture recipe with a suitable
`-t` and a unique raw filename. Preserve aspect ratio when normalising clips,
for example `scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2`.
Use a readable application zoom and leave a caption-safe bottom area. Never crop
away evidence needed to interpret a result. Concatenate the audited cut list into
`joined.mp4`, then burn in the committed captions copied into the media directory:

```powershell
& $ffmpeg -n -i joined.mp4 -vf "subtitles=week12-captions.srt:force_style='FontName=Arial,FontSize=24,Outline=2,MarginV=30'" -an -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart release-c-demo.mp4
& $ffprobe -v error -show_streams -show_format -of json release-c-demo.mp4
& $ffmpeg -v error -i release-c-demo.mp4 -f null -
Get-FileHash release-c-demo.mp4 -Algorithm SHA256
```

Retain raw footage, edited intermediate, exact captions, cut list, executed command
log and checksums outside Git. Record final absolute location, duration, size,
SHA-256, R and tool versions in the review record. The full video must decode,
contain H.264 video and no audio stream, and remain legible at normal playback.
Review the complete timeline, not just representative screenshots.

## Privacy, claims and final review

Author review records pass/fail and corrections for each item; blanks are not passes:

- Only synthetic account IDs and inputs appear; no historical row-level records.
- No credentials, account profiles, unrelated windows, notification content or personal information appear.
- Captions do not obscure probabilities, explanations, queue counts or alert state.
- Quantitative claims reconcile with the [claims inventory](claims-inventory.md) and identify the population/fixture.
- Historical measurements, synthetic checks, approximate planning estimates and hypothetical proposals remain distinguishable.
- Linux evidence is visibly labelled; cloud mappings are conceptual; PostgreSQL transitions are not claimed.
- Education-related G3 conditions, human control, local-use boundary and open G5 remain explicit.
- Every required scene is present; duration, decode, checksum and source commit are verified.

Record all findings, corrections and unresolved criteria in `week12-review.md`.
Run Ruff, mypy, applicable tests and existing coverage gates; render Markdown and
Mermaid and check links. Require all four Linux CI jobs plus historical Release B
verification on final P. The PR carries final commit and check URLs, avoiding a
self-referential hash in a committed review record. Do not reuse R's CI result as
P's result. Recheck protected evidence and model bytes.

Only when the package and reviewed video are concrete, prepare
`docs/reviews/release-c-owner-review.md` identifying P, the evidence inventory and
video SHA-256. The owner watches and explicitly accepts that package. Until then,
do not tick owner-review items or mark Release C complete. No automatic merge,
public hosting, tags or production approval is part of this work.
