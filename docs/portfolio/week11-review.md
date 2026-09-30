# Week 11 implementation review record

**Status:** implementation prepared for review; Release C remains planned.
No owner acceptance, video approval or G5 production approval is recorded here.

## Baseline and artifact inventory

Local main was fast-forwarded to merged review-fix commit
`9b0635559ec80d9f0d1198be5f978fd0999bff24`, preserving the three fix commits and
source branches. Its [four CI jobs](https://github.com/shaloy-lewis/credit-card-default-prediction/actions/runs/36594629955)
and [historical release verifier](https://github.com/shaloy-lewis/credit-card-default-prediction/actions/runs/36594629957)
passed before the Week 11 branch was created.

| Planned change | Review artifacts | Commit record |
| --- | --- | --- |
| Study and calculator | [Protocol](intervention-study.md), [method and examples](planning-calculator.md), [Python source](../../src/credit_risk/portfolio/planning.py), [tests](../../tests/unit/portfolio/test_planning.py) | `3469e5f` |
| Architecture and narrative | [README overview](../../README.md), [case study](case-study.md), [Mermaid architecture](architecture.md), [claims inventory](claims-inventory.md) | `6e27570` |
| Demonstration and handoff | [Timed script](demo-script.md), [helper](../../src/credit_risk/portfolio/demo.py), [tests](../../tests/unit/portfolio/test_demo.py), this record | Commit introducing this record; exact head and final checks are recorded in the Week 11 PR. |

The PR for `codex/release-c-week11` is the final-revision verification ledger:
record exact head, all four CI links/results, historical verification, native
walkthrough receipt/commit and document rendering checks there. The helper
records its precise commit and hashes in ignored runtime receipts. This avoids
a self-referential committed file pretending to authenticate its own commit.
No subsequent source changes may reuse a previous head's passing checks.

## Local checks before the final commit

- Ruff formatting/lint and mypy pass (82 maintained source files).
- The ordinary suite passes 887 tests; five Windows symlink tests are skipped.
- The artifact suite passes 119 tests; one Windows symlink test is skipped.
- The portfolio package passes 71 tests, including its actual-model walkthrough,
  with 98.14% branch coverage. Linux CI must exercise the platform-specific paths.
- Eleven Markdown files render; 202 links are parsed and all relative destinations
  resolve. All 13 owner acceptance items remain unchecked.
- Protected reports, configurations and the model manifest match the approved
  snapshot; the unchanged model artifact verifies.

The final PR records the clean-commit native walkthrough, diagram rendering and
remote CI results after this commit. Browser automation is unavailable in this
session; the native UI is checked with Streamlit's application test harness
against the live API. The final visual walkthrough and recording review remain
Week 12 work.

## Acceptance evidence required at handoff

- Calculator reference examples, validation failures, sensitivity and inversion tests pass; isolated execution excludes site packages, model/data reads and networking.
- New-package branch coverage exceeds the existing 90% gate; Ruff and mypy pass.
- Actual batch CLI events, 400-row scoring, a 40-row queue, no-rewrite reuse and expected monitoring investigation pass with the reviewed model.
- Native API returns `0.190382`; the unchanged UI displays `0.1904`, the standard band and explanations; the native helper produces an exact-commit receipt.
- Markdown and Mermaid render; relative links resolve; numeric claims match the inventory.
- All four Linux CI jobs and historical Release B verification pass on the final revision.
- Historical reports, approvals, configurations, model manifests and model bytes remain unchanged; no training, tuning, calibration fitting, uncertainty regeneration or sealed-test evaluation occurs.

These are implementation checks, not owner-review ticks. The
[Release C acceptance checklist](release-c-acceptance-plan.md#completion-checklist-and-owner-decision)
retains its original authority and remains open.

## Week 12 work still required

Repeat a full fresh-checkout rehearsal; record and review the 4-6 minute MP4;
record its checksum, recording commit and retrieval location outside Git;
complete privacy/claims review; write the system-design walkthrough and three
evidence-backed resume bullets; obtain the project owner's explicit acceptance.
G3 conditions remain in force and G5 remains open.
