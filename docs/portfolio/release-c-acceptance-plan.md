# Release C acceptance plan

**Status: planned; no Release C deliverable is accepted by this plan.**
**Owner:** Shaloy Lewis, project owner.
**Capacity:** the [roadmap](../roadmap.md)'s remaining two 10-hour work packages
(Weeks 11 and 12).
**Audience:** Senior Data Scientist and Senior ML Engineer reviewers.

## Week 11 implementation candidates

The [study protocol](intervention-study.md), [planning calculator](planning-calculator.md),
[architecture](architecture.md), [case study](case-study.md) and
[claims inventory](claims-inventory.md), [demo script](demo-script.md) and
[Week 11 review record](week11-review.md) are prepared for review. They do not
constitute owner acceptance; the completion checklist below remains open.

## Week 12 preparation

The [rehearsal and recording guide](week12-recording-guide.md),
[timed captions](week12-captions.srt) and [interview packet](interview-packet.md)
prepare the final work package. Rehearsal outcomes, media checks and findings
must be recorded separately before owner acceptance can be requested.

## Scope and inherited boundaries

Release C packages the implemented local portfolio product into a reproducible,
evidence-backed case study. Release B is already approved for local portfolio
use and G4 is closed for that scope. Reuse its authenticated platform, monitoring,
service and incident evidence; this communication work does not regenerate it.
The [governance status](../governance/model-governance-status.md),
[Release B owner review](../reviews/release-b-owner-review.md) and
[owner approval](../../configs/releases/release_b_owner_approval_v1.json) retain
their authority. G3 conditions remain in force and G5 production review stays open.

Keep the unchanged `selected_v1` model, 19 predictors, identity calibration,
risk bands and 10% review policy. Training, tuning, calibration fitting,
bootstrap regeneration and sealed-test evaluation remain prohibited. Verify
historical evidence without rescoring its cohorts; use synthetic operational
fixtures for the repeatable demonstration. Preserve authenticated evidence and
model bytes, and keep account mappings and row-level historical scores outside Git.

Human-owned outreach is the demonstration use case. Do not imply lending approval,
automated customer decisions, fairness certification, prospective performance or
causal impact. Explanations describe model attribution. Drift prompts human
investigation without automatic retraining, promotion or threshold changes.

This plan defines acceptance only. It does not implement the calculator, create
the video, approve Release C, deploy cloud services or extend production scope.

## Deliverables and acceptance evidence

Every quantitative claim must identify its source, population or fixture,
measurement context and limitations. Distinguish **measured historical results**,
**measured synthetic demonstrations**, **simulated planning estimates** and
**hypothetical proposals** in the material itself. Model lift is not causal impact.

| Deliverable | Acceptance criteria | Evidence required for review |
| --- | --- | --- |
| Portfolio narrative | A two-minute README overview and concise executive case study explain the decision, individual ownership, measured results, limitations and operating restrictions. Both audiences can locate the scientific and engineering evidence. Quantitative claims link to authenticated evidence and carry the appropriate claim label. | README and case-study paths at the review commit; claim-to-source checklist; recorded owner review of the two-minute overview. |
| Architecture | A rendered Mermaid diagram describes the implemented data, inference, registry, storage, monitoring and human decision paths. It separates Phase 7 SQLite promotion/rollback from Phase 8 PostgreSQL fixed-state bootstrap, persistence and restart recovery. Azure/Databricks mappings are explicitly conceptual; no PostgreSQL promotion/rollback or deployed-cloud claim. | Mermaid source and rendering check; component-to-code/configuration references; review against the implemented system and the roadmap's conceptual cloud mapping. |
| Intervention study design | A hypothetical protocol specifies eligibility, randomisation unit, treatment/control arms, primary and secondary outcomes, customer-harm guardrails, intention-to-treat analysis, label maturity and stopping rules. Address contamination and duplicate participation. Neither the model nor the proposed intervention is claimed to reduce default. | Versioned protocol, declared assumptions and owner review; outcomes and maturity definitions distinguish proposed future collection from the historical dataset. |
| Planning calculator | A standalone Python sample-size/MDE calculator documents method, assumptions, inputs, units and outputs, aligned with the study design. It identifies the effect scale, significance level, power, allocation and any attrition or clustering assumptions. Outputs are labelled planning estimates. | Script, usage examples and method reference; tests for independent reference calculations, invalid inputs and expected sensitivity to effect size and power; example output and test results at the review commit. |
| Demonstration | A reviewed local MP4 runs for 4–6 minutes and follows a reproducible script. Show the local API/UI and a synthetic batch live, including prediction `0.190382`, explanations, the review queue and monitoring. Clearly label authenticated Linux CI evidence used for platform persistence and recovery. Public hosting is optional. | MP4 outside Git; duration, SHA-256, recording commit and retrieval location in the review record; script with commands and expected outputs; claims/privacy review and owner viewing decision. |
| Reproducibility and interview material | A fresh-checkout rehearsal verifies the approved evidence and repeats the synthetic demonstration. Include a privacy/claims checklist, a system-design walkthrough and three evidence-backed résumé bullets, covering scientific judgement and engineering ownership. | Rehearsal environment, checkout commit, commands, outcomes and CI links; completed privacy/claims checklist; walkthrough and three bullets with traceable evidence; video checksum verification. |

## Reproducibility and demonstration contract

1. Start from a fresh checkout of the recorded implementation commit. Record OS,
   Python and pinned environment versions, hardware, artifact retrieval steps and
   explicit local API/UI startup commands. Retrieve the exact reviewed model;
   never retrain it. Keep credentials and runtime outputs in ignored storage.
2. Authenticate the unchanged Release B dossier and detached owner approval in
   the separate approved checkout at `7e571fc4fbb4e6cb99b0d66f8e7d72b24feccff4`,
   using the fixed external digest anchors below. Require
   `approved_local_portfolio_release` and closed G4 for local scope. Verification
   authenticates that historical release; it is not approval of the patched
   implementation. Record the current implementation commit and its separate
   passing CI results for the live synthetic demonstration.
3. Repeat the scripted API/UI request and deterministic synthetic batch, checking
   prediction `0.190382`, explanation categories, queue policy and monitoring.
   Record expected alerts and human disposition: the approved synthetic batch
   intentionally differs from the reference and can require investigation.
   Preserve completed batch outputs when an alert fires.
4. Show the authenticated Linux CI records for persistent platform startup,
   restart and recovery with their run links and implementation identities.
   Clearly separate recorded CI evidence from services shown live on the recording
   machine. The Phase 7 rollback demonstration remains SQLite-specific.
5. Record failures and resolutions, then rerun affected rehearsal steps. Review
   the full video for synthetic-only customer displays, absent secrets and
   unsupported claims. Record any edits and verify the final MP4 checksum.

Follow the [historical checkout setup](../../README.md#release-b-sign-off), then
run this command inside that checkout using its own pinned environment. Do not
change historical source hashes to make verification pass in a patched checkout:

```sh
uv run credit-risk release verify-b \
  --expected-manifest-sha256 f5fa342b89e06c016ea7b632c8a252c472502186ab52ac68aba6992d652c8c88 \
  --approval configs/releases/release_b_owner_approval_v1.json \
  --approval-sha256 9a40d17f038056e3e475762810ead7887cdf764be836dc6387bc19638e4f8f1f
```

The [Release B report](../../reports/releases/release_b_v1/release-b-report.md)
and its [manifest](../../reports/releases/release_b_v1/evidence-manifest.json)
remain immutable. Their candidate status is historical; the separate authenticated
owner decision establishes approval. Do not rewrite the dossier to change status.

## Work order and ownership

The implementer prepares artifacts and records verification. The project owner
reviews claims, limitations, the full recording and the completed package, and
alone accepts Release C. Where the owner also authors a deliverable, record
completion and acceptance as separate decisions. A generated checklist or green
CI run cannot approve the package.

| Work package | Sequence and approximate effort | Review checkpoint |
| --- | --- | --- |
| Week 11 — impact and communication, 10 h | Study design (3 h); calculator and tests (3 h); architecture (1 h); narrative (2 h); demo script (1 h). | Protocol and calculator assumptions agree; diagram matches implementation; claims have sources; script is executable. All six deliverables have an identified artifact or remaining action. |
| Week 12 — hardening and interview packet, 10 h | Fresh-checkout rehearsal (3 h); recording (3 h); claims/privacy review (2 h); system-design walkthrough and résumé material (2 h). | Rehearsal and required CI pass; the reviewed MP4 meets duration/content requirements; evidence inventory and owner acceptance are recorded. |

Reuse Release B evidence throughout. If an exit criterion cannot be met within
the estimate, record the outstanding work and keep Release C planned; do not
weaken acceptance or treat elapsed effort as completion.

## Completion checklist and owner decision

Maintain a review record with paths, exact commits, evidence/CI links, test and
rehearsal outcomes, video location and checksum, findings and their dispositions.
Future artifacts are not acceptance evidence until completed and reviewed.

- [ ] Two-minute README and executive case study pass the claims and ownership review.
- [ ] Mermaid source renders and matches the implemented system and phase boundaries.
- [ ] Hypothetical intervention protocol covers every required design and harm control.
- [ ] Python calculator method, examples and meaningful tests pass at the review commit.
- [ ] Fresh-checkout rehearsal verifies approved Release B evidence and repeats the synthetic demo.
- [ ] Local MP4 is 4–6 minutes, contains all required scenes and distinguishes live work from Linux CI evidence.
- [ ] Video SHA-256, recording commit and accessible out-of-Git location are recorded.
- [ ] Privacy/claims checklist is complete; operating restrictions and G3 conditions remain prominent.
- [ ] System-design walkthrough and three résumé bullets link to supporting evidence.
- [ ] Required CI passes at the final package commit; Markdown rendering and relative links are checked.
- [ ] Every review finding has a disposition, with no failed mandatory criterion.
- [ ] Project owner explicitly accepts the completed package, bound to its exact commit, evidence inventory and video checksum.
- [ ] Only after acceptance, update the roadmap, governance status, progress log and README to mark Release C complete; G5 remains open.

The owner decision records reviewer, date, accepted commit, evidence inventory,
video checksum, scope and retained restrictions. Until every item passes and
that explicit decision exists, Release C remains planned.
