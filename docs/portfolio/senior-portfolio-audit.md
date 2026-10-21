# Senior portfolio audit

**Audit date:** 2026-10-21. **Scope:** independently developed local portfolio;
not a production-readiness or Release C approval.

Inspected baseline: `834d5f2fd25a36b5957504f39a552f4f122969c9` on
`codex/pre-recording-cleanup`, initially clean. Improvements use the separate
`codex/senior-portfolio-review` branch. No push, merge or owner decision is implied.
The assessment below describes the baseline; the disposition column identifies
the permitted follow-up, rather than calling existing work missing.

## Evidence-linked gap matrix

“Complete” means evidence exists for the stated **local** scope, not that it
establishes real-world suitability. P0 protects interpretation and reproducibility;
P1 improves defensibility; P2 requires future evidence or a demonstrated need.

| Dimension / baseline evidence class | Existing implementation and strengths | Specific gap and credibility consequence | Type / priority | Improvement and current permission |
| --- | --- | --- | --- | --- |
| 1. Business framing — Complete | [Accepted brief](../product-brief.md), [case study](case-study.md): named hypothetical operations owner, monthly next-month target, fixed capacity, prohibited lending actions. | README puts substantial release history before audience routing; stakeholder actions and capacity trade-offs require several documents. | Presentation / P0 | Shorten entry narrative; add audience paths and a findings-to-action table. Permitted; no policy change. |
| 2. Data collection and quality — Complete | [Data card](../data/data-card.md), [validation policy](../data/validation-policy.md), [schema](../../src/credit_risk/data/schema.py), [reviewed lineage lock](../../configs/data/split_v1.lock.json): pinned acquisition, quarantine, deterministic schema/splits, accepted anomalies and duplicate policy. | Older cards describe retired compatibility workflows in historical tense/context that can be mistaken for current instructions. Quality, eligibility and predictor boundaries are scattered. | Discoverability / P1 | Add a current scientific reading guide distinguishing these boundaries; retain historical documents and digests. Permitted. |
| 3. EDA and statistical insight — Partially demonstrated | Data-card composition/anomalies; [baseline report](../../reports/modeling/baseline_v1/baseline-report.md); [calibration/capacity evidence](../../reports/releases/release_a_v1/release-a-report.md); [subgroup report](../../reports/governance/phase5_v1/fairness-report.md). | Substantial aggregate diagnostics exist, but no concise interpretation connects findings to modeling choices. No authenticated feature-level association study or duplicate-sensitivity result was located; do not imply one exists. | Presentation / P1; evidence / P2 | Explain only established aggregates and what they cannot show. Any new row-level EDA, duplicate sensitivity or historical scoring needs separately approved scope; not executed. |
| 4. Features and leakage — Complete | [Availability review](../data/feature-availability.md), [19-feature contract](../../configs/modeling/feature_contract_v1.json), [dataset interface](../../src/credit_risk/modeling/dataset.py), [split tests](../../tests/integration/test_selection_split.py): demographic separation and training-only preprocessing. | Need a readable rationale for keeping raw monetary values, status encoding, no imputation/clipping and proxy risk. Six history columns can be mistaken for longitudinal evaluation. | Presentation / P1 | Explain existing decisions and residual duplicate/proxy/availability limitations. Permitted; no new features or fitting. |
| 5. Model methodology and validation — Complete | [Selection protocol](../modeling/selection-protocol.md), [Release A](../../reports/releases/release_a_v1/release-a-report.md), [consumed-test protection](../../src/credit_risk/modeling/final_test_workflow.py): fixed comparison, guardrails, exact winner, validation-only uncertainty, one authorized test. | README advertises selection/freezing as setup operations; small winning margin and conditional uncertainty deserve clearer interpretation. | Presentation with operational consequence / P0 | Replace runnable training instructions with historical explanation; document exact selection arithmetic, metrics and uncertainty limits. Permitted. |
| 6. Business value and decision policy — Partially demonstrated | Authenticated 5/10/20% capacity tables; [trial protocol](intervention-study.md); existing [sample-size/MDE calculator](planning-calculator.md) and [tests](../../tests/unit/portfolio/test_planning.py). | Explicit review burden, missed-event opportunity cost and economic break-even sensitivity are absent. Actual treatment benefit/cost and financial impact are unobserved. | Analytical framing / P1; observed impact blocked | Add aggregate-only decision analysis and explicitly assumed economics. Permitted. Real impact requires approved prospective study and new records; blocked in this scope. |
| 7. ML engineering and reproducibility — Complete | [Shared engine](../../src/credit_risk/inference/engine.py), [batch](../../src/credit_risk/inference/batch.py), [artifact distribution](../artifacts/storage-architecture.md), [CI](../../.github/workflows/ci.yml), [September rehearsal](week12-review.md): contracts, idempotency, lineage, health, scans and recovery. | README quickstart mixes immutable history with executable setup and has an outdated narrower mypy command. No established defect warrants an inference rewrite. Linux container checks do not establish throughput or current-branch CI success. | Documentation correctness / P0 | Provide synthetic-only fast start and CI-aligned local checks; describe scale, concurrency and recovery limits. Permitted; preserve runtime and dependency lock. |
| 8. Governance, fairness and monitoring — Complete for local scope | [G3 decision](../../reports/governance/phase5_v1/g3-review.md), [governance status](../governance/model-governance-status.md), [runbooks](../operations/release-b-runbooks.md), [delayed labels](../operations/delayed-label-contract.md), [Release B approval](../../configs/releases/release_b_owner_approval_v1.json). | Technical tests can be mistaken for certification; q90 subgroup diagnostics differ from the operational exact-count queue. Prospective subgroup harms, mature labels and G5 remain unresolved. | Presentation / P1; production evidence blocked | Explain methodology, support limits, proxy risk, human escalation and distinct gates. Permitted; no gate closure or retrospective evidence edits. |
| 9. Communication and documentation — Partially demonstrated | Existing case study, architecture Mermaid, claims inventory, demo helper, recording guide and functional rehearsal. | No explicit recruiter/science/engineering routes. Final MP4, visual UI review and full-video claims/privacy review remain missing in the Week 12 record. | Presentation / P0; media blocked by assets/review | Improve existing narrative/navigation and use current follow-up checklist. Actual footage and human review remain required; do not recreate completed preparation or fake media. |
| 10. Senior interview readiness — Partially demonstrated | Existing ten-minute walkthrough, eight discussion prompts, exactly three evidence-linked resume bullets and personal ownership statement. | The requested twenty questions are not all answered explicitly; weak coverage of alternatives, six-week priorities and conditional evidence can undermine a difficult interview. | Presentation / P1 | Extend the existing packet with concise stakeholder answers and technical reasoning linked to evidence. Permitted; retain three bullets and local-project framing. |

## Verified anchors and important qualifications

- [Final-test summary](../../reports/modeling/final_test_v1/summary.json):
  6,000 accounts, 1,327 observed defaults; AP 0.542867, Brier 0.136304,
  lift at 10% 3.089676, selected 600 with 410 defaults. These are historical
  aggregate findings. No new scoring is needed to discuss them.
- [Release A](../../reports/releases/release_a_v1/release-a-report.md): the
  500-resample intervals describe validation predictions, not test uncertainty,
  algorithm superiority or transportability.
- The final four-fit comparison followed earlier development-CV experiments.
  It was not the project's entire research budget or a wholly untouched
  validation cohort. [Scientific decisions](scientific-decisions.md) explains
  this limitation without reopening the test.
- [Phase 7](../../reports/registry/phase7_v1/registry-release-report.md) supports
  SQLite transitions; [Phase 8](../../reports/platform/phase8_v1/evidence-manifest.json)
  supports PostgreSQL/MinIO fixed-state persistence and recovery. These are
  separate demonstrations, not evidence of PostgreSQL promotion/rollback.
- The immutable Release B dossier retains historical candidate wording; its
  separate digest-bound owner approval establishes local acceptance. Later
  implementation changes require their own checks. G3 remains conditional,
  G4 closed locally, G5 open, and Release C planned.

## Implementation order and explicit deferrals

1. **P0:** preserve protected-byte baseline; correct README reproduction scope;
   put the decision, ownership, verified result, limitations and audience paths
   first. Keep historical operating instructions and approval anchors available.
2. **P1:** add [decision analysis](decision-analysis.md) and
   [scientific reasoning](scientific-decisions.md); extend the existing case
   study, architecture, claims inventory and interview packet instead of making
   competing case studies, calculators or notebooks.
3. **P1:** run existing applicable quality gates, check new Markdown/diagrams and
   links, and compare all protected bytes with the pre-edit baseline. Record
   actual outcomes below, including environmental blockers.
4. **P2 / deferred:** representative data, temporal/external validation,
   approved future training/calibration, prospective effectiveness, treatment
   targeting, economic cost measurement and PostgreSQL release transitions.
   These need new evidence and approval; a communication upgrade cannot supply
   them. Add infrastructure only for a measured requirement.

## Release C handoff, without changing acceptance

The [Week 12 record](week12-review.md) already establishes the September 30
functional fresh-checkout rehearsal at its recorded source; do not repeat it as
new evidence or rewrite its outcomes. The [recording guide](week12-recording-guide.md)
already supplies commands, isolated destinations, preflight, ffprobe/checksum and
privacy review instructions. The next package still needs:

| Remaining action | Required evidence / owner |
| --- | --- |
| Freeze the next recording source after implementation checks | Exact committed revision and its four Linux CI jobs plus historical verification; implementer prepares, owner reviews. Current local checks do not substitute. |
| Repeat affected final-revision rehearsal steps | Fresh checkout and independent artifact/environment retrieval under the existing guide; preserve the prior rehearsal. |
| Capture actual 4–6 minute video and visually inspect UI | Successful dedicated-window preflight, real scenes, final MP4 outside Git, duration, size, decode and SHA-256. No video completion asserted by this audit. |
| Review all footage for privacy and claims | Full playback, legible captions and scene-by-scene findings; author checks and owner viewing recorded separately. |
| Accept completed package | Owner decision bound to package revision, inventory and video checksum, only after all existing criteria pass. No acceptance is inferred here. |

The [existing acceptance checklist](release-c-acceptance-plan.md#completion-checklist-and-owner-decision)
remains authoritative and unchecked. Production adoption separately needs a
representative population, approved operational eligibility, security/privacy
design, mature labels, staffing, harm review and governance authorization.

## Verification of this improvement

Author-run checks on 2026-10-21, using the existing Windows environment, Python
3.12.13 and uv 0.11.28. This is a working-tree verification record, not new Linux
CI evidence, fresh-checkout acceptance or owner approval. Commands below use the
existing `.venv/Scripts/python.exe` directly unless otherwise stated.

| Check actually executed | Result and practical limit |
| --- | --- |
| `uv lock --check --offline --cache-dir .uv-cache` | Passed; lock resolves 151 packages. Used the existing `.cache/tools/uv/uv.exe`. |
| `uv sync --locked --all-extras --dev --offline --cache-dir .uv-cache --dry-run` | Passed; checked 149 installed packages and would make no changes. This checks the current environment, not a new installation or independent remote download. |
| `python -m ruff format --check api.py app.py src/credit_risk tests` | Passed; 187 files already formatted. |
| `python -m ruff check api.py app.py src/credit_risk tests` | Passed; installed Ruff 0.16.2. |
| `python -m mypy` with the full maintained-path list from CI | Passed; no issues in 82 source files. README now includes the portfolio/assurance/monitoring boundaries previously omitted from its command. |
| `python -m pytest -m "not training and not artifact" --cov --cov-report=term-missing` | **889 passed, 5 skipped, 153 deselected**; 137.18 seconds. Skips explicitly concern unavailable Windows symlinks. Combined statement/branch coverage reported 79% with artifact tests excluded; this is not the separate package coverage gate. |
| `python -m pytest -m "artifact and not training"` | **120 passed, 1 skipped, 926 deselected**; 73.75 seconds. The skip concerns Windows symlinks. Uses reviewed bytes and synthetic inputs/temporary fixtures; no protected-cohort rescoring. |
| `python -m pytest tests/unit/portfolio -m "not training and not artifact" --cov=credit_risk.portfolio --cov-branch --cov-report=term-missing --cov-fail-under=90` | **70 passed, 1 deselected; 98.14% coverage**, above the 90% branch-enabled gate. These tests overlap the ordinary suite and are not 70 additional unique passes. |
| `python -m pre_commit validate-config`; `python -m pre_commit run --all-files` | Configuration passed. Initial default-cache access and then sandboxed hook download failed; rerun with a project-local cache and authorized network access passed both pinned Ruff hooks. No source changes resulted. |
| `credit-risk artifacts verify` | Passed; one reviewed native model file verified, without deserialization. |
| Markdown/aggregate one-off review | Nine documents parsed/rendered; **346** links checked, including local destinations and heading anchors; table cell counts, twenty interview answers, three resume bullets and all thirteen unchecked acceptance items verified. Published aggregate subtraction and hypothetical economic arithmetic independently checked with exact decimal arithmetic. |
| Mermaid and local document rendering | All three diagrams rendered to SVG/PNG; nine local HTML previews rendered at 1440-pixel width with no horizontal overflow or broken images. Selected document previews and all diagrams visually inspected. Initial sandbox browser startup timed out; authorized local headless rendering succeeded. This is documentation QA, not application visual review or video footage. |
| `git diff --check` and protected-byte comparison | Passed. A pre-edit baseline covers **341** tracked/model files. Only six existing Markdown files differ; three new Markdown files are added. All **279** specifically enumerated model/config/report/source/test/workflow/evaluator/dependency files match their baseline hashes; all other baseline files outside the edited documents also match. |
| Docker/current Linux CI | **Not executed successfully here:** `docker info` could not connect to the Docker engine; the daemon is unavailable and the sandbox also denied the default Docker config. No current image build, vulnerability scan, restart rehearsal or new remote CI result is asserted. |

Both test suites retained existing third-party warnings about Starlette's httpx
test client and a Pydantic protected namespace. No dependency or runtime change
was made to suppress them. Training-marked tests were excluded; replay tests
only verify rejection by the tombstone. Historical uncertainty was not regenerated.
Remaining CI package coverage jobs were not separately repeated for this
documentation-only change; full final-revision Linux CI remains required under
the existing acceptance plan.

One-off baselines, JUnit output, logs, arithmetic/link-check code and rendered
previews remain in ignored `.cache/senior-portfolio-review/`; they are local
author-review support, not newly authenticated release evidence. The earlier
Week 12 record and its evidence inventory retain their original bytes and status.

The selected model SHA-256 remains
`844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c`;
the selected manifest remains
`df5ce6ce07b268f57fa3bf72c97cd32f8ebb66695d7157139942c91e46d7cd88`.
No contract, approved risk band, policy, authorization, receipt, trust anchor or
approval changed. No model fitting, calibration fit, protected historical scoring,
test replay, bootstrap regeneration, push, merge or owner acceptance occurred.

## Readiness after these improvements

| Area | Assessment | Remaining limitation |
| --- | --- | --- |
| Scientific rigor | Demonstrated with limitations | Strong frozen within-snapshot evidence; development reuse, conditional uncertainty, unknown duplicate dependence and no external/temporal validation remain. |
| Business impact reasoning | Strongly demonstrated | Capacity burden, independent assumed economics and causal design are explicit. **Actual business impact is not yet demonstrated.** |
| Software engineering | Strongly demonstrated | For the tested local scope; current remote container/scan checks and production scale/security are separate evidence. |
| Governance | Demonstrated with limitations | Traceable local approvals and controls; G3 conditions and G5 remain unresolved for real use. |
| Communication | Demonstrated with limitations | Written paths, science/business reasoning and visuals are prepared; actual video, visual UI review and owner acceptance remain open. |
| Interview readiness | Strongly demonstrated | Twenty evidence-backed answers, walkthrough and exactly three scoped bullets support discussion; they do not substitute for real deployment or impact experience. |

The written portfolio is ready for a skeptical senior-level discussion. Formal
Release C acceptance and any production-readiness claim remain outstanding.
