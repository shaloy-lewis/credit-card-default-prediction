# Model-governance status

| Gate | Status | Evidence | Remaining boundary |
| --- | --- | --- | --- |
| G0 scope approval | Closed | Product brief and ADR 0001 | Material scope changes require a new decision record |
| G1 data readiness | Closed | Data card, schema gate, lineage, split lock | Production data representativeness remains unproven |
| G2 model candidate | Closed | Authenticated Release A dossier binding four-fit selection, bundle, validation-only uncertainty, capacity evidence, and one consumed final test | Final-test reevaluation is permanently prohibited |
| G3 promotion review | Closed with conditions | Corrected Phase 5 model card, subgroup review, explanations, risk register, checklist | Education triggers require human review; no fairness/compliance claim |
| G4 release readiness | Open — owner sign-off pending | Authenticated Phase 6 parity, Phase 7 SQLite registry/rollback, Phase 8 persistent platform, robustness, monitoring, service acceptance and eight incident drills | Explicit owner approval of the verified dossier and proposed restrictions for local portfolio scope |
| G5 ongoing review | Open | Not started | Production drift, outcomes, incidents, and approval renewal |

The API and Streamlit application are local technical demonstrations. G3 closure does not
represent external approval or production promotion. Phase 5 did not change the selected model,
API contract, container artifacts, final-test authorization, or consumed-test receipts.
Release A is complete as a defensible-model milestone, not as a production-readiness
claim. Its authenticated dossier is under `reports/releases/release_a_v1/`.
Phase 6 adds authenticated technical parity evidence without changing the model,
policy, authorization, or receipts. Phase 7 then demonstrates local registry
promotion and rollback for two transparent revisions of those identical bytes.
Phase 8 adds persistent local PostgreSQL, MinIO, and MLflow prerequisites without
changing those bytes or their approvals. None of these phases constitutes external
production release approval. The official Release B Linux run passed every
mandatory live control. The final dossier and its exact-commit CI are independently verified. G4 remains
open until the owner approves its exact digest and all operating restrictions.
See the [owner review candidate](../reviews/release-b-owner-review.md).
