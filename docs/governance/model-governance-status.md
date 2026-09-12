# Model-governance status

| Gate | Status | Evidence | Remaining boundary |
| --- | --- | --- | --- |
| G0 scope approval | Closed | Product brief and ADR 0001 | Material scope changes require a new decision record |
| G1 data readiness | Closed | Data card, schema gate, lineage, split lock | Production data representativeness remains unproven |
| G2 model candidate | Closed | Four-fit selection, bundle, one consumed final test | Final-test reevaluation is permanently prohibited |
| G3 promotion review | Closed with conditions | Corrected Phase 5 model card, subgroup review, explanations, risk register, checklist | Education triggers require human review; no fairness/compliance claim |
| G4 release readiness | Open | Not started | Batch/API parity, registry, scanning, rollback, and runbooks |
| G5 ongoing review | Open | Not started | Production drift, outcomes, incidents, and approval renewal |

The API and Streamlit application are local technical demonstrations. G3 closure does not
represent external approval or production promotion. Phase 5 did not change the selected model,
API contract, container artifacts, final-test authorization, or consumed-test receipts.
