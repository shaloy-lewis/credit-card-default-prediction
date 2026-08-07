# ADR 0001: Use a batch-first scoring architecture

**Status:** Accepted
**Date:** 2026-08-07
**Decision owner:** Project owner

**Approval:** Accepted by the project owner on 2026-08-07.

## Context

The available predictors represent billing, payment, and repayment-status
history over monthly periods. The proposed operating decision is also monthly:
rank eligible existing accounts for a capacity-constrained intervention queue
after the billing snapshot closes.

The original project exposes a real-time FastAPI endpoint, but an online request
is not the natural trigger for the underlying business decision. Treating the
API as the product would leave portfolio completeness, retries, duplicate
handling, lineage, and the intervention policy unspecified.

## Decision

The primary production abstraction will be an idempotent monthly batch scorer.
For a supplied `as_of_date`, input snapshot, model version, and policy version,
it will produce:

- one score per eligible account;
- calibrated probability and risk band;
- reviewed reason categories;
- policy outcome and rank;
- model, data, and policy lineage; and
- a batch manifest containing counts, checksums, failures, and run metadata.

FastAPI remains a secondary, versioned interface for integration testing,
single-record demonstrations, and controlled what-if analysis. Streamlit will
consume the API rather than load model artifacts independently.

## Rationale

- The feature refresh and decision cadence are monthly.
- A batch queue matches a finite operations capacity and enables ranking across
  the eligible population.
- Batch manifests make completeness, retries, idempotency, and auditability
  visible.
- One shared inference package can preserve parity across training, batch, API,
  and UI paths.
- The design can map to a Databricks Job or Workflow without requiring a paid
  Azure deployment for the portfolio demonstration.

## Options considered

### Online-first API

Retain the current API as the main product and score accounts on demand.

Rejected as the primary path because it does not naturally implement monthly
population ranking, capacity allocation, or completeness controls. It remains
valuable as a secondary interface.

### Streaming inference

Score continuously from payment or transaction events.

Rejected because the dataset does not contain a real event stream and the
agreed decision cadence is monthly. A streaming demonstration would manufacture
a requirement and distract from model and governance evidence.

### Notebook-generated score file

Generate a CSV directly from an analysis notebook.

Rejected because notebooks provide weak contracts for retries, testing,
lineage, packaging, and controlled release.

## Consequences

### Positive

- The architecture is aligned to the decision rather than the existing UI.
- Operational controls such as rerun safety, quarantine, and completeness become
  testable portfolio evidence.
- The policy layer can rank the entire eligible population consistently.
- Local Docker execution remains sufficient and free.

### Costs and constraints

- A batch schema, manifest, and idempotency key must be designed and tested.
- The API and batch scorer must use the same versioned inference bundle.
- The demonstration needs a synthetic account identifier and scoring date even
  though the modelling dataset is a single historical snapshot.
- The synthetic operational envelope must be clearly separated from model
  performance claims.

## Revisit triggers

Reconsider this decision if a future dataset and product requirement provide:

- event-level inputs with a sub-month decision deadline;
- a customer interaction that requires immediate scoring;
- measured value from intra-cycle interventions; or
- throughput or availability requirements that cannot be met by scheduled batch
  processing plus the secondary API.

Until one of these conditions exists, streaming infrastructure, Kubernetes, and
an online feature store remain out of scope.
