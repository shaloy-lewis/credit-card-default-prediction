# ADR 0003: Close Release A with existing aggregate evidence

**Status:** accepted

## Context

Release A is the end-of-Week-5 defensible-model milestone. Its reviewed evidence
already covers reproducible data and splits, scientific baselines, a fixed
four-model comparison, identity-calibration diagnostics, prediction-only
uncertainty, capacity-aware evaluation, and one authorized final-test pass.

The evidence was nevertheless difficult to audit as a release. The validation
bootstrap intervals existed only in ignored runtime storage, and the data,
baseline, selection, bundle, and final-test evidence had no common authenticated
manifest. The product brief also listed stress testing in G2 even though the
Release A milestone did not authorize or publish a stress-analysis package.

## Decision

Release A will be closed through a non-computational evidence workflow. It will:

- verify the exact reviewed source artifacts and their SHA-256 digests;
- copy the existing aggregate bootstrap interval file byte-for-byte into the
  committed release dossier;
- assemble a deterministic summary, report, and evidence manifest from existing
  aggregate evidence only; and
- require an external manifest digest when verifying the dossier.

No model or row-level prediction file may be loaded. Model fitting, prediction,
bootstrap regeneration, calibration fitting, cross-validation, tuning, test-row
selection, and final-test reevaluation are prohibited.

The Release A G2 boundary is the milestone's five published criteria. Robustness,
missingness, range, category, and population-shift stress evidence is deferred to
G4/Release B. This is a retrospective correction to the documentation, not a
claim that stress testing was previously completed.

## Consequences

- Release A gains one clean-checkout-verifiable audit package without changing
  the selected model or any historical result.
- Validation uncertainty becomes reviewable as aggregate evidence. It is not
  described as final-test uncertainty.
- The one-time final test remains immutable and permanently consumed.
- G1 and G2 remain closed, G3 remains closed with conditions, and G4 remains open.
- Existing limitations remain binding: the evidence is based on a 2005 Taiwan
  snapshot and does not establish causal impact, India-specific validity,
  regulatory compliance, fairness certification, or production suitability.
