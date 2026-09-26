# Delayed-label evaluation contract

Status: design only; no longitudinal observations are supplied by the source dataset.

A future outcome feed must identify the scored account and billing cutoff, the
next-month outcome window, observation timestamp, label version and binary default
definition. Join only to the digest-verified scoring cohort; never infer outcomes
from prediction bands or intervention status. Audit demographics stay separate.

The owner must declare source-specific maturity and reporting-lag rules before
any report. Until that interval has elapsed and the complete cohort is accounted
for, report pending/unmatched/late labels and coverage; suppress performance claims.
Reject duplicate account/cutoff labels, contradictory revisions and unknown accounts.
A correction requires an explicit new label version and retained prior report.

A future mature-cohort report measures AP, Brier, calibration bins, capacity lift
and supported subgroup outcomes with denominators and missing-label counts.
Intervention effects require a separate causal design. Current validation outcomes
are historical diagnostics and cannot stand in for delayed production labels.
No automatic retraining, promotion or threshold change follows an alert.
