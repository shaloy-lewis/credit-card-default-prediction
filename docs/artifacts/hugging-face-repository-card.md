---
license: mit
library_name: catboost
tags:
  - tabular-classification
  - credit-risk
  - model-governance
  - mlops
---

# Governed credit-card default early-warning artifacts

This public model repository distributes exact binary artifacts for the
[`credit-card-default-prediction`](https://github.com/shaloy-lewis/credit-card-default-prediction)
portfolio project. The GitHub repository remains the source of validation
authority through reviewed manifests and SHA-256 digests.

## Selected release

`selected_v1/model.cbm` is the unchanged CatBoost winner selected by the
project's four-fit, no-tuning governance workflow. Its intended use is
human-owned outreach prioritisation. It is not approved for adverse action,
autonomous credit decisions, production use, India-specific claims, or
regulatory-compliance claims.

See the GitHub repository for the model card, Release A dossier, subgroup review,
final-test evidence, inference contract, registry controls, and limitations.

## Dataset

The project does not redistribute training data here. Data is acquired directly
from the UCI Default of Credit Card Clients dataset using a checksum-pinned
source manifest. The dataset is attributed under CC BY 4.0 and represents a
historical 2005 Taiwanese population.

## Legacy compatibility files

Files under `legacy_v1/` use Python pickle semantics and can execute arbitrary
code. Public hosting does not establish their safety. They must be retrieved
explicitly and loaded only after exact verification against the Git-tracked
legacy trust manifest.

## Reproducibility

Consumers retrieve an exact full Hugging Face commit and then independently
verify the file SHA-256 values. Normal model loaders remain local-only and never
download artifacts implicitly.
