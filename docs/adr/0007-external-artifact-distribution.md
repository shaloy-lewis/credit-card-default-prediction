# ADR 0007: External binary-artifact distribution

## Status

Accepted for implementation after Phase 8.

## Context

The reviewed CatBoost winner and two retired compatibility pickles were stored
directly in Git. Their current size is modest, but continuing that pattern makes
future model releases scale poorly and obscures the separation between source,
data, binary distribution, and model release control.

The selected model already has an authoritative Git-tracked manifest and is
referenced by immutable scientific, governance, inference, registry, and final
test evidence. Those references and bytes cannot be rewritten.

## Decision

- GitHub retains code, configuration, manifests, checksums, tests, documentation,
  lockfiles, and aggregate evidence.
- UCI remains the sole canonical dataset source.
- A public Hugging Face model repository distributes exact reviewed binary
  artifacts from a full immutable commit SHA.
- An explicit `credit-risk artifacts pull` step materialises binaries at their
  historical local paths. Application loaders remain local-only and never make
  network requests.
- Git-tracked manifests authenticate downloaded bytes. Hugging Face provides
  transport and storage, not trust.
- Legacy pickles remain explicit compatibility artifacts. Their bytes must match
  the reviewed legacy manifest before deserialization; public hosting does not
  make pickle safe.
- Docker downloads and verifies the public selected model at build time. Runtime
  images remain self-contained and do not include the Hub client or its cache.
- MLflow, PostgreSQL, and MinIO retain their separate registration, promotion,
  deployment, and rollback roles.
- Git history is not rewritten. Only future tracking of the three binaries ends.

## Consequences

A fresh clone needs the optional `artifacts` dependency and an explicit pull
before local inference or evidence workflows that hash the physical model. Data
reproduction remains independent. Already-built images continue to work during
a Hub outage; new downloads fail closed rather than falling back to mutable or
unauthenticated sources.
