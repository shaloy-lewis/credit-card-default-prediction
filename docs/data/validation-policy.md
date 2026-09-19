# Data validation and quarantine policy

- **Applies to:** acquisition and canonical build for `uci_credit_default_v1`
- **Objective:** prevent unverified source bytes or incompatible records from
  entering trusted processed data

## Trust zones

| Zone | Purpose | Trust rule |
| --- | --- | --- |
| Download staging | Temporary bytes from the network | Untrusted until exact size and SHA-256 pass |
| `data/quarantine/uci_credit_default/v1/` | Content-addressed failed downloads | Evidence for diagnosis only; never an input to build or training |
| `data/raw/uci_credit_default/v1/` | Immutable verified source CSV | Write only after the complete source contract passes; a valid existing file makes fetch a no-op |
| `data/processed/uci_credit_default/v1/` | Canonical CSV and quality report | Trusted only after schema and quality checks pass |
| `data/splits/uci_credit_default/v1/` | Split assignments and manifest | Trusted only after exclusivity, stratification, and determinism checks pass |
| `configs/data/split_v1.lock.json` | Reviewed lineage reference | Committed reference copied from an approved runtime split manifest; ordinary builds do not rewrite it |

All data zones are excluded from Git. Configuration, contracts, tests, and
governance documents are committed; downloaded records are reproduced from the
pinned UCI source.

## Acquisition controls

`credit-risk data fetch` is the explicit acquisition operation. `data build`
invokes the same idempotent acquisition/no-op before processing, so either may
access the network when the raw snapshot is absent. Both use the versioned
source manifest at `configs/data/uci_credit_default_v1.json`, a 30-second request
timeout, at most three attempts for transient failures, and no force-overwrite
mode.

The fetch workflow must:

1. stream to staging rather than the trusted raw path;
2. enforce the pinned byte size and SHA-256;
3. atomically publish valid bytes to the content-addressed raw path; and
4. treat an already-valid raw CSV as an idempotent no-op.

A response body, redirect target, or filename is never trusted merely because
the request succeeded. The checksum is the identity.

`credit-risk data build --offline` requires the content-addressed raw CSV and
structurally bypasses network acquisition. It fails if the raw source is absent
or corrupt.

`credit-risk data verify` is strictly offline. It verifies the raw source,
canonical dataset, quality report, split assignments, runtime split manifest,
and reviewed lock, and exits nonzero if an expected file is missing, corrupt,
stale, or incompatible. It must not repair or redownload data implicitly.

## Validation severities

| Severity | Examples | Required outcome |
| --- | --- | --- |
| Reject | Source size/hash mismatch | Do not publish to raw; quarantine observed bytes; return nonzero |
| Reject | Missing/extra/reordered columns; row-count mismatch; non-integral values; nulls; duplicate/missing IDs; target outside `{0,1}` | Do not publish processed outputs; preserve the previous valid build; return nonzero |
| Reject | Values outside the versioned accepted domain without a reviewed contract change | Stop rather than coerce, drop, clip, or map silently |
| Accepted anomaly | `EDUCATION` 0/5/6, `MARRIAGE` 0, repayment status -2/0, negative bill amounts | Preserve and include counts in the validation report |
| Review signal | Non-ID duplicate records or material class-distribution change in a future dataset version | Report explicitly; do not infer customer duplication or delete rows automatically |

Validation findings use stable rule IDs. Severity `error` is a hard gate and
raises a data-contract error; no canonical output is promoted. Severity
`warning` preserves the source values, produces report status
`passed_with_warnings`, and exits successfully. Reports include counts and at
most 20 sample account IDs so evidence stays useful without dumping records.

The accepted anomalies are exceptions only for this checksum-pinned source.
They are not broad rules that make the validator accept the same values in an
unknown future dataset. A changed source requires a new manifest and contract
version, review of observed differences, and fresh G1 approval.

## Canonicalization policy

Canonicalization is deterministic and lossless with respect to source rows and
values. It maps UCI's generic CSV headers to explicit canonical names, normalizes
serialization, and preserves the columns needed to derive governed predictor
and audit views in the modelling phase. It must not silently impute, cap,
deduplicate, consolidate undocumented categories, or fit a statistical
transformation.

Every processed build records at least:

- source manifest version, URL, size, and source hash;
- schema/contract version and split-configuration hash;
- canonical row/column counts and target counts;
- canonical and split-assignment hashes; and
- tool/package version or code revision when available.

The runtime `split_manifest.json` records hashes produced by a build. The
committed `configs/data/split_v1.lock.json` is a reviewed reference copied from
an accepted runtime manifest, not an output silently rewritten by every build.
The lock remains the machine-readable source of truth. The versioned data card
repeats its reviewed Phase 1 fingerprints as human-readable release evidence;
future revisions must be checked against the lock.

## Quarantine and recovery

On a fingerprint failure, the observed download is stored at
`data/quarantine/uci_credit_default/v1/<observed-sha256>/data.csv`.
This preserves evidence and prevents filename collisions without blessing the
file as raw data. Logs and errors report expected versus observed size/hash and
the quarantine location; they must not include record contents.

Schema or canonical-build failures remain outside the trusted processed output.
A failed run must not partially overwrite the last valid raw snapshot,
canonical dataset, split assignments, or runtime split manifest. An ordinary
run never writes the reviewed lock. Recovery is deliberate:

1. inspect the structured failure and quarantined fingerprint;
2. confirm whether the problem is corruption, an upstream source change, or a
   proposed contract change;
3. retry unchanged only for a transient transfer problem; or
4. for a genuine source revision, create and review a new manifest/contract
   version rather than editing pinned hashes in place.

Quarantine is not an archival retention promise. A maintainer may remove it
after diagnosis using an explicit, reviewed cleanup action; the pipeline never
promotes or deletes quarantined data automatically.

## Reproducible operator workflow

```bash
uv sync --locked --all-extras --dev
uv run credit-risk data fetch
uv run credit-risk data build
uv run credit-risk data verify
uv run pytest
```

After `build`, inspect the emitted quality report and runtime split manifest
rather than relying on console success alone, then use offline `verify` to
compare the complete state with the reviewed lock. A clean checkout with the
same manifest, split configuration, lockfile, and source bytes must reproduce
the same canonical and split hashes.

The legacy `credit-risk train` workflow is compatibility-only until a later
phase consumes the canonical data, versioned splits, and experiment lineage.
Running it is not part of G1 and its artifacts must not be cited as results of
the Phase 1 data contract.

## G1 decision rule

G1 passes only when all of the following evidence is present and reviewed:

- official source attribution, licence, immutable fingerprint, and raw lineage;
- successful offline source and schema validation;
- a data card and approved feature-availability/leakage boundary;
- deterministic holdout/fold assignments with tests for exclusivity and
  reproducibility;
- runtime split manifest and reviewed lock tying source, configuration,
  canonical data, and splits together; and
- negative tests proving corruption and schema violations fail closed.

G1 failure blocks baseline modelling. G1 success authorizes the baseline phase;
it does not approve a predictive model, operating threshold, or production use.
