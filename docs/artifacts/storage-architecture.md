# Selected-model storage and retrieval

## Responsibility boundaries

| System | Responsibility |
| --- | --- |
| GitHub | Source, tests, configuration, reviewed manifests, cryptographic digests, documentation, and aggregate evidence |
| UCI | Canonical public training dataset |
| Hugging Face Hub | Public distribution of the exact reviewed selected-model binary from a pinned commit |
| Local ignored paths | Materialised model files, UCI data products, caches, quarantine, MLflow, and deployment state |
| MLflow/PostgreSQL/MinIO | Local release registration, aliases, promotion, deployment, and rollback |

Hugging Face repository `ShaloyL/credit-card-default-prediction` is pinned to
immutable commit `f73ca4ee7a2c2d2ea51741e75fccf66ae7a4a640`. Hugging Face is a transport,
not a validation authority. The only supported artifact is
`selected_v1/model.cbm`, and it is accepted only when its bytes match
`models/selected_v1/manifest.json` through the selected-only v2 distribution
lock.

The pinned remote revision also contains retired legacy files published during
the v1 migration. They are inert historical bytes: the current repository has
no contract, command, loader, or CI path that downloads or deserializes them.

## Fresh checkout: inference only

```bash
uv sync --locked --extra artifacts
uv run credit-risk artifacts pull
uv run credit-risk artifacts verify
uv run uvicorn api:app --host 0.0.0.0 --port 8080
```

This path does not download or build the UCI dataset. Pulling an already valid
artifact is a network-free, no-rewrite operation. `--offline` permits only a
cache hit; `verify` is always offline and never deserializes the model.

## Data and evidence reproduction

```bash
uv sync --locked --extra artifacts --extra data --extra modeling --dev
uv run credit-risk artifacts pull
uv run credit-risk data fetch
uv run credit-risk data build
uv run credit-risk data verify
```

The UCI source is independently pinned by URL, size, SHA-256, schema, row count,
and target distribution. It is not mirrored to Hugging Face.

## Maintainer publication

Set `HF_TOKEN` or authenticate through the standard Hugging Face client, then:

```bash
uv run credit-risk artifacts publish \
  --repo-id OWNER/credit-card-default-prediction
```

Publication verifies the local selected-model bytes first, refuses conflicting
remote paths, uploads no training data, verifies the returned full commit
anonymously, and writes an ignored candidate lock for review. The reviewed lock
at `configs/artifacts/hf_distribution_v2.lock.json` continues to reference the
already published immutable revision; this cleanup does not modify that remote
repository.

There is no force option, no token command-line argument, and no mutable `main`
revision in the retrieval contract.
