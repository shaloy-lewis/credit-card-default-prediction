"""Small deterministic contracts shared by governed-data unit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.data.acquisition import resolve_raw_data_path
from credit_risk.data.manifest import DatasetManifest, SplitConfig, load_dataset_manifest
from credit_risk.data.schema import SOURCE_COLUMNS, SOURCE_TO_CANONICAL, ValidationExpectations


def source_frame(row_count: int = 100) -> pd.DataFrame:
    """Return a valid balanced source-shaped frame with sequential IDs."""

    ids = np.arange(1, row_count + 1)
    values: dict[str, object] = {
        "ID": ids,
        "X1": 50_000 + ids,
        "X2": np.where(ids % 2, 1, 2),
        "X3": np.where(ids % 2, 1, 2),
        "X4": np.where(ids % 2, 1, 2),
        "X5": 21 + ids % 50,
    }
    values.update({f"X{index}": np.full(row_count, -1) for index in range(6, 12)})
    values.update({f"X{index}": 1_000 + ids for index in range(12, 18)})
    values.update({f"X{index}": 100 + ids for index in range(18, 24)})
    values["Y"] = np.resize(np.array([0, 1], dtype=int), row_count)
    return pd.DataFrame(values, columns=SOURCE_COLUMNS)


def expectations(row_count: int = 100) -> ValidationExpectations:
    """Return validation expectations matching :func:`source_frame`."""

    zero_count = (row_count + 1) // 2
    return ValidationExpectations(
        row_count=row_count,
        target_counts=((0, zero_count), (1, row_count - zero_count)),
        account_id_min=1,
        account_id_max=row_count,
    )


def split_config(row_count: int = 100) -> SplitConfig:
    """Return the sealed 80/20 and 5x3 protocol for balanced fixture rows."""

    if row_count != 100:
        raise ValueError("The shared split fixture currently supports exactly 100 rows.")
    return SplitConfig.model_validate(
        {
            "config_version": 1,
            "dataset_id": "fixture_credit_default",
            "dataset_version": "v1",
            "id_column": "account_id",
            "target_column": "default_next_month",
            "sort_by": ["account_id"],
            "holdout": {
                "method": "stratified_shuffle_split",
                "test_fraction": 0.2,
                "random_state": 42,
            },
            "cross_validation": {
                "method": "repeated_stratified_k_fold",
                "n_splits": 5,
                "n_repeats": 3,
                "random_state": 42,
            },
            "expected_counts": {
                "total": 100,
                "development": {"total": 80, "target_counts": {"0": 40, "1": 40}},
                "test": {"total": 20, "target_counts": {"0": 10, "1": 10}},
            },
        }
    )


def canonical_split_frame() -> pd.DataFrame:
    """Return the two columns consumed by the split builder."""

    return pd.DataFrame(
        {
            "account_id": range(1, 101),
            "default_next_month": [0, 1] * 50,
        }
    )


def manifest_payload() -> dict[str, Any]:
    """Return a mutable copy of the checked-in official manifest JSON."""

    return json.loads(Path("configs/data/uci_credit_default_v1.json").read_text(encoding="utf-8"))


def split_payload() -> dict[str, Any]:
    """Return a mutable copy of the checked-in split JSON."""

    return json.loads(Path("configs/data/split_v1.json").read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    """Write stable JSON for a temporary test contract."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def acquisition_manifest(payload: bytes, *, sha256: str | None = None) -> DatasetManifest:
    """Return a synthetic manifest that pins ``payload``."""

    official = load_dataset_manifest()
    return official.model_copy(
        update={
            "dataset_id": "synthetic_acquisition_test",
            "dataset_version": "v1",
            "source": official.source.model_copy(
                update={
                    "url": "https://example.test/data.csv",
                    "size_bytes": len(payload),
                    "sha256": sha256 or hashlib.sha256(payload).hexdigest(),
                }
            ),
        }
    )


def write_workflow_contract(root: Path, source: pd.DataFrame) -> tuple[Path, Path, Path]:
    """Write a complete offline fixture and return data/manifest/split paths."""

    config_dir = root / "configs"
    data_root = root / "data"
    source_bytes = source.to_csv(index=False, lineterminator="\n").encode("utf-8")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "fixture_credit_default",
        "dataset_version": "v1",
        "title": "Synthetic credit default workflow fixture",
        "repository_id": 350,
        "dataset_page_url": "https://example.invalid/datasets/350",
        "doi": "10.0000/example",
        "creator": "Test fixture",
        "citation": "Synthetic test fixture.",
        "license_name": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source": {
            "url": "https://example.invalid/data.csv",
            "filename": "data.csv",
            "media_type": "text/csv",
            "size_bytes": len(source_bytes),
            "sha256": source_sha256,
        },
        "expectations": {
            "row_count": 100,
            "column_count": len(SOURCE_TO_CANONICAL),
            "source_columns": list(SOURCE_TO_CANONICAL),
            "target_column": "Y",
            "target_counts": {"0": 50, "1": 50},
        },
        "canonical_contract": {
            "columns": [
                {
                    "source_name": source_name,
                    "canonical_name": canonical_name,
                    "logical_dtype": "integer",
                }
                for source_name, canonical_name in SOURCE_TO_CANONICAL.items()
            ]
        },
    }
    manifest_path = config_dir / "manifest.json"
    split_path = config_dir / "split.json"
    write_json(manifest_path, manifest)
    split = split_payload()
    split["dataset_id"] = "fixture_credit_default"
    split["dataset_version"] = "v1"
    split["expected_counts"] = {
        "total": 100,
        "development": {"total": 80, "target_counts": {"0": 40, "1": 40}},
        "test": {"total": 20, "target_counts": {"0": 10, "1": 10}},
    }
    write_json(split_path, split)
    raw_path = resolve_raw_data_path(load_dataset_manifest(manifest_path), data_root)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(source_bytes)
    return data_root, manifest_path, split_path
