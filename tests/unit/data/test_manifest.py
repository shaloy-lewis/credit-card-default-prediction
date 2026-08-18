"""Tests for checked-in source and split configuration contracts."""

import hashlib
import json

import pytest

from credit_risk.data.manifest import (
    DEFAULT_DATASET_MANIFEST_PATH,
    DEFAULT_SPLIT_CONFIG_PATH,
    ManifestLoadError,
    load_dataset_manifest,
    load_split_config,
)
from tests.unit.data.helpers import manifest_payload, split_payload, write_json

# Update this digest only after an official rebuild, offline verification, and
# explicit review of the complete generated split lock.
REVIEWED_SPLIT_LOCK_SHA256 = "b2312380fa46924ca414acbcfef63b0435d1321083e87e4df5ec04f18736093d"


def test_checked_in_dataset_manifest_pins_official_csv() -> None:
    manifest = load_dataset_manifest()

    assert manifest.schema_version == "1.0.0"
    assert manifest.dataset_id == "uci_credit_default"
    assert manifest.dataset_version == "v1"
    assert manifest.source.url == "https://archive.ics.uci.edu/static/public/350/data.csv"
    assert manifest.source.filename == "data.csv"
    assert manifest.source.size_bytes == 2_897_080
    assert manifest.source.sha256 == (
        "45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0"
    )
    assert manifest.expectations.source_columns == (
        "ID",
        *(f"X{index}" for index in range(1, 24)),
        "Y",
    )
    assert manifest.expectations.target_counts == {"0": 23_364, "1": 6_636}


def test_manifest_canonical_mapping_has_exact_order_and_integer_types() -> None:
    manifest = load_dataset_manifest()
    columns = manifest.canonical_contract.columns

    assert tuple(column.source_name for column in columns) == manifest.expectations.source_columns
    assert columns[0].canonical_name == "account_id"
    assert columns[-1].canonical_name == "default_next_month"
    assert all(column.logical_dtype == "integer" for column in columns)


def test_checked_in_split_config_locks_protocol_and_expected_counts() -> None:
    config = load_split_config()

    assert config.holdout.test_fraction == 0.2
    assert config.holdout.random_state == 42
    assert config.cross_validation.n_splits == 5
    assert config.cross_validation.n_repeats == 3
    assert config.cross_validation.random_state == 42
    assert config.expected_counts.development.total == 24_000
    assert config.expected_counts.development.target_counts == {"0": 18_691, "1": 5_309}
    assert config.expected_counts.test.total == 6_000
    assert config.expected_counts.test.target_counts == {"0": 4_673, "1": 1_327}


def test_checked_in_split_lock_pins_reviewed_g1_evidence() -> None:
    lock_path = DEFAULT_SPLIT_CONFIG_PATH.with_suffix(".lock.json")
    lock_bytes = lock_path.read_bytes()

    assert hashlib.sha256(lock_bytes).hexdigest() == REVIEWED_SPLIT_LOCK_SHA256
    lock = json.loads(lock_bytes)

    assert set(lock) == {
        "algorithm",
        "assignment",
        "config",
        "dataset_id",
        "dataset_version",
        "lineage",
        "schema_version",
    }
    assert lock["dataset_id"] == "uci_credit_default"
    assert lock["dataset_version"] == "v1"
    assert lock["schema_version"] == "1.0.0"
    assert lock["lineage"] == {
        "canonical_sha256": ("75b2a746781a584b0456f843f1f269190b51e90983cba44c4ed6c4a8685e6c1c"),
        "source_sha256": ("45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0"),
        "split_config_sha256": hashlib.sha256(DEFAULT_SPLIT_CONFIG_PATH.read_bytes()).hexdigest(),
    }
    assert lock["algorithm"] == {
        "cross_validation": "repeated_stratified_k_fold",
        "holdout": "stratified_shuffle_split",
        "scikit_learn_version": "1.4.2",
    }
    assert lock["config"]["holdout"] == {"random_state": 42, "test_fraction": 0.2}
    assert lock["config"]["cross_validation"] == {
        "n_repeats": 3,
        "n_splits": 5,
        "random_state": 42,
    }
    assert lock["assignment"]["rows"] == 30_000
    assert lock["assignment"]["partition_counts"] == {
        "development": {"rows": 24_000, "target_counts": {"0": 18_691, "1": 5_309}},
        "test": {"rows": 6_000, "target_counts": {"0": 4_673, "1": 1_327}},
    }
    assert lock["assignment"]["sha256"] == (
        "2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e"
    )
    assert "timestamp" not in json.dumps(lock).lower()


def test_manifest_rejects_non_https_source(tmp_path) -> None:
    payload = json.loads(DEFAULT_DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["source"]["url"] = "http://example.test/data.csv"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestLoadError, match="absolute HTTPS URL"):
        load_dataset_manifest(manifest_path)


def test_manifest_rejects_cross_platform_filename_traversal(tmp_path) -> None:
    payload = json.loads(DEFAULT_DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["source"]["filename"] = "..\\data.csv"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestLoadError, match="plain CSV filename"):
        load_dataset_manifest(manifest_path)


def test_manifest_rejects_mapping_order_drift(tmp_path) -> None:
    payload = json.loads(DEFAULT_DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    columns = payload["canonical_contract"]["columns"]
    columns[0], columns[1] = columns[1], columns[0]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestLoadError, match="exactly match source_columns order"):
        load_dataset_manifest(manifest_path)


def test_split_config_rejects_inconsistent_expected_counts(tmp_path) -> None:
    payload_path = DEFAULT_DATASET_MANIFEST_PATH.parent / "split_v1.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["expected_counts"]["test"]["total"] = 5_999
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestLoadError, match="target_counts must sum to total"):
        load_split_config(split_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", "https://user:secret@example.test/data.csv"),
        ("url", "data.csv"),
        ("filename", ""),
        ("filename", "../data.csv"),
        ("filename", "data.txt"),
    ],
)
def test_manifest_rejects_unsafe_source_locations(tmp_path, field: str, value: str) -> None:
    payload = manifest_payload()
    payload["source"][field] = value
    path = tmp_path / "manifest.json"
    write_json(path, payload)

    with pytest.raises(ManifestLoadError):
        load_dataset_manifest(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["expectations"].update(column_count=24),
            "column_count must equal",
        ),
        (
            lambda payload: payload["expectations"]["source_columns"].__setitem__(1, "ID"),
            "source_columns must be unique",
        ),
        (
            lambda payload: payload["expectations"].update(target_column="missing"),
            "target_column must be present",
        ),
        (
            lambda payload: payload["expectations"].update(target_counts={"0": 30_000}),
            "exactly the binary labels",
        ),
        (
            lambda payload: payload["expectations"].update(target_counts={"0": 30_000, "1": 0}),
            "at least one record",
        ),
        (
            lambda payload: payload["expectations"].update(target_counts={"0": 23_363, "1": 6_636}),
            "must sum to row_count",
        ),
    ],
)
def test_manifest_rejects_internally_inconsistent_expectations(
    tmp_path, mutate, message: str
) -> None:
    payload = manifest_payload()
    mutate(payload)
    path = tmp_path / "manifest.json"
    write_json(path, payload)

    with pytest.raises(ManifestLoadError, match=message):
        load_dataset_manifest(path)


@pytest.mark.parametrize(
    ("index", "field", "replacement", "message"),
    [
        (1, "source_name", "ID", "source names must be unique"),
        (1, "canonical_name", "account_id", "target names must be unique"),
        (1, "canonical_name", "different_name", "official canonical mapping differs"),
        (1, "logical_dtype", "number", "official canonical columns must use integer"),
    ],
)
def test_manifest_rejects_canonical_contract_drift(
    tmp_path,
    index: int,
    field: str,
    replacement: str,
    message: str,
) -> None:
    payload = manifest_payload()
    payload["canonical_contract"]["columns"][index][field] = replacement
    path = tmp_path / "manifest.json"
    write_json(path, payload)

    with pytest.raises(ManifestLoadError, match=message):
        load_dataset_manifest(path)


@pytest.mark.parametrize("field", ["dataset_page_url", "license_url"])
def test_manifest_rejects_non_https_documentation_urls(tmp_path, field: str) -> None:
    payload = manifest_payload()
    payload[field] = "http://example.test/docs"
    path = tmp_path / "manifest.json"
    write_json(path, payload)

    with pytest.raises(ManifestLoadError, match="documentation URLs"):
        load_dataset_manifest(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["expected_counts"]["development"].update(
                target_counts={"0": 24_000}
            ),
            "exactly the binary labels",
        ),
        (
            lambda payload: payload["expected_counts"]["development"].update(
                target_counts={"0": 24_000, "1": 0}
            ),
            "at least one record",
        ),
        (
            lambda payload: payload["expected_counts"].update(total=29_999),
            "development and test counts must sum",
        ),
        (
            lambda payload: payload.update(sort_by=[]),
            "sort_by must contain",
        ),
        (
            lambda payload: payload.update(sort_by=["account_id", "account_id"]),
            "sort_by columns must be unique",
        ),
        (
            lambda payload: payload["holdout"].update(test_fraction=0.25),
            "expected holdout count must match",
        ),
    ],
)
def test_split_config_rejects_governance_inconsistencies(tmp_path, mutate, message: str) -> None:
    payload = split_payload()
    mutate(payload)
    path = tmp_path / "split.json"
    write_json(path, payload)

    with pytest.raises(ManifestLoadError, match=message):
        load_split_config(path)


@pytest.mark.parametrize("contents", ["not-json", "[]"])
def test_loaders_wrap_invalid_json_as_actionable_manifest_error(tmp_path, contents: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ManifestLoadError, match="Unable to load"):
        load_dataset_manifest(path)


def test_loader_wraps_missing_file_as_actionable_manifest_error(tmp_path) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(ManifestLoadError, match="Unable to load"):
        load_split_config(path)
