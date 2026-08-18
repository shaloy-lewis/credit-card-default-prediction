"""End-to-end evidence for the offline governed-data workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import credit_risk.data.workflow as workflow
from credit_risk.data.acquisition import resolve_raw_data_path
from credit_risk.data.manifest import load_dataset_manifest
from credit_risk.data.schema import SOURCE_TO_CANONICAL
from credit_risk.data.workflow import DataWorkflowError, build_dataset, verify_dataset

pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class _FixtureContract:
    data_root: Path
    manifest_path: Path
    split_config_path: Path


def test_offline_build_and_repeated_verification_are_idempotent(tmp_path: Path) -> None:
    contract = _write_fixture_contract(tmp_path, _source_frame())

    built = build_dataset(
        contract.data_root,
        contract.manifest_path,
        contract.split_config_path,
        offline=True,
    )
    assert set(built.changed_paths) == {
        built.paths.canonical,
        built.paths.quality_report,
        built.paths.split_assignments,
        built.paths.split_manifest,
    }
    assert not built.source_downloaded
    assert not built.reviewed_lock_verified

    reviewed_lock = contract.split_config_path.with_suffix(".lock.json")
    reviewed_lock.write_bytes(built.paths.split_manifest.read_bytes())
    mtimes = {
        path: path.stat().st_mtime_ns
        for path in (
            built.paths.canonical,
            built.paths.quality_report,
            built.paths.split_assignments,
            built.paths.split_manifest,
        )
    }

    first_verification = verify_dataset(
        contract.data_root,
        contract.manifest_path,
        contract.split_config_path,
    )
    second_verification = verify_dataset(
        contract.data_root,
        contract.manifest_path,
        contract.split_config_path,
    )
    rebuilt = build_dataset(
        contract.data_root,
        contract.manifest_path,
        contract.split_config_path,
        offline=True,
    )

    assert first_verification == second_verification
    assert first_verification.reviewed_lock_verified
    assert rebuilt.reviewed_lock_verified
    assert rebuilt.changed_paths == ()
    assert {path: path.stat().st_mtime_ns for path in mtimes} == mtimes


def test_verification_requires_an_exact_reviewed_split_lock(tmp_path: Path) -> None:
    contract = _write_fixture_contract(tmp_path, _source_frame())
    built = build_dataset(
        contract.data_root,
        contract.manifest_path,
        contract.split_config_path,
        offline=True,
    )

    with pytest.raises(DataWorkflowError, match="Reviewed split lock is missing"):
        verify_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
        )

    reviewed_lock = contract.split_config_path.with_suffix(".lock.json")
    reviewed_lock.write_text('{"reviewed": false}\n', encoding="utf-8", newline="\n")
    with pytest.raises(DataWorkflowError, match="differs from the reviewed lock"):
        verify_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
        )

    reviewed_lock.write_bytes(built.paths.split_manifest.read_bytes())
    assert verify_dataset(
        contract.data_root,
        contract.manifest_path,
        contract.split_config_path,
    ).reviewed_lock_verified


def test_shuffled_source_rows_have_identical_canonical_and_assignment_hashes(
    tmp_path: Path,
) -> None:
    source = _source_frame()
    first = _write_fixture_contract(tmp_path / "ordered", source)
    second = _write_fixture_contract(
        tmp_path / "shuffled",
        source.sample(frac=1.0, random_state=7).reset_index(drop=True),
    )

    ordered = build_dataset(
        first.data_root,
        first.manifest_path,
        first.split_config_path,
        offline=True,
    )
    shuffled = build_dataset(
        second.data_root,
        second.manifest_path,
        second.split_config_path,
        offline=True,
    )

    assert ordered.source_sha256 != shuffled.source_sha256
    assert ordered.canonical_sha256 == shuffled.canonical_sha256
    assert ordered.assignment_sha256 == shuffled.assignment_sha256
    assert ordered.paths.canonical.read_bytes() == shuffled.paths.canonical.read_bytes()
    assert (
        ordered.paths.split_assignments.read_bytes()
        == shuffled.paths.split_assignments.read_bytes()
    )


def test_failed_contract_build_preserves_last_valid_products(tmp_path: Path) -> None:
    valid_contract = _write_fixture_contract(tmp_path, _source_frame())
    built = build_dataset(
        valid_contract.data_root,
        valid_contract.manifest_path,
        valid_contract.split_config_path,
        offline=True,
    )
    products = (
        built.paths.canonical,
        built.paths.quality_report,
        built.paths.split_assignments,
        built.paths.split_manifest,
    )
    valid_bytes = {path: path.read_bytes() for path in products}

    invalid_source = _source_frame()
    invalid_source.loc[0, "X18"] = -1
    invalid_contract = _write_fixture_contract(
        tmp_path,
        invalid_source,
        manifest_name="invalid_manifest.json",
    )

    with pytest.raises(DataWorkflowError, match="schema.payment_amount.nonnegative"):
        build_dataset(
            invalid_contract.data_root,
            invalid_contract.manifest_path,
            invalid_contract.split_config_path,
            offline=True,
        )
    with pytest.raises(DataWorkflowError, match="schema.payment_amount.nonnegative"):
        build_dataset(
            invalid_contract.data_root,
            invalid_contract.manifest_path,
            invalid_contract.split_config_path,
            offline=True,
        )

    assert {path: path.read_bytes() for path in products} == valid_bytes
    reports = list(
        (tmp_path / "data" / "quarantine" / "fixture_credit_default" / "v1").glob(
            "validation/*/quality_report.json"
        )
    )
    assert len(reports) == 1
    failure_report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert failure_report["status"] == "failed"
    assert max(len(issue["account_ids"]) for issue in failure_report["issues"]) <= 20


def test_offline_commands_never_call_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write_fixture_contract(tmp_path, _source_frame())

    def unexpected_fetch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline build attempted acquisition")

    monkeypatch.setattr(workflow, "fetch_source", unexpected_fetch)
    result = build_dataset(
        contract.data_root,
        contract.manifest_path,
        contract.split_config_path,
        offline=True,
    )
    assert result.paths.canonical.is_file()


def test_offline_commands_fail_without_or_with_corrupt_raw_data(tmp_path: Path) -> None:
    contract = _write_fixture_contract(tmp_path, _source_frame())
    manifest = load_dataset_manifest(contract.manifest_path)
    raw_path = resolve_raw_data_path(manifest, contract.data_root)
    raw_path.unlink()

    with pytest.raises(DataWorkflowError, match="Offline build requires"):
        build_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
            offline=True,
        )
    with pytest.raises(DataWorkflowError, match="Offline source verification failed"):
        verify_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
        )

    raw_path.write_bytes(b"corrupt")
    with pytest.raises(DataWorkflowError, match="offline source verification.*integrity check"):
        build_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
            offline=True,
        )
    assert raw_path.read_bytes() == b"corrupt"


def test_verification_reports_missing_changed_and_nonfile_products(tmp_path: Path) -> None:
    contract = _write_fixture_contract(tmp_path, _source_frame())
    built = build_dataset(
        contract.data_root,
        contract.manifest_path,
        contract.split_config_path,
        offline=True,
    )
    contract.split_config_path.with_suffix(".lock.json").write_bytes(
        built.paths.split_manifest.read_bytes()
    )
    built.paths.canonical.write_bytes(b"changed")
    built.paths.quality_report.unlink()
    built.paths.split_assignments.unlink()
    built.paths.split_assignments.mkdir()

    with pytest.raises(DataWorkflowError, match="missing or changed generated products") as error:
        verify_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
        )
    assert "has sha256=" in str(error.value)
    assert "is missing" in str(error.value)
    assert "is not a regular file" in str(error.value)


def test_workflow_rejects_inconsistent_governed_configs(tmp_path: Path) -> None:
    contract = _write_fixture_contract(tmp_path, _source_frame())

    split_payload = _read_json(contract.split_config_path)
    split_payload["dataset_version"] = "different"
    _write_json(contract.split_config_path, split_payload)
    with pytest.raises(DataWorkflowError, match="different dataset snapshots"):
        build_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
            offline=True,
        )

    contract = _write_fixture_contract(tmp_path, _source_frame())
    manifest_payload = _read_json(contract.manifest_path)
    manifest_payload["canonical_contract"]["columns"][1]["canonical_name"] = (
        "different_credit_limit"
    )
    _write_json(contract.manifest_path, manifest_payload)
    with pytest.raises(DataWorkflowError, match="mapping does not match canonical schema"):
        build_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
            offline=True,
        )

    contract = _write_fixture_contract(tmp_path, _source_frame())
    split_payload = _read_json(contract.split_config_path)
    split_payload["id_column"] = "customer_id"
    split_payload["sort_by"] = ["customer_id"]
    _write_json(contract.split_config_path, split_payload)
    with pytest.raises(DataWorkflowError, match="must use account_id"):
        build_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
            offline=True,
        )

    with pytest.raises(DataWorkflowError, match="Unable to load"):
        build_dataset(
            contract.data_root,
            tmp_path / "missing-manifest.json",
            contract.split_config_path,
            offline=True,
        )


def test_workflow_wraps_parser_contract_and_split_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write_fixture_contract(tmp_path / "parser", _source_frame())
    _replace_source_bytes(contract, b"\xff\xfe\xfd")
    with pytest.raises(DataWorkflowError, match="Unable to parse the pinned source CSV"):
        build_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
            offline=True,
        )

    contract = _write_fixture_contract(tmp_path / "contract", _source_frame())
    monkeypatch.setattr(
        workflow,
        "canonicalize_source_frame",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad canonical config")),
    )
    with pytest.raises(DataWorkflowError, match="Canonical contract configuration is invalid"):
        build_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
            offline=True,
        )
    monkeypatch.undo()

    contract = _write_fixture_contract(tmp_path / "split", _source_frame())
    monkeypatch.setattr(
        workflow,
        "build_split_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("sklearn rejected input")),
    )
    with pytest.raises(DataWorkflowError, match="Sealed split validation failed"):
        build_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
            offline=True,
        )


def test_build_rejects_nonfile_lock_and_product_destination(tmp_path: Path) -> None:
    contract = _write_fixture_contract(tmp_path / "lock", _source_frame())
    reviewed_lock = contract.split_config_path.with_suffix(".lock.json")
    reviewed_lock.mkdir()
    with pytest.raises(DataWorkflowError, match="Reviewed split lock is not a regular file"):
        build_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
            offline=True,
        )

    contract = _write_fixture_contract(tmp_path / "destination", _source_frame())
    canonical_destination = (
        contract.data_root / "processed" / "fixture_credit_default" / "v1" / "canonical.csv"
    )
    canonical_destination.mkdir(parents=True)
    with pytest.raises(DataWorkflowError, match="destination is not a regular file"):
        build_dataset(
            contract.data_root,
            contract.manifest_path,
            contract.split_config_path,
            offline=True,
        )


@pytest.mark.parametrize("existing_destination", [False, True])
def test_failed_promotion_rolls_back_prior_product(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_destination: bool,
) -> None:
    first_destination = tmp_path / "products" / "first.txt"
    second_destination = tmp_path / "products" / "second.txt"
    first_stage = tmp_path / "stage" / "first.txt"
    second_stage = tmp_path / "stage" / "second.txt"
    first_stage.parent.mkdir()
    first_stage.write_bytes(b"new-first")
    second_stage.write_bytes(b"new-second")
    if existing_destination:
        first_destination.parent.mkdir()
        first_destination.write_bytes(b"old-first")

    real_writer = workflow.write_bytes_atomically
    write_count = 0

    def fail_second_write(path: str | Path, content: bytes) -> str:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise OSError("simulated promotion failure")
        return real_writer(path, content)

    monkeypatch.setattr(workflow, "write_bytes_atomically", fail_second_write)
    with pytest.raises(DataWorkflowError, match="Unable to promote staged data products"):
        workflow._promote_staged({first_destination: first_stage, second_destination: second_stage})

    if existing_destination:
        assert first_destination.read_bytes() == b"old-first"
    else:
        assert not first_destination.exists()
    assert not second_destination.exists()


def _source_frame() -> pd.DataFrame:
    rows = 100
    values: dict[str, list[int]] = {
        "ID": list(range(1, rows + 1)),
        "X1": [50_000 + index for index in range(rows)],
        "X2": [1 + index % 2 for index in range(rows)],
        "X3": [1 + index % 2 for index in range(rows)],
        "X4": [1 + index % 2 for index in range(rows)],
        "X5": [21 + index % 50 for index in range(rows)],
    }
    for source_column in (f"X{index}" for index in range(6, 12)):
        values[source_column] = [-1] * rows
    for source_column in (f"X{index}" for index in range(12, 18)):
        values[source_column] = [1_000] * rows
    for source_column in (f"X{index}" for index in range(18, 24)):
        values[source_column] = [100] * rows
    values["Y"] = [0] * 50 + [1] * 50
    return pd.DataFrame(values, columns=list(SOURCE_TO_CANONICAL))


def _write_fixture_contract(
    root: Path,
    source: pd.DataFrame,
    *,
    manifest_name: str = "manifest.json",
) -> _FixtureContract:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
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
    split_config = {
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

    manifest_path = config_dir / manifest_name
    split_config_path = config_dir / "split.json"
    _write_json(manifest_path, manifest)
    _write_json(split_config_path, split_config)
    loaded_manifest = load_dataset_manifest(manifest_path)
    raw_path = resolve_raw_data_path(loaded_manifest, data_root)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(source_bytes)
    return _FixtureContract(data_root, manifest_path, split_config_path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _replace_source_bytes(contract: _FixtureContract, content: bytes) -> None:
    manifest_payload = _read_json(contract.manifest_path)
    source_payload = manifest_payload["source"]
    assert isinstance(source_payload, dict)
    source_payload["size_bytes"] = len(content)
    source_payload["sha256"] = hashlib.sha256(content).hexdigest()
    _write_json(contract.manifest_path, manifest_payload)
    raw_path = resolve_raw_data_path(
        load_dataset_manifest(contract.manifest_path),
        contract.data_root,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
