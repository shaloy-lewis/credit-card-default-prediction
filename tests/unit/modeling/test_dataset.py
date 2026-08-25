"""Tests for the verified, development-only modelling data boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.data.manifest import (
    DatasetManifest,
    SplitConfig,
    load_dataset_manifest,
    load_split_config,
)
from credit_risk.data.workflow import DataWorkflowResult, build_dataset, verify_dataset
from credit_risk.modeling.contracts import (
    AUDIT_COLUMNS,
    PREDICTOR_COLUMNS,
    FeatureContract,
    load_feature_contract,
)
from credit_risk.modeling.dataset import (
    GovernedDevelopmentData,
    ModelingDataError,
    _build_development_view,
    _file_sha256,
    _read_csv,
    _read_verified_csv,
    _validate_contract_parity,
    load_governed_development_data,
)
from tests.unit.data.helpers import source_frame, write_json, write_workflow_contract


def _write_synthetic_modeling_contract(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data_root, manifest_path, split_path = write_workflow_contract(tmp_path, source_frame())
    result = build_dataset(data_root, manifest_path, split_path, offline=True)
    lock_path = split_path.with_suffix(".lock.json")
    lock_path.write_bytes(result.paths.split_manifest.read_bytes())

    official = json.loads(
        Path("configs/modeling/feature_contract_v1.json").read_text(encoding="utf-8")
    )
    official["contract_id"] = "fixture_credit_default_features_v1"
    official["dataset"] = {
        "dataset_id": "fixture_credit_default",
        "dataset_version": "v1",
    }
    official["expected_development"] = {
        "rows": 80,
        "target_counts": {"0": 40, "1": 40},
    }
    official["lineage"] = {
        "source_sha256": result.source_sha256,
        "dataset_manifest_sha256": result.dataset_manifest_sha256,
        "canonical_sha256": result.canonical_sha256,
        "split_config_sha256": result.split_config_sha256,
        "assignment_sha256": result.assignment_sha256,
        "reviewed_split_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    }
    feature_path = tmp_path / "configs" / "feature_contract.json"
    write_json(feature_path, official)
    return data_root, manifest_path, split_path, feature_path


@dataclass(frozen=True)
class ViewInputs:
    canonical: pd.DataFrame
    assignments: pd.DataFrame
    contract: FeatureContract
    manifest: DatasetManifest
    split_config: SplitConfig
    verification: DataWorkflowResult
    feature_contract_sha256: str
    reviewed_lock_sha256: str


@pytest.fixture
def view_inputs(tmp_path: Path) -> ViewInputs:
    data_root, manifest_path, split_path, feature_path = _write_synthetic_modeling_contract(
        tmp_path
    )
    verification = verify_dataset(data_root, manifest_path, split_path)
    manifest = load_dataset_manifest(manifest_path)
    return ViewInputs(
        canonical=pd.read_csv(verification.paths.canonical),
        assignments=pd.read_csv(verification.paths.split_assignments),
        contract=load_feature_contract(feature_path),
        manifest=manifest,
        split_config=load_split_config(split_path),
        verification=verification,
        feature_contract_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        reviewed_lock_sha256=hashlib.sha256(
            verification.paths.reviewed_split_lock.read_bytes()
        ).hexdigest(),
    )


def _build_view(inputs: ViewInputs) -> GovernedDevelopmentData:
    return _build_development_view(
        canonical=inputs.canonical,
        assignments=inputs.assignments,
        contract=inputs.contract,
        expected_canonical_columns=tuple(
            column.canonical_name for column in inputs.manifest.canonical_contract.columns
        ),
        feature_contract_sha256=inputs.feature_contract_sha256,
        verification=inputs.verification,
        reviewed_lock_sha256=inputs.reviewed_lock_sha256,
    )


@pytest.fixture
def governed_data(tmp_path: Path):
    paths = _write_synthetic_modeling_contract(tmp_path)
    return load_governed_development_data(
        data_root=paths[0],
        feature_contract_path=paths[3],
        manifest_path=paths[1],
        split_config_path=paths[2],
    )


def test_loader_exposes_only_governed_development_views(
    governed_data: GovernedDevelopmentData,
) -> None:
    data = governed_data

    assert data.predictors.shape == (80, 19)
    assert tuple(data.predictors.columns) == PREDICTOR_COLUMNS
    assert data.predictors.index.equals(data.account_ids)
    assert data.target.index.equals(data.account_ids)
    assert tuple(data.audit.columns) == (
        "account_id",
        "default_next_month",
        *AUDIT_COLUMNS,
    )
    assert data.audit.index.equals(data.account_ids)
    assert tuple(data.assignments.columns) == (
        "partition",
        "cv_fold_r0",
        "cv_fold_r1",
        "cv_fold_r2",
    )
    assert data.assignments["partition"].eq("development").all()
    assert set(data.target.unique()) == {0, 1}
    assert not hasattr(data, "test")
    assert data.X is data.predictors
    assert data.y is data.target


def test_loader_never_exposes_test_account_ids(tmp_path: Path) -> None:
    data_root, manifest_path, split_path, feature_path = _write_synthetic_modeling_contract(
        tmp_path
    )
    data = load_governed_development_data(data_root, feature_path, manifest_path, split_path)
    all_assignments = pd.read_csv(
        data_root / "splits" / "fixture_credit_default" / "v1" / "split_assignments.csv"
    )
    test_ids = set(all_assignments.loc[all_assignments["partition"].eq("test"), "account_id"])

    assert len(test_ids) == 20
    assert test_ids.isdisjoint(set(data.account_ids))


def test_every_reviewed_fold_is_disjoint_and_complete(
    governed_data: GovernedDevelopmentData,
) -> None:
    data = governed_data

    for repeat_index in range(3):
        validation_ids: list[int] = []
        for fold_index in range(5):
            fold = data.fold(repeat_index, fold_index)
            assert len(fold.train_account_ids) == 64
            assert len(fold.validation_account_ids) == 16
            assert fold.train_account_ids.intersection(fold.validation_account_ids).empty
            assert fold.X_train.index.equals(fold.train_account_ids)
            assert fold.X_validation.index.equals(fold.validation_account_ids)
            assert fold.y_train.index.equals(fold.train_account_ids)
            assert fold.y_validation.index.equals(fold.validation_account_ids)
            validation_ids.extend(int(value) for value in fold.validation_account_ids)
        assert sorted(validation_ids) == sorted(int(value) for value in data.account_ids)


@pytest.mark.parametrize(
    ("repeat_index", "fold_index", "message"),
    [
        (-1, 0, "repeat_index"),
        (3, 0, "repeat_index"),
        (0, -1, "fold_index"),
        (0, 5, "fold_index"),
        (True, 0, "repeat_index"),
    ],
)
def test_fold_rejects_invalid_indices(
    governed_data: GovernedDevelopmentData,
    repeat_index: int,
    fold_index: int,
    message: str,
) -> None:
    with pytest.raises(ModelingDataError, match=message):
        governed_data.fold(repeat_index, fold_index)


def test_fold_rejects_reviewed_assignment_without_requested_fold(
    governed_data: GovernedDevelopmentData,
) -> None:
    assignments = governed_data.assignments.copy()
    assignments["cv_fold_r0"] = 0
    changed = replace(governed_data, assignments=assignments)

    with pytest.raises(ModelingDataError, match="contain no rows"):
        changed.fold(0, 1)


def test_loader_rejects_feature_lineage_drift(tmp_path: Path) -> None:
    data_root, manifest_path, split_path, feature_path = _write_synthetic_modeling_contract(
        tmp_path
    )
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    payload["lineage"]["canonical_sha256"] = "0" * 64
    write_json(feature_path, payload)

    with pytest.raises(ModelingDataError, match="canonical_sha256"):
        load_governed_development_data(data_root, feature_path, manifest_path, split_path)


def test_loader_fails_closed_when_generated_assignments_change(tmp_path: Path) -> None:
    data_root, manifest_path, split_path, feature_path = _write_synthetic_modeling_contract(
        tmp_path
    )
    assignment_path = (
        data_root / "splits" / "fixture_credit_default" / "v1" / "split_assignments.csv"
    )
    assignments = assignment_path.read_text(encoding="utf-8")
    assignment_path.write_text(assignments.replace("development", "test", 1), encoding="utf-8")

    with pytest.raises(ModelingDataError, match="verification failed"):
        load_governed_development_data(data_root, feature_path, manifest_path, split_path)


@pytest.mark.parametrize(
    ("target_file", "column", "message"),
    [
        ("canonical", "account_id", "verification failed"),
        ("assignments", "account_id", "verification failed"),
        ("assignments", "partition", "verification failed"),
    ],
)
def test_loader_reports_missing_required_columns_without_key_error(
    tmp_path: Path,
    target_file: str,
    column: str,
    message: str,
) -> None:
    data_root, manifest_path, split_path, feature_path = _write_synthetic_modeling_contract(
        tmp_path
    )
    if target_file == "canonical":
        path = data_root / "processed" / "fixture_credit_default" / "v1" / "canonical.csv"
    else:
        path = data_root / "splits" / "fixture_credit_default" / "v1" / "split_assignments.csv"
    frame = pd.read_csv(path).drop(columns=column)
    frame.to_csv(path, index=False, lineterminator="\n")

    with pytest.raises(ModelingDataError, match=message) as raised:
        load_governed_development_data(data_root, feature_path, manifest_path, split_path)
    assert not isinstance(raised.value.__cause__, KeyError)


@pytest.mark.parametrize(
    "case",
    [
        "manifest_identity",
        "split_identity",
        "identifier",
        "target",
        "cv_shape",
        "development_rows",
        "target_counts",
    ],
)
def test_contract_parity_rejects_every_cross_contract_drift(
    view_inputs: ViewInputs,
    case: str,
) -> None:
    contract = view_inputs.contract
    manifest = view_inputs.manifest
    split_config = view_inputs.split_config
    if case == "manifest_identity":
        manifest = manifest.model_copy(update={"dataset_version": "v2"})
    elif case == "split_identity":
        split_config = split_config.model_copy(update={"dataset_version": "v2"})
    elif case == "identifier":
        split_config = split_config.model_copy(update={"id_column": "different_id"})
    elif case == "target":
        split_config = split_config.model_copy(update={"target_column": "different_target"})
    elif case == "cv_shape":
        split_config = split_config.model_copy(
            update={
                "cross_validation": split_config.cross_validation.model_copy(
                    update={"n_repeats": 2}
                )
            }
        )
    elif case == "development_rows":
        split_config = split_config.model_copy(
            update={
                "expected_counts": split_config.expected_counts.model_copy(
                    update={
                        "development": split_config.expected_counts.development.model_copy(
                            update={"total": 79}
                        )
                    }
                )
            }
        )
    else:
        split_config = split_config.model_copy(
            update={
                "expected_counts": split_config.expected_counts.model_copy(
                    update={
                        "development": split_config.expected_counts.development.model_copy(
                            update={"target_counts": {"0": 39, "1": 41}}
                        )
                    }
                )
            }
        )

    with pytest.raises(ModelingDataError):
        _validate_contract_parity(contract, manifest, split_config)


@pytest.mark.parametrize(
    "case",
    [
        "missing_canonical",
        "missing_assignment",
        "canonical_order",
        "uncovered_canonical",
        "assignment_order",
        "null_id",
        "fractional_id",
        "duplicate_id",
        "canonical_unsorted",
        "assignment_unsorted",
        "incomplete_id_coverage",
        "invalid_partitions",
        "development_rows",
        "target_counts",
        "fractional_fold",
        "missing_fold",
        "nonnumeric_predictor",
    ],
)
def test_development_view_fails_closed_on_boundary_drift(
    view_inputs: ViewInputs,
    case: str,
) -> None:
    canonical = view_inputs.canonical.copy()
    assignments = view_inputs.assignments.copy()
    expected_columns = tuple(
        column.canonical_name for column in view_inputs.manifest.canonical_contract.columns
    )
    if case == "missing_canonical":
        canonical = canonical.drop(columns="account_id")
    elif case == "missing_assignment":
        assignments = assignments.drop(columns="partition")
    elif case == "canonical_order":
        canonical = canonical.loc[:, list(reversed(canonical.columns))]
    elif case == "uncovered_canonical":
        canonical["unexpected_column"] = 1
        expected_columns = (*expected_columns, "unexpected_column")
    elif case == "assignment_order":
        assignments = assignments.loc[:, list(reversed(assignments.columns))]
    elif case == "null_id":
        canonical.loc[0, "account_id"] = None
    elif case == "fractional_id":
        canonical["account_id"] = canonical["account_id"].astype("float64")
        canonical.loc[0, "account_id"] = 1.5
    elif case == "duplicate_id":
        assignments.loc[1, "account_id"] = assignments.loc[0, "account_id"]
    elif case == "canonical_unsorted":
        canonical = canonical.iloc[::-1].reset_index(drop=True)
    elif case == "assignment_unsorted":
        assignments = assignments.iloc[::-1].reset_index(drop=True)
    elif case == "incomplete_id_coverage":
        assignments.loc[len(assignments) - 1, "account_id"] = 101
    elif case == "invalid_partitions":
        assignments["partition"] = "development"
        for column in ("cv_fold_r0", "cv_fold_r1", "cv_fold_r2"):
            assignments[column] = assignments[column].fillna(0)
    elif case == "development_rows":
        development_index = assignments.index[assignments["partition"].eq("development")][0]
        assignments.loc[development_index, "partition"] = "test"
        assignments.loc[development_index, ["cv_fold_r0", "cv_fold_r1", "cv_fold_r2"]] = None
    elif case == "target_counts":
        development_id = int(
            assignments.loc[assignments["partition"].eq("development"), "account_id"].iloc[0]
        )
        current = int(
            canonical.loc[canonical["account_id"].eq(development_id), "default_next_month"].iloc[0]
        )
        canonical.loc[canonical["account_id"].eq(development_id), "default_next_month"] = (
            1 - current
        )
    elif case == "fractional_fold":
        development_index = assignments.index[assignments["partition"].eq("development")][0]
        assignments.loc[development_index, "cv_fold_r0"] = 0.5
    elif case == "missing_fold":
        assignments.loc[assignments["cv_fold_r0"].eq(4), "cv_fold_r0"] = 3
    else:
        canonical["credit_limit_ntd"] = canonical["credit_limit_ntd"].astype("object")
        canonical.loc[0, "credit_limit_ntd"] = "not-numeric"

    changed = replace(view_inputs, canonical=canonical, assignments=assignments)
    with pytest.raises(ModelingDataError):
        _build_development_view(
            canonical=changed.canonical,
            assignments=changed.assignments,
            contract=changed.contract,
            expected_canonical_columns=expected_columns,
            feature_contract_sha256=changed.feature_contract_sha256,
            verification=changed.verification,
            reviewed_lock_sha256=changed.reviewed_lock_sha256,
        )


def test_file_helpers_wrap_io_failures(tmp_path: Path) -> None:
    with pytest.raises(ModelingDataError, match="Unable to read"):
        _read_csv(tmp_path / "missing.csv", "missing fixture")
    with pytest.raises(ModelingDataError, match="Unable to hash"):
        _file_sha256(tmp_path)


def test_verified_csv_parses_only_bytes_bound_to_the_expected_hash(tmp_path: Path) -> None:
    path = tmp_path / "verified.csv"
    content = b"account_id,value\n1,2\n"
    path.write_bytes(content)

    parsed = _read_verified_csv(path, "fixture", hashlib.sha256(content).hexdigest())
    assert parsed.to_dict(orient="records") == [{"account_id": 1, "value": 2}]

    path.write_bytes(b"account_id,value\n1,changed\n")
    with pytest.raises(ModelingDataError, match="changed before modelling"):
        _read_verified_csv(path, "fixture", hashlib.sha256(content).hexdigest())


def test_loader_rejects_unapproved_lock_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, manifest_path, split_path, feature_path = _write_synthetic_modeling_contract(
        tmp_path
    )
    result = verify_dataset(data_root, manifest_path, split_path)
    monkeypatch.setattr(
        "credit_risk.modeling.dataset.verify_dataset",
        lambda *_args, **_kwargs: replace(result, reviewed_lock_verified=False),
    )

    with pytest.raises(ModelingDataError, match="did not approve"):
        load_governed_development_data(data_root, feature_path, manifest_path, split_path)
