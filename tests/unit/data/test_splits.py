"""Tests for sealed holdout and repeated cross-validation assignments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from credit_risk.data.manifest import SplitConfig
from credit_risk.data.splits import (
    SplitValidationError,
    _resolve_config,
    _validate_assignments,
    build_split_artifacts,
    split_manifest_bytes,
    write_split_artifacts,
)


def _canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": range(1, 101),
            "default_next_month": [0] * 80 + [1] * 20,
        }
    )


def _config() -> SplitConfig:
    return SplitConfig.model_validate(
        {
            "config_version": 1,
            "dataset_id": "uci_credit_default",
            "dataset_version": "test",
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
                "development": {"total": 80, "target_counts": {"0": 64, "1": 16}},
                "test": {"total": 20, "target_counts": {"0": 16, "1": 4}},
            },
        }
    )


def test_split_is_wide_stratified_and_test_folds_are_null() -> None:
    artifacts = build_split_artifacts(
        _canonical(),
        _config(),
        source_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    assignments = artifacts.assignments

    assert assignments.columns.tolist() == [
        "account_id",
        "partition",
        "cv_fold_r0",
        "cv_fold_r1",
        "cv_fold_r2",
    ]
    assert assignments["partition"].value_counts().to_dict() == {
        "development": 80,
        "test": 20,
    }
    test_rows = assignments[assignments["partition"] == "test"]
    development = assignments[assignments["partition"] == "development"]
    assert test_rows.filter(like="cv_fold").isna().all().all()
    assert development.filter(like="cv_fold").notna().all().all()
    for column in development.filter(like="cv_fold"):
        assert set(development[column]) == set(range(5))
    assert artifacts.manifest["lineage"]["split_config_sha256"] == "b" * 64
    assert artifacts.manifest["dataset_version"] == "test"


def test_split_is_invariant_to_input_row_order() -> None:
    canonical = _canonical()

    first = build_split_artifacts(canonical, _config())
    second = build_split_artifacts(canonical.sample(frac=1, random_state=99), _config())

    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    assert first.assignment_sha256 == second.assignment_sha256
    assert first.lock_sha256 == second.lock_sha256
    assert split_manifest_bytes(first.manifest) == split_manifest_bytes(second.manifest)


def test_split_manifest_contains_exact_counts_and_assignment_hash() -> None:
    artifacts = build_split_artifacts(_canonical(), _config())

    assignment = artifacts.manifest["assignment"]
    assert assignment["sha256"] == artifacts.assignment_sha256
    assert assignment["partition_counts"]["development"]["target_counts"] == {
        "0": 64,
        "1": 16,
    }
    assert assignment["partition_counts"]["test"]["target_counts"] == {
        "0": 16,
        "1": 4,
    }
    assert len(artifacts.lock_sha256) == 64


def test_write_split_artifacts_preserves_locked_bytes(tmp_path: Path) -> None:
    artifacts = build_split_artifacts(_canonical(), _config())

    written = write_split_artifacts(artifacts, tmp_path)

    assert written.assignments_path.name == "split_assignments.csv"
    assert written.manifest_path.name == "split_manifest.json"
    assert b"\r\n" not in written.assignments_path.read_bytes()
    assert written.manifest_path.read_bytes() == split_manifest_bytes(artifacts.manifest)


@pytest.mark.parametrize("mutation", ["duplicate_id", "one_class", "wrong_count"])
def test_split_rejects_invalid_inputs(mutation: str) -> None:
    canonical = _canonical()
    if mutation == "duplicate_id":
        canonical.loc[1, "account_id"] = 1
    elif mutation == "one_class":
        canonical["default_next_month"] = 0
    else:
        canonical = canonical.iloc[:-1]

    with pytest.raises(SplitValidationError):
        build_split_artifacts(canonical, _config())


def _with_nested_update(config: SplitConfig, field: str, **updates: Any) -> SplitConfig:
    nested = getattr(config, field).model_copy(update=updates)
    return config.model_copy(update={field: nested})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config.model_copy(update={"id_column": "customer_id"}), "account_id"),
        (
            lambda config: config.model_copy(update={"target_column": "target"}),
            "default_next_month",
        ),
        (lambda config: config.model_copy(update={"sort_by": ("account_id", "age")}), "sorted"),
        (
            lambda config: _with_nested_update(config, "holdout", test_fraction=0.25),
            "test_fraction",
        ),
        (
            lambda config: _with_nested_update(config, "holdout", random_state=7),
            "random_state",
        ),
        (
            lambda config: _with_nested_update(config, "holdout", method="random"),
            "Holdout method",
        ),
        (
            lambda config: _with_nested_update(config, "cross_validation", n_splits=4),
            "exactly 5 folds",
        ),
        (
            lambda config: _with_nested_update(config, "cross_validation", n_repeats=2),
            "exactly 5 folds",
        ),
        (
            lambda config: _with_nested_update(config, "cross_validation", random_state=7),
            "random_state",
        ),
        (
            lambda config: _with_nested_update(
                config, "cross_validation", method="stratified_k_fold"
            ),
            "repeated_stratified_k_fold",
        ),
        (
            lambda config: config.model_copy(
                update={"expected_counts": config.expected_counts.model_copy(update={"total": 101})}
            ),
            "partition totals",
        ),
    ],
)
def test_split_rejects_any_drift_from_the_sealed_algorithm(mutation, message: str) -> None:
    with pytest.raises(SplitValidationError, match=message):
        build_split_artifacts(_canonical(), mutation(_config()))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="account_id"), "missing columns"),
        (
            lambda frame: frame.assign(
                account_id=frame["account_id"].astype("Float64").mask(frame.index == 0)
            ),
            "must be non-null",
        ),
    ],
)
def test_split_rejects_missing_or_null_identity_inputs(mutation, message: str) -> None:
    with pytest.raises(SplitValidationError, match=message):
        build_split_artifacts(mutation(_canonical()), _config())


def test_split_rejects_development_class_too_small_for_five_folds() -> None:
    canonical = pd.DataFrame(
        {
            "account_id": range(1, 101),
            "default_next_month": [0] * 96 + [1] * 4,
        }
    )

    with pytest.raises(SplitValidationError, match="at least n_splits"):
        build_split_artifacts(canonical, _config())


def test_split_rejects_expected_partition_count_drift() -> None:
    config = _config()
    development = config.expected_counts.development.model_copy(
        update={"target_counts": {"0": 63, "1": 17}}
    )
    expected = config.expected_counts.model_copy(update={"development": development})
    drifted = config.model_copy(update={"expected_counts": expected})

    with pytest.raises(SplitValidationError, match="development partition counts"):
        build_split_artifacts(_canonical(), drifted)


def test_default_config_is_the_official_30000_row_contract() -> None:
    with pytest.raises(SplitValidationError, match="exactly 30000"):
        build_split_artifacts(_canonical(), None)


def test_malformed_duck_typed_config_is_rejected_actionably() -> None:
    with pytest.raises(SplitValidationError, match="must expose dataset"):
        build_split_artifacts(_canonical(), cast(Any, object()))


def test_non_integer_expected_target_counts_are_rejected_actionably() -> None:
    config = _config()
    development = config.expected_counts.development.model_copy(
        update={"target_counts": cast(Any, {"zero": "many"})}
    )
    malformed = config.model_copy(
        update={
            "expected_counts": config.expected_counts.model_copy(
                update={"development": development}
            )
        }
    )

    with pytest.raises(SplitValidationError, match="must expose dataset"):
        build_split_artifacts(_canonical(), malformed)


def test_write_rejects_assignment_digest_drift(tmp_path: Path, monkeypatch) -> None:
    artifacts = build_split_artifacts(_canonical(), _config())
    monkeypatch.setattr("credit_risk.data.splits.write_canonical_csv", lambda *_: "bad")

    with pytest.raises(SplitValidationError, match="assignment digest"):
        write_split_artifacts(artifacts, tmp_path)


def test_write_rejects_manifest_digest_drift(tmp_path: Path, monkeypatch) -> None:
    artifacts = build_split_artifacts(_canonical(), _config())
    monkeypatch.setattr("credit_risk.data.splits.write_bytes_atomically", lambda *_: "bad")

    with pytest.raises(SplitValidationError, match="manifest digest"):
        write_split_artifacts(artifacts, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda frame, base: pd.concat([frame.iloc[:-1], frame.iloc[[0]]], ignore_index=True),
            "every account exactly once",
        ),
        (
            lambda frame, base: frame.assign(
                account_id=[999, *frame["account_id"].iloc[1:].tolist()]
            ),
            "do not match canonical",
        ),
        (
            lambda frame, base: frame.assign(
                partition=["other", *frame["partition"].iloc[1:].tolist()]
            ),
            "exactly development and test",
        ),
        (
            lambda frame, base: frame.assign(
                cv_fold_r0=frame["cv_fold_r0"].mask(frame["partition"] == "development")
            ),
            "Development rows",
        ),
        (
            lambda frame, base: frame.assign(
                cv_fold_r0=frame["cv_fold_r0"].mask(frame["partition"] == "test", 0)
            ),
            "Test rows",
        ),
        (
            lambda frame, base: frame.assign(
                cv_fold_r0=frame["cv_fold_r0"].mask(frame["partition"] == "development", 0)
            ),
            "does not contain every configured fold",
        ),
    ],
)
def test_assignment_validator_rejects_corrupted_evidence(mutation, message: str) -> None:
    canonical = _canonical()
    config = _config()
    artifacts = build_split_artifacts(canonical, config)
    resolved = _resolve_config(config)
    fold_columns = ["cv_fold_r0", "cv_fold_r1", "cv_fold_r2"]

    with pytest.raises(SplitValidationError, match=message):
        _validate_assignments(
            canonical,
            mutation(artifacts.assignments.copy(), canonical),
            resolved,
            fold_columns,
        )


def test_assignment_validator_rejects_class_imbalanced_fold() -> None:
    canonical = _canonical()
    config = _config()
    artifacts = build_split_artifacts(canonical, config)
    assignments = artifacts.assignments.copy()
    labels = canonical.set_index("account_id")["default_next_month"]
    development_zero_ids = [
        account_id
        for account_id in assignments.loc[assignments["partition"] == "development", "account_id"]
        if labels.loc[account_id] == 0
    ]
    assignments.loc[assignments["account_id"].isin(development_zero_ids), "cv_fold_r0"] = 0
    for fold, account_id in enumerate(development_zero_ids[:4], start=1):
        assignments.loc[assignments["account_id"] == account_id, "cv_fold_r0"] = fold

    with pytest.raises(SplitValidationError, match="not class-balanced"):
        _validate_assignments(
            canonical,
            assignments,
            _resolve_config(config),
            ["cv_fold_r0", "cv_fold_r1", "cv_fold_r2"],
        )
