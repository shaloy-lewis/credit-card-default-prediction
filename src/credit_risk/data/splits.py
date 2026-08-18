"""Deterministic sealed-holdout and repeated cross-validation assignments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from credit_risk.data.manifest import SplitConfig
from credit_risk.data.schema import (
    SCHEMA_VERSION,
    dataframe_sha256,
    write_bytes_atomically,
    write_canonical_csv,
)

SPLIT_ASSIGNMENTS_FILENAME = "split_assignments.csv"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"

DEFAULT_ID_COLUMN = "account_id"
DEFAULT_TARGET_COLUMN = "default_next_month"
DEFAULT_TEST_FRACTION = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 3


class SplitValidationError(ValueError):
    """Raised when split inputs, configuration, or assignments are invalid."""


@dataclass(frozen=True, slots=True)
class SplitArtifacts:
    """One account-level assignment table and its deterministic evidence lock."""

    assignments: pd.DataFrame
    manifest: dict[str, Any]
    assignment_sha256: str
    lock_sha256: str


@dataclass(frozen=True, slots=True)
class WrittenSplitArtifacts:
    """Paths written by :func:`write_split_artifacts`."""

    assignments_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class _ResolvedSplitConfig:
    dataset_id: str
    dataset_version: str
    id_column: str
    target_column: str
    sort_by: tuple[str, ...]
    holdout_method: str
    test_fraction: float
    holdout_random_state: int
    cross_validation_method: str
    n_splits: int
    n_repeats: int
    cross_validation_random_state: int
    expected_total: int
    expected_development_total: int
    expected_development_target_counts: tuple[tuple[int, int], ...]
    expected_test_total: int
    expected_test_target_counts: tuple[tuple[int, int], ...]


def build_split_artifacts(
    canonical: pd.DataFrame,
    config: SplitConfig | None = None,
    *,
    source_sha256: str | None = None,
    canonical_sha256: str | None = None,
    config_sha256: str | None = None,
) -> SplitArtifacts:
    """Build the locked 80/20 holdout and 5x3 repeated stratified folds.

    The config is duck-typed to the checked-in ``SplitConfig`` so its Pydantic
    model remains the single source of truth.
    """

    resolved = _resolve_config(config)
    _validate_config(resolved)
    base = _validate_and_sort_input(canonical, resolved)

    development_ids, test_ids = train_test_split(
        base[resolved.id_column].to_numpy(),
        test_size=resolved.test_fraction,
        random_state=resolved.holdout_random_state,
        shuffle=True,
        stratify=base[resolved.target_column].to_numpy(),
    )
    development_id_set = {int(value) for value in development_ids}
    test_id_set = {int(value) for value in test_ids}

    assignments = base.loc[:, [resolved.id_column]].copy()
    assignments["partition"] = np.where(
        assignments[resolved.id_column].isin(test_id_set), "test", "development"
    )
    fold_columns = [f"cv_fold_r{repeat}" for repeat in range(resolved.n_repeats)]
    for column in fold_columns:
        assignments[column] = pd.Series(pd.NA, index=assignments.index, dtype="Int8")

    development = base[base[resolved.id_column].isin(development_id_set)].reset_index(drop=True)
    development_class_counts = development[resolved.target_column].value_counts()
    if int(development_class_counts.min()) < resolved.n_splits:
        raise SplitValidationError(
            "Every development target class must contain at least n_splits rows."
        )

    repeated_cv = RepeatedStratifiedKFold(
        n_splits=resolved.n_splits,
        n_repeats=resolved.n_repeats,
        random_state=resolved.cross_validation_random_state,
    )
    placeholder_features = np.zeros((len(development), 1), dtype=np.int8)
    assignment_rows_by_id = assignments.reset_index().set_index(resolved.id_column)["index"]
    for split_index, (_, validation_indices) in enumerate(
        repeated_cv.split(placeholder_features, development[resolved.target_column])
    ):
        repeat = split_index // resolved.n_splits
        fold = split_index % resolved.n_splits
        validation_ids = development.iloc[validation_indices][resolved.id_column]
        assignment_rows = assignment_rows_by_id.loc[validation_ids].to_numpy()
        assignments.loc[assignment_rows, fold_columns[repeat]] = fold

    assignments = assignments.sort_values(resolved.id_column, kind="stable").reset_index(drop=True)
    _validate_assignments(base, assignments, resolved, fold_columns)

    assignment_sha256 = dataframe_sha256(assignments)
    canonical_digest = canonical_sha256 or dataframe_sha256(base.loc[:, canonical.columns])
    split_config_digest = config_sha256 or _resolved_config_sha256(resolved)
    manifest = _build_manifest(
        base=base,
        assignments=assignments,
        config=resolved,
        fold_columns=fold_columns,
        source_sha256=source_sha256,
        canonical_sha256=canonical_digest,
        config_sha256=split_config_digest,
        assignment_sha256=assignment_sha256,
    )
    lock_sha256 = hashlib.sha256(split_manifest_bytes(manifest)).hexdigest()
    return SplitArtifacts(
        assignments=assignments,
        manifest=manifest,
        assignment_sha256=assignment_sha256,
        lock_sha256=lock_sha256,
    )


def split_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """Serialize a split manifest deterministically without timestamps."""

    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_split_artifacts(
    artifacts: SplitArtifacts,
    output_dir: str | Path,
) -> WrittenSplitArtifacts:
    """Atomically write the account-level assignments and evidence manifest."""

    directory = Path(output_dir)
    assignments_path = directory / SPLIT_ASSIGNMENTS_FILENAME
    manifest_path = directory / SPLIT_MANIFEST_FILENAME
    written_assignment_sha = write_canonical_csv(artifacts.assignments, assignments_path)
    if written_assignment_sha != artifacts.assignment_sha256:
        raise SplitValidationError("Written split assignment digest changed unexpectedly.")
    written_lock_sha = write_bytes_atomically(
        manifest_path, split_manifest_bytes(artifacts.manifest)
    )
    if written_lock_sha != artifacts.lock_sha256:
        raise SplitValidationError("Written split manifest digest changed unexpectedly.")
    return WrittenSplitArtifacts(assignments_path, manifest_path)


def _resolve_config(config: SplitConfig | None) -> _ResolvedSplitConfig:
    if config is None:
        return _ResolvedSplitConfig(
            dataset_id="uci_credit_default",
            dataset_version="v1",
            id_column=DEFAULT_ID_COLUMN,
            target_column=DEFAULT_TARGET_COLUMN,
            sort_by=(DEFAULT_ID_COLUMN,),
            holdout_method="stratified_shuffle_split",
            test_fraction=DEFAULT_TEST_FRACTION,
            holdout_random_state=DEFAULT_RANDOM_STATE,
            cross_validation_method="repeated_stratified_k_fold",
            n_splits=DEFAULT_N_SPLITS,
            n_repeats=DEFAULT_N_REPEATS,
            cross_validation_random_state=DEFAULT_RANDOM_STATE,
            expected_total=30_000,
            expected_development_total=24_000,
            expected_development_target_counts=((0, 18_691), (1, 5_309)),
            expected_test_total=6_000,
            expected_test_target_counts=((0, 4_673), (1, 1_327)),
        )

    try:
        holdout = config.holdout
        cross_validation = config.cross_validation
        expected_counts = config.expected_counts
        development = expected_counts.development
        test = expected_counts.test
        raw_sort_by = config.sort_by
        sort_by = (raw_sort_by,) if isinstance(raw_sort_by, str) else tuple(raw_sort_by)
        return _ResolvedSplitConfig(
            dataset_id=str(config.dataset_id),
            dataset_version=str(config.dataset_version),
            id_column=str(config.id_column),
            target_column=str(config.target_column),
            sort_by=sort_by,
            holdout_method=str(holdout.method),
            test_fraction=float(holdout.test_fraction),
            holdout_random_state=int(holdout.random_state),
            cross_validation_method=str(cross_validation.method),
            n_splits=int(cross_validation.n_splits),
            n_repeats=int(cross_validation.n_repeats),
            cross_validation_random_state=int(cross_validation.random_state),
            expected_total=int(expected_counts.total),
            expected_development_total=int(development.total),
            expected_development_target_counts=_integer_count_pairs(development.target_counts),
            expected_test_total=int(test.total),
            expected_test_target_counts=_integer_count_pairs(test.target_counts),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise SplitValidationError(
            "Split config must expose dataset, field, holdout, CV, and expected-count values."
        ) from error


def _validate_config(config: _ResolvedSplitConfig) -> None:
    if config.id_column != DEFAULT_ID_COLUMN or config.target_column != DEFAULT_TARGET_COLUMN:
        raise SplitValidationError(
            "Phase 1 split config must use account_id and default_next_month."
        )
    if config.sort_by != (config.id_column,):
        raise SplitValidationError("Split input must be sorted only by account_id.")
    if config.test_fraction != DEFAULT_TEST_FRACTION:
        raise SplitValidationError("The sealed holdout test_fraction must be exactly 0.20.")
    if config.holdout_random_state != DEFAULT_RANDOM_STATE:
        raise SplitValidationError("The sealed holdout random_state must be exactly 42.")
    if config.holdout_method != "stratified_shuffle_split":
        raise SplitValidationError("Holdout method must be stratified_shuffle_split.")
    if config.n_splits != DEFAULT_N_SPLITS or config.n_repeats != DEFAULT_N_REPEATS:
        raise SplitValidationError("Cross-validation must use exactly 5 folds and 3 repeats.")
    if config.cross_validation_random_state != DEFAULT_RANDOM_STATE:
        raise SplitValidationError("Cross-validation random_state must be exactly 42.")
    if config.cross_validation_method != "repeated_stratified_k_fold":
        raise SplitValidationError("Cross-validation method must be repeated_stratified_k_fold.")
    if config.expected_development_total + config.expected_test_total != config.expected_total:
        raise SplitValidationError("Locked partition totals must equal the dataset total.")


def _validate_and_sort_input(
    canonical: pd.DataFrame,
    config: _ResolvedSplitConfig,
) -> pd.DataFrame:
    missing = sorted({config.id_column, config.target_column} - set(canonical.columns))
    if missing:
        raise SplitValidationError(f"Canonical split input is missing columns: {missing}.")
    if canonical[config.id_column].isna().any() or canonical[config.target_column].isna().any():
        raise SplitValidationError("Split ID and target columns must be non-null.")
    if canonical[config.id_column].duplicated().any():
        raise SplitValidationError("Split input account_id values must be unique.")
    if set(canonical[config.target_column].unique()) != {0, 1}:
        raise SplitValidationError("Split input target must contain exactly classes 0 and 1.")
    if len(canonical) != config.expected_total:
        raise SplitValidationError(
            f"Split input must contain exactly {config.expected_total} validated rows."
        )
    return canonical.sort_values(list(config.sort_by), kind="stable").reset_index(drop=True)


def _validate_assignments(
    base: pd.DataFrame,
    assignments: pd.DataFrame,
    config: _ResolvedSplitConfig,
    fold_columns: list[str],
) -> None:
    all_ids = set(int(value) for value in base[config.id_column])
    if len(assignments) != len(base) or assignments[config.id_column].duplicated().any():
        raise SplitValidationError("Assignments must contain every account exactly once.")
    if set(int(value) for value in assignments[config.id_column]) != all_ids:
        raise SplitValidationError("Assignments do not match canonical account IDs.")
    if set(assignments["partition"]) != {"development", "test"}:
        raise SplitValidationError("Partitions must be exactly development and test.")

    labels = base.set_index(config.id_column)[config.target_column]
    joined = assignments.join(labels, on=config.id_column)
    _validate_expected_partition(
        joined,
        "development",
        config.expected_development_total,
        config.expected_development_target_counts,
        config.target_column,
    )
    _validate_expected_partition(
        joined,
        "test",
        config.expected_test_total,
        config.expected_test_target_counts,
        config.target_column,
    )

    development = joined[joined["partition"] == "development"]
    test = joined[joined["partition"] == "test"]
    if development[fold_columns].isna().any().any():
        raise SplitValidationError("Development rows must have one fold in every repeat.")
    if test[fold_columns].notna().any().any():
        raise SplitValidationError("Test rows must have null CV folds.")
    for column in fold_columns:
        if set(int(value) for value in development[column].unique()) != set(range(config.n_splits)):
            raise SplitValidationError(f"{column} does not contain every configured fold.")
        for target in (0, 1):
            fold_counts = (
                development[development[config.target_column] == target].groupby(column).size()
            )
            if int(fold_counts.max() - fold_counts.min()) > 1:
                raise SplitValidationError("CV folds are not class-balanced within one row.")


def _build_manifest(
    *,
    base: pd.DataFrame,
    assignments: pd.DataFrame,
    config: _ResolvedSplitConfig,
    fold_columns: list[str],
    source_sha256: str | None,
    canonical_sha256: str,
    config_sha256: str,
    assignment_sha256: str,
) -> dict[str, Any]:
    labels = base.set_index(config.id_column)[config.target_column]
    joined = assignments.join(labels, on=config.id_column)
    partition_counts = {
        partition: {
            "rows": len(rows),
            "target_counts": _string_key_counts(rows[config.target_column]),
        }
        for partition, rows in joined.groupby("partition", sort=True)
    }
    fold_counts = {
        column: {
            str(fold): {
                "rows": len(rows),
                "target_counts": _string_key_counts(rows[config.target_column]),
            }
            for fold, rows in joined[joined["partition"] == "development"].groupby(column)
        }
        for column in fold_columns
    }
    lineage = {
        "canonical_sha256": canonical_sha256,
        "split_config_sha256": config_sha256,
    }
    if source_sha256 is not None:
        lineage["source_sha256"] = source_sha256
    return {
        "algorithm": {
            "cross_validation": config.cross_validation_method,
            "holdout": config.holdout_method,
            "scikit_learn_version": sklearn.__version__,
        },
        "assignment": {
            "filename": SPLIT_ASSIGNMENTS_FILENAME,
            "fold_counts": fold_counts,
            "partition_counts": partition_counts,
            "rows": len(assignments),
            "sha256": assignment_sha256,
        },
        "config": {
            "cross_validation": {
                "n_repeats": config.n_repeats,
                "n_splits": config.n_splits,
                "random_state": config.cross_validation_random_state,
            },
            "holdout": {
                "random_state": config.holdout_random_state,
                "test_fraction": config.test_fraction,
            },
            "id_column": config.id_column,
            "sort_by": list(config.sort_by),
            "target_column": config.target_column,
        },
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "lineage": lineage,
        "schema_version": SCHEMA_VERSION,
    }


def _validate_expected_partition(
    assignments: pd.DataFrame,
    partition: str,
    expected_total: int,
    expected_counts: tuple[tuple[int, int], ...],
    target_column: str,
) -> None:
    rows = assignments[assignments["partition"] == partition]
    actual_counts = tuple(
        sorted(
            (int(label), int(count)) for label, count in rows[target_column].value_counts().items()
        )
    )
    if len(rows) != expected_total or actual_counts != expected_counts:
        raise SplitValidationError(
            f"{partition} partition counts do not match the locked split config."
        )


def _integer_count_pairs(counts: Mapping[Any, int]) -> tuple[tuple[int, int], ...]:
    try:
        return tuple(sorted((int(label), int(count)) for label, count in dict(counts).items()))
    except (TypeError, ValueError) as error:
        raise SplitValidationError("Expected target counts must be an integer mapping.") from error


def _string_key_counts(values: pd.Series) -> dict[str, int]:
    return {
        str(int(label)): int(count)
        for label, count in sorted(values.value_counts().items(), key=lambda item: item[0])
    }


def _resolved_config_sha256(config: _ResolvedSplitConfig) -> str:
    payload = {
        "cross_validation": {
            "method": config.cross_validation_method,
            "n_repeats": config.n_repeats,
            "n_splits": config.n_splits,
            "random_state": config.cross_validation_random_state,
        },
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "expected_counts": {
            "development": dict(config.expected_development_target_counts),
            "development_total": config.expected_development_total,
            "test": dict(config.expected_test_target_counts),
            "test_total": config.expected_test_total,
            "total": config.expected_total,
        },
        "holdout": {
            "method": config.holdout_method,
            "random_state": config.holdout_random_state,
            "test_fraction": config.test_fraction,
        },
        "id_column": config.id_column,
        "sort_by": list(config.sort_by),
        "target_column": config.target_column,
    }
    content = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()
