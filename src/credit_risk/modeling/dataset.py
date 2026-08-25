"""Leakage-safe development views backed by verified Phase 1 products."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.data.manifest import (
    DEFAULT_DATASET_MANIFEST_PATH,
    DEFAULT_SPLIT_CONFIG_PATH,
    DatasetManifest,
    SplitConfig,
    load_dataset_manifest,
    load_split_config,
)
from credit_risk.data.workflow import DataWorkflowError, DataWorkflowResult, verify_dataset
from credit_risk.modeling.contracts import (
    DEFAULT_FEATURE_CONTRACT_PATH,
    FeatureContract,
    parse_feature_contract,
)


class ModelingDataError(RuntimeError):
    """Raised when a governed modelling view cannot be constructed safely."""


@dataclass(frozen=True, slots=True)
class ModelingLineage:
    """Hash-rich evidence attached to every governed development view."""

    dataset_id: str
    dataset_version: str
    source_sha256: str
    dataset_manifest_sha256: str
    canonical_sha256: str
    quality_report_sha256: str
    split_config_sha256: str
    assignment_sha256: str
    split_manifest_sha256: str
    reviewed_split_lock_sha256: str
    feature_contract_sha256: str


@dataclass(frozen=True, slots=True)
class FoldSlice:
    """One reviewed development training/validation fold."""

    repeat_index: int
    fold_index: int
    train_account_ids: pd.Index
    validation_account_ids: pd.Index
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series


@dataclass(frozen=True, slots=True)
class GovernedDevelopmentData:
    """Development-only predictor, label, audit, and reviewed-fold views."""

    account_ids: pd.Index
    predictors: pd.DataFrame
    target: pd.Series
    audit: pd.DataFrame
    assignments: pd.DataFrame
    lineage: ModelingLineage
    n_splits: int
    n_repeats: int

    @property
    def X(self) -> pd.DataFrame:
        """Return the governed predictor view."""

        return self.predictors

    @property
    def y(self) -> pd.Series:
        """Return the governed development target."""

        return self.target

    def fold(self, repeat_index: int, fold_index: int) -> FoldSlice:
        """Return train/validation data from one reviewed assignment column."""

        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
            raise ModelingDataError("repeat_index must be an integer")
        if isinstance(fold_index, bool) or not isinstance(fold_index, int):
            raise ModelingDataError("fold_index must be an integer")
        if repeat_index < 0 or repeat_index >= self.n_repeats:
            raise ModelingDataError(f"repeat_index must be between 0 and {self.n_repeats - 1}")
        if fold_index < 0 or fold_index >= self.n_splits:
            raise ModelingDataError(f"fold_index must be between 0 and {self.n_splits - 1}")

        assignment_column = f"cv_fold_r{repeat_index}"
        validation_mask = self.assignments[assignment_column].eq(fold_index)
        if not validation_mask.any():
            raise ModelingDataError(
                f"Reviewed assignments contain no rows for repeat {repeat_index}, fold {fold_index}"
            )
        train_ids = self.assignments.index[~validation_mask].copy()
        validation_ids = self.assignments.index[validation_mask].copy()

        return FoldSlice(
            repeat_index=repeat_index,
            fold_index=fold_index,
            train_account_ids=train_ids,
            validation_account_ids=validation_ids,
            X_train=self.predictors.loc[train_ids].copy(),
            X_validation=self.predictors.loc[validation_ids].copy(),
            y_train=self.target.loc[train_ids].copy(),
            y_validation=self.target.loc[validation_ids].copy(),
        )


def load_governed_development_data(
    data_root: str | Path = "data",
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    manifest_path: str | Path = DEFAULT_DATASET_MANIFEST_PATH,
    split_config_path: str | Path = DEFAULT_SPLIT_CONFIG_PATH,
) -> GovernedDevelopmentData:
    """Verify Phase 1 offline, then expose only governed development data.

    The sealed test partition is checked for assignment integrity but is never
    returned by this interface.
    """

    contract_path = Path(feature_contract_path)
    try:
        contract_bytes = contract_path.read_bytes()
        contract = parse_feature_contract(contract_bytes, source=contract_path)
        manifest = load_dataset_manifest(manifest_path)
        split_config = load_split_config(split_config_path)
        verification = verify_dataset(data_root, manifest_path, split_config_path)
    except (OSError, ValueError, DataWorkflowError) as error:
        raise ModelingDataError(f"Governed data verification failed: {error}") from error

    _validate_contract_parity(contract, manifest, split_config)
    feature_contract_sha256 = _sha256(contract_bytes)
    reviewed_lock_sha256 = _file_sha256(verification.paths.reviewed_split_lock)
    observed_lineage = {
        "source_sha256": verification.source_sha256,
        "dataset_manifest_sha256": verification.dataset_manifest_sha256,
        "canonical_sha256": verification.canonical_sha256,
        "split_config_sha256": verification.split_config_sha256,
        "assignment_sha256": verification.assignment_sha256,
        "reviewed_split_lock_sha256": reviewed_lock_sha256,
    }
    expected_lineage = contract.lineage.model_dump()
    mismatches = [
        name for name, expected in expected_lineage.items() if observed_lineage[name] != expected
    ]
    if mismatches:
        details = ", ".join(
            f"{name}: expected={expected_lineage[name]}, observed={observed_lineage[name]}"
            for name in mismatches
        )
        raise ModelingDataError(f"Feature-contract lineage mismatch: {details}")
    if not verification.reviewed_lock_verified:
        raise ModelingDataError("Phase 1 verification did not approve the reviewed split lock")

    canonical = _read_verified_csv(
        verification.paths.canonical,
        "canonical dataset",
        verification.canonical_sha256,
    )
    assignments = _read_verified_csv(
        verification.paths.split_assignments,
        "split assignments",
        verification.assignment_sha256,
    )
    return _build_development_view(
        canonical=canonical,
        assignments=assignments,
        contract=contract,
        expected_canonical_columns=tuple(
            column.canonical_name for column in manifest.canonical_contract.columns
        ),
        feature_contract_sha256=feature_contract_sha256,
        verification=verification,
        reviewed_lock_sha256=reviewed_lock_sha256,
    )


def _validate_contract_parity(
    contract: FeatureContract,
    manifest: DatasetManifest,
    split_config: SplitConfig,
) -> None:
    manifest_identity = (manifest.dataset_id, manifest.dataset_version)
    contract_identity = (contract.dataset.dataset_id, contract.dataset.dataset_version)
    if manifest_identity != contract_identity:
        raise ModelingDataError(
            "Feature contract and dataset manifest identify different snapshots: "
            f"{contract_identity} versus {manifest_identity}"
        )
    split_identity = (
        split_config.dataset_id,
        split_config.dataset_version,
    )
    if split_identity != contract_identity:
        raise ModelingDataError(
            "Feature contract and split config identify different snapshots: "
            f"{contract_identity} versus {split_identity}"
        )
    if split_config.id_column != contract.columns.id_column:
        raise ModelingDataError("Feature contract identifier differs from the split config")
    if split_config.target_column != contract.columns.target_column:
        raise ModelingDataError("Feature contract target differs from the split config")
    cross_validation = split_config.cross_validation
    if (
        cross_validation.n_splits != contract.cross_validation.n_splits
        or cross_validation.n_repeats != contract.cross_validation.n_repeats
    ):
        raise ModelingDataError("Feature contract CV shape differs from the split config")
    expected_development = split_config.expected_counts.development
    if expected_development.total != contract.expected_development.rows:
        raise ModelingDataError("Feature contract development count differs from the split config")
    if dict(expected_development.target_counts) != contract.expected_development.target_counts:
        raise ModelingDataError(
            "Feature contract development target counts differ from the split config"
        )


def _read_csv(path: Path, description: str) -> pd.DataFrame:
    try:
        content = path.read_bytes()
        return pd.read_csv(io.BytesIO(content), encoding="utf-8", low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ModelingDataError(f"Unable to read verified {description} {path}: {error}") from error


def _read_verified_csv(
    path: Path,
    description: str,
    expected_sha256: str,
) -> pd.DataFrame:
    """Parse the same bytes whose digest is bound to the verified lineage."""

    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelingDataError(f"Unable to read verified {description} {path}: {error}") from error
    observed_sha256 = _sha256(content)
    if observed_sha256 != expected_sha256:
        raise ModelingDataError(
            f"Verified {description} changed before modelling: "
            f"expected_sha256={expected_sha256}, observed_sha256={observed_sha256}"
        )
    try:
        return pd.read_csv(io.BytesIO(content), encoding="utf-8", low_memory=False)
    except (UnicodeError, pd.errors.ParserError) as error:
        raise ModelingDataError(
            f"Unable to parse verified {description} {path}: {error}"
        ) from error


def _build_development_view(
    *,
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
    contract: FeatureContract,
    expected_canonical_columns: tuple[str, ...],
    feature_contract_sha256: str,
    verification: DataWorkflowResult,
    reviewed_lock_sha256: str,
) -> GovernedDevelopmentData:
    id_column = contract.columns.id_column
    target_column = contract.columns.target_column
    expected_assignment_columns = (
        id_column,
        "partition",
        *(f"cv_fold_r{repeat}" for repeat in range(contract.cross_validation.n_repeats)),
    )
    missing_canonical = sorted(set(expected_canonical_columns) - set(canonical.columns))
    if missing_canonical:
        raise ModelingDataError(
            f"Verified canonical dataset is missing required columns: {missing_canonical}"
        )
    missing_assignments = sorted(set(expected_assignment_columns) - set(assignments.columns))
    if missing_assignments:
        raise ModelingDataError(
            f"Verified split assignments are missing required columns: {missing_assignments}"
        )
    if tuple(canonical.columns) != expected_canonical_columns:
        raise ModelingDataError("Verified canonical columns differ from the dataset manifest order")
    governed_columns = {
        id_column,
        target_column,
        *contract.columns.predictor_columns,
        *contract.columns.audit_columns,
    }
    if governed_columns != set(canonical.columns):
        missing = sorted(governed_columns - set(canonical.columns))
        unexpected = sorted(set(canonical.columns) - governed_columns)
        raise ModelingDataError(
            f"Feature contract does not cover canonical columns: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if tuple(assignments.columns) != expected_assignment_columns:
        raise ModelingDataError(
            "Split assignment columns differ from the reviewed repeat contract: "
            f"expected={expected_assignment_columns}, observed={tuple(assignments.columns)}"
        )
    _validate_identifier_column(canonical, id_column, "canonical dataset")
    _validate_identifier_column(assignments, id_column, "split assignments")
    if not canonical[id_column].is_monotonic_increasing:
        raise ModelingDataError("Canonical account IDs must be sorted in ascending order")
    if not assignments[id_column].is_monotonic_increasing:
        raise ModelingDataError("Split assignment account IDs must be sorted in ascending order")

    joined = canonical.merge(
        assignments,
        on=id_column,
        how="outer",
        sort=True,
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        left_only = int(joined["_merge"].eq("left_only").sum())
        right_only = int(joined["_merge"].eq("right_only").sum())
        raise ModelingDataError(
            "Canonical and assignment IDs do not have complete coverage: "
            f"canonical_only={left_only}, assignment_only={right_only}"
        )
    joined = joined.drop(columns="_merge")
    partitions = set(joined["partition"].dropna().astype(str))
    if partitions != {"development", "test"}:
        raise ModelingDataError(
            f"Assignments must contain exactly development and test partitions, got {partitions}"
        )

    development = joined.loc[joined["partition"].eq("development")].copy()
    if len(development) != contract.expected_development.rows:
        raise ModelingDataError(
            "Development row count differs from the feature contract: "
            f"expected={contract.expected_development.rows}, observed={len(development)}"
        )
    observed_target_counts = {
        str(label): int(count)
        for label, count in development[target_column].value_counts().sort_index().items()
    }
    if observed_target_counts != contract.expected_development.target_counts:
        raise ModelingDataError(
            "Development target counts differ from the feature contract: "
            f"expected={contract.expected_development.target_counts}, "
            f"observed={observed_target_counts}"
        )

    fold_columns = list(expected_assignment_columns[2:])
    for fold_column in fold_columns:
        numeric = pd.to_numeric(development[fold_column], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ModelingDataError(f"Development fold assignments must be integral: {fold_column}")
        integer_folds = numeric.astype("int8")
        expected_folds = set(range(contract.cross_validation.n_splits))
        observed_folds = set(int(value) for value in integer_folds.unique())
        if observed_folds != expected_folds:
            raise ModelingDataError(
                f"{fold_column} must contain every reviewed fold {sorted(expected_folds)}, "
                f"got {sorted(observed_folds)}"
            )
        development[fold_column] = integer_folds

    development = development.sort_values(id_column, kind="mergesort").reset_index(drop=True)
    account_ids = pd.Index(development[id_column].astype("int64"), name=id_column)
    predictors = development.loc[:, list(contract.columns.predictor_columns)].copy()
    predictors.index = account_ids
    numeric_predictors = predictors.apply(pd.to_numeric, errors="coerce")
    if (
        numeric_predictors.isna().any().any()
        or not np.isfinite(numeric_predictors.to_numpy(dtype="float64")).all()
    ):
        raise ModelingDataError("Predictor view contains non-numeric or non-finite values")

    target = development[target_column].astype("int8").copy()
    target.index = account_ids
    audit = development.loc[:, [id_column, target_column, *contract.columns.audit_columns]].copy()
    audit.index = account_ids
    governed_assignments = development.loc[:, ["partition", *fold_columns]].copy()
    governed_assignments.index = account_ids

    lineage = ModelingLineage(
        dataset_id=contract.dataset.dataset_id,
        dataset_version=contract.dataset.dataset_version,
        source_sha256=verification.source_sha256,
        dataset_manifest_sha256=verification.dataset_manifest_sha256,
        canonical_sha256=verification.canonical_sha256,
        quality_report_sha256=verification.quality_report_sha256,
        split_config_sha256=verification.split_config_sha256,
        assignment_sha256=verification.assignment_sha256,
        split_manifest_sha256=verification.split_manifest_sha256,
        reviewed_split_lock_sha256=reviewed_lock_sha256,
        feature_contract_sha256=feature_contract_sha256,
    )
    return GovernedDevelopmentData(
        account_ids=account_ids,
        predictors=predictors,
        target=target,
        audit=audit,
        assignments=governed_assignments,
        lineage=lineage,
        n_splits=contract.cross_validation.n_splits,
        n_repeats=contract.cross_validation.n_repeats,
    )


def _validate_identifier_column(frame: pd.DataFrame, column: str, description: str) -> None:
    if frame[column].isna().any():
        raise ModelingDataError(f"{description} contains null {column} values")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ModelingDataError(f"{description} contains non-integral {column} values")
    if numeric.duplicated().any():
        raise ModelingDataError(f"{description} contains duplicate {column} values")


def _file_sha256(path: Path) -> str:
    try:
        return _sha256(path.read_bytes())
    except OSError as error:
        raise ModelingDataError(f"Unable to hash reviewed file {path}: {error}") from error


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
