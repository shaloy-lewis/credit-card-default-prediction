"""End-to-end orchestration for governed Phase 1 data products.

The workflow deliberately keeps acquisition, validation, and splitting as
separate contracts.  This module joins them into a transaction: every output
is computed and checked before any previously valid product is replaced.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from credit_risk.data.acquisition import (
    DataAcquisitionError,
    fetch_source,
    resolve_raw_data_path,
    verify_source_file,
)
from credit_risk.data.manifest import (
    DatasetManifest,
    ManifestLoadError,
    SplitConfig,
    load_dataset_manifest,
    load_split_config,
)
from credit_risk.data.schema import (
    SOURCE_TO_CANONICAL,
    DataContractError,
    canonicalize_source_frame,
    dataframe_csv_bytes,
    quality_report_bytes,
    write_bytes_atomically,
)
from credit_risk.data.splits import (
    SPLIT_ASSIGNMENTS_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    SplitArtifacts,
    build_split_artifacts,
    split_manifest_bytes,
)

CANONICAL_FILENAME = "canonical.csv"
QUALITY_REPORT_FILENAME = "quality_report.json"
VALIDATION_FAILURE_FILENAME = "quality_report.json"


class DataWorkflowError(RuntimeError):
    """Raised when a complete data build or verification cannot be trusted."""


@dataclass(frozen=True, slots=True)
class DataProductPaths:
    """Resolved raw, generated, and reviewed-lock locations for one snapshot."""

    raw: Path
    canonical: Path
    quality_report: Path
    split_assignments: Path
    split_manifest: Path
    reviewed_split_lock: Path


@dataclass(frozen=True, slots=True)
class DataWorkflowResult:
    """Hash-rich evidence returned by a successful build or verification."""

    paths: DataProductPaths
    source_sha256: str
    dataset_manifest_sha256: str
    canonical_sha256: str
    quality_report_sha256: str
    split_config_sha256: str
    assignment_sha256: str
    split_manifest_sha256: str
    reviewed_lock_verified: bool
    source_downloaded: bool
    changed_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _Contracts:
    manifest: DatasetManifest
    split_config: SplitConfig
    manifest_path: Path
    split_config_path: Path
    manifest_sha256: str
    split_config_sha256: str


@dataclass(frozen=True, slots=True)
class _ComputedProducts:
    canonical: bytes
    quality_report: bytes
    split_assignments: bytes
    split_manifest: bytes
    canonical_sha256: str
    quality_report_sha256: str
    assignment_sha256: str
    split_manifest_sha256: str


def build_dataset(
    data_root: str | Path,
    manifest_path: str | Path,
    split_config_path: str | Path,
    *,
    offline: bool = False,
) -> DataWorkflowResult:
    """Build canonical data and sealed assignments as one validated transaction.

    A reviewed ``*.lock.json`` next to the split config is treated as an
    approval boundary when present.  The workflow compares it byte-for-byte
    but never creates or modifies it.
    """

    root = Path(data_root)
    contracts = _load_contracts(manifest_path, split_config_path)
    paths = _resolve_product_paths(root, contracts)

    if offline and not paths.raw.exists():
        raise DataWorkflowError(
            f"Offline build requires the pinned raw source at {paths.raw}. "
            "Run `credit-risk data fetch` while online first."
        )

    try:
        if offline:
            verify_source_file(paths.raw, contracts.manifest)
            raw_path = paths.raw
            source_downloaded = False
        else:
            acquisition = fetch_source(contracts.manifest, root)
            raw_path = acquisition.path
            source_downloaded = acquisition.downloaded
    except DataAcquisitionError as error:
        mode = "offline source verification" if offline else "source acquisition"
        raise DataWorkflowError(f"Data {mode} failed: {error}") from error

    products = _compute_products(
        raw_path=raw_path,
        data_root=root,
        contracts=contracts,
    )
    reviewed_lock_verified = _compare_reviewed_lock(
        paths.reviewed_split_lock,
        products.split_manifest,
        required=False,
    )

    payloads = {
        paths.canonical: products.canonical,
        paths.quality_report: products.quality_report,
        paths.split_assignments: products.split_assignments,
        paths.split_manifest: products.split_manifest,
    }
    changed_paths = _stage_and_promote(root, payloads)
    return _result(
        paths=paths,
        contracts=contracts,
        products=products,
        reviewed_lock_verified=reviewed_lock_verified,
        source_downloaded=source_downloaded,
        changed_paths=changed_paths,
    )


def verify_dataset(
    data_root: str | Path,
    manifest_path: str | Path,
    split_config_path: str | Path,
) -> DataWorkflowResult:
    """Recompute and verify every raw-to-split product without network access."""

    root = Path(data_root)
    contracts = _load_contracts(manifest_path, split_config_path)
    paths = _resolve_product_paths(root, contracts)
    try:
        verify_source_file(paths.raw, contracts.manifest)
    except DataAcquisitionError as error:
        raise DataWorkflowError(f"Offline source verification failed: {error}") from error

    products = _compute_products(raw_path=paths.raw, data_root=root, contracts=contracts)
    expected = {
        paths.canonical: products.canonical,
        paths.quality_report: products.quality_report,
        paths.split_assignments: products.split_assignments,
        paths.split_manifest: products.split_manifest,
    }
    differences = [
        _byte_difference(path, content)
        for path, content in expected.items()
        if not path.is_file() or path.read_bytes() != content
    ]
    if differences:
        raise DataWorkflowError(
            "Offline verification found missing or changed generated products: "
            + "; ".join(differences)
            + ". Re-run `credit-risk data build` after reviewing the source and config."
        )

    reviewed_lock_verified = _compare_reviewed_lock(
        paths.reviewed_split_lock,
        products.split_manifest,
        required=True,
    )
    return _result(
        paths=paths,
        contracts=contracts,
        products=products,
        reviewed_lock_verified=reviewed_lock_verified,
        source_downloaded=False,
        changed_paths=(),
    )


def _load_contracts(
    manifest_path: str | Path,
    split_config_path: str | Path,
) -> _Contracts:
    source_manifest_path = Path(manifest_path)
    split_path = Path(split_config_path)
    try:
        manifest_bytes = source_manifest_path.read_bytes()
        split_config_bytes = split_path.read_bytes()
        manifest = load_dataset_manifest(source_manifest_path)
        split_config = load_split_config(split_path)
    except (ManifestLoadError, OSError) as error:
        raise DataWorkflowError(
            f"Unable to load the governed data configuration: {error}"
        ) from error

    if (manifest.dataset_id, manifest.dataset_version) != (
        split_config.dataset_id,
        split_config.dataset_version,
    ):
        raise DataWorkflowError(
            "Dataset manifest and split config identify different dataset snapshots: "
            f"{manifest.dataset_id}/{manifest.dataset_version} versus "
            f"{split_config.dataset_id}/{split_config.dataset_version}."
        )
    manifest_mapping = tuple(
        (column.source_name, column.canonical_name)
        for column in manifest.canonical_contract.columns
    )
    if manifest_mapping != tuple(SOURCE_TO_CANONICAL.items()):
        raise DataWorkflowError(
            "Dataset manifest mapping does not match canonical schema version 1.0.0."
        )
    if split_config.id_column != "account_id" or split_config.target_column != (
        "default_next_month"
    ):
        raise DataWorkflowError(
            "Split config must use account_id and default_next_month from the canonical schema."
        )

    return _Contracts(
        manifest=manifest,
        split_config=split_config,
        manifest_path=source_manifest_path,
        split_config_path=split_path,
        manifest_sha256=_sha256(manifest_bytes),
        split_config_sha256=_sha256(split_config_bytes),
    )


def _resolve_product_paths(root: Path, contracts: _Contracts) -> DataProductPaths:
    dataset_id = contracts.manifest.dataset_id
    dataset_version = contracts.manifest.dataset_version
    processed_dir = root / "processed" / dataset_id / dataset_version
    split_dir = root / "splits" / dataset_id / dataset_version
    return DataProductPaths(
        raw=resolve_raw_data_path(contracts.manifest, root),
        canonical=processed_dir / CANONICAL_FILENAME,
        quality_report=processed_dir / QUALITY_REPORT_FILENAME,
        split_assignments=split_dir / SPLIT_ASSIGNMENTS_FILENAME,
        split_manifest=split_dir / SPLIT_MANIFEST_FILENAME,
        reviewed_split_lock=contracts.split_config_path.with_suffix(".lock.json"),
    )


def _compute_products(
    *,
    raw_path: Path,
    data_root: Path,
    contracts: _Contracts,
) -> _ComputedProducts:
    try:
        source = pd.read_csv(raw_path, encoding="utf-8", low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise DataWorkflowError(
            f"Unable to parse the pinned source CSV {raw_path}: {error}"
        ) from error

    try:
        canonical = canonicalize_source_frame(source, contracts.manifest)
    except DataContractError as error:
        failure_path = _write_contract_failure(data_root, contracts.manifest, error)
        raise DataWorkflowError(f"{error}. Deterministic failure report: {failure_path}") from error
    except (TypeError, ValueError) as error:
        raise DataWorkflowError(f"Canonical contract configuration is invalid: {error}") from error

    canonical_bytes = dataframe_csv_bytes(canonical.data)
    canonical_sha256 = _sha256(canonical_bytes)
    if canonical_sha256 != canonical.sha256:
        raise DataWorkflowError("Canonical serialization changed after validation.")
    report_bytes = quality_report_bytes(canonical.report)

    try:
        split = build_split_artifacts(
            canonical.data,
            contracts.split_config,
            source_sha256=contracts.manifest.source.sha256,
            canonical_sha256=canonical_sha256,
            config_sha256=contracts.split_config_sha256,
        )
    except ValueError as error:
        raise DataWorkflowError(f"Sealed split validation failed: {error}") from error

    assignment_bytes = dataframe_csv_bytes(split.assignments)
    split_bytes = split_manifest_bytes(split.manifest)
    _assert_split_digests(split, assignment_bytes, split_bytes)
    return _ComputedProducts(
        canonical=canonical_bytes,
        quality_report=report_bytes,
        split_assignments=assignment_bytes,
        split_manifest=split_bytes,
        canonical_sha256=canonical_sha256,
        quality_report_sha256=_sha256(report_bytes),
        assignment_sha256=split.assignment_sha256,
        split_manifest_sha256=split.lock_sha256,
    )


def _assert_split_digests(
    split: SplitArtifacts,
    assignment_bytes: bytes,
    split_manifest: bytes,
) -> None:
    if _sha256(assignment_bytes) != split.assignment_sha256:
        raise DataWorkflowError("Split assignment serialization changed after validation.")
    if _sha256(split_manifest) != split.lock_sha256:
        raise DataWorkflowError("Split manifest serialization changed after validation.")


def _write_contract_failure(
    data_root: Path,
    manifest: DatasetManifest,
    error: DataContractError,
) -> Path:
    content = quality_report_bytes(error.report)
    report_sha256 = _sha256(content)
    path = (
        data_root
        / "quarantine"
        / manifest.dataset_id
        / manifest.dataset_version
        / "validation"
        / report_sha256
        / VALIDATION_FAILURE_FILENAME
    )
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise DataWorkflowError(
                f"Validation quarantine path contains conflicting content: {path}"
            )
        return path
    write_bytes_atomically(path, content)
    return path


def _compare_reviewed_lock(path: Path, expected: bytes, *, required: bool) -> bool:
    if not path.exists():
        if required:
            raise DataWorkflowError(
                f"Reviewed split lock is missing: {path}. Generate the data products, review "
                "the split evidence, and commit an exact copy at this path before verification."
            )
        return False
    if not path.is_file():
        raise DataWorkflowError(f"Reviewed split lock is not a regular file: {path}")
    observed = path.read_bytes()
    if observed != expected:
        raise DataWorkflowError(
            "Generated split evidence differs from the reviewed lock "
            f"{path}: generated_sha256={_sha256(expected)}, "
            f"reviewed_sha256={_sha256(observed)}. Review the lineage or dependency change; "
            "the workflow will not update the lock automatically."
        )
    return True


def _stage_and_promote(root: Path, payloads: dict[Path, bytes]) -> tuple[Path, ...]:
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase1-stage-", dir=root) as temporary_directory:
        staging_root = Path(temporary_directory)
        staged: dict[Path, Path] = {}
        for index, (destination, content) in enumerate(payloads.items()):
            stage_path = staging_root / f"{index:02d}-{destination.name}"
            write_bytes_atomically(stage_path, content)
            if stage_path.read_bytes() != content:
                raise DataWorkflowError(
                    f"Staged data product failed byte verification: {stage_path}"
                )
            staged[destination] = stage_path
        return _promote_staged(staged)


def _promote_staged(staged: dict[Path, Path]) -> tuple[Path, ...]:
    changed: list[Path] = []
    previous: dict[Path, bytes | None] = {}
    for destination, stage_path in staged.items():
        if destination.exists() and not destination.is_file():
            raise DataWorkflowError(
                f"Generated-product destination is not a regular file: {destination}"
            )
        staged_bytes = stage_path.read_bytes()
        if destination.is_file() and destination.read_bytes() == staged_bytes:
            continue
        previous[destination] = destination.read_bytes() if destination.is_file() else None

    try:
        for destination in previous:
            write_bytes_atomically(destination, staged[destination].read_bytes())
            changed.append(destination)
    except OSError as error:
        rollback_errors: list[str] = []
        for destination in reversed(changed):
            try:
                prior_content = previous[destination]
                if prior_content is None:
                    destination.unlink(missing_ok=True)
                else:
                    write_bytes_atomically(destination, prior_content)
            except OSError as rollback_error:
                rollback_errors.append(f"{destination}: {rollback_error}")
        detail = f" Rollback errors: {'; '.join(rollback_errors)}" if rollback_errors else ""
        raise DataWorkflowError(
            f"Unable to promote staged data products: {error}.{detail}"
        ) from error
    return tuple(changed)


def _byte_difference(path: Path, expected: bytes) -> str:
    expected_sha256 = _sha256(expected)
    if not path.exists():
        return f"{path} is missing (expected_sha256={expected_sha256})"
    if not path.is_file():
        return f"{path} is not a regular file (expected_sha256={expected_sha256})"
    return f"{path} has sha256={_sha256(path.read_bytes())} (expected_sha256={expected_sha256})"


def _result(
    *,
    paths: DataProductPaths,
    contracts: _Contracts,
    products: _ComputedProducts,
    reviewed_lock_verified: bool,
    source_downloaded: bool,
    changed_paths: tuple[Path, ...],
) -> DataWorkflowResult:
    return DataWorkflowResult(
        paths=paths,
        source_sha256=contracts.manifest.source.sha256,
        dataset_manifest_sha256=contracts.manifest_sha256,
        canonical_sha256=products.canonical_sha256,
        quality_report_sha256=products.quality_report_sha256,
        split_config_sha256=contracts.split_config_sha256,
        assignment_sha256=products.assignment_sha256,
        split_manifest_sha256=products.split_manifest_sha256,
        reviewed_lock_verified=reviewed_lock_verified,
        source_downloaded=source_downloaded,
        changed_paths=changed_paths,
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
