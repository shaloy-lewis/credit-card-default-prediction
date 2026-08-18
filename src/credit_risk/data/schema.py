"""Canonical schema and deterministic quality reporting for the UCI dataset."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from credit_risk.data.manifest import (
    DatasetExpectations as ManifestDatasetExpectations,
)
from credit_risk.data.manifest import DatasetManifest

SCHEMA_VERSION = "1.0.0"
MAX_REPORTED_ACCOUNT_IDS = 20

SOURCE_TO_CANONICAL: dict[str, str] = {
    "ID": "account_id",
    "X1": "credit_limit_ntd",
    "X2": "sex_code",
    "X3": "education_code",
    "X4": "marital_status_code",
    "X5": "age_years",
    "X6": "repayment_status_lag_0",
    "X7": "repayment_status_lag_1",
    "X8": "repayment_status_lag_2",
    "X9": "repayment_status_lag_3",
    "X10": "repayment_status_lag_4",
    "X11": "repayment_status_lag_5",
    "X12": "bill_amount_ntd_lag_0",
    "X13": "bill_amount_ntd_lag_1",
    "X14": "bill_amount_ntd_lag_2",
    "X15": "bill_amount_ntd_lag_3",
    "X16": "bill_amount_ntd_lag_4",
    "X17": "bill_amount_ntd_lag_5",
    "X18": "payment_amount_ntd_lag_0",
    "X19": "payment_amount_ntd_lag_1",
    "X20": "payment_amount_ntd_lag_2",
    "X21": "payment_amount_ntd_lag_3",
    "X22": "payment_amount_ntd_lag_4",
    "X23": "payment_amount_ntd_lag_5",
    "Y": "default_next_month",
}

SOURCE_COLUMNS = tuple(SOURCE_TO_CANONICAL)
CANONICAL_COLUMNS = tuple(SOURCE_TO_CANONICAL.values())
REPAYMENT_STATUS_COLUMNS = tuple(
    column for column in CANONICAL_COLUMNS if column.startswith("repayment_status_")
)
BILL_AMOUNT_COLUMNS = tuple(
    column for column in CANONICAL_COLUMNS if column.startswith("bill_amount_ntd_")
)
PAYMENT_AMOUNT_COLUMNS = tuple(
    column for column in CANONICAL_COLUMNS if column.startswith("payment_amount_ntd_")
)

CANONICAL_DTYPES: dict[str, str] = {
    "account_id": "int64",
    "credit_limit_ntd": "int64",
    "sex_code": "int8",
    "education_code": "int8",
    "marital_status_code": "int8",
    "age_years": "int16",
    **{column: "int8" for column in REPAYMENT_STATUS_COLUMNS},
    **{column: "int64" for column in BILL_AMOUNT_COLUMNS},
    **{column: "int64" for column in PAYMENT_AMOUNT_COLUMNS},
    "default_next_month": "int8",
}


@dataclass(frozen=True, slots=True)
class ValidationExpectations:
    """Dataset-level invariants that distinguish the locked source snapshot."""

    row_count: int = 30_000
    target_counts: tuple[tuple[int, int], ...] = ((0, 23_364), (1, 6_636))
    account_id_min: int = 1
    account_id_max: int = 30_000
    require_contiguous_account_ids: bool = True
    source_columns: tuple[str, ...] = SOURCE_COLUMNS
    source_to_canonical: tuple[tuple[str, str], ...] = tuple(SOURCE_TO_CANONICAL.items())
    canonical_dtypes: tuple[tuple[str, str], ...] = tuple(CANONICAL_DTYPES.items())

    def target_count_mapping(self) -> dict[int, int]:
        return dict(self.target_counts)


OFFICIAL_EXPECTATIONS = ValidationExpectations()


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One stable, machine-readable data-quality finding."""

    rule_id: str
    severity: Literal["error", "warning"]
    message: str
    count: int
    account_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_ids": list(self.account_ids),
            "count": self.count,
            "message": self.message,
            "rule_id": self.rule_id,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Deterministic result of validating one canonical dataset."""

    status: Literal["passed", "passed_with_warnings", "failed"]
    row_count: int
    column_count: int
    target_counts: Mapping[str, int]
    issues: tuple[ValidationIssue, ...] = ()
    schema_version: str = SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.status != "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_count": self.column_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "row_count": self.row_count,
            "schema_version": self.schema_version,
            "status": self.status,
            "target_counts": dict(sorted(self.target_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    """Validated canonical data and its reproducibility evidence."""

    data: pd.DataFrame
    report: QualityReport
    sha256: str


class DataContractError(ValueError):
    """Raised when a hard data-quality gate fails."""

    def __init__(self, report: QualityReport):
        self.report = report
        rule_ids = ", ".join(issue.rule_id for issue in report.issues if issue.severity == "error")
        super().__init__(f"Canonical data contract failed: {rule_ids}")


def _build_pandera_schema() -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {
        "account_id": pa.Column(pa.Int64, checks=pa.Check.greater_than(0), nullable=False),
        "credit_limit_ntd": pa.Column(pa.Int64, checks=pa.Check.greater_than(0), nullable=False),
        "sex_code": pa.Column(pa.Int8, checks=pa.Check.isin([1, 2]), nullable=False),
        "education_code": pa.Column(pa.Int8, checks=pa.Check.isin(range(0, 7)), nullable=False),
        "marital_status_code": pa.Column(
            pa.Int8, checks=pa.Check.isin(range(0, 4)), nullable=False
        ),
        "age_years": pa.Column(pa.Int16, checks=pa.Check.in_range(18, 100), nullable=False),
    }
    columns.update(
        {
            column: pa.Column(
                pa.Int8,
                checks=pa.Check.in_range(-2, 9),
                nullable=False,
            )
            for column in REPAYMENT_STATUS_COLUMNS
        }
    )
    columns.update({column: pa.Column(pa.Int64, nullable=False) for column in BILL_AMOUNT_COLUMNS})
    columns.update(
        {
            column: pa.Column(
                pa.Int64,
                checks=pa.Check.greater_than_or_equal_to(0),
                nullable=False,
            )
            for column in PAYMENT_AMOUNT_COLUMNS
        }
    )
    columns["default_next_month"] = pa.Column(pa.Int8, checks=pa.Check.isin([0, 1]), nullable=False)
    return pa.DataFrameSchema(columns, strict=True, ordered=True, coerce=False)


CANONICAL_SCHEMA = _build_pandera_schema()


def canonicalize_source_frame(
    source: pd.DataFrame,
    expectations: ValidationExpectations | DatasetManifest | ManifestDatasetExpectations = (
        OFFICIAL_EXPECTATIONS
    ),
) -> CanonicalizationResult:
    """Strictly rename and cast the official source frame, then validate it.

    No modelling transformation, category repair, imputation, or outlier handling
    occurs at this boundary. Hard failures raise :class:`DataContractError`.
    """

    resolved = _resolve_expectations(expectations)
    structural_issues = _source_structure_issues(source, resolved)
    if structural_issues:
        raise DataContractError(_quality_report(source, structural_issues))

    value_issues = _raw_value_issues(source)
    if value_issues:
        raise DataContractError(_quality_report(source, value_issues))

    canonical = (
        source.rename(columns=dict(resolved.source_to_canonical)).loc[:, CANONICAL_COLUMNS].copy()
    )
    try:
        canonical = canonical.astype("int64")
    except (OverflowError, TypeError, ValueError):
        issue = ValidationIssue(
            "schema.values.int64_range",
            "error",
            "Source values could not be represented as signed 64-bit integers.",
            len(source),
        )
        raise DataContractError(_quality_report(source, [issue])) from None
    canonical = canonical.sort_values("account_id", kind="stable").reset_index(drop=True)
    wide_issues = _canonical_value_issues(canonical, resolved, check_dtypes=False)
    if wide_issues:
        raise DataContractError(_quality_report(canonical, wide_issues))
    canonical = canonical.astype(dict(resolved.canonical_dtypes))
    report = validate_canonical_frame(canonical, resolved)
    return CanonicalizationResult(
        data=canonical,
        report=report,
        sha256=dataframe_sha256(canonical),
    )


def validate_canonical_frame(
    canonical: pd.DataFrame,
    expectations: ValidationExpectations | DatasetManifest | ManifestDatasetExpectations = (
        OFFICIAL_EXPECTATIONS
    ),
) -> QualityReport:
    """Validate canonical data and return warnings; raise on any hard gate."""

    resolved = _resolve_expectations(expectations)
    issues: list[ValidationIssue] = []
    issues.extend(_canonical_structure_issues(canonical))
    if not issues:
        issues.extend(_canonical_value_issues(canonical, resolved))

    errors = [issue for issue in issues if issue.severity == "error"]
    if not errors:
        try:
            CANONICAL_SCHEMA.validate(canonical, lazy=True)
        except pa.errors.SchemaErrors as error:
            issues.append(
                ValidationIssue(
                    rule_id="schema.pandera",
                    severity="error",
                    message="Canonical values or dtypes violate the strict Pandera schema.",
                    count=len(error.failure_cases),
                )
            )

    issues.extend(_accepted_anomaly_warnings(canonical) if not errors else ())
    report = _quality_report(canonical, issues)
    if not report.passed:
        raise DataContractError(report)
    return report


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a frame with stable UTF-8 and LF line endings."""

    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Return the digest of the deterministic CSV representation."""

    return hashlib.sha256(dataframe_csv_bytes(frame)).hexdigest()


def quality_report_bytes(report: QualityReport) -> bytes:
    """Serialize a report deterministically, excluding volatile timestamps."""

    return (json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_canonical_csv(frame: pd.DataFrame, path: str | Path) -> str:
    """Atomically write canonical data and return its SHA-256 digest."""

    content = dataframe_csv_bytes(frame)
    _atomic_write_bytes(Path(path), content)
    return hashlib.sha256(content).hexdigest()


def write_quality_report(report: QualityReport, path: str | Path) -> str:
    """Atomically write a deterministic quality report and return its digest."""

    content = quality_report_bytes(report)
    _atomic_write_bytes(Path(path), content)
    return hashlib.sha256(content).hexdigest()


def write_bytes_atomically(path: str | Path, content: bytes) -> str:
    """Atomically write arbitrary deterministic bytes and return their digest."""

    _atomic_write_bytes(Path(path), content)
    return hashlib.sha256(content).hexdigest()


def _resolve_expectations(
    expectations: ValidationExpectations | DatasetManifest | ManifestDatasetExpectations,
) -> ValidationExpectations:
    if isinstance(expectations, ValidationExpectations):
        return expectations

    if isinstance(expectations, DatasetManifest):
        manifest_expectations = expectations.expectations
        contract = expectations.canonical_contract
    else:
        manifest_expectations = expectations
        contract = None
    try:
        row_count = int(manifest_expectations.row_count)
        column_count = int(manifest_expectations.column_count)
        source_columns = tuple(manifest_expectations.source_columns)
        raw_target_counts = dict(manifest_expectations.target_counts)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "expectations must be ValidationExpectations or a validated DatasetManifest"
        ) from error

    if contract is None:
        mapping = SOURCE_TO_CANONICAL
    else:
        columns = tuple(contract.columns)
        mapping = {str(column.source_name): str(column.canonical_name) for column in columns}
        logical_dtypes = {str(column.logical_dtype) for column in columns}
        if logical_dtypes != {"integer"}:
            raise ValueError("The Phase 1 canonical contract must contain only integer columns.")

    if column_count != len(source_columns):
        raise ValueError("Manifest column_count does not match source_columns.")
    if source_columns != tuple(mapping):
        raise ValueError("Manifest source columns and canonical mapping are not ordered equally.")
    if tuple(mapping.values()) != CANONICAL_COLUMNS:
        raise ValueError("Manifest canonical names do not match schema version 1.0.0.")

    target_counts = tuple(
        sorted((int(label), int(count)) for label, count in raw_target_counts.items())
    )
    return ValidationExpectations(
        row_count=row_count,
        target_counts=target_counts,
        account_id_min=1,
        account_id_max=row_count,
        require_contiguous_account_ids=True,
        source_columns=source_columns,
        source_to_canonical=tuple(mapping.items()),
        canonical_dtypes=tuple(CANONICAL_DTYPES.items()),
    )


def _source_structure_issues(
    source: pd.DataFrame,
    expectations: ValidationExpectations,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    columns = list(source.columns)
    duplicate_count = len(columns) - len(set(columns))
    if duplicate_count:
        issues.append(
            ValidationIssue(
                "schema.source_columns.duplicate",
                "error",
                "Source columns must be unique.",
                duplicate_count,
            )
        )
    if tuple(columns) != expectations.source_columns:
        missing = sorted(set(expectations.source_columns) - set(columns))
        extra = sorted(set(columns) - set(expectations.source_columns))
        detail = f"missing={missing}, extra={extra}"
        if not missing and not extra:
            detail = "the official columns are not in their required order"
        issues.append(
            ValidationIssue(
                "schema.source_columns.exact",
                "error",
                f"Source columns must exactly match the official contract; {detail}.",
                len(missing) + len(extra) or 1,
            )
        )
    return issues


def _canonical_structure_issues(canonical: pd.DataFrame) -> list[ValidationIssue]:
    columns = list(canonical.columns)
    issues: list[ValidationIssue] = []
    duplicate_count = len(columns) - len(set(columns))
    if duplicate_count:
        issues.append(
            ValidationIssue(
                "schema.canonical_columns.duplicate",
                "error",
                "Canonical columns must be unique.",
                duplicate_count,
            )
        )
    if tuple(columns) != CANONICAL_COLUMNS:
        missing = sorted(set(CANONICAL_COLUMNS) - set(columns))
        extra = sorted(set(columns) - set(CANONICAL_COLUMNS))
        detail = f"missing={missing}, extra={extra}"
        if not missing and not extra:
            detail = "canonical columns are not in their required order"
        issues.append(
            ValidationIssue(
                "schema.canonical_columns.exact",
                "error",
                f"Canonical columns must exactly match schema {SCHEMA_VERSION}; {detail}.",
                len(missing) + len(extra) or 1,
            )
        )
    return issues


def _raw_value_issues(source: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    null_mask = source.isna().any(axis=1)
    if null_mask.any():
        issues.append(
            _row_issue(
                source,
                null_mask,
                "schema.values.null",
                "All source values must be non-null.",
            )
        )

    nonnumeric_columns = [
        column
        for column in SOURCE_COLUMNS
        if is_bool_dtype(source[column].dtype)
        or is_complex_dtype(source[column].dtype)
        or not is_numeric_dtype(source[column].dtype)
    ]
    if nonnumeric_columns:
        issues.append(
            ValidationIssue(
                "schema.values.numeric",
                "error",
                f"All source columns must contain numeric values: {nonnumeric_columns}.",
                len(nonnumeric_columns),
            )
        )
        return issues

    numeric = source.loc[:, SOURCE_COLUMNS].to_numpy(dtype=np.float64)
    finite_rows = np.isfinite(numeric).all(axis=1)
    if not finite_rows.all():
        issues.append(
            _row_issue(
                source,
                pd.Series(~finite_rows, index=source.index),
                "schema.values.finite",
                "All source values must be finite.",
            )
        )
    fractional_rows = np.logical_and(np.isfinite(numeric), numeric != np.trunc(numeric)).any(axis=1)
    if fractional_rows.any():
        issues.append(
            _row_issue(
                source,
                pd.Series(fractional_rows, index=source.index),
                "schema.values.integral",
                "All source values must be mathematically integral.",
            )
        )
    int64_info = np.iinfo(np.int64)
    out_of_range_rows = pd.Series(False, index=source.index)
    for column in SOURCE_COLUMNS:
        out_of_range_rows |= source[column].map(
            lambda value: bool(
                pd.notna(value)
                and np.isfinite(value)
                and (int(value) < int64_info.min or int(value) > int64_info.max)
            )
        )
    if out_of_range_rows.any():
        issues.append(
            _row_issue(
                source,
                pd.Series(out_of_range_rows, index=source.index),
                "schema.values.int64_range",
                "All source values must fit in a signed 64-bit integer.",
            )
        )
    return issues


def _canonical_value_issues(
    canonical: pd.DataFrame,
    expectations: ValidationExpectations,
    *,
    check_dtypes: bool = True,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if len(canonical) != expectations.row_count:
        issues.append(
            ValidationIssue(
                "dataset.row_count",
                "error",
                f"Expected {expectations.row_count} rows, found {len(canonical)}.",
                abs(len(canonical) - expectations.row_count),
            )
        )

    bad_dtype_columns = (
        [
            column
            for column, expected_dtype in expectations.canonical_dtypes
            if str(canonical[column].dtype) != expected_dtype
        ]
        if check_dtypes
        else []
    )
    if bad_dtype_columns:
        issues.append(
            ValidationIssue(
                "schema.dtypes.exact",
                "error",
                f"Canonical dtypes do not match the contract: {bad_dtype_columns}.",
                len(bad_dtype_columns),
            )
        )
        return issues

    duplicate_id_mask = canonical["account_id"].duplicated(keep=False)
    if duplicate_id_mask.any():
        issues.append(
            _row_issue(
                canonical,
                duplicate_id_mask,
                "schema.account_id.unique",
                "account_id must be unique.",
            )
        )

    nonpositive_id_mask = canonical["account_id"] <= 0
    if nonpositive_id_mask.any():
        issues.append(
            _row_issue(
                canonical,
                nonpositive_id_mask,
                "schema.account_id.positive",
                "account_id must be positive.",
            )
        )

    if expectations.require_contiguous_account_ids:
        expected_ids = set(range(expectations.account_id_min, expectations.account_id_max + 1))
        actual_ids = set(canonical["account_id"].astype(int))
        if actual_ids != expected_ids:
            issues.append(
                ValidationIssue(
                    "dataset.account_id.sequence",
                    "error",
                    "account_id values do not match the locked contiguous source sequence.",
                    len(actual_ids.symmetric_difference(expected_ids)),
                )
            )

    domain_checks: tuple[tuple[str, pd.Series, str], ...] = (
        (
            "schema.credit_limit.positive",
            canonical["credit_limit_ntd"] <= 0,
            "credit_limit_ntd must be positive.",
        ),
        (
            "schema.sex.domain",
            ~canonical["sex_code"].isin([1, 2]),
            "sex_code must be one of {1, 2}.",
        ),
        (
            "schema.education.domain",
            ~canonical["education_code"].isin(range(0, 7)),
            "education_code must be an integer from 0 through 6.",
        ),
        (
            "schema.marital_status.domain",
            ~canonical["marital_status_code"].isin(range(0, 4)),
            "marital_status_code must be an integer from 0 through 3.",
        ),
        (
            "schema.age.range",
            ~canonical["age_years"].between(18, 100),
            "age_years must be between 18 and 100 inclusive.",
        ),
        (
            "schema.repayment_status.domain",
            ~canonical.loc[:, REPAYMENT_STATUS_COLUMNS].isin(range(-2, 10)).all(axis=1),
            "Repayment status values must be integers from -2 through 9.",
        ),
        (
            "schema.payment_amount.nonnegative",
            (canonical.loc[:, PAYMENT_AMOUNT_COLUMNS] < 0).any(axis=1),
            "Payment amounts must be nonnegative.",
        ),
        (
            "schema.target.domain",
            ~canonical["default_next_month"].isin([0, 1]),
            "default_next_month must be binary {0, 1}.",
        ),
    )
    for rule_id, mask, message in domain_checks:
        if mask.any():
            issues.append(_row_issue(canonical, mask, rule_id, message))

    actual_target_counts = {
        int(key): int(value)
        for key, value in canonical["default_next_month"].value_counts().items()
        if int(key) in (0, 1)
    }
    if set(actual_target_counts) != {0, 1}:
        issues.append(
            ValidationIssue(
                "dataset.target.classes",
                "error",
                "Both target classes 0 and 1 must be present.",
                2 - len(set(actual_target_counts)),
            )
        )
    expected_target_counts = expectations.target_count_mapping()
    if actual_target_counts != expected_target_counts:
        issues.append(
            ValidationIssue(
                "dataset.target.counts",
                "error",
                f"Expected target counts {expected_target_counts}, found {actual_target_counts}.",
                sum(
                    abs(actual_target_counts.get(label, 0) - expected_count)
                    for label, expected_count in expected_target_counts.items()
                ),
            )
        )
    return issues


def _accepted_anomaly_warnings(canonical: pd.DataFrame) -> Sequence[ValidationIssue]:
    if tuple(canonical.columns) != CANONICAL_COLUMNS:
        return ()
    warnings: list[ValidationIssue] = []
    warning_checks: tuple[tuple[str, pd.Series, str], ...] = (
        (
            "warning.education.undocumented_code",
            canonical["education_code"].isin([0, 5, 6]),
            "Education codes 0, 5, and 6 are retained source anomalies.",
        ),
        (
            "warning.marital_status.undocumented_code",
            canonical["marital_status_code"] == 0,
            "Marital-status code 0 is retained as a source anomaly.",
        ),
        (
            "warning.repayment_status.undocumented_code",
            canonical.loc[:, REPAYMENT_STATUS_COLUMNS].isin([-2, 0]).any(axis=1),
            "Repayment-status codes -2 and 0 are retained source anomalies.",
        ),
        (
            "warning.bill_amount.negative",
            (canonical.loc[:, BILL_AMOUNT_COLUMNS] < 0).any(axis=1),
            "Negative bill amounts are retained rather than silently corrected.",
        ),
        (
            "warning.features.duplicate",
            canonical.drop(columns="account_id").duplicated(keep="first"),
            "Duplicate non-identifier rows are retained and reported.",
        ),
    )
    for rule_id, mask, message in warning_checks:
        if mask.any():
            warnings.append(_row_issue(canonical, mask, rule_id, message, severity="warning"))
    return warnings


def _row_issue(
    frame: pd.DataFrame,
    mask: pd.Series,
    rule_id: str,
    message: str,
    *,
    severity: Literal["error", "warning"] = "error",
) -> ValidationIssue:
    normalized_mask = pd.Series(mask, index=frame.index).fillna(True).astype(bool)
    return ValidationIssue(
        rule_id=rule_id,
        severity=severity,
        message=message,
        count=int(normalized_mask.sum()),
        account_ids=_sample_account_ids(frame.loc[normalized_mask]),
    )


def _sample_account_ids(frame: pd.DataFrame) -> tuple[int, ...]:
    id_column = "account_id" if "account_id" in frame else "ID" if "ID" in frame else None
    if id_column is None or list(frame.columns).count(id_column) != 1:
        return ()
    numeric_ids = pd.to_numeric(frame[id_column], errors="coerce")
    numeric_ids = numeric_ids[np.isfinite(numeric_ids)]
    numeric_ids = numeric_ids[numeric_ids == np.trunc(numeric_ids)]
    return tuple(sorted({int(value) for value in numeric_ids})[:MAX_REPORTED_ACCOUNT_IDS])


def _quality_report(frame: pd.DataFrame, issues: Sequence[ValidationIssue]) -> QualityReport:
    ordered_issues = tuple(
        sorted(issues, key=lambda issue: (issue.severity != "error", issue.rule_id))
    )
    has_errors = any(issue.severity == "error" for issue in ordered_issues)
    has_warnings = any(issue.severity == "warning" for issue in ordered_issues)
    status: Literal["passed", "passed_with_warnings", "failed"]
    if has_errors:
        status = "failed"
    elif has_warnings:
        status = "passed_with_warnings"
    else:
        status = "passed"

    target_name = (
        "default_next_month" if "default_next_month" in frame else "Y" if "Y" in frame else None
    )
    target_counts: dict[str, int] = {}
    if target_name is not None and list(frame.columns).count(target_name) == 1:
        target_counts = {
            str(key): int(value)
            for key, value in frame[target_name].value_counts(dropna=False).items()
            if pd.notna(key)
        }
    return QualityReport(
        status=status,
        row_count=len(frame),
        column_count=len(frame.columns),
        target_counts=target_counts,
        issues=ordered_issues,
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
