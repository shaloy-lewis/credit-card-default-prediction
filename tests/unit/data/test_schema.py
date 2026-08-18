"""Tests for the strict canonical data contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.manifest import load_dataset_manifest
from credit_risk.data.schema import (
    BILL_AMOUNT_COLUMNS,
    CANONICAL_COLUMNS,
    PAYMENT_AMOUNT_COLUMNS,
    REPAYMENT_STATUS_COLUMNS,
    SOURCE_COLUMNS,
    CanonicalizationResult,
    DataContractError,
    QualityReport,
    ValidationExpectations,
    ValidationIssue,
    canonicalize_source_frame,
    dataframe_csv_bytes,
    dataframe_sha256,
    quality_report_bytes,
    validate_canonical_frame,
    write_bytes_atomically,
    write_canonical_csv,
    write_quality_report,
)


def _source_frame(row_count: int = 10) -> pd.DataFrame:
    ids = np.arange(1, row_count + 1)
    values: dict[str, object] = {
        "ID": ids,
        "X1": 10_000 + ids * 1_000,
        "X2": np.where(ids % 2, 1, 2),
        "X3": np.ones(row_count, dtype=int),
        "X4": np.ones(row_count, dtype=int),
        "X5": 20 + ids,
    }
    values.update({f"X{index}": np.full(row_count, -1) for index in range(6, 12)})
    values.update({f"X{index}": ids * 100 + index for index in range(12, 18)})
    values.update({f"X{index}": ids * 10 + index for index in range(18, 24)})
    values["Y"] = np.where(ids % 2, 0, 1)
    return pd.DataFrame(values, columns=SOURCE_COLUMNS)


def _expectations(row_count: int = 10) -> ValidationExpectations:
    return ValidationExpectations(
        row_count=row_count,
        target_counts=((0, row_count // 2), (1, row_count // 2)),
        account_id_min=1,
        account_id_max=row_count,
    )


def _rule_ids(error: DataContractError) -> set[str]:
    return {issue.rule_id for issue in error.report.issues}


def test_canonicalize_uses_manifest_mapping_and_stable_dtypes() -> None:
    manifest = load_dataset_manifest()
    source = _source_frame(10)
    custom_manifest = manifest.model_copy(
        update={
            "expectations": manifest.expectations.model_copy(
                update={
                    "row_count": 10,
                    "target_counts": {"0": 5, "1": 5},
                }
            )
        }
    )

    result = canonicalize_source_frame(source.sample(frac=1, random_state=7), custom_manifest)

    assert isinstance(result, CanonicalizationResult)
    assert tuple(result.data.columns) == CANONICAL_COLUMNS
    assert result.data["account_id"].tolist() == list(range(1, 11))
    assert str(result.data["sex_code"].dtype) == "int8"
    assert str(result.data["age_years"].dtype) == "int16"
    assert result.report.status == "passed"
    assert len(result.sha256) == 64


@pytest.mark.parametrize(
    ("mutate", "rule_id"),
    [
        (lambda frame: frame.drop(columns="X23"), "schema.source_columns.exact"),
        (
            lambda frame: frame.loc[:, [*SOURCE_COLUMNS[1:], SOURCE_COLUMNS[0]]],
            "schema.source_columns.exact",
        ),
    ],
)
def test_canonicalize_rejects_wrong_source_structure(mutate, rule_id: str) -> None:
    with pytest.raises(DataContractError) as caught:
        canonicalize_source_frame(mutate(_source_frame()), _expectations())

    assert rule_id in _rule_ids(caught.value)


@pytest.mark.parametrize(
    ("column", "value", "rule_id"),
    [
        ("X1", np.nan, "schema.values.null"),
        ("X1", 10_000.5, "schema.values.integral"),
        ("X5", 101, "schema.age.range"),
        ("X6", 256, "schema.repayment_status.domain"),
        ("X18", -1, "schema.payment_amount.nonnegative"),
    ],
)
def test_canonicalize_rejects_invalid_values_without_narrow_integer_wrap(
    column: str,
    value: float,
    rule_id: str,
) -> None:
    source = _source_frame()
    if not float(value).is_integer():
        source[column] = source[column].astype(float)
    source.loc[0, column] = value

    with pytest.raises(DataContractError) as caught:
        canonicalize_source_frame(source, _expectations())

    assert rule_id in _rule_ids(caught.value)


def test_accepted_anomalies_are_retained_and_sample_ids_are_bounded() -> None:
    row_count = 30
    source = _source_frame(row_count)
    source["X3"] = 0
    source["X4"] = 0
    source["X6"] = -2
    source["X12"] = -100
    expectations = ValidationExpectations(
        row_count=row_count,
        target_counts=((0, 15), (1, 15)),
        account_id_min=1,
        account_id_max=row_count,
    )

    result = canonicalize_source_frame(source, expectations)

    assert result.report.status == "passed_with_warnings"
    issues = {issue.rule_id: issue for issue in result.report.issues}
    assert issues["warning.education.undocumented_code"].count == row_count
    assert len(issues["warning.education.undocumented_code"].account_ids) == 20
    assert result.data.loc[0, "education_code"] == 0
    assert result.data.loc[0, "bill_amount_ntd_lag_0"] == -100


def test_duplicate_non_identifier_rows_are_warned_but_retained() -> None:
    source = _source_frame()
    feature_and_target_columns = list(SOURCE_COLUMNS[1:])
    source.loc[2, feature_and_target_columns] = source.loc[0, feature_and_target_columns].to_numpy()

    result = canonicalize_source_frame(source, _expectations())

    duplicate_issue = next(
        issue for issue in result.report.issues if issue.rule_id == "warning.features.duplicate"
    )
    assert duplicate_issue.count == 1
    assert duplicate_issue.account_ids == (3,)
    assert len(result.data) == 10


def test_values_outside_int64_raise_stable_contract_error() -> None:
    source = _source_frame()
    source["X1"] = source["X1"].astype("uint64")
    source.loc[0, "X1"] = np.uint64(2**63)

    with pytest.raises(DataContractError) as caught:
        canonicalize_source_frame(source, _expectations())

    assert "schema.values.int64_range" in _rule_ids(caught.value)


def test_quality_outputs_are_deterministic_utf8_lf(tmp_path: Path) -> None:
    result = canonicalize_source_frame(_source_frame(), _expectations())
    csv_path = tmp_path / "canonical.csv"
    report_path = tmp_path / "quality.json"

    assert write_canonical_csv(result.data, csv_path) == result.sha256
    report_hash = write_quality_report(result.report, report_path)

    assert b"\r\n" not in csv_path.read_bytes()
    assert report_path.read_bytes() == quality_report_bytes(result.report)
    assert len(report_hash) == 64
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "passed"


def _canonical_frame(row_count: int = 10) -> pd.DataFrame:
    return canonicalize_source_frame(_source_frame(row_count), _expectations(row_count)).data


@pytest.mark.parametrize(
    ("mutation", "rule_id"),
    [
        (lambda frame: frame.assign(X1=np.inf), "schema.values.finite"),
        (lambda frame: frame.assign(X1="not numeric"), "schema.values.numeric"),
        (lambda frame: frame.assign(ID=[1, 1, *range(3, 11)]), "schema.account_id.unique"),
        (lambda frame: frame.assign(ID=[0, *range(2, 11)]), "schema.account_id.positive"),
        (lambda frame: frame.assign(ID=[11, *range(2, 11)]), "dataset.account_id.sequence"),
        (lambda frame: frame.assign(X1=0), "schema.credit_limit.positive"),
        (lambda frame: frame.assign(X2=3), "schema.sex.domain"),
        (lambda frame: frame.assign(X3=7), "schema.education.domain"),
        (lambda frame: frame.assign(X4=4), "schema.marital_status.domain"),
        (lambda frame: frame.assign(X5=17), "schema.age.range"),
        (lambda frame: frame.assign(X6=-3), "schema.repayment_status.domain"),
        (lambda frame: frame.assign(X18=-1), "schema.payment_amount.nonnegative"),
        (lambda frame: frame.assign(Y=2), "schema.target.domain"),
        (lambda frame: frame.iloc[:-1].copy(), "dataset.row_count"),
    ],
)
def test_every_canonical_hard_gate_has_a_stable_rule_id(mutation, rule_id: str) -> None:
    with pytest.raises(DataContractError) as caught:
        canonicalize_source_frame(mutation(_source_frame()), _expectations())

    assert rule_id in _rule_ids(caught.value)


def test_target_class_presence_and_snapshot_counts_are_both_enforced() -> None:
    one_class = _source_frame()
    one_class["Y"] = 0

    with pytest.raises(DataContractError) as caught:
        canonicalize_source_frame(one_class, _expectations())

    assert {"dataset.target.classes", "dataset.target.counts"} <= _rule_ids(caught.value)

    count_drift = _source_frame()
    count_drift.loc[0, "Y"] = 1
    with pytest.raises(DataContractError) as caught:
        canonicalize_source_frame(count_drift, _expectations())
    assert "dataset.target.counts" in _rule_ids(caught.value)


@pytest.mark.parametrize("duplicate_name", ["ID", "Y"])
def test_duplicate_identity_or_target_headers_return_a_structured_error(
    duplicate_name: str,
) -> None:
    source = _source_frame()
    columns = list(source.columns)
    replacement_index = 1 if duplicate_name == "ID" else -2
    columns[replacement_index] = duplicate_name
    source.columns = columns

    with pytest.raises(DataContractError) as caught:
        canonicalize_source_frame(source, _expectations())

    assert {"schema.source_columns.duplicate", "schema.source_columns.exact"} <= _rule_ids(
        caught.value
    )


def test_complex_values_are_rejected_as_non_numeric_without_a_traceback() -> None:
    source = _source_frame()
    source["X1"] = source["X1"].astype("complex128")

    with pytest.raises(DataContractError) as caught:
        canonicalize_source_frame(source, _expectations())

    assert "schema.values.numeric" in _rule_ids(caught.value)


@pytest.mark.parametrize(
    ("source_column", "canonical_column", "values", "warning_rule"),
    [
        ("X3", "education_code", [0, 5, 6], "warning.education.undocumented_code"),
        ("X4", "marital_status_code", [0], "warning.marital_status.undocumented_code"),
        (
            "X6",
            "repayment_status_lag_0",
            [-2, 0],
            "warning.repayment_status.undocumented_code",
        ),
        ("X12", "bill_amount_ntd_lag_0", [-500], "warning.bill_amount.negative"),
    ],
)
def test_each_accepted_anomaly_is_warned_and_preserved_exactly(
    source_column: str,
    canonical_column: str,
    values: list[int],
    warning_rule: str,
) -> None:
    source = _source_frame()
    for index, value in enumerate(values):
        source.loc[index, source_column] = value

    result = canonicalize_source_frame(source, _expectations())

    warning = next(issue for issue in result.report.issues if issue.rule_id == warning_rule)
    assert warning.count == len(values)
    assert result.data.loc[: len(values) - 1, canonical_column].tolist() == values


def test_large_signed_bill_balances_and_payment_zero_are_not_mutated() -> None:
    source = _source_frame()
    source["X12"] = source["X12"].astype("int64")
    source.loc[0, "X12"] = np.iinfo(np.int64).min + 1
    source.loc[0, "X18"] = 0

    result = canonicalize_source_frame(source, _expectations())

    assert result.data.loc[0, BILL_AMOUNT_COLUMNS[0]] == np.iinfo(np.int64).min + 1
    assert result.data.loc[0, PAYMENT_AMOUNT_COLUMNS[0]] == 0


def test_warning_report_orders_rules_and_bounds_sorted_account_ids() -> None:
    source = _source_frame(30)
    source["X3"] = 0
    result = canonicalize_source_frame(source.sample(frac=1, random_state=4), _expectations(30))

    issue = next(
        item
        for item in result.report.issues
        if item.rule_id == "warning.education.undocumented_code"
    )
    assert issue.account_ids == tuple(range(1, 21))
    assert [item.rule_id for item in result.report.issues] == sorted(
        item.rule_id for item in result.report.issues
    )


@pytest.mark.parametrize("shape", ["missing", "reordered", "duplicate"])
def test_validate_canonical_frame_rejects_structural_drift(shape: str) -> None:
    canonical = _canonical_frame()
    if shape == "missing":
        canonical = canonical.drop(columns="payment_amount_ntd_lag_5")
    elif shape == "reordered":
        canonical = canonical.loc[:, [*CANONICAL_COLUMNS[1:], CANONICAL_COLUMNS[0]]]
    else:
        columns = list(canonical.columns)
        columns[1] = "account_id"
        canonical.columns = columns

    with pytest.raises(DataContractError) as caught:
        validate_canonical_frame(canonical, _expectations())

    assert any(rule.startswith("schema.canonical_columns") for rule in _rule_ids(caught.value))


def test_validate_canonical_frame_rejects_dtype_drift_before_pandera() -> None:
    canonical = _canonical_frame()
    canonical["sex_code"] = canonical["sex_code"].astype("int64")

    with pytest.raises(DataContractError) as caught:
        validate_canonical_frame(canonical, _expectations())

    assert "schema.dtypes.exact" in _rule_ids(caught.value)


def test_manifest_expectations_and_manifest_contract_are_both_supported() -> None:
    manifest = load_dataset_manifest()
    expectations = manifest.expectations.model_copy(
        update={"row_count": 10, "target_counts": {"0": 5, "1": 5}}
    )
    custom_manifest = manifest.model_copy(update={"expectations": expectations})

    from_expectations = canonicalize_source_frame(_source_frame(), expectations)
    from_manifest = canonicalize_source_frame(_source_frame(), custom_manifest)

    pd.testing.assert_frame_equal(from_expectations.data, from_manifest.data)


def test_invalid_expectation_objects_and_contracts_fail_actionably() -> None:
    with pytest.raises(TypeError, match="validated DatasetManifest"):
        canonicalize_source_frame(_source_frame(), object())  # type: ignore[arg-type]

    manifest = load_dataset_manifest()
    bad_column = manifest.canonical_contract.columns[0].model_copy(
        update={"logical_dtype": "number"}
    )
    bad_contract = manifest.canonical_contract.model_copy(
        update={"columns": (bad_column, *manifest.canonical_contract.columns[1:])}
    )
    bad_manifest = manifest.model_copy(update={"canonical_contract": bad_contract})
    with pytest.raises(ValueError, match="only integer columns"):
        canonicalize_source_frame(_source_frame(), bad_manifest)


def test_serializers_and_quality_models_are_stable(tmp_path: Path) -> None:
    canonical = _canonical_frame()
    content = dataframe_csv_bytes(canonical)
    arbitrary_path = tmp_path / "nested" / "payload.bin"

    assert dataframe_sha256(canonical) == write_bytes_atomically(arbitrary_path, content)
    assert arbitrary_path.read_bytes() == content
    assert QualityReport(
        status="passed",
        row_count=0,
        column_count=0,
        target_counts={},
    ).passed
    assert not QualityReport(
        status="failed",
        row_count=0,
        column_count=0,
        target_counts={},
        issues=(ValidationIssue("rule", "error", "message", 1),),
    ).passed


def test_all_lag_groups_have_six_ordered_columns() -> None:
    assert REPAYMENT_STATUS_COLUMNS == tuple(f"repayment_status_lag_{index}" for index in range(6))
    assert BILL_AMOUNT_COLUMNS == tuple(f"bill_amount_ntd_lag_{index}" for index in range(6))
    assert PAYMENT_AMOUNT_COLUMNS == tuple(f"payment_amount_ntd_lag_{index}" for index in range(6))
