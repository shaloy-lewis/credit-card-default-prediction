"""Tests for strict Phase 2 modelling configuration contracts."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from credit_risk.modeling.contracts import (
    AUDIT_COLUMNS,
    FORBIDDEN_PREDICTOR_COLUMNS,
    PREDICTOR_COLUMNS,
    REPAYMENT_RULE_WEIGHTS,
    REPAYMENT_STATUS_COLUMNS,
    BaselineExperimentConfig,
    FeatureContract,
    ModelingContractError,
    load_baseline_config,
    load_feature_contract,
)
from tests.unit.data.helpers import write_json


def _feature_payload() -> dict[str, object]:
    return json.loads(Path("configs/modeling/feature_contract_v1.json").read_text(encoding="utf-8"))


def _baseline_payload() -> dict[str, object]:
    return json.loads(Path("configs/modeling/baseline_v1.json").read_text(encoding="utf-8"))


def test_checked_in_contracts_load_without_warnings() -> None:
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        feature = load_feature_contract()
        baseline = load_baseline_config()

    assert observed == []
    assert feature.contract_id == "uci_credit_default_features_v1"
    assert feature.columns.predictor_columns == PREDICTOR_COLUMNS
    assert feature.columns.audit_columns == AUDIT_COLUMNS
    assert feature.columns.forbidden_predictor_columns == FORBIDDEN_PREDICTOR_COLUMNS
    assert baseline.experiment_name == "credit-risk-baseline-v1"
    assert baseline.dataset_manifest_path == "configs/data/uci_credit_default_v1.json"
    assert baseline.split_config_path == "configs/data/split_v1.json"
    assert baseline.evaluation.primary_metric == "average_precision"
    assert baseline.evaluation.probability_metrics == ("brier_score", "log_loss")
    assert baseline.evaluation.capacities == (0.05, 0.1, 0.2)
    assert baseline.baselines.repayment_rule.recency_weights == REPAYMENT_RULE_WEIGHTS


def test_checked_in_baseline_pins_complete_feature_contract_bytes() -> None:
    feature_bytes = Path("configs/modeling/feature_contract_v1.json").read_bytes()

    assert (
        load_baseline_config().feature_contract_sha256 == hashlib.sha256(feature_bytes).hexdigest()
    )


def test_baseline_loader_resolves_repository_relative_contract_outside_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/modeling/baseline_v1.json").resolve()
    monkeypatch.chdir(tmp_path)

    baseline = load_baseline_config(config_path)

    assert baseline.experiment_id == "baseline_v1"
    assert baseline.feature_contract_path == "configs/modeling/feature_contract_v1.json"


def test_checked_in_baseline_pins_the_metric_decision_hierarchy() -> None:
    evaluation = load_baseline_config().evaluation

    assert evaluation.primary_metric == "average_precision"
    assert evaluation.probability_guardrail == "brier_score"
    assert evaluation.primary_capacity_metric.metric == "lift"
    assert evaluation.primary_capacity_metric.capacity == 0.1


def test_contract_models_are_frozen_and_reject_extra_fields() -> None:
    contract = load_feature_contract()
    with pytest.raises(ValidationError, match="frozen"):
        contract.contract_id = "changed"  # type: ignore[misc]

    payload = _feature_payload()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FeatureContract.model_validate(payload)


@pytest.mark.parametrize(
    ("mutate", "field_name"),
    [
        (
            lambda payload: payload["columns"]["predictor_columns"].reverse(),
            "predictor_columns",
        ),
        (
            lambda payload: payload["expected_development"].update(rows=23_999),
            "rows",
        ),
        (
            lambda payload: payload["lineage"].update(canonical_sha256="0" * 64),
            "canonical_sha256",
        ),
        (
            lambda payload: payload["cross_validation"].update(n_repeats=2),
            "n_repeats",
        ),
    ],
)
def test_official_contract_cannot_drift(
    mutate: Callable[[dict[str, object]], None],
    field_name: str,
) -> None:
    payload = _feature_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=field_name):
        FeatureContract.model_validate(payload)


def test_smaller_nonofficial_contract_is_supported() -> None:
    payload = _feature_payload()
    payload["dataset"] = {"dataset_id": "fixture_credit_default", "dataset_version": "v1"}
    payload["expected_development"] = {
        "rows": 80,
        "target_counts": {"0": 40, "1": 40},
    }
    payload["lineage"] = {
        name: str(index) * 64
        for index, name in enumerate(
            (
                "source_sha256",
                "dataset_manifest_sha256",
                "canonical_sha256",
                "split_config_sha256",
                "assignment_sha256",
                "reviewed_split_lock_sha256",
            )
        )
    }

    contract = FeatureContract.model_validate(payload)

    assert contract.expected_development.rows == 80
    assert contract.columns.predictor_columns == PREDICTOR_COLUMNS


def test_feature_boundary_rejects_audit_predictor_overlap() -> None:
    payload = _feature_payload()
    payload["dataset"]["dataset_id"] = "fixture_credit_default"
    payload["columns"]["predictor_columns"].append("sex_code")

    with pytest.raises(ValidationError, match="must be disjoint"):
        FeatureContract.model_validate(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["columns"].update(predictor_columns=[]),
            "must not be empty",
        ),
        (
            lambda payload: payload["columns"].update(
                predictor_columns=[*PREDICTOR_COLUMNS, PREDICTOR_COLUMNS[0]]
            ),
            "must contain unique",
        ),
        (
            lambda payload: payload["columns"].update(
                predictor_columns=[*PREDICTOR_COLUMNS[:-1], "bad-column"]
            ),
            "invalid column name",
        ),
        (
            lambda payload: payload["columns"].update(
                predictor_columns=[*PREDICTOR_COLUMNS, "account_id"]
            ),
            "identifier and target",
        ),
        (
            lambda payload: payload["columns"].update(
                forbidden_predictor_columns=["account_id", "default_next_month"]
            ),
            "identifier, target",
        ),
    ],
)
def test_feature_column_contract_rejects_unsafe_boundaries(
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    payload = _feature_payload()
    payload["dataset"]["dataset_id"] = "fixture_credit_default"
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        FeatureContract.model_validate(payload)


@pytest.mark.parametrize(
    ("target_counts", "message"),
    [
        ({"0": 80}, "exactly labels"),
        ({"0": 80, "1": 0}, "at least one"),
        ({"0": 76, "1": 4}, "at least n_splits"),
    ],
)
def test_development_contract_rejects_invalid_class_counts(
    target_counts: dict[str, int],
    message: str,
) -> None:
    payload = _feature_payload()
    payload["dataset"]["dataset_id"] = "fixture_credit_default"
    payload["expected_development"] = {"rows": 80, "target_counts": target_counts}

    with pytest.raises(ValidationError, match=message):
        FeatureContract.model_validate(payload)


def test_baseline_config_rejects_changed_feature_contract(tmp_path: Path) -> None:
    feature_path = tmp_path / "feature_contract_v1.json"
    baseline_path = tmp_path / "baseline_v1.json"
    write_json(feature_path, _feature_payload())
    baseline = _baseline_payload()
    baseline["feature_contract_path"] = "feature_contract_v1.json"
    baseline["feature_contract_sha256"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    write_json(baseline_path, baseline)
    feature_path.write_text(feature_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ModelingContractError, match="hash mismatch"):
        load_baseline_config(baseline_path)


@pytest.mark.parametrize(
    ("field", "path", "message"),
    [
        ("feature_contract_path", "../feature.json", "safe relative JSON path"),
        ("dataset_manifest_path", "/absolute/manifest.json", "safe relative JSON path"),
        ("dataset_manifest_path", "C:/absolute/manifest.json", "safe relative JSON path"),
        ("split_config_path", "C:\\split.json", "safe relative JSON path"),
        ("split_config_path", "split.txt", "safe relative JSON path"),
    ],
)
def test_baseline_config_rejects_unsafe_governed_paths(
    field: str,
    path: str,
    message: str,
) -> None:
    payload = _baseline_payload()
    payload[field] = path

    with pytest.raises(ValidationError, match=message):
        BaselineExperimentConfig.model_validate(payload)


def test_baseline_config_rejects_protocol_drift() -> None:
    payload = _baseline_payload()
    payload["baselines"]["logistic"]["class_weight"] = "balanced"

    with pytest.raises(ValidationError):
        BaselineExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            "repayment_rule",
            "status_columns",
            list(reversed(REPAYMENT_STATUS_COLUMNS)),
            "status",
        ),
        ("repayment_rule", "recency_weights", [1, 2, 3, 4, 5, 6], "weights"),
        ("evaluation", "discrimination_metrics", ["roc_auc"], "discrimination_metrics"),
        ("evaluation", "probability_metrics", ["log_loss"], "probability_metrics"),
        ("evaluation", "capacities", [0.1], "capacities"),
        ("evaluation", "repeat_summary", ["mean"], "repeat_summary"),
        ("logistic", "max_iter", 100, "logistic baseline"),
    ],
)
def test_baseline_protocol_rejects_component_drift(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _baseline_payload()
    if section == "evaluation":
        payload["evaluation"][field] = value
    else:
        payload["baselines"][section][field] = value

    with pytest.raises(ValidationError, match=message):
        BaselineExperimentConfig.model_validate(payload)


def test_contract_loaders_wrap_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(ModelingContractError, match="Unable to load"):
        load_feature_contract(tmp_path / "missing.json")

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not json", encoding="utf-8")
    with pytest.raises(ModelingContractError, match="Unable to load"):
        load_feature_contract(invalid_path)


def test_baseline_loader_reports_missing_feature_contract(tmp_path: Path) -> None:
    baseline = _baseline_payload()
    baseline["feature_contract_path"] = "missing-feature.json"
    baseline_path = tmp_path / "baseline.json"
    write_json(baseline_path, baseline)

    with pytest.raises(ModelingContractError, match="Unable to read feature contract"):
        load_baseline_config(baseline_path)


def test_baseline_loader_does_not_alias_missing_nested_contract_by_basename(
    tmp_path: Path,
) -> None:
    feature_path = tmp_path / "feature_contract_v1.json"
    baseline_path = tmp_path / "baseline_v1.json"
    write_json(feature_path, _feature_payload())
    baseline = _baseline_payload()
    baseline["feature_contract_path"] = "missing/nested/feature_contract_v1.json"
    baseline["feature_contract_sha256"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    write_json(baseline_path, baseline)

    with pytest.raises(ModelingContractError, match="Unable to read feature contract"):
        load_baseline_config(baseline_path)
