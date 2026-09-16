"""Phase 6 inference contract tests."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import credit_risk.inference.contracts as contracts


def test_load_reviewed_inference_config() -> None:
    config = contracts.load_inference_config()

    assert config.protocol_id == "phase6_v1"
    assert config.input.columns == ("account_id", *config.prediction.feature_order)
    assert config.policy.review_capacity_fraction == 0.1


def test_config_rejects_missing_and_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(contracts.InferenceContractError, match="Unable to read"):
        contracts.load_inference_config(tmp_path / "missing.json")

    changed = tmp_path / "changed.json"
    changed.write_text("{}", encoding="utf-8")
    with pytest.raises(contracts.InferenceContractError, match="digest mismatch"):
        contracts.load_inference_config(changed)

    monkeypatch.setattr(
        contracts, "PHASE6_CONFIG_SHA256", hashlib.sha256(changed.read_bytes()).hexdigest()
    )
    with pytest.raises(contracts.InferenceContractError, match="Invalid inference config"):
        contracts.load_inference_config(changed)


def test_operational_features_are_strict() -> None:
    payload = {
        "credit_limit_ntd": 100000,
        **{f"repayment_status_lag_{lag}": 0 for lag in range(6)},
        **{f"bill_amount_ntd_lag_{lag}": -100 for lag in range(6)},
        **{f"payment_amount_ntd_lag_{lag}": 0 for lag in range(6)},
    }
    assert contracts.OperationalFeatures.model_validate(payload).credit_limit_ntd == 100000

    with pytest.raises(ValidationError):
        contracts.OperationalFeatures.model_validate({**payload, "credit_limit_ntd": "100000"})
    with pytest.raises(ValidationError):
        contracts.OperationalFeatures.model_validate({**payload, "age_years": 40})
    with pytest.raises(ValidationError):
        contracts.OperationalFeatures.model_validate({**payload, "repayment_status_lag_0": 10})


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("feature_order", "feature order"),
        ("input_columns", "batch columns"),
        ("threshold_keys", "thresholds are incomplete"),
        ("threshold_order", "thresholds must be ordered"),
        ("category_names", "reason categories changed"),
        ("category_partition", "partition the operational features"),
        ("capacity", "fixed at 10 percent"),
        ("exit_codes", "exit codes differ"),
        ("prohibitions", "prohibitions are incomplete"),
    ),
)
def test_semantic_contract_rejects_policy_drift(case: str, message: str) -> None:
    payload = json.loads(contracts.DEFAULT_INFERENCE_CONFIG_PATH.read_text(encoding="utf-8"))
    if case == "feature_order":
        payload["prediction"]["feature_order"].reverse()
    elif case == "input_columns":
        payload["input"]["columns"].pop()
    elif case == "threshold_keys":
        del payload["prediction"]["risk_band_thresholds"]["q80"]
    elif case == "threshold_order":
        payload["prediction"]["risk_band_thresholds"]["q80"] = 0.9
    elif case == "category_names":
        payload["explanation"]["reason_categories"]["other"] = payload["explanation"][
            "reason_categories"
        ].pop("credit_capacity")
    elif case == "category_partition":
        payload["explanation"]["reason_categories"]["credit_capacity"] = ["repayment_status_lag_0"]
    elif case == "capacity":
        payload["policy"]["review_capacity_fraction"] = 0.2
    elif case == "exit_codes":
        payload["batch"]["exit_codes"]["completed_with_rejections"] = 0
    else:
        payload["prohibitions"].remove("sealed_test_scoring")

    with pytest.raises(ValidationError, match=message):
        contracts.InferenceConfig.model_validate_json(json.dumps(payload))
