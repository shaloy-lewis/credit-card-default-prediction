"""Integrity proof for the inference protocol frozen before implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "inference" / "phase6_v1.json"

# Change only after a new inference protocol and explicit review.
EXPECTED_CONFIG_SHA256 = "84227bb48c7ba812bdf2a2752ed151ede2ce5daa398911af9311491a852500a8"


def test_phase6_protocol_is_complete_and_prediction_only() -> None:
    content = CONFIG_PATH.read_bytes()
    assert hashlib.sha256(content).hexdigest() == EXPECTED_CONFIG_SHA256
    config = json.loads(content)

    assert config["status"] == "frozen_before_implementation"
    assert config["bundle"] == {
        "bundle_id": "selected_v1",
        "model_id": "catboost_fixed",
        "manifest_path": "models/selected_v1/manifest.json",
        "manifest_sha256": "df5ce6ce07b268f57fa3bf72c97cd32f8ebb66695d7157139942c91e46d7cd88",
        "model_path": "models/selected_v1/model.cbm",
        "model_sha256": "844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c",
        "calibration": "identity",
    }
    assert len(config["prediction"]["feature_order"]) == 19
    assert config["input"]["columns"] == [
        "account_id",
        *config["prediction"]["feature_order"],
    ]
    assert config["input"]["row_error_policy"] == "score_valid_rows"
    assert config["policy"]["review_capacity_fraction"] == pytest.approx(0.1)
    assert config["policy"]["review_capacity_rounding"] == "floor"
    assert config["batch"]["exit_codes"] == {
        "completed": 0,
        "completed_with_rejections": 3,
        "failed": 1,
    }
    assert config["batch"]["reserved_snapshot_ids"] == [".", ".."]
    assert config["governance"]["pre_push_review_amendment"] == (
        "batch_integrity_and_path_safety_only_no_model_or_policy_change"
    )
    assert config["api"]["prediction_path"] == "/v1/predict"
    assert config["api"]["removed_prediction_path"] == "/predict"
    assert config["explanation"]["top_reason_count"] == 2
    assert config["explanation"]["additivity_absolute_tolerance"] == pytest.approx(1e-10)
    assert set(config["explanation"]["reason_categories"]) == {
        "billing_balance",
        "credit_capacity",
        "payment_behaviour",
        "repayment_status",
    }
    assert config["logging"]["prohibited_fields"] == [
        "features",
        "account_id",
        "probability",
        "shap_contributions",
        "target",
        "demographics",
        "local_path",
    ]
    assert {
        "model_fitting",
        "parameter_tuning",
        "calibration_fitting",
        "bootstrap_generation",
        "final_test_loading",
        "sealed_test_scoring",
    } <= set(config["prohibitions"])
