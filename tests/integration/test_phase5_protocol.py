"""Integrity proof for the governance protocol frozen before official evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "governance" / "phase5_v1.json"

# Change only after a new governance protocol and explicit review.
EXPECTED_CONFIG_SHA256 = "1717abd20e5dad6819d2f67fc13eecfa38a8800decf9e6954ffd0c74a913f68c"


def test_phase5_protocol_is_complete_and_validation_only() -> None:
    content = CONFIG_PATH.read_bytes()
    assert hashlib.sha256(content).hexdigest() == EXPECTED_CONFIG_SHA256
    config = json.loads(content)

    assert config["status"] == "frozen_before_official_evidence"
    assert config["population"] == {
        "assignment_column": "cv_fold_r0",
        "development_rows": 24000,
        "partition": "development_validation_only",
        "rows": 4800,
        "target_counts": {"0": 3738, "1": 1062},
        "validation_fold": 0,
    }
    assert config["features"]["demographic_policy"] == "audit_only_excluded_from_estimator"
    assert len(config["features"]["predictor_columns"]) == 19
    assert config["features"]["audit_columns"] == [
        "sex_code",
        "education_code",
        "marital_status_code",
        "age_years",
    ]
    assert config["prediction"]["q90"] == pytest.approx(0.5858430725164706)
    assert config["prediction"]["metric_absolute_tolerance"] == pytest.approx(1e-12)
    assert config["explanation"]["sample_rows"] == 1000
    assert config["explanation"]["shap_output_columns"] == 20
    assert config["explanation"]["additivity_absolute_tolerance"] == pytest.approx(1e-10)
    assert config["fairness"]["bootstrap"]["resamples"] == 500
    assert config["fairness"]["bootstrap"]["metrics"] == [
        "mean_probability",
        "calibration_in_the_large",
        "brier_score",
        "selection_rate_at_q90",
        "true_positive_rate_at_q90",
        "false_positive_rate_at_q90",
    ]
    assert config["fairness"]["prevalence_interval"] == {
        "confidence_level": 0.95,
        "method": "wilson_score",
        "z_value": pytest.approx(1.959963984540054),
    }
    assert config["test_boundary"] == {
        "full_dataset_integrity_verification": "required",
        "test_explanation_generation": "prohibited",
        "test_partition_return": "prohibited",
        "test_partition_selection": "prohibited",
        "test_prediction_generation": "prohibited",
        "test_prediction_loading": "prohibited",
        "test_subgroup_analysis": "prohibited",
    }
    assert config["review"]["g3_result"] == "closed_with_conditions"
    assert config["review"]["expected_triggers"] == [
        {
            "axis": "education_code",
            "direction": "below_lower_bound",
            "group": "1",
            "metric": "selection_rate_ratio",
        },
        {
            "axis": "education_code",
            "direction": "above_upper_bound",
            "group": "3",
            "metric": "selection_rate_ratio",
        },
    ]
    assert {
        "training",
        "refitting",
        "parameter_tuning",
        "cross_validation",
        "calibration_fitting",
        "test_partition_selection",
        "test_partition_return",
        "test_prediction_generation",
        "test_prediction_loading",
        "final_test_prediction_loading",
    } <= set(config["prohibitions"])
