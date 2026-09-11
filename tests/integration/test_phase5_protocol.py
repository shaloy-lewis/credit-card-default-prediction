"""Integrity proof for the governance protocol frozen before official evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "governance" / "phase5_v1.json"

# Change only after a new governance protocol and explicit review.
EXPECTED_CONFIG_SHA256 = "990f33b1f1389f0666a75400a5677f3bdac4b8a09a531b43e6262b5a44cb0e12"


def test_phase5_protocol_is_complete_and_validation_only() -> None:
    content = CONFIG_PATH.read_bytes()
    assert hashlib.sha256(content).hexdigest() == EXPECTED_CONFIG_SHA256
    config = json.loads(content)

    assert config["status"] == "frozen_before_official_evidence"
    assert config["population"] == {
        "assignment_column": "cv_fold_r0",
        "partition": "development_validation_only",
        "rows": 4800,
        "sealed_test_access": "prohibited",
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
        "test_partition_loading",
        "final_test_prediction_loading",
    } <= set(config["prohibitions"])
