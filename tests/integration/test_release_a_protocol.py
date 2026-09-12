"""Integrity proof for the frozen Release A audit-closure protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "releases" / "release_a_v1.json"

# Change only after a new release protocol and explicit review of every bound artifact.
EXPECTED_CONFIG_SHA256 = "9dc78bbe1ee7c116e6c0362987d935b7258e720139943bd0203efcc5fb0b3920"


def test_release_a_protocol_binds_reviewed_evidence_without_computation() -> None:
    content = CONFIG_PATH.read_bytes()
    assert hashlib.sha256(content).hexdigest() == EXPECTED_CONFIG_SHA256
    config = json.loads(content)

    assert config["schema_version"] == "1.0.0"
    assert config["release_id"] == "release_a_v1"
    assert config["milestone"] == "defensible_model"
    assert config["status"] == "frozen_for_audit_closure"
    assert set(config["source_artifacts"]) == {
        "data_manifest",
        "split_lock",
        "split_config",
        "feature_contract",
        "baseline_config",
        "baseline_summary",
        "baseline_report",
        "selection_config",
        "selection_summary",
        "selection_report",
        "bundle_manifest",
        "bundle_model",
        "final_test_authorization",
        "final_test_approval",
        "final_test_started_receipt",
        "final_test_completed_receipt",
        "final_test_summary",
        "final_test_report",
        "executed_evaluator_source",
    }
    for artifact in config["source_artifacts"].values():
        path = REPOSITORY_ROOT / artifact["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    assert config["uncertainty_source"] == {
        "path": (
            "experiment/mlflow/selection-runtime/"
            "f7c99f257fe756f6db6bac449a7ef4f48a899ea4/bootstrap_intervals.json"
        ),
        "sha256": "187004bd3cb646279977b24e34d16bb8f097f27edeb0edc88a1ad1bc6c50ffa2",
        "population": "development_validation_only",
        "method": "stratified_prediction_only_percentile_bootstrap",
        "confidence_level": 0.95,
        "resamples": 500,
        "random_state": 42,
    }
    assert config["release_criteria"] == [
        "reproducible_data_and_split_protocol",
        "fixed_four_model_comparison_and_exact_serialized_winner",
        "identity_calibration_prediction_only_uncertainty_and_capacity_evaluation",
        "single_authorized_final_test_with_frozen_gates_and_no_rerun",
        "unsupported_claims_prohibited",
    ]
    assert config["governance"] == {
        "full_dataset_integrity_verification_permitted": True,
        "model_loading": "prohibited",
        "prediction": "prohibited",
        "training": "prohibited",
        "refitting": "prohibited",
        "parameter_tuning": "prohibited",
        "cross_validation": "prohibited",
        "calibration_fitting": "prohibited",
        "bootstrap_generation": "prohibited",
        "test_partition_selection": "prohibited",
        "test_predictions_loading": "prohibited",
        "final_test_reevaluation": "prohibited",
        "stress_evidence": "deferred_to_g4_release_b",
    }
    assert config["outputs"] == [
        "summary.json",
        "release-a-report.md",
        "validation-uncertainty.json",
        "evidence-manifest.json",
    ]
