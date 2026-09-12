"""Complete-file and semantic integrity for the authenticated Release A dossier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "reports" / "releases" / "release_a_v1"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "releases" / "release_a_v1.json"

# Change only after a clean, zero-computation Release A build, external-digest
# verification, and explicit review of the complete source and output allowlists.
EXPECTED_DIGESTS = {
    "evidence-manifest.json": "7e65c7b854de15742f05c4b8c2de891f50512518f8eb2339241f87f98754edf7",
    "release-a-report.md": "7d5873bfdf804782fcf2bdd21a54ff23b75f46c011b49adfedccd45214173a86",
    "summary.json": "a8cfdd1f4e4655235082430fdc0cd76e0034ec1797b3cdf9b0f08f5ff8719acb",
    "validation-uncertainty.json": (
        "187004bd3cb646279977b24e34d16bb8f097f27edeb0edc88a1ad1bc6c50ffa2"
    ),
}
EXPECTED_CONFIG_SHA256 = "9dc78bbe1ee7c116e6c0362987d935b7258e720139943bd0203efcc5fb0b3920"


def test_release_a_evidence_is_byte_identical_and_allowlisted() -> None:
    paths = {name: EVIDENCE_ROOT / name for name in EXPECTED_DIGESTS}

    assert {path.name for path in EVIDENCE_ROOT.iterdir() if path.is_file()} == set(paths)
    assert {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    } == EXPECTED_DIGESTS


def test_release_a_manifest_authenticates_the_complete_evidence_chain() -> None:
    config = json.loads(CONFIG_PATH.read_bytes())
    manifest = json.loads((EVIDENCE_ROOT / "evidence-manifest.json").read_bytes())

    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == EXPECTED_CONFIG_SHA256
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["release_id"] == "release_a_v1"
    assert manifest["configuration_sha256"] == EXPECTED_CONFIG_SHA256
    assert manifest["implementation_git_commit"] == ("20186ad25f7adcb751cf05a142a291377f67d8b6")
    assert manifest["source_artifacts"] == config["source_artifacts"]
    assert set(manifest["source_artifacts"]) == {
        "baseline_config",
        "baseline_report",
        "baseline_summary",
        "bundle_manifest",
        "bundle_model",
        "data_manifest",
        "executed_evaluator_source",
        "feature_contract",
        "final_test_approval",
        "final_test_authorization",
        "final_test_completed_receipt",
        "final_test_report",
        "final_test_started_receipt",
        "final_test_summary",
        "selection_config",
        "selection_report",
        "selection_summary",
        "split_config",
        "split_lock",
    }
    assert set(manifest["artifacts"]) == set(EXPECTED_DIGESTS) - {"evidence-manifest.json"}
    assert {name: item["sha256"] for name, item in manifest["artifacts"].items()} == {
        name: digest
        for name, digest in EXPECTED_DIGESTS.items()
        if name != "evidence-manifest.json"
    }
    assert all(item["row_level_data"] is False for item in manifest["artifacts"].values())
    assert manifest["boundaries_verified"] == {
        "bootstrap_generated": False,
        "final_test_reevaluated": False,
        "model_deserialized": False,
        "prediction_generated": False,
        "test_partition_selected": False,
        "test_predictions_loaded": False,
        "training_performed": False,
    }


def test_release_a_dossier_closes_the_defensible_model_milestone() -> None:
    summary_bytes = (EVIDENCE_ROOT / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    uncertainty_bytes = (EVIDENCE_ROOT / "validation-uncertainty.json").read_bytes()
    uncertainty = json.loads(uncertainty_bytes)
    report = (EVIDENCE_ROOT / "release-a-report.md").read_text(encoding="utf-8")

    assert summary["schema_version"] == "1.0.0"
    assert summary["release_id"] == "release_a_v1"
    assert summary["milestone"] == "defensible_model"
    assert summary["status"] == "complete"
    assert summary["lineage"]["git_dirty"] is False
    assert summary["lineage"]["release_config_sha256"] == EXPECTED_CONFIG_SHA256
    assert summary["data"] == {
        "assignment_sha256": ("2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e"),
        "canonical_sha256": ("75b2a746781a584b0456f843f1f269190b51e90983cba44c4ed6c4a8685e6c1c"),
        "dataset_id": "uci_credit_default",
        "development_rows": 24000,
        "modeling_use": "none",
        "rows": 30000,
        "source_sha256": ("45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0"),
        "test_rows": 6000,
        "verification": "offline_complete_snapshot_integrity",
    }

    assert [item["model_id"] for item in summary["baselines"]["models"]] == [
        "fold_prevalence",
        "repayment_burden_rule",
        "logistic_l2",
    ]
    assert [item["model_id"] for item in summary["selection"]["models"]] == [
        "logistic_l2",
        "random_forest",
        "hist_gradient_boosting",
        "catboost_fixed",
    ]
    assert summary["selection"]["fit_count"] == 4
    assert summary["selection"]["parameter_tuning"] is False
    assert summary["selection"]["cross_validation_iteration"] is False
    assert summary["selection"]["winner_refitted"] is False
    assert summary["selection"]["selected_model_id"] == "catboost_fixed"

    assert summary["calibration"]["method"] == "identity"
    assert summary["calibration"]["calibrator_fitted"] is False
    assert len(summary["calibration"]["reliability_bins"]) == 10
    assert summary["calibration"]["expected_calibration_error_10_equal_count_bins"] == (
        pytest.approx(0.013557513854215497)
    )

    assert summary["uncertainty"]["population"] == "development_validation_only"
    assert summary["uncertainty"]["final_test_intervals_computed"] is False
    assert summary["uncertainty"]["resamples"] == 500
    assert summary["uncertainty"]["random_state"] == 42
    assert summary["uncertainty"]["metrics"] == uncertainty["metrics"]
    expected_intervals = {
        "average_precision": (0.5565104548302114, 0.5254308022482898, 0.5877548367667342),
        "brier_score": (0.13353854208377516, 0.12882632019280904, 0.1379238496379374),
        "lift_at_0_1": (3.2109227871939736, 3.027071563088512, 3.3759416195856873),
    }
    for metric, (point, lower, upper) in expected_intervals.items():
        assert uncertainty["metrics"][metric] == {
            "lower": pytest.approx(lower),
            "point": pytest.approx(point),
            "upper": pytest.approx(upper),
        }

    for population, expected_counts in (
        ("validation", [240, 480, 960]),
        ("final_test", [300, 600, 1200]),
    ):
        capacities = summary["capacity"][population]
        assert [item["capacity"] for item in capacities] == [0.05, 0.1, 0.2]
        assert [item["selected_count"] for item in capacities] == expected_counts
        assert all(
            set(item) >= {"precision", "recall", "lift", "expected_true_positives"}
            for item in capacities
        )

    final_test = summary["final_test"]
    assert final_test["status"] == "complete"
    assert final_test["g2_status"] == "closed"
    assert final_test["evaluation_count"] == final_test["maximum_evaluations"] == 1
    assert final_test["permanently_consumed"] is True
    assert final_test["metrics"]["discrimination"]["average_precision"] == pytest.approx(
        0.5428673518681313
    )
    assert final_test["metrics"]["probability"]["brier_score"] == pytest.approx(0.1363037019973075)
    assert final_test["metrics"]["capacities"][1]["lift"] == pytest.approx(3.089675960813866)
    assert set(final_test["gates"]) == {"average_precision", "brier_score", "lift_at_0_1"}
    assert all(gate["passed"] is True for gate in final_test["gates"].values())

    assert set(summary["release_criteria"].values()) == {"passed"}
    assert summary["evidence_boundary"] == {
        "bootstrap_generated": False,
        "final_test_reevaluated": False,
        "full_dataset_integrity_verification_performed": True,
        "model_deserialized": False,
        "prediction_generated": False,
        "stress_evidence": "deferred_to_g4_release_b",
        "test_partition_selected": False,
        "test_predictions_loaded": False,
        "training_performed": False,
    }
    assert all(value is False for value in summary["claims"].values())

    deterministic_text = summary_bytes.decode("utf-8") + report
    for forbidden in (
        "C:\\Users",
        "/home/",
        "account_id,",
        "run_id",
        "mlflow.db",
        "timestamp",
    ):
        assert forbidden not in deterministic_text
    assert "Status: **complete**" in report
    assert "validation intervals, not final-test intervals" in report
    assert "The evaluation cannot be rerun." in report
    assert "deferred to G4/Release B" in report
    assert "do not establish causal" in report
    assert "India-specific validity" in report
    assert "regulatory compliance" in report
    assert "production suitability" in report
