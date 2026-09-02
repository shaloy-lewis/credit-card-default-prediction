"""Complete-file and semantic integrity for the reviewed one-pass release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.modeling.selected_bundle import BundleManifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "reports" / "modeling" / "selection_v1"
BUNDLE_ROOT = REPOSITORY_ROOT / "models" / "selected_v1"
SUMMARY_PATH = EVIDENCE_ROOT / "summary.json"
REPORT_PATH = EVIDENCE_ROOT / "selection-report.md"
MANIFEST_PATH = BUNDLE_ROOT / "manifest.json"
MODEL_PATH = BUNDLE_ROOT / "model.cbm"

# Change only after a new protocol, clean bounded run, explicit evidence review,
# and a separately approved replacement release decision.
EXPECTED_DIGESTS = {
    "summary.json": "8c11b1d443c782a8ef14aa3e708e3fffa064ecb4c9fe58d3e51a6effa46efbd7",
    "selection-report.md": "16c8748e76002ebedd5c41938df7364e493af590461d03a926a2aab3d801cee1",
    "manifest.json": "df5ce6ce07b268f57fa3bf72c97cd32f8ebb66695d7157139942c91e46d7cd88",
    "model.cbm": "844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c",
}


def test_reviewed_selection_files_are_byte_identical_and_allowlisted() -> None:
    paths = {
        "summary.json": SUMMARY_PATH,
        "selection-report.md": REPORT_PATH,
        "manifest.json": MANIFEST_PATH,
        "model.cbm": MODEL_PATH,
    }
    assert {path.name for path in EVIDENCE_ROOT.iterdir() if path.is_file()} == {
        "summary.json",
        "selection-report.md",
    }
    assert {path.name for path in BUNDLE_ROOT.iterdir() if path.is_file()} == {
        "manifest.json",
        "model.cbm",
    }
    assert {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    } == EXPECTED_DIGESTS


def test_selection_evidence_proves_four_fit_holdout_blind_release() -> None:
    summary_bytes = SUMMARY_PATH.read_bytes()
    summary = json.loads(summary_bytes)
    report = REPORT_PATH.read_text(encoding="utf-8")
    manifest = BundleManifest.model_validate_json(MANIFEST_PATH.read_bytes())

    assert summary["status"] == "complete"
    assert summary["experiment_id"] == "selection_v1"
    assert summary["protocol"] == {
        "calibration": "identity",
        "cross_validation_iteration": False,
        "fit_count": 4,
        "parameter_tuning": False,
        "selection_config_sha256": (
            "2c85c6c0c07fa875256f2d861e2ded24a96532395c93f975d40886b0d6dc8c09"
        ),
        "winner_refitted": False,
    }
    assert summary["population"] == {
        "holdout_accessed": False,
        "partition": "development_only",
        "sealed_test_rows": 6000,
        "training_rows": 19200,
        "validation_rows": 4800,
    }
    assert summary["holdout"] == {
        "authorization_frozen": False,
        "evaluated": False,
        "g2_status": "open",
    }
    assert summary["reproducibility"]["git_commit"] == ("f7c99f257fe756f6db6bac449a7ef4f48a899ea4")
    assert summary["reproducibility"]["git_dirty"] is False
    assert summary["reproducibility"]["data_lineage"]["assignment_sha256"] == (
        "2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e"
    )
    assert [model["model_id"] for model in summary["models"]] == [
        "logistic_l2",
        "random_forest",
        "hist_gradient_boosting",
        "catboost_fixed",
    ]
    assert summary["selection"]["selected_model_id"] == "catboost_fixed"
    selected = summary["models"][-1]
    metrics = selected["validation_metrics"]
    assert metrics["discrimination"]["average_precision"] == pytest.approx(0.5565104548302114)
    assert metrics["probability"]["brier_score"] == pytest.approx(0.13353854208377516)
    assert metrics["capacities"][1]["lift"] == pytest.approx(3.2109227871939736)
    assert selected["decision"] == {
        "brier_guardrail_passed": True,
        "eligible": True,
        "lift_guardrail_passed": True,
        "model_id": "catboost_fixed",
        "within_equivalence_band": True,
    }
    assert summary["selected_model"]["calibration_diagnostics"]["calibrator_fitted"] is False
    assert summary["runtime_artifacts"] == {
        "bootstrap_intervals_sha256": (
            "187004bd3cb646279977b24e34d16bb8f097f27edeb0edc88a1ad1bc6c50ffa2"
        ),
        "validation_predictions_sha256": (
            "9637e6ca404c58e4b7edb0f3f1c035279b2e6efb2ee27a6cda1cf8433b438312"
        ),
    }

    assert manifest.selected_model_id == "catboost_fixed"
    assert manifest.model_filename == "model.cbm"
    assert manifest.model_sha256 == EXPECTED_DIGESTS["model.cbm"]
    assert manifest.feature_order == PREDICTOR_COLUMNS
    assert manifest.class_order == (0, 1)
    assert manifest.fit_count == 4
    assert manifest.winner_refitted is False
    assert manifest.holdout_evaluated is False
    assert summary["bundle"] == {
        "manifest_sha256": EXPECTED_DIGESTS["manifest.json"],
        "model_sha256": EXPECTED_DIGESTS["model.cbm"],
        "trusted_local_serialization": True,
    }
    assert EXPECTED_DIGESTS["summary.json"] in report
    assert "Selected model: **catboost_fixed**" in report
    assert "test partition remains sealed" in report

    deterministic_text = (
        summary_bytes.decode("utf-8") + report + MANIFEST_PATH.read_text(encoding="utf-8")
    )
    for forbidden in ("C:\\Users", "run_id", "parent_run", "mlflow.db", "timestamp"):
        assert forbidden not in deterministic_text
