"""Complete-file and semantic integrity for the reviewed Phase 5 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "reports" / "governance" / "phase5_v1"

# Change only after a clean prediction-only build, offline verification, and explicit
# review of all aggregate evidence. Never update these for provisional evidence.
EXPECTED_DIGESTS = {
    "evidence-manifest.json": "127e24fe77ebbf8bcf039edceb47dc9c7c93ebe0667c69132f1be3f6d9a1e57e",
    "fairness-report.md": "b50eb501675ab42094e4370fdee416a50d529936422a7e091f243dfc9c73a2d3",
    "g3-review.md": "6330736f8460f2ce01cf9448751dad3631558bf09bb971bbdbae96f5cb74c939",
    "governance-report.md": "b0cd6a5b0b46e77bbccfee78b9d1794bdc378a43c57cd2b088e914935ad4f30c",
    "model-card.md": "6ff8af6e92be11434fa54f99bcae0c844f0b953bee59f4ec7762064032e89171",
    "risk-register.md": "239b35a44d46b153b9d55174a54f31632b30ded06daa50fbb2a6053478b3cdc0",
    "summary.json": "9431910060c38c5c50fe58508871dd85c5fbf118f3c1af00ffaa8807420ee123",
}


def test_phase5_evidence_is_byte_identical_and_allowlisted() -> None:
    paths = {name: EVIDENCE_ROOT / name for name in EXPECTED_DIGESTS}

    assert {path.name for path in EVIDENCE_ROOT.iterdir() if path.is_file()} == set(paths)
    assert {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    } == EXPECTED_DIGESTS


def test_phase5_evidence_closes_g3_with_review_conditions_only() -> None:
    summary_bytes = (EVIDENCE_ROOT / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    manifest = json.loads((EVIDENCE_ROOT / "evidence-manifest.json").read_bytes())
    reports = "\n".join(
        (EVIDENCE_ROOT / filename).read_text(encoding="utf-8")
        for filename in EXPECTED_DIGESTS
        if filename.endswith(".md")
    )

    assert summary["status"] == "complete"
    assert summary["lineage"]["git_commit"] == "8989374d1d218cb850a2b291c5ba0d386181b4aa"
    assert summary["lineage"]["git_dirty"] is False
    assert summary["population"] == {
        "partition": "development_validation_only",
        "rows": 4800,
        "target_counts": {"0": 3738, "1": 1062},
        "unique_accounts": 4800,
        "validation_fold": 0,
    }
    assert summary["execution"] == {
        "calibration_fitting_performed": False,
        "cross_validation_performed": False,
        "final_test_predictions_loaded": False,
        "parameter_tuning_performed": False,
        "refitting_performed": False,
        "sealed_test_accessed": False,
        "training_performed": False,
        "validation_prediction_passes": 1,
    }
    assert summary["model"] == {
        "audit_fields_passed_to_estimator": False,
        "audit_input_invariance_verified": True,
        "bundle_id": "selected_v1",
        "calibration": "identity",
        "model_id": "catboost_fixed",
        "predictor_count": 19,
    }
    assert summary["validation_metrics"]["average_precision"] == pytest.approx(0.5565104548302114)
    assert summary["validation_metrics"]["brier_score"] == pytest.approx(0.13353854208377516)
    assert summary["validation_metrics"]["lift_at_0_1"] == pytest.approx(3.2109227871939736)
    assert summary["explanations"]["sample_rows"] == 1000
    assert sum(summary["explanations"]["stratum_counts"].values()) == 1000
    assert summary["explanations"]["max_raw_additivity_error"] <= 1e-10
    assert summary["explanations"]["max_sigmoid_probability_error"] <= 1e-10
    assert summary["explanations"]["row_level_values_committed"] is False
    assert {item["name"] for item in summary["explanations"]["reason_category_summary"]} == {
        "credit_capacity",
        "repayment_status",
        "billing_balance",
        "payment_behaviour",
    }
    assert [
        (item["axis"], item["group"], item["metric"], item["direction"])
        for item in summary["subgroup_review"]["triggers"]
    ] == [
        ("education_code", "1", "selection_rate_ratio", "below_lower_bound"),
        ("education_code", "3", "selection_rate_ratio", "above_upper_bound"),
    ]
    assert summary["subgroup_review"]["bootstrap"] == {
        "confidence_level": 0.95,
        "method": "within_group_stratified_percentile",
        "resamples": 500,
        "seed": 42,
    }
    assert summary["g3"]["result"] == "closed_with_conditions"
    assert summary["g3"]["fairness_certification_claimed"] is False
    assert summary["g3"]["regulatory_compliance_claimed"] is False
    assert summary["final_test_aggregate_reference"]["row_level_predictions_loaded"] is False
    assert summary["runtime_artifacts"]["committed"] is False
    assert summary["runtime_artifacts"]["row_level_data_committed"] is False

    assert manifest["configuration_sha256"] == (
        "990f33b1f1389f0666a75400a5677f3bdac4b8a09a531b43e6262b5a44cb0e12"
    )
    assert set(manifest["artifacts"]) == set(EXPECTED_DIGESTS) - {"evidence-manifest.json"}
    assert {name: item["sha256"] for name, item in manifest["artifacts"].items()} == {
        name: digest
        for name, digest in EXPECTED_DIGESTS.items()
        if name != "evidence-manifest.json"
    }
    assert manifest["prohibitions_verified"] == {
        "final_test_predictions_loaded": False,
        "fitting_performed": False,
        "sealed_test_accessed": False,
    }

    deterministic_text = summary_bytes.decode("utf-8") + reports
    for forbidden in (
        "C:\\Users",
        "/home/",
        "account_id,",
        "run_id",
        "mlflow.db",
        "timestamp",
        "fair model",
        "unbiased",
        "regulatory approval",
    ):
        assert forbidden not in deterministic_text
    assert "not a fairness certification" in reports
    assert "human-owned outreach" in reports
