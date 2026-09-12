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
    "evidence-manifest.json": "6df8745f6deefcd138d7d1c821e6ad38ca7762aa7581fa0fae28042cf7f2b853",
    "fairness-report.md": "5ce92624532618fd1493a3ff0b9dd12657fba77a04a44a6ed8c624439337fd3e",
    "g3-review.md": "f8ed2da0f5cbea29eb0107cc2ffb9cc2c40f1d40ba42d9ab95eed2a06ca38bbf",
    "governance-report.md": "3bc6ed154b4ae850810db4945f16642c00aa3413c8af314fcf1739748a1282fb",
    "model-card.md": "6ff8af6e92be11434fa54f99bcae0c844f0b953bee59f4ec7762064032e89171",
    "risk-register.md": "239b35a44d46b153b9d55174a54f31632b30ded06daa50fbb2a6053478b3cdc0",
    "summary.json": "9173ce26d9821aea6fc1744dc07b3d504888e074532703f9dd80a3e706e97c73",
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
    assert summary["lineage"] == {
        "assignment_sha256": ("2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e"),
        "bundle_manifest_sha256": (
            "df5ce6ce07b268f57fa3bf72c97cd32f8ebb66695d7157139942c91e46d7cd88"
        ),
        "canonical_sha256": ("75b2a746781a584b0456f843f1f269190b51e90983cba44c4ed6c4a8685e6c1c"),
        "configuration_sha256": (
            "1717abd20e5dad6819d2f67fc13eecfa38a8800decf9e6954ffd0c74a913f68c"
        ),
        "feature_contract_sha256": (
            "8978277ae1c92b6f0b8daed94cccf3cd51d8e6cae0aa9c0620d8cfb813384a4b"
        ),
        "final_test_summary_sha256": (
            "8b5e018f5e29a5128285afb877e0adaeca35f4b450061cac21e08ea3a51bda56"
        ),
        "git_commit": "226b7d7dd295d2d07b0eb567b228bfc2615ac5c7",
        "git_dirty": False,
        "model_sha256": "844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c",
        "reviewed_split_lock_sha256": (
            "b2312380fa46924ca414acbcfef63b0435d1321083e87e4df5ec04f18736093d"
        ),
        "selection_summary_sha256": (
            "8c11b1d443c782a8ef14aa3e708e3fffa064ecb4c9fe58d3e51a6effa46efbd7"
        ),
        "source_sha256": "45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0",
    }
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
        "parameter_tuning_performed": False,
        "refitting_performed": False,
        "training_performed": False,
        "validation_prediction_passes": 1,
    }
    expected_boundary = {
        "final_test_predictions_loaded": False,
        "full_dataset_integrity_verification_performed": True,
        "test_explanations_generated": False,
        "test_partition_returned": False,
        "test_partition_selected": False,
        "test_predictions_generated": False,
        "test_subgroup_analysis_performed": False,
    }
    assert summary["data_boundary"] == expected_boundary
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
    assert summary["subgroup_review"]["uncertainty"] == {
        "performance_intervals": {
            "confidence_level": 0.95,
            "method": "within_group_stratified_percentile",
            "metrics": [
                "mean_probability",
                "calibration_in_the_large",
                "brier_score",
                "selection_rate_at_q90",
                "true_positive_rate_at_q90",
                "false_positive_rate_at_q90",
            ],
            "resamples": 500,
            "seed": 42,
        },
        "prevalence_interval": {
            "confidence_level": 0.95,
            "method": "wilson_score",
            "z_value": 1.959963984540054,
        },
    }
    supported_groups = [
        group for group in summary["subgroup_review"]["groups"] if group["status"] == "reviewed"
    ]
    assert supported_groups
    for group in supported_groups:
        prevalence = group["metrics"]["target_prevalence"]
        interval = group["confidence_intervals"]["target_prevalence"]
        assert interval["lower"] < prevalence < interval["upper"]
    assert summary["g3"]["result"] == "closed_with_conditions"
    assert summary["g3"]["fairness_certification_claimed"] is False
    assert summary["g3"]["regulatory_compliance_claimed"] is False
    assert summary["final_test_aggregate_reference"]["row_level_predictions_loaded"] is False
    assert summary["runtime_artifacts"] == {
        "committed": False,
        "hashes": {
            "sampled_shap_values.csv": (
                "fb0ac7f76802aa985fe1a9e5b4164d96cc8f51379bfff0e1f28a3e7f81ded2b4"
            ),
            "subgroup_bootstrap.json": (
                "67387e8d1d35ed2ea13d0a8111c5a958df9c1bc214b188712e318cb3227793ee"
            ),
            "validation_predictions.csv": (
                "e0ad88980350bc003950624291bf4e98e92f61cfed0a824ffea12e355b99788a"
            ),
        },
        "row_level_data_committed": False,
    }

    assert manifest["configuration_sha256"] == (
        "1717abd20e5dad6819d2f67fc13eecfa38a8800decf9e6954ffd0c74a913f68c"
    )
    assert set(manifest["artifacts"]) == set(EXPECTED_DIGESTS) - {"evidence-manifest.json"}
    assert {name: item["sha256"] for name, item in manifest["artifacts"].items()} == {
        name: digest
        for name, digest in EXPECTED_DIGESTS.items()
        if name != "evidence-manifest.json"
    }
    assert manifest["prohibitions_verified"] == {
        "fitting_performed": False,
        **expected_boundary,
    }
    assert manifest["runtime_artifacts"] == {
        name: {"committed": False, "sha256": digest}
        for name, digest in summary["runtime_artifacts"]["hashes"].items()
    }
    assert all(item["row_level_data"] is False for item in manifest["artifacts"].values())

    deterministic_text = summary_bytes.decode("utf-8") + reports
    assert "sealed_test_accessed" not in deterministic_text
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
    fairness_report = (EVIDENCE_ROOT / "fairness-report.md").read_text(encoding="utf-8")
    assert "Prevalence [95% CI]" in fairness_report
    assert "Mean probability [95% CI]" in fairness_report
    assert "0.2358 [0.2154, 0.2574]" in fairness_report
