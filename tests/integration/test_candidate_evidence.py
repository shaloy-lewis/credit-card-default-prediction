"""Integrity checks for the reviewed Phase 3 candidate evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "reports" / "modeling" / "candidate_v1"
SUMMARY_PATH = EVIDENCE_ROOT / "summary.json"
REPORT_PATH = EVIDENCE_ROOT / "candidate-report.md"

# Change these digests only after a clean official two-run rebuild, byte-level
# reproducibility verification, and explicit review of the complete evidence.
EXPECTED_SUMMARY_SHA256 = "55aaa971417bddbcad00b8bdf388f74baa13f6ad96304dd108227e48de23ea83"
EXPECTED_REPORT_SHA256 = "156967cfda68ddf6c49e4f1e1666266d69261c58628820df3a2a821e560b17c2"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def test_reviewed_candidate_evidence_is_reproducible_governed_and_holdout_blind() -> None:
    summary_bytes = SUMMARY_PATH.read_bytes()
    report_bytes = REPORT_PATH.read_bytes()

    assert _sha256(summary_bytes) == EXPECTED_SUMMARY_SHA256
    assert _sha256(report_bytes) == EXPECTED_REPORT_SHA256

    summary = json.loads(summary_bytes)
    assert summary["schema_version"] == "1.0.0"
    assert summary["data"] == {
        "development_rows": 24_000,
        "holdout_evaluated": False,
        "n_repeats": 3,
        "n_splits": 5,
        "partition": "development",
    }
    assert summary["fit_budget"] == {
        "completed_fold_fits": 150,
        "diagnostic_fold_fits": 30,
        "evaluated_variants": 10,
        "maximum_fold_fits": 150,
        "search_fold_fits": 120,
    }

    evidence_policy = summary["evidence_policy"]
    assert evidence_policy == {
        "independent_executions_required": 2,
        "required_byte_identical_artifacts": [
            "summary.json",
            "candidate-report.md",
            "oof_predictions.csv",
            "fold_diagnostics.json",
        ],
        "third_fit_pass_for_publication": "prohibited",
        "tracking_roots_must_be_independent": True,
    }

    lineage = summary["lineage"]
    assert lineage["source_sha256"] == (
        "45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0"
    )
    assert lineage["canonical_sha256"] == (
        "75b2a746781a584b0456f843f1f269190b51e90983cba44c4ed6c4a8685e6c1c"
    )
    assert lineage["assignment_sha256"] == (
        "2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e"
    )
    assert lineage["feature_contract_sha256"] == (
        "8978277ae1c92b6f0b8daed94cccf3cd51d8e6cae0aa9c0620d8cfb813384a4b"
    )
    assert lineage["reviewed_split_lock_sha256"] == (
        "b2312380fa46924ca414acbcfef63b0435d1321083e87e4df5ec04f18736093d"
    )

    reproducibility = summary["reproducibility"]
    assert reproducibility["git_dirty"] is False
    assert reproducibility["git_commit_sha"] == ("2b46d4c3d0e2c37c7b8ef056244c5870d7b098b6")
    assert reproducibility["candidate_config_sha256"] == (
        "4bd9a404064d410e0339e0638464aaf6c1ac0bca632156a47af14a822d7cb5f3"
    )
    assert reproducibility["feature_contract_sha256"] == lineage["feature_contract_sha256"]

    runtime = summary["runtime_artifacts"]
    assert runtime["contains_fitted_models"] is False
    assert runtime["contains_holdout_rows"] is False
    assert runtime["oof_prediction_rows"] == 720_000
    assert runtime["oof_predictions_sha256"] == (
        "94ee8a56b731008722a63a9913f696dbe1bc827f64e1e903208bf98a0c44fd46"
    )
    assert runtime["fold_diagnostics_sha256"] == (
        "a3b5a2e64c7a145ce408d2e6a77ffc2034fe71e76387271a9ff9bdf19e5a2174"
    )

    variants = summary["variants"]
    assert len(variants) == 10
    selected = variants["operational_full__cb_cfg_006"]
    assert selected["configuration_id"] == "cb_cfg_006"
    assert selected["feature_view"] == "operational_full"
    assert selected["role"] == "search"
    assert selected["parameters"] == {
        "bagging_temperature": 0.0,
        "depth": 4,
        "iterations": 300,
        "l2_leaf_reg": 12.0,
        "learning_rate": 0.03,
        "random_strength": 0.0,
    }
    assert selected["eligible_for_advancement"] is True
    assert selected["gate_outcome"] == {
        "average_precision_passed": True,
        "brier_score_passed": True,
        "eligible": True,
        "lift_at_0_1_passed": True,
        "repeat_stability_passed": True,
    }
    repeat = selected["repeat_summaries"]
    assert repeat["average_precision"]["mean"] == pytest.approx(0.556419296250262)
    assert repeat["average_precision"]["standard_deviation"] == pytest.approx(0.000821123128344397)
    assert repeat["brier_score"]["mean"] == pytest.approx(0.134101450753693)
    assert repeat["capacity_0_1.lift"]["mean"] == pytest.approx(3.20210962516481)

    decision = summary["selection"]
    assert decision == {
        "catboost_advances": True,
        "diagnostic_views_eligible_for_advancement": False,
        "equivalent_configuration_ids": [
            "cb_cfg_001",
            "cb_cfg_002",
            "cb_cfg_003",
            "cb_cfg_004",
            "cb_cfg_006",
            "cb_cfg_007",
        ],
        "gate_id": "balanced_v1",
        "selected_configuration_id": "cb_cfg_006",
        "selected_model_id": "catboost_v1",
    }

    forbidden_operational_keys = {
        "artifact_uri",
        "end_time",
        "run_id",
        "start_time",
        "timestamp",
        "tracking_uri",
    }
    assert _all_keys(summary).isdisjoint(forbidden_operational_keys)
    for content in (summary_bytes, report_bytes):
        assert b"C:\\Users\\" not in content
        assert b"/home/" not in content

    report = report_bytes.decode("utf-8")
    assert report.startswith("# Governed CatBoost candidate report\n")
    assert f"- **Deterministic summary SHA-256:** `{EXPECTED_SUMMARY_SHA256}`" in report
    assert "**Completed fold fits:** `150`" in report
    assert "**Selected configuration:** `cb_cfg_006`" in report
    assert "**Phase 4 candidate:** `catboost_v1`" in report
    assert "**CatBoost advances:** `true`" in report
    assert "the holdout was not fitted, scored, or evaluated" in report
