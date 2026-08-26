"""Integrity checks for the reviewed Phase 2 baseline evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "reports" / "modeling" / "baseline_v1"
SUMMARY_PATH = EVIDENCE_ROOT / "summary.json"
REPORT_PATH = EVIDENCE_ROOT / "baseline-report.md"

# Change these digests only after a clean official rebuild, reproducibility
# verification, and explicit review of the complete baseline evidence.
EXPECTED_SUMMARY_SHA256 = "11e0332fc9df6f7abf36080a8d09304b3e975f34ad060f70f8611f4fc0ad69d6"
EXPECTED_REPORT_SHA256 = "2830b4080f954e773dfdf0c37ed6eaabeaa31917f32071c783ec36abafb63a10"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def test_reviewed_baseline_evidence_is_byte_identical_and_development_only() -> None:
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
    }
    assert summary["experiment"]["experiment_id"] == "baseline_v1"
    assert summary["experiment"]["partition"] == "development"
    assert summary["experiment"]["primary_metric"] == "average_precision"
    assert summary["experiment"]["probability_guardrail"] == "brier_score"
    assert summary["experiment"]["primary_capacity_metric"] == {
        "capacity": 0.1,
        "metric": "lift",
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
    assert reproducibility["git_commit_sha"] == ("c695c600b5d48263b40c56b81be7b66f1edb9f2f")
    assert reproducibility["baseline_config_sha256"] == (
        "1666691fffea7d10debd233ed26114af74737bb8f66e0d442a7f4233d68762e0"
    )

    runtime = summary["runtime_artifacts"]
    assert runtime["contains_fitted_models"] is False
    assert runtime["contains_holdout_rows"] is False
    assert runtime["oof_predictions_sha256"] == (
        "c8ec30bec3c323ed0cfbe050aa3313ac356eb5d717ab305dee1b4365a0e51abe"
    )
    assert runtime["logistic_diagnostics_sha256"] == (
        "9a6c0ebe027fe00eda305d319bdf4dd1c7dfc84e470f3a8a7e00cf387ffda425"
    )

    logistic = summary["models"]["logistic_l2"]["repeat_summaries"]
    assert logistic["average_precision"]["mean"] == pytest.approx(0.541294, abs=5e-7)
    assert logistic["roc_auc"]["mean"] == pytest.approx(0.767968, abs=5e-7)
    assert logistic["brier_score"]["mean"] == pytest.approx(0.136362, abs=5e-7)
    assert logistic["capacity_0_1.lift"]["mean"] == pytest.approx(3.156903, abs=5e-7)

    forbidden_operational_keys = {
        "artifact_uri",
        "end_time",
        "run_id",
        "start_time",
        "timestamp",
        "tracking_uri",
    }
    assert _all_keys(summary).isdisjoint(forbidden_operational_keys)
    assert b"C:\\Users\\" not in summary_bytes
    assert b"/home/" not in summary_bytes

    report = report_bytes.decode("utf-8")
    assert report.startswith("# Governed baseline experiment report\n")
    assert f"- **Deterministic summary SHA-256:** `{EXPECTED_SUMMARY_SHA256}`" in report
    assert "sealed development folds only" in report
    assert "holdout rows were not exposed" in report
