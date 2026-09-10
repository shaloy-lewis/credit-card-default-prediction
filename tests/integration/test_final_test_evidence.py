"""Complete-file and semantic integrity for the one-time final-test release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "reports" / "modeling" / "final_test_v1"
EXPECTED_DIGESTS = {
    "summary.json": "8b5e018f5e29a5128285afb877e0adaeca35f4b450061cac21e08ea3a51bda56",
    "final-test-report.md": "cb2410b288b1b11e73b37c591b3a129e2dce1abc8ba9ce2969217f15344aab54",
    "evaluation-started.json": "2eb8e4ae9f31c1b5afa0cd7cd63a2db4ed7103eca692cdc8530a1156c7103aef",
    "evaluation-completed.json": "eca1cfb2741645f1eb40a84c6a8a0528a31e101d67ae7e664a317d5d696bf581",
}


def test_final_test_evidence_is_byte_identical_and_allowlisted() -> None:
    paths = {name: EVIDENCE_ROOT / name for name in EXPECTED_DIGESTS}

    assert {path.name for path in EVIDENCE_ROOT.iterdir() if path.is_file()} == set(paths)
    assert {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    } == EXPECTED_DIGESTS


def test_final_test_evidence_closes_g2_without_model_changes_or_row_data() -> None:
    summary_bytes = (EVIDENCE_ROOT / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    started = json.loads((EVIDENCE_ROOT / "evaluation-started.json").read_bytes())
    completed = json.loads((EVIDENCE_ROOT / "evaluation-completed.json").read_bytes())
    report = (EVIDENCE_ROOT / "final-test-report.md").read_text(encoding="utf-8")

    assert summary["status"] == "complete"
    assert summary["g2_status"] == "closed"
    assert summary["population"] == {
        "assignment_sha256": "2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e",
        "partition": "test",
        "rows": 6000,
        "target_counts": {"0": 4673, "1": 1327},
        "unique_accounts": 6000,
    }
    assert summary["execution"] == {
        "cross_validation_performed": False,
        "evaluation_count": 1,
        "maximum_evaluations": 1,
        "refitting_performed": False,
        "retuning_performed": False,
        "training_performed": False,
    }
    assert summary["metrics"]["discrimination"]["average_precision"] == pytest.approx(
        0.5428673518681313
    )
    assert summary["metrics"]["probability"]["brier_score"] == pytest.approx(0.1363037019973075)
    lift = next(
        item["lift"] for item in summary["metrics"]["capacities"] if item["capacity"] == 0.1
    )
    assert lift == pytest.approx(3.089675960813866)
    assert all(gate["passed"] is True for gate in summary["gates"].values())
    assert summary["model"] == {
        "bundle_id": "selected_v1",
        "calibration": "identity",
        "manifest_sha256": "df5ce6ce07b268f57fa3bf72c97cd32f8ebb66695d7157139942c91e46d7cd88",
        "model_id": "catboost_fixed",
        "model_sha256": "844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c",
    }
    assert summary["runtime_artifacts"] == {
        "row_level_data_committed": False,
        "test_predictions_sha256": "645753a7f226670eb113dc60094c426e26763c8e3c76e466a6099691b56eb484",
    }
    assert started["evaluation_count"] == 1
    assert started["git_commit"] == "d001d21e8aca112dba20475e2d98ac4f06d50824"
    assert started["status"] == "started"
    assert completed == {
        "evaluation_count": 1,
        "evaluation_id": summary["evaluation_id"],
        "g2_closed": True,
        "report_sha256": EXPECTED_DIGESTS["final-test-report.md"],
        "schema_version": "1.0.0",
        "status": "complete",
        "summary_sha256": EXPECTED_DIGESTS["summary.json"],
        "test_predictions_sha256": summary["runtime_artifacts"]["test_predictions_sha256"],
    }
    assert "G2 status:** closed" in report
    assert "Training, refitting, retuning, and cross-validation:** not performed" in report

    deterministic_text = summary_bytes.decode("utf-8") + report
    for forbidden in ("C:\\Users", "run_id", "mlflow.db", "timestamp", "account_id,"):
        assert forbidden not in deterministic_text
