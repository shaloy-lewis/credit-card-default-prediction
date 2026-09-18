"""Complete-file and semantic integrity checks for published Phase 6 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from credit_risk.inference.evidence import verify_inference_evidence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "reports" / "inference" / "phase6_v1"
EXPECTED_DIGESTS = {
    "evidence-manifest.json": "919087229d20fe83c1846da65d5901ea103ac3182c9c2975c3490424a49f4df8",
    "inference-parity-report.md": "6132a645385aec8377b94c308c69c693ff3e86cc1e4218948744d7fcf8b9ee50",
    "summary.json": "3f5e9744ef4aa49661c900bfb52b3eee687ac27d4fc62903c94a5d69ac9f6afa",
}


@pytest.mark.artifact
def test_published_phase6_files_match_reviewed_digests() -> None:
    assert {path.name for path in EVIDENCE_ROOT.iterdir()} == set(EXPECTED_DIGESTS)
    for name, expected in EXPECTED_DIGESTS.items():
        assert hashlib.sha256((EVIDENCE_ROOT / name).read_bytes()).hexdigest() == expected

    verified = verify_inference_evidence(
        evidence_root=Path("reports/inference/phase6_v1"),
        expected_manifest_sha256=EXPECTED_DIGESTS["evidence-manifest.json"],
    )
    assert verified.summary_sha256 == EXPECTED_DIGESTS["summary.json"]


def test_published_phase6_evidence_preserves_parity_and_boundaries() -> None:
    summary = json.loads((EVIDENCE_ROOT / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE_ROOT / "evidence-manifest.json").read_text(encoding="utf-8"))

    assert summary["implementation_git_commit"] == "f6b37afea4f5f6fb28a4ffbff9444c3a795d98f9"
    assert summary["lineage"] == {
        "bundle_manifest_sha256": "df5ce6ce07b268f57fa3bf72c97cd32f8ebb66695d7157139942c91e46d7cd88",
        "config_sha256": "84227bb48c7ba812bdf2a2752ed151ede2ce5daa398911af9311491a852500a8",
        "fixture_sha256": "326d2fad6845f5ecf026b14a23bb8c5798d2ace7a776ccfd0f7bd5b5926bfd4a",
        "model_sha256": "844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c",
    }
    assert summary["population"] == {
        "input_rows": 20,
        "rejected_rows": 0,
        "row_level_data_published": False,
        "synthetic": True,
        "valid_rows": 20,
    }
    assert summary["batch"]["selected_rows"] == 2
    assert summary["batch"]["review_capacity_fraction"] == 0.1
    assert summary["batch"]["idempotent_rerun_reused"] is True
    assert summary["batch"]["idempotent_rerun_files_unchanged"] is True
    assert summary["parity"]["offline_to_batch"]["maximum_offline_batch_probability_error"] == 0.0
    assert summary["parity"]["offline_to_api"]["maximum_offline_api_probability_error"] <= 5e-7
    assert summary["parity"]["offline_to_api"]["removed_predict_endpoint_status"] == 404
    assert summary["boundaries"] == {
        "calibration_fitting_performed": False,
        "model_fitting_performed": False,
        "model_or_policy_changed": False,
        "parameter_tuning_performed": False,
        "safe_aggregate_logging_only": True,
        "sealed_test_loaded_or_scored": False,
    }
    assert summary["g4_status"] == "open"
    assert manifest["boundaries_verified"]["row_level_output_published"] is False
    serialized = json.dumps({"manifest": manifest, "summary": summary}, sort_keys=True)
    assert "acct-" not in serialized
    assert "C:\\" not in serialized
