"""Complete-file and semantic integrity checks for published Phase 6 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from credit_risk.inference.evidence import verify_inference_evidence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "reports" / "inference" / "phase6_v1"
EXPECTED_DIGESTS = {
    "evidence-manifest.json": "d870d04ce247458ed559dc80b7493c42d8e51ffa2feef7fa4883f44f291c6819",
    "inference-parity-report.md": "6132a645385aec8377b94c308c69c693ff3e86cc1e4218948744d7fcf8b9ee50",
    "summary.json": "2fe101d3e99335e4ecd35c59f81dd4e5672bc4df248a17056c76cd5e72216ef9",
}


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

    assert summary["implementation_git_commit"] == "7cb8f058de25f07252459dce6849a02dabc7ee55"
    assert summary["lineage"] == {
        "bundle_manifest_sha256": "df5ce6ce07b268f57fa3bf72c97cd32f8ebb66695d7157139942c91e46d7cd88",
        "config_sha256": "12fd8a8d0afc3e6394b03801da8991942b986f6a6e3676cb908b233919133ce9",
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
