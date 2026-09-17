"""Complete-file and semantic integrity checks for published Phase 7 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from credit_risk.registry.workflow import verify_registry_evidence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / "reports" / "registry" / "phase7_v1"
EXPECTED_DIGESTS = {
    "evidence-manifest.json": "ce36f33da60fe6470d28a76b8053d102e74731115d069c4d476d0c2abbc47da9",
    "promotion-checklist.md": "8af86710b1b8d52d6c4df7f3cacf392357d9ad1689ce8bbe21f5dea2c74fe407",
    "registry-release-report.md": "d918d77337b6f5a09c3be27f0cc49ebf96f8be103fd59164cccaf147d39fbb26",
    "rollback-runbook.md": "37ae777c9b8c9345298ed0de814b770927ebe9df81e634624b3f8e937f450f2c",
    "summary.json": "f88324b6258e014bf399ef6bb5da246e927edd614a14cd2cf2770d1824d61865",
}


def test_published_phase7_files_match_reviewed_digests() -> None:
    assert {path.name for path in EVIDENCE_ROOT.iterdir()} == set(EXPECTED_DIGESTS)
    for name, expected in EXPECTED_DIGESTS.items():
        assert hashlib.sha256((EVIDENCE_ROOT / name).read_bytes()).hexdigest() == expected

    verified = verify_registry_evidence(
        evidence_root=Path("reports/registry/phase7_v1"),
        expected_manifest_sha256=EXPECTED_DIGESTS["evidence-manifest.json"],
    )
    assert verified.summary_sha256 == EXPECTED_DIGESTS["summary.json"]


def test_published_phase7_evidence_preserves_release_boundaries() -> None:
    summary = json.loads((EVIDENCE_ROOT / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE_ROOT / "evidence-manifest.json").read_text(encoding="utf-8"))

    assert summary["implementation_git_commit"] == ("cb63b396695dbaa997515e36a6e3f8c895476e67")
    assert summary["execution_git_commit"] == ("b26ef3ed81ff9dc8eeca41dfe833809dfe0af191")
    assert summary["configuration_sha256"] == (
        "83a29f8e927e91336bfc39779f27c5bb27de91194b5e1b8889acfd95dafe925b"
    )
    assert summary["transitions"] == {
        "initial": {"candidate": "2", "champion": "1"},
        "promoted": {"champion": "2", "rollback": "1"},
        "rolled_back": {"champion": "1", "rollback": "2"},
    }
    assert summary["final_state"] == {
        "active_registry_version": "1",
        "active_revision": "phase7_rev_001",
        "aliases": {"champion": "1", "rollback": "2"},
    }
    assert summary["model_bytes_unchanged_between_revisions"] is True
    assert {revision["model_sha256"] for revision in summary["revisions"]} == {
        "844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c"
    }
    smoke = summary["smoke_parity"]
    assert smoke["fixture_sha256"] == (
        "2da5a264485d4b3c9c3334ef74471dde7fb86a24c9e9d8b68aa311fc79aef3db"
    )
    assert smoke["expected_probability_six_decimals"] == 0.190382
    assert smoke["outputs_identical"] is True
    assert smoke["prediction_only"] is True
    assert smoke["sealed_test_fixture"] is False
    assert [item["release_revision"] for item in smoke["revisions"]] == [
        "phase7_rev_001",
        "phase7_rev_002",
    ]
    assert {item["probability_six_decimals"] for item in smoke["revisions"]} == {0.190382}
    assert {item["risk_band"] for item in smoke["revisions"]} == {"standard"}
    assert len({item["output_sha256"] for item in smoke["revisions"]}) == 1
    assert summary["image_scan"] == {
        "ignore_unfixed": True,
        "passed": True,
        "scanner": "trivy",
        "severity": ["HIGH", "CRITICAL"],
        "waiver_used": False,
    }
    assert summary["boundaries"] == {
        "fit_count": 0,
        "local_paths_published": False,
        "model_changed": False,
        "row_level_data_published": False,
        "runtime_mlflow_in_api": False,
        "sealed_test_accessed": False,
        "timestamps_published": False,
    }
    assert summary["claims"] == {
        "different_model_versions_compared": False,
        "g4_closed": False,
        "production_readiness_claimed": False,
    }
    assert manifest["boundaries_verified"] == summary["boundaries"]
    assert all(not receipt["published"] for receipt in manifest["runtime_receipts"].values())
    serialized = json.dumps({"manifest": manifest, "summary": summary}, sort_keys=True)
    assert "C:\\" not in serialized
    assert "sqlite:///" not in serialized
