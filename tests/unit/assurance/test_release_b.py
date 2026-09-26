"""Release B verifies dependencies and requires a separate owner decision."""

import json

import pytest

from credit_risk.assurance import evidence as ev
from credit_risk.release import release_b as rb

SHA = "a" * 64
COMMIT = "b" * 40


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(rb, "clean_commit", lambda: COMMIT)
    monkeypatch.setattr(rb, "historical_evidence", lambda: {})
    monkeypatch.setattr(rb, "source_map", lambda *args: {})
    config = {
        "release_id": "release_b_v1",
        "evidence": {
            name: {"root": f"reports/test/{name}", "expected_manifest_sha256": SHA}
            for name in rb.NEW_EVIDENCE
        },
        "ci": {"path": "experiment/ci.json", "sha256": SHA},
    }
    summaries = {
        "platform": {"status": "platform_verified"},
        "robustness": {
            "scenarios": [{"scenario": "sensitivity"}],
            "subsets": [{"scenario": "subset"}],
        },
        "reference": {"status": "reference_ready"},
        "batch_monitoring": {"status": "investigate"},
        "service_monitoring": {"status": "measured"},
        "benchmark": {"status": "targets_frozen"},
        "acceptance": {"status": "passed"},
        "incidents": {
            "drills": [
                {"drill": name, "detected": True, "contained": True}
                for name in (
                    "missing_columns",
                    "invalid_values",
                    "duplicate_ids",
                    "population_shift",
                    "prediction_shift",
                    "artifact_integrity",
                    "phase7_sqlite_rollback",
                    "service_interruption",
                )
            ]
        },
    }
    ci = {
        "repository": "shaloy-lewis/credit-card-default-prediction",
        "head_sha": COMMIT,
        "conclusion": "success",
        "run_id": 1,
        "jobs": {name: "success" for name in rb.REQUIRED_JOBS},
    }

    def read(path):
        if str(path).endswith("release_b_v1.json"):
            return config
        if str(path).endswith("ci-receipt.json"):
            return ci
        return {"outputs": {}}

    monkeypatch.setattr(rb, "read_json", read)
    monkeypatch.setattr(rb, "hash_file", lambda path: SHA)
    monkeypatch.setattr(
        rb, "verify", lambda root, expected, kind: summaries[str(root).split("/")[-1]]
    )
    import credit_risk.platform.evidence
    import credit_risk.robustness.workflow

    monkeypatch.setattr(credit_risk.platform.evidence, "verify_evidence", lambda *args: {})
    monkeypatch.setattr(credit_risk.robustness.workflow, "verify_evidence", lambda *args: {})
    published = {}

    def publish(destination, **kwargs):
        published.update(kwargs)
        return SHA

    monkeypatch.setattr(rb, "publish", publish)
    return config, summaries, ci, published


def test_build_is_zero_scoring_and_cannot_approve_itself(candidate):
    config, summaries, ci, published = candidate
    assert rb.build(expected_ci_sha256=SHA) == SHA
    summary = published["summary"]
    assert summary["status"] == "pending_owner_signoff"
    assert summary["g4_status"] == "open"
    assert all("\\" not in key for key in published["sources"])
    template = json.loads(published["extra"]["owner-approval-template.json"])
    assert template["decision"] == "pending"
    assert template["dossier_sha256"] is None
    assert "risk:education_disparities" in template["dispositions"]
    risks = json.loads(published["extra"]["current-risk-disposition.json"])
    assert risks["historical_register_rewritten"] is False
    assert len([row for row in risks["risks"] if row["historical_entry"]]) == 10
    assert all(row["decision"] == "pending" for row in risks["risks"])


@pytest.mark.parametrize(
    "case",
    [
        "contract",
        "missing_sha",
        "acceptance",
        "service",
        "sample",
        "drills",
        "control",
        "ci_hash",
        "ci_failed",
    ],
)
def test_build_refuses_missing_or_failed_evidence(candidate, monkeypatch, case):
    config, summaries, ci, published = candidate
    if case == "contract":
        config["evidence"].pop("platform")
    elif case == "missing_sha":
        config["evidence"]["platform"]["expected_manifest_sha256"] = None
    elif case == "acceptance":
        summaries["acceptance"]["status"] = "failed"
    elif case == "service":
        summaries["service_monitoring"]["status"] = "insufficient_data"
    elif case == "sample":
        summaries["batch_monitoring"]["status"] = "insufficient_data"
    elif case == "drills":
        summaries["incidents"]["drills"].pop()
    elif case == "control":
        summaries["incidents"]["drills"][0]["detected"] = False
    elif case == "ci_hash":
        monkeypatch.setattr(rb, "hash_file", lambda path: "f" * 64)
    elif case == "ci_failed":
        ci["jobs"][next(iter(ci["jobs"]))] = "failure"
    with pytest.raises(ev.EvidenceError):
        rb.build(expected_ci_sha256=SHA)
    assert not published


def test_owner_approval_must_bind_dossier_and_every_disposition(candidate, monkeypatch):
    config, summaries, ci, published = candidate
    rb.build(expected_ci_sha256=SHA)
    summary = published["summary"]
    original_verify = rb.verify
    monkeypatch.setattr(
        rb,
        "verify",
        lambda root, expected, kind: (
            summary if kind == "release_b_v1" else original_verify(root, expected, kind)
        ),
    )
    pending = rb.verify_b("reports/releases/test", SHA)
    assert pending["g4_status"] == "open"
    approval = {
        "decision": "approved",
        "owner": "test owner",
        "scope": "local_portfolio_only",
        "dossier_sha256": SHA,
        "g3_conditions_retained": True,
        "dispositions": {
            key: {"decision": "accepted_with_restriction", "restriction": "local demo only"}
            for key in summary["required_dispositions"]
        },
    }
    monkeypatch.setattr(rb, "read_json", lambda path: approval)
    assert (
        rb.verify_b("reports/releases/test", SHA, "configs/owner.json", SHA)["g4_status"]
        == "closed"
    )
    with pytest.raises(ev.EvidenceError):
        rb.verify_b("reports/releases/test", SHA, "configs/owner.json", "f" * 64)
    approval["dossier_sha256"] = "c" * 64
    with pytest.raises(ev.EvidenceError):
        rb.verify_b("reports/releases/test", SHA, "configs/owner.json", SHA)
    approval["dossier_sha256"] = SHA
    next(iter(approval["dispositions"].values()))["restriction"] = ""
    with pytest.raises(ev.EvidenceError):
        rb.verify_b("reports/releases/test", SHA, "configs/owner.json", SHA)
