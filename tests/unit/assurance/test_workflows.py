"""Exercise new orchestration with synthetic inputs and isolated publication roots."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.assurance import evidence as ev
from credit_risk.assurance import validation
from credit_risk.inference.engine import InferenceResult, ReasonAttribution
from credit_risk.monitoring import workflow as monitoring
from credit_risk.platform import evidence as platform_evidence
from credit_risk.release import release_b
from credit_risk.robustness import workflow as robustness

REAL_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    for module in (monitoring, robustness, platform_evidence, release_b):
        monkeypatch.setattr(module, "ROOT", tmp_path, raising=False)
        monkeypatch.setattr(module, "clean_commit", lambda: "a" * 40)
    files = set(
        monitoring.SOURCE_FILES
        + platform_evidence.SOURCE_FILES
        + [
            robustness.CONFIG,
            "configs/data/split_v1.lock.json",
        ]
    )
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_ROOT / name, target)
    return tmp_path


def features(rows=400):
    frame = pd.read_csv(REAL_ROOT / "tests/fixtures/inference_batch_v1.csv").iloc[:, 1:]
    frame = frame.iloc[np.arange(rows) % len(frame)].copy()
    frame.index = pd.Index([f"synthetic-{i:05}" for i in range(rows)])
    return frame


class Engine:
    def __init__(self):
        from credit_risk.inference.contracts import load_inference_config

        self.config = load_inference_config()

    def score(self, frame):
        n = len(frame)
        return InferenceResult(
            np.linspace(0.1, 0.9, n),
            ("standard",) * n,
            tuple((ReasonAttribution("credit_capacity", "neutral", 0.0),) * 2 for _ in range(n)),
            np.zeros((n, 4)),
            0.0,
            0.0,
        )


def test_reference_publication_and_no_overwrite(sandbox, monkeypatch):
    frame = features(4800)
    monkeypatch.setattr(
        monitoring, "validation_cohort", lambda root: (frame, pd.Series([0, 1] * 2400))
    )
    monkeypatch.setattr(monitoring, "InferenceEngine", Engine)
    monkeypatch.setattr(monitoring, "check_baseline", lambda *args: {})
    sha = monitoring.reference()
    result = monitoring.verify_report(monitoring.REFERENCE, sha, "reference")
    assert result["rows"] == 4800
    assert result["profiles"]["probability"]["rows"] == 4800
    with pytest.raises(ev.EvidenceError, match="exists"):
        monitoring.reference()
    with pytest.raises(ev.EvidenceError, match="Unknown"):
        monitoring.verify_report(monitoring.REFERENCE, sha, "bad")


def test_robustness_full_grid_never_fits_or_publishes_counterfactual_metrics(sandbox, monkeypatch):
    frame = features(4800)
    monkeypatch.setattr(
        robustness,
        "validation_cohort",
        lambda root: (frame, pd.Series([0, 1] * 2400, index=frame.index)),
    )
    monkeypatch.setattr(robustness, "InferenceEngine", Engine)
    monkeypatch.setattr(robustness, "check_baseline", lambda *args: {"status": "measured"})
    sha = robustness.build()
    result = robustness.verify_evidence(robustness.DEFAULT_OUTPUT, sha)
    assert result["status"] == "pending_owner_review"
    assert len(result["scenarios"]) == 14
    assert len(result["subsets"]) == 2
    assert "account_id" not in (sandbox / robustness.DEFAULT_OUTPUT / "summary.json").read_text()
    with pytest.raises(ev.EvidenceError, match="overwrite"):
        robustness.build()
    summary = json.loads((sandbox / robustness.DEFAULT_OUTPUT / "summary.json").read_bytes())
    summary["scenarios"].pop()
    monkeypatch.setattr(robustness, "verify", lambda *args: summary)
    with pytest.raises(ev.EvidenceError, match="Incomplete"):
        robustness.verify_evidence("x", "a" * 64)


def test_robustness_changed_protocol_refused(sandbox):
    config = sandbox / robustness.CONFIG
    config.write_text("{}")
    with pytest.raises(ev.EvidenceError, match="protocol"):
        robustness.build()


def test_monitor_batch_reconciles_input_and_publishes_no_accounts(sandbox, monkeypatch):
    from credit_risk.inference.batch import parse_batch_csv
    from credit_risk.inference.contracts import load_inference_config
    from credit_risk.monitoring.drift import profile

    frame = features(400)
    input_path = sandbox / "experiment/input.csv"
    input_path.parent.mkdir()
    frame.rename_axis("account_id").reset_index().to_csv(input_path, index=False)
    parsed = parse_batch_csv(input_path.read_bytes(), load_inference_config())
    run = sandbox / "experiment/batch"
    run.mkdir()
    scores = pd.DataFrame(
        {
            "account_id": parsed.account_ids,
            "probability_of_default": np.linspace(0.1, 0.9, 400),
            "risk_band": "standard",
        }
    )
    scores.to_csv(run / "scores.csv", index=False)
    ref = {
        "model_sha256": Engine().config.bundle.model_sha256,
        "profiles": {c: profile(frame[c]) for c in frame},
    }
    ref["profiles"]["probability"] = profile(scores.probability_of_default)
    reference_root = sandbox / "reports/monitoring/ref"
    reference_root.mkdir(parents=True)
    (reference_root / "evidence-manifest.json").write_bytes(b"{}")
    monkeypatch.setattr(monitoring, "verify", lambda *args: ref)
    manifest = {
        "input_sha256": ev.hash_file(input_path),
        "batch_id": "a" * 64,
        "status": "completed",
    }
    monkeypatch.setattr(monitoring, "verify_batch_run", lambda *args, **kwargs: manifest)
    sha = monitoring.batch(
        str(input_path), str(run), str(reference_root), "b" * 64, "reports/monitoring/batch"
    )
    result = ev.verify("reports/monitoring/batch", sha, "monitor_batch_v1")
    assert result["status"] == "clear"
    assert result["counts"]["valid"] == 400
    assert "synthetic-" not in (sandbox / "reports/monitoring/batch/summary.json").read_text()
    manifest["input_sha256"] = "f" * 64
    with pytest.raises(ev.EvidenceError, match="input"):
        monitoring.batch(
            str(input_path), str(run), str(reference_root), "b" * 64, "reports/monitoring/other"
        )


def test_service_publication(sandbox):
    logs = sandbox / "experiment/events.jsonl"
    logs.parent.mkdir()
    logs.write_text(
        json.dumps({"event": "api_request_completed", "status": "200", "duration_ms": 7})
    )
    sha = monitoring.service(str(logs), "reports/monitoring/service")
    assert ev.verify("reports/monitoring/service", sha, "monitor_service_v1")["request_count"] == 1


def platform_receipt(sandbox):
    from credit_risk.platform.contracts import load_platform_config

    config = load_platform_config()
    state = {
        "status": "ready",
        "registered_model_name": "credit-risk-default",
        "aliases": {"champion": "1", "rollback": "2"},
        "object_sha256": {
            "manifest.json": config.bundle.manifest_sha256,
            "model.cbm": config.bundle.model_sha256,
        },
        "active_revision": "phase7_rev_001",
        "fit_count": 0,
        "sealed_test_accessed": False,
    }
    folder = sandbox / "experiment/platform/test"
    folder.mkdir(parents=True)
    receipt = {
        "implementation_commit": "a" * 40,
        "sources": ev.source_map(platform_evidence.SOURCE_FILES),
        "states": [state] * 3,
        "health": {"api": True, "mlflow": True, "ui": True},
        "smoke_probabilities": [0.190382, 0.190382],
        "images": {s: "sha256:" + "b" * 64 for s in platform_evidence.SERVICES},
        "scan_sha256": {},
        "sbom_sha256": {},
    }
    for service in platform_evidence.SERVICES:
        scan = {"Metadata": {"ImageID": receipt["images"][service]}, "Results": []}
        sbom = {"bomFormat": "CycloneDX", "components": [{"name": "test"}]}
        for suffix, value in (("scan", scan), ("sbom", sbom)):
            path = folder / f"{service}-{suffix}.json"
            path.write_bytes(ev.encode(value))
            receipt[f"{suffix}_sha256"][service] = ev.hash_file(path)
    (folder / "receipt.json").write_bytes(ev.encode(receipt))
    return receipt, folder


def test_platform_receipts_bind_image_and_recovery(sandbox):
    receipt, folder = platform_receipt(sandbox)
    sha = platform_evidence.publish_evidence(str(folder))
    assert platform_evidence.verify_evidence(platform_evidence.OUTPUT, sha)["states_verified"] == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("implementation_commit", "f" * 40),
        ("sources", {}),
        ("states", []),
        ("health", {}),
        ("smoke_probabilities", [0.1, 0.1]),
        ("images", {}),
    ],
)
def test_platform_rejects_incomplete_receipt(sandbox, key, value):
    receipt, folder = platform_receipt(sandbox)
    receipt[key] = value
    with pytest.raises(ev.EvidenceError):
        platform_evidence.validate_receipt(receipt, "a" * 40)


@pytest.mark.parametrize("kind", ["image", "vulnerability", "sbom", "scan_hash", "sbom_hash"])
def test_platform_scan_blocks_bad_provenance_or_findings(sandbox, kind):
    receipt, folder = platform_receipt(sandbox)
    scan_path = folder / "api-scan.json"
    scan = json.loads(scan_path.read_bytes())
    if kind == "image":
        scan["Metadata"]["ImageID"] = "sha256:" + "c" * 64
    elif kind == "vulnerability":
        scan["Results"] = [{"Vulnerabilities": [{"Severity": "HIGH", "FixedVersion": "2"}]}]
    scan_path.write_bytes(ev.encode(scan))
    receipt["scan_sha256"]["api"] = ev.hash_file(scan_path)
    if kind == "sbom":
        (folder / "api-sbom.json").write_bytes(b"{}")
    elif kind == "scan_hash":
        receipt["scan_sha256"]["api"] = "f" * 64
    elif kind == "sbom_hash":
        receipt["sbom_sha256"]["api"] = "f" * 64
    (folder / "receipt.json").write_bytes(ev.encode(receipt))
    with pytest.raises(ev.EvidenceError):
        platform_evidence.publish_evidence(str(folder))


def test_validation_no_outcomes_for_unsupported_cohort():
    assert validation.metrics(np.zeros(20), np.ones(20) * 0.2)["status"] == "insufficient_support"


def test_ci_gate_requires_all_exact_commit_jobs():
    ci = {
        "repository": "shaloy-lewis/credit-card-default-prediction",
        "head_sha": "a" * 40,
        "run_id": 1,
        "conclusion": "success",
        "jobs": {name: "success" for name in release_b.REQUIRED_JOBS},
    }
    release_b.check_ci(ci, "a" * 40)
    with pytest.raises(ev.EvidenceError):
        release_b.check_ci(ci, "b" * 40)
    ci["jobs"].pop(next(iter(ci["jobs"])))
    with pytest.raises(ev.EvidenceError):
        release_b.check_ci(ci, "a" * 40)
