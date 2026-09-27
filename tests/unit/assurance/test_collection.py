"""Evidence collection records measurements; CI capture reads actual run conclusions."""

import io
import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from credit_risk.assurance import collect, validation
from credit_risk.assurance import evidence as ev
from credit_risk.release import ci_evidence
from credit_risk.release.release_b import REQUIRED_JOBS


def test_ci_capture_exact_commit_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(ci_evidence, "clean_commit", lambda: "a" * 40)
    run = {
        "name": "CI",
        "head_sha": "a" * 40,
        "id": 20,
        "conclusion": "success",
        "html_url": "https://github.com/test",
    }

    def github(path):
        if path.startswith("actions/runs?"):
            return {"workflow_runs": [run]}
        return {"jobs": [{"name": name, "conclusion": "success"} for name in REQUIRED_JOBS]}

    monkeypatch.setattr(ci_evidence, "github", github)
    result = ci_evidence.capture()
    assert result["status"] == "ci_verified"
    with pytest.raises(FileExistsError):
        ci_evidence.capture()
    run["head_sha"] = "b" * 40
    with pytest.raises(ev.EvidenceError, match="No CI"):
        ci_evidence.capture(output="experiment/release_b/other.json")


def test_ci_github_transport_keeps_token_out_of_output(monkeypatch):
    class Response(io.BytesIO):
        pass

    requests = []

    def open_request(request, **kwargs):
        requests.append(request)
        return Response(b'{"jobs":[]}')

    monkeypatch.setattr(ci_evidence.urllib.request, "urlopen", open_request)
    monkeypatch.setenv("GITHUB_TOKEN", "test-only-token")
    assert ci_evidence.github("actions/runs/1/jobs") == {"jobs": []}
    assert requests[0].get_header("Authorization") == "Bearer test-only-token"


def test_validation_boundary_delegates_only_to_reviewed_development_loader(monkeypatch):
    import credit_risk.governance.workflow as gov
    import credit_risk.modeling.dataset as dataset

    monkeypatch.setattr(dataset, "load_governed_development_data", lambda **kwargs: object())
    monkeypatch.setattr(
        dataset,
        "load_governed_test_data",
        lambda **kwargs: pytest.fail("sealed test must remain unreachable"),
    )
    calls = []
    monkeypatch.setattr(
        gov, "_validate_development_boundary", lambda *args: calls.append("boundary")
    )
    monkeypatch.setattr(gov, "_validate_data_lineage", lambda *args: calls.append("lineage"))
    frame = pd.DataFrame({"x": np.zeros(4800)})
    target = pd.Series([0, 1] * 2400)
    monkeypatch.setattr(gov, "_validation_slice", lambda *args: (frame, target, None))
    actual, labels = validation.validation_cohort()
    assert len(actual) == len(labels) == 4800
    assert calls == ["boundary", "lineage"]
    monkeypatch.setattr(
        gov, "_validation_slice", lambda *args: (frame.iloc[:10], target[:10], None)
    )
    with pytest.raises(ev.EvidenceError):
        validation.validation_cohort()


def test_baseline_metrics_reject_drift(monkeypatch):
    import credit_risk.governance.contracts as gov

    config = gov.load_governance_config()
    monkeypatch.setattr(
        validation, "metrics", lambda *args: dict(config.prediction.expected_validation_metrics)
    )
    assert validation.check_baseline(None, None) == config.prediction.expected_validation_metrics
    monkeypatch.setattr(
        validation,
        "metrics",
        lambda *args: {key: 0 for key in config.prediction.expected_validation_metrics},
    )
    with pytest.raises(ev.EvidenceError):
        validation.check_baseline(None, None)


def test_collector_sequences_new_packages_and_keeps_release_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "ROOT", tmp_path)
    operations = []
    for name in [
        "platform",
        "publish_evidence",
        "robustness",
        "reference",
        "benchmark",
        "acceptance",
        "incidents",
        "batch",
        "service",
    ]:

        def fake(*args, _name=name, **kwargs):
            operations.append(_name)
            if _name == "incidents":
                # MLflow logging configuration closes existing file handlers.
                for handler in collect.LOGGER.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.close()
                collect.LOGGER.info('{"event":"batch_attempt_completed","status":"failed"}')
            return "a" * 64

        monkeypatch.setattr(collect, name, fake)
    monkeypatch.setattr(collect, "InferenceEngine", lambda: SimpleNamespace(config={}))
    monkeypatch.setattr(collect, "synthetic_batch", lambda path: path.write_bytes(b"synthetic"))
    monkeypatch.setattr(
        collect,
        "run_batch",
        lambda **kwargs: SimpleNamespace(
            run_root=tmp_path / "experiment/run",
            status="completed",
            valid_rows=10000,
            rejected_rows=0,
        ),
    )
    monkeypatch.setattr(
        collect,
        "command",
        lambda *args: (
            'startup\n{"event":"api_request_completed","status":"200","duration_ms":1}\n{}'
        ),
    )
    from credit_risk.release.release_b import NEW_EVIDENCE

    monkeypatch.setattr(
        collect, "read_json", lambda *args: {"evidence": {key: {} for key in NEW_EVIDENCE}}
    )
    monkeypatch.setattr(collect, "request", lambda *args: {"status": "ready"})
    collect.main()
    assert operations == [
        "platform",
        "publish_evidence",
        "robustness",
        "reference",
        "benchmark",
        "acceptance",
        "incidents",
        "batch",
        "service",
    ]
    contract = json.loads(
        (tmp_path / "experiment/release_b/proposed-release-contract.json").read_bytes()
    )
    assert contract["status"] == "evidence_collected_ci_and_owner_review_pending"
    captured = (tmp_path / "experiment/release_b/incident-events.jsonl").read_text()
    assert '"status":"failed"' in captured


def test_collector_authenticates_committed_reports_without_scoring(tmp_path, monkeypatch):
    from credit_risk.release.release_b import NEW_EVIDENCE

    monkeypatch.setattr(collect, "ROOT", tmp_path)
    (tmp_path / "reports/platform/phase8_v1").mkdir(parents=True)
    monkeypatch.setattr(collect, "platform", lambda: pytest.fail("Must not rerun official scoring"))
    monkeypatch.setattr(collect, "verify_platform", lambda *a: {})
    monkeypatch.setattr(collect, "verify_robustness", lambda *a: {})
    contract = {
        "evidence": {
            name: {"root": f"reports/{name}/v1", "expected_manifest_sha256": "a" * 64}
            for name in NEW_EVIDENCE
        }
    }
    monkeypatch.setattr(collect, "read_json", lambda *a: contract)
    verified = []
    monkeypatch.setattr(collect, "verify", lambda *a: verified.append(a))
    collect.main()
    assert len(verified) == len(NEW_EVIDENCE)
    monkeypatch.setattr(
        collect, "verify", lambda *a: (_ for _ in ()).throw(ev.EvidenceError("tampered"))
    )
    with pytest.raises(ev.EvidenceError, match="tampered"):
        collect.main()
