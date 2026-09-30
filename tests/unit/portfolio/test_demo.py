"""Exercise synthetic demonstration boundaries without touching historical evidence."""

import csv
import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from credit_risk.assurance import evidence as ev
from credit_risk.inference.batch import run_batch
from credit_risk.inference.contracts import load_inference_config
from credit_risk.monitoring import workflow as monitoring
from credit_risk.portfolio import demo
from tests.unit.inference.test_batch import _Engine

REAL_ROOT = Path(__file__).resolve().parents[3]
REAL_BATCH_CLI = demo._batch_cli


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    sources = [
        *monitoring.SOURCE_FILES,
        "tests/fixtures/prediction_request.json",
        "tests/fixtures/inference_batch_v1.csv",
        "reports/monitoring/reference_v1/evidence-manifest.json",
        "reports/monitoring/reference_v1/summary.json",
    ]
    for name in sources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_ROOT / name, path)
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(demo, "clean_commit", lambda: "a" * 40)
    monkeypatch.setattr(monitoring, "clean_commit", lambda: "a" * 40)
    monkeypatch.setattr(demo, "_ready", lambda url: None)
    config = load_inference_config()

    def predict(payload, *, request_id, **kwargs):
        assert payload["credit_limit_ntd"] == 1000000
        return {
            "probability_of_default": 0.190382,
            "risk_band": "standard",
            "trace_id": request_id,
            "manifest_sha256": config.bundle.manifest_sha256,
            "reasons": [
                {"category": "credit_capacity", "contribution_raw_log_odds": -0.2},
                {"category": "repayment_status", "contribution_raw_log_odds": 0.3},
            ],
        }

    def batch(source, run_id, runtime, receipt):
        result = run_batch(
            input_path=source,
            as_of_date=demo.AS_OF_DATE,
            snapshot_id=run_id,
            output_root=runtime / "batches",
            config=config,
            engine=_Engine(),
        )
        return {"batch_id": result.batch_id, "trace_id": uuid4().hex, "status": result.status}

    monkeypatch.setattr(demo, "predict_v1", predict)
    monkeypatch.setattr(demo, "_batch_cli", batch)
    return tmp_path


def test_walkthrough_reconciles_real_batch_and_monitoring(sandbox):
    result = demo.run_demo("unit-demo")
    assert result["prediction"] == 0.190382
    assert result["rows"] == 400 and result["selected_rows"] == 40
    assert result["monitoring_status"] == "investigate"
    assert result["idempotent_no_rewrite"] and not result["automatic_model_change"]
    receipt = sandbox / result["runtime_root"] / "receipt.json"
    assert json.loads(receipt.read_bytes()) == result
    source = receipt.parent / "synthetic.csv"
    original = source.read_bytes()
    with pytest.raises(demo.DemoError, match="already exists"):
        demo.run_demo("unit-demo")
    assert source.read_bytes() == original
    assert not any("account" in key for key in result)


def test_synthetic_fixture_is_deterministic_and_unique():
    content = (REAL_ROOT / "tests/fixtures/inference_batch_v1.csv").read_bytes()
    first = demo.synthetic_input(content)
    assert first == demo.synthetic_input(content)
    rows = list(csv.DictReader(io.StringIO(first.decode())))
    assert len(rows) == len({row["account_id"] for row in rows}) == 400
    assert all(row["account_id"].startswith("synthetic-") for row in rows)
    with pytest.raises(demo.DemoError, match="fixture"):
        demo.synthetic_input(b"wrong\n")


@pytest.mark.parametrize("run_id", ["", "..", "../outside", "bad/id", "x" * 49])
def test_unsafe_identity_fails_before_access(monkeypatch, run_id):
    monkeypatch.setattr(demo, "clean_commit", lambda: pytest.fail("no repository access"))
    with pytest.raises(demo.DemoError, match="run_id"):
        demo.run_demo(run_id)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com",
        "http://user:pass@localhost:8080",
        "http://localhost:8080/path",
        "http://localhost:8080/?query=1",
        "http://localhost:8080/#fragment",
    ],
)
def test_external_or_ambiguous_api_urls_are_rejected(url):
    with pytest.raises(demo.DemoError, match="loopback"):
        demo.run_demo("safe", url)


def test_dirty_code_and_existing_report_are_preserved(sandbox, monkeypatch):
    def dirty():
        raise ev.EvidenceError("dirty implementation")

    with monkeypatch.context() as changes:
        changes.setattr(demo, "clean_commit", dirty)
        with pytest.raises(ev.EvidenceError, match="dirty"):
            demo.run_demo("dirty")
    assert not (sandbox / "experiment").exists()
    existing = sandbox / "reports/monitoring/release_c_demo/existing"
    existing.mkdir(parents=True)
    (existing / "sentinel").write_bytes(b"preserve")
    with pytest.raises(demo.DemoError, match="already exists"):
        demo.run_demo("existing")
    assert (existing / "sentinel").read_bytes() == b"preserve"


def test_symlink_destination_is_refused(sandbox, monkeypatch):
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p.name == "linked" or original(p))
    with pytest.raises(ev.EvidenceError, match="Symlinked"):
        demo.run_demo("linked")
    assert not (sandbox / "experiment").exists()


@pytest.mark.parametrize(
    "fault", ["reference", "prediction", "counts", "rewrite", "trace", "monitor"]
)
def test_failed_invariants_never_publish_success_receipt(sandbox, monkeypatch, fault):
    if fault in {"reference", "monitor"}:
        original_verify = demo.verify

        def verify(root, digest, kind):
            value = original_verify(root, digest, kind)
            if fault == "reference" and kind == "monitor_reference_v1":
                value["model_sha256"] = "0" * 64
            if fault == "monitor" and kind == "monitor_batch_v1":
                value["status"] = "clear"
            return value

        monkeypatch.setattr(demo, "verify", verify)
    elif fault == "prediction":
        original_predict = demo.predict_v1

        def predict(*args, **kwargs):
            value = original_predict(*args, **kwargs)
            value["probability_of_default"] = 0.9
            return value

        monkeypatch.setattr(demo, "predict_v1", predict)
    elif fault == "counts":
        original_verify_batch = demo.verify_batch_run

        def verify_batch(*args, **kwargs):
            value = original_verify_batch(*args, **kwargs)
            value["counts"]["valid_rows"] = 399
            return value

        monkeypatch.setattr(demo, "verify_batch_run", verify_batch)
    else:
        original_batch = demo._batch_cli

        def batch(source, run_id, runtime, receipt):
            value = original_batch(source, run_id, runtime, receipt)
            if fault == "trace":
                value["trace_id"] = "b" * 32
            if fault == "rewrite" and "reuse" in receipt:
                path = runtime / "batches" / demo.AS_OF_DATE / run_id / "scores.csv"
                path.write_bytes(path.read_bytes() + b"\n")
            return value

        monkeypatch.setattr(demo, "_batch_cli", batch)
    with pytest.raises(demo.DemoError):
        demo.run_demo(fault)
    assert not (sandbox / f"experiment/portfolio/week11/{fault}/receipt.json").exists()


@pytest.mark.parametrize(
    "fault", [None, "fields", "missing", "duplicate", "exit", "trace", "batch"]
)
def test_cli_receipts_capture_actual_allowlisted_events(tmp_path, monkeypatch, fault):
    event = {
        "event": "batch_attempt_completed",
        "trace_id": "a" * 32,
        "batch_id": "b" * 64,
        "status": "completed",
    }
    if fault == "fields":
        event["account_id"] = "must-not-log"
    if fault == "trace":
        event["trace_id"] = "invalid"
    if fault == "batch":
        event.pop("batch_id")
    lines = ["human output", "[]", "{}"]
    if fault != "missing":
        lines.append(json.dumps(event))
    if fault == "duplicate":
        lines.append(json.dumps(event))

    def execute(args, **kwargs):
        assert args[:5] == [demo.sys.executable, "-m", "credit_risk.cli", "inference", "batch"]
        assert kwargs["timeout"] == 120
        return SimpleNamespace(returncode=1 if fault == "exit" else 0, stdout="\n".join(lines))

    monkeypatch.setattr(demo.subprocess, "run", execute)
    if fault:
        with pytest.raises(demo.DemoError):
            demo._batch_cli(tmp_path / "input.csv", "safe", tmp_path, "events.jsonl")
    else:
        assert demo._batch_cli(tmp_path / "input.csv", "safe", tmp_path, "events.jsonl") == event
        assert json.loads((tmp_path / "events.jsonl").read_bytes()) == event


def test_cli_timeout_is_controlled(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("batch", 120)

    monkeypatch.setattr(demo.subprocess, "run", timeout)
    with pytest.raises(demo.DemoError, match="could not complete"):
        demo._batch_cli(tmp_path / "input.csv", "safe", tmp_path, "events.jsonl")


def test_readiness_contract(monkeypatch):
    monkeypatch.setattr(
        demo.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b'{"status":"ready"}')
    )
    demo._ready("http://localhost:8080")
    monkeypatch.setattr(
        demo.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b'{"status":"down"}')
    )
    with pytest.raises(demo.DemoError, match="not ready"):
        demo._ready("http://localhost:8080")


def test_cli_result_and_expected_failure(sandbox, capsys):
    assert demo.main(["--run-id", "cli-example"]) == 0
    output = capsys.readouterr().out
    # Existing batch engine emits its own safe logs; the final pretty JSON is last.
    assert "synthetic_walkthrough_passed_not_release_approval" in output
    assert demo.main(["--run-id", "../outside"]) == 1
    assert "Demo failed:" in capsys.readouterr().err


@pytest.mark.artifact
def test_reviewed_model_walkthrough_uses_actual_cli_events(sandbox, monkeypatch):
    from fastapi.testclient import TestClient

    from credit_risk.inference.api import create_app
    from credit_risk.inference.contracts import CreditRiskResponse

    shutil.copyfile(
        REAL_ROOT / "models/selected_v1/model.cbm", sandbox / "models/selected_v1/model.cbm"
    )
    monkeypatch.setattr(demo, "_batch_cli", REAL_BATCH_CLI)
    with TestClient(create_app(bundle_root=sandbox / "models/selected_v1")) as client:

        def ready(url):
            assert client.get("/ready").json() == {"status": "ready"}

        def predict(payload, *, request_id, **kwargs):
            response = client.post(
                "/v1/predict", json=payload, headers={"X-Request-ID": request_id}
            )
            assert response.status_code == 200
            return CreditRiskResponse.model_validate_json(response.content).model_dump(mode="json")

        monkeypatch.setattr(demo, "_ready", ready)
        monkeypatch.setattr(demo, "predict_v1", predict)
        result = demo.run_demo("reviewed-model")
    assert result["prediction"] == 0.190382
    assert result["monitoring_status"] == "investigate"
    folder = sandbox / result["runtime_root"]
    first = [
        json.loads(line) for line in (folder / "batch-first-events.jsonl").read_text().splitlines()
    ]
    second = [
        json.loads(line) for line in (folder / "batch-reuse-events.jsonl").read_text().splitlines()
    ]
    assert len(first) == 2 and len(second) == 1
    assert first[-1]["batch_id"] == second[0]["batch_id"] == result["batch_id"]
    assert first[-1]["trace_id"] != second[0]["trace_id"]
