"""Fault drills prove rejection, isolation and controlled recovery."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from credit_risk.assurance import evidence as ev
from credit_risk.incidents import workflow as incidents
from credit_risk.inference.contracts import load_inference_config
from credit_risk.inference.engine import InferenceError
from credit_risk.inference.logging import ALLOWED_LOG_FIELDS
from credit_risk.monitoring.workflow import summarize_events


def test_schema_and_drift_drills_are_detected():
    outcomes = incidents.data_drills(SimpleNamespace(config=load_inference_config()))
    assert {x["drill"] for x in outcomes} == {
        "missing_columns",
        "invalid_values",
        "duplicate_ids",
        "population_shift",
        "prediction_shift",
    }
    assert all(x["detected"] and x["contained"] for x in outcomes)


def test_artifact_failure_recovery_is_isolated(tmp_path, monkeypatch):
    source = tmp_path / "models/selected_v1"
    source.mkdir(parents=True)
    (source / "manifest.json").write_bytes(b"manifest")
    (source / "model.cbm").write_bytes(b"reviewed")
    fixture = tmp_path / "tests/fixtures/inference_batch_v1.csv"
    fixture.parent.mkdir(parents=True)
    real_fixture = Path(__file__).resolve().parents[3] / "tests/fixtures/inference_batch_v1.csv"
    fixture.write_bytes(real_fixture.read_bytes())
    monkeypatch.setattr(incidents, "ROOT", tmp_path)

    class Engine:
        def __init__(self, bundle_root=None):
            if bundle_root:
                p = bundle_root / "model.cbm"
                if not p.exists() or p.read_bytes() != b"reviewed":
                    raise InferenceError("rejected")
            self.config = load_inference_config()

        def score(self, frame):
            return SimpleNamespace(probabilities=np.zeros(len(frame)))

    monkeypatch.setattr(incidents, "InferenceEngine", Engine)
    runtime = tmp_path / "experiment"
    runtime.mkdir()
    result = incidents.artifact_drill(runtime, Engine())
    assert result["recovered"]
    assert (source / "model.cbm").read_bytes() == b"reviewed"


def test_incident_runner_restores_service_and_publishes_pending_review(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(incidents, "clean_commit", lambda: "a" * 40)
    monkeypatch.setattr(incidents, "verify", lambda *a: {"targets": {"recovery_seconds": 1000}})
    monkeypatch.setattr(incidents, "InferenceEngine", lambda: None)
    monkeypatch.setattr(incidents, "data_drills", lambda *args: [])
    monkeypatch.setattr(incidents, "artifact_drill", lambda *a: {"drill": "artifact_integrity"})
    monkeypatch.setattr(incidents, "rollback_paths", lambda *a: {"drill": "phase7_sqlite_rollback"})
    monkeypatch.setattr(incidents, "platform_state", lambda: {"same": True})
    monkeypatch.setattr(incidents, "read_json", lambda *a: {})
    monkeypatch.setattr(incidents, "source_map", lambda *a: {})
    commands = []

    def command(args):
        commands.append(args)
        return "c" * 64 if args[-3:] == ["ps", "-q", "api"] else ""

    monkeypatch.setattr(incidents, "command", command)
    monkeypatch.setattr(incidents, "await_ready", lambda: 1)

    def request(path, payload=None):
        if path == "/ready":
            raise OSError("offline")
        return {"probability_of_default": 0.190382}

    monkeypatch.setattr(incidents, "request", request)
    sha = incidents.build("b" * 64)
    result = ev.verify(incidents.OUTPUT, sha, incidents.KIND)
    assert result["status"] == "pending_owner_review"
    assert result["drills"][-1]["recovered"] is True
    assert commands[-1] == ["docker", "start", "c" * 64]
    with pytest.raises(ev.EvidenceError, match="exists"):
        incidents.build("b" * 64)


@pytest.mark.artifact
def test_real_invalid_batches_leave_verified_receipts(tmp_path, monkeypatch):
    from credit_risk.inference.engine import InferenceEngine

    events = []
    monkeypatch.setattr(
        incidents, "emit_event", lambda event, **fields: events.append({"event": event, **fields})
    )
    outcomes = incidents.data_drills(InferenceEngine(), tmp_path)
    assert outcomes[0]["batch_status"] == "failed"
    assert outcomes[1]["rejected_rows"] == 1
    assert outcomes[2]["rejected_rows"] == 2
    assert all(len(x["input_sha256"]) == 64 for x in outcomes[:3])
    assert all(len(x["batch_id"]) == 64 for x in outcomes[:3])
    assert all(x["recovered"] and x["verified"] for x in outcomes)
    assert all(len(x["recovery_batch_id"]) == 64 for x in outcomes[:3])
    terminal = [x for x in events if x["event"] == "batch_attempt_completed"]
    assert len(terminal) == len({x["trace_id"] for x in terminal}) == 6
    assert [x["trace_id"] for x in terminal[::2]] == [x["trace_id"] for x in outcomes[:3]]
    assert [x["trace_id"] for x in terminal[1::2]] == [x["recovery_trace_id"] for x in outcomes[:3]]
    assert all(set(x) <= ALLOWED_LOG_FIELDS for x in events)
    summary = summarize_events([json.dumps(x) for x in events])
    assert summary["batch_attempt_count"] == 6
    assert summary["batch_failure_count"] == 1
    assert len(events) == 11  # Six attempts and five successfully published batch runs.


@pytest.mark.parametrize(
    "fault", [None, "identity", "published_missing", "status", "count", "recovery"]
)
def test_batch_drills_reconcile_published_rejections(tmp_path, monkeypatch, fault):
    config = load_inference_config()
    calls = []
    manifests = {}

    def attempt(source, name, output_root):
        calls.append(name)
        batch_id = incidents._batch_id(
            incidents._batch_identity(
                input_sha256=incidents.digest(source.read_bytes()),
                as_of_date="2026-09-30",
                snapshot_id=name,
                config_sha256=incidents.PHASE6_CONFIG_SHA256,
                manifest_sha256=config.bundle.manifest_sha256,
                model_sha256=config.bundle.model_sha256,
            )
        )
        status = "completed"
        count = 0
        if name == "missing_columns":
            status = "failed"
            if fault == "published_missing":
                (output_root / "2026-09-30" / name).mkdir(parents=True)
        elif not name.startswith("restored-"):
            status = "completed" if fault == "status" else "completed_with_rejections"
            count = 1 if name == "invalid_values" else 2
        if fault == "identity":
            batch_id = "0" * 64
        manifests[name] = {
            "batch_id": batch_id,
            "status": status,
            "counts": {"rejected_rows": 10 if fault == "count" else count},
        }
        if name.startswith("restored-") and fault == "recovery":
            status = "failed"
        return {
            "status": status,
            "rejection_count": count,
            "batch_id": batch_id,
            "trace_id": "f" * 32,
        }

    monkeypatch.setattr(incidents, "_batch_attempt", attempt)
    monkeypatch.setattr(incidents, "verify_batch_run", lambda path, **k: manifests[path.name])
    if fault:
        with pytest.raises(ev.EvidenceError):
            incidents.data_drills(SimpleNamespace(config=config), tmp_path)
    else:
        outcomes = incidents.data_drills(SimpleNamespace(config=config), tmp_path)
        assert calls == [
            "missing_columns",
            "restored-missing_columns",
            "invalid_values",
            "restored-invalid_values",
            "duplicate_ids",
            "restored-duplicate_ids",
        ]
        assert [x["rejected_rows"] for x in outcomes[1:3]] == [1, 2]


@pytest.mark.parametrize(
    "fault", [None, "missing", "duplicate", "fields", "trace", "batch", "status", "exit"]
)
def test_cli_drill_accepts_only_correlated_events(tmp_path, monkeypatch, fault):
    event = {
        "event": "batch_attempt_completed",
        "status": "failed",
        "operation": "batch",
        "trace_id": "a" * 32,
        "batch_id": "b" * 64,
        "duration_ms": 2.0,
    }
    if fault == "fields":
        event["account_id"] = "must-not-be-forwarded"
    elif fault == "trace":
        event.pop("trace_id")
    elif fault == "batch":
        event["batch_id"] = "invalid"
    elif fault == "status":
        event.pop("status")
    lines = ["human readable output", "[]", '{"event":"other"}']
    if fault != "missing":
        lines.append(json.dumps(event))
    if fault == "duplicate":
        lines.append(json.dumps(event))
    completed = SimpleNamespace(stdout="\n".join(lines), returncode=0 if fault == "exit" else 1)
    invoked = []

    def run(args, **kwargs):
        invoked.append((args, kwargs))
        return completed

    forwarded = []
    monkeypatch.setattr(incidents.subprocess, "run", run)
    monkeypatch.setattr(
        incidents,
        "emit_event",
        lambda event, **fields: forwarded.append({"event": event, **fields}),
    )
    if fault:
        with pytest.raises(ev.EvidenceError):
            incidents._batch_attempt(tmp_path / "input.csv", "drill", tmp_path / "batches")
        assert not forwarded
    else:
        result = incidents._batch_attempt(tmp_path / "input.csv", "drill", tmp_path / "batches")
        assert result == event
        assert forwarded == [event]
        assert invoked[0][0][:5] == [
            incidents.sys.executable,
            "-m",
            "credit_risk.cli",
            "inference",
            "batch",
        ]
        assert invoked[0][1]["capture_output"] is True
        assert invoked[0][1]["timeout"] == 120


@pytest.mark.parametrize("error", [OSError("unavailable"), subprocess.TimeoutExpired("cli", 120)])
def test_cli_drill_subprocess_failure_is_controlled(tmp_path, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(incidents.subprocess, "run", fail)
    with pytest.raises(ev.EvidenceError, match="did not complete"):
        incidents._batch_attempt(tmp_path / "input.csv", "drill", tmp_path / "batches")


@pytest.mark.artifact
def test_real_rollback_drill_uses_phase7_relative_paths(tmp_path, monkeypatch):
    import shutil

    from credit_risk.registry import workflow as registry
    from tests.integration.test_registry_workflow import _copy_repository_contract

    _copy_repository_contract(tmp_path)
    for name in ("phase7_promotion_approval.json", "phase7_rollback_approval.json"):
        shutil.copyfile(
            incidents.ROOT / "configs/registry" / name, tmp_path / "configs/registry" / name
        )
    monkeypatch.setattr(incidents, "ROOT", tmp_path)
    monkeypatch.setattr(registry, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(registry, "_git_is_ancestor", lambda *a: True)
    # Pass absolute roots as the incident collector does; Phase 7 must get relative paths.
    result = incidents.rollback_paths(
        tmp_path / "experiment/registry/drill", tmp_path / "experiment/deployments/drill"
    )
    assert result["recovered"] and result["prediction"] == 0.190382
    assert result["deployment_pointer_failures_refused"] == 2
