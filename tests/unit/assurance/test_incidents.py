"""Fault drills prove rejection, isolation and controlled recovery."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from credit_risk.assurance import evidence as ev
from credit_risk.incidents import workflow as incidents
from credit_risk.inference.contracts import load_inference_config
from credit_risk.inference.engine import InferenceError


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
    monkeypatch.setattr(incidents, "data_drills", lambda engine: [])
    monkeypatch.setattr(incidents, "artifact_drill", lambda *a: {"drill": "artifact_integrity"})
    monkeypatch.setattr(incidents, "rollback_paths", lambda *a: {"drill": "phase7_sqlite_rollback"})
    monkeypatch.setattr(incidents, "platform_state", lambda: {"same": True})
    monkeypatch.setattr(incidents, "read_json", lambda *a: {})
    monkeypatch.setattr(incidents, "source_map", lambda *a: {})
    commands = []
    monkeypatch.setattr(incidents, "command", lambda args: commands.append(args))
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
    assert commands[-1][-2:] == ["start", "api"]
    with pytest.raises(ev.EvidenceError, match="exists"):
        incidents.build("b" * 64)
