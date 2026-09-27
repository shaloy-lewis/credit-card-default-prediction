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
def test_real_invalid_batches_leave_verified_receipts(tmp_path):
    from credit_risk.inference.engine import InferenceEngine

    outcomes = incidents.data_drills(InferenceEngine(), tmp_path)
    assert outcomes[0]["batch_status"] == "failed"
    assert outcomes[1]["rejected_rows"] == 1
    assert outcomes[2]["rejected_rows"] == 2
    assert all(len(x["input_sha256"]) == 64 for x in outcomes[:3])
    assert all(len(x["batch_id"]) == 64 for x in outcomes[:3])
    assert all(x["recovered"] and x["verified"] for x in outcomes)
    assert all(len(x["recovery_batch_id"]) == 64 for x in outcomes[:3])


@pytest.mark.parametrize("invalid_result", [False, True])
def test_batch_drills_reconcile_published_rejections(tmp_path, monkeypatch, invalid_result):
    from credit_risk.inference.batch import BatchInferenceError

    calls = []

    def run_batch(**kwargs):
        name = kwargs["snapshot_id"]
        calls.append(name)
        if name == "missing_columns":
            raise BatchInferenceError("rejected schema")
        if name.startswith("restored-"):
            return SimpleNamespace(
                status="completed", rejected_rows=0, run_root=tmp_path, batch_id="c" * 64
            )
        return SimpleNamespace(
            status="completed" if invalid_result else "completed_with_rejections",
            rejected_rows=1 if name == "invalid_values" else 2,
            run_root=tmp_path,
            batch_id="a" * 64,
        )

    monkeypatch.setattr(incidents, "run_batch", run_batch)
    monkeypatch.setattr(incidents, "verify_batch_run", lambda *a, **k: {"input_sha256": "b" * 64})
    if invalid_result:
        with pytest.raises(ev.EvidenceError, match="expected batch evidence"):
            incidents.data_drills(SimpleNamespace(config=load_inference_config()), tmp_path)
    else:
        outcomes = incidents.data_drills(SimpleNamespace(config=load_inference_config()), tmp_path)
        assert calls == [
            "missing_columns",
            "restored-missing_columns",
            "invalid_values",
            "restored-invalid_values",
            "duplicate_ids",
            "restored-duplicate_ids",
        ]
        assert [x["rejected_rows"] for x in outcomes[1:3]] == [1, 2]


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
