"""Operational commands are measured, bounded and fail closed."""

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from credit_risk.assurance import runtime
from credit_risk.assurance.evidence import EvidenceError
from credit_risk.monitoring import benchmark
from credit_risk.platform import rehearsal


def test_command_sanitizes_failures(monkeypatch):
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=" ok\n"))
    assert runtime.command(["safe"]) == "ok"

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["secret"], stderr="password=secret")

    monkeypatch.setattr(runtime.subprocess, "run", fail)
    with pytest.raises(EvidenceError) as error:
        runtime.command(["safe"])
    assert "password" not in str(error.value)


@pytest.mark.parametrize(
    "content,expected", [(b'{"status":"ready"}', {"status": "ready"}), (b"ok", "ok")]
)
def test_http_response_formats(monkeypatch, content, expected):
    class Response(io.BytesIO):
        pass

    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k: Response(content))
    assert runtime.request("/ready") == expected
    assert runtime.request("/v1/predict", {"safe": 1}) == expected


def test_wait_until_ready_and_timeout(monkeypatch):
    readings = iter([0, 0, 1])
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: next(readings))
    monkeypatch.setattr(runtime, "request", lambda *a: {"status": "ready"})
    assert runtime.await_ready() == 1
    readings = iter([0, 0, 1000])
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: next(readings))
    monkeypatch.setattr(runtime.time, "sleep", lambda x: None)
    monkeypatch.setattr(runtime, "request", lambda *a: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(EvidenceError):
        runtime.await_ready()


def test_platform_state_and_immutable_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "command", lambda *a: 'progress\n{"object_sha256": {}}')
    assert runtime.platform_state() == {"object_sha256": {}}
    monkeypatch.setattr(runtime, "command", lambda *a: 'progress\n{"other": 1}')
    with pytest.raises(EvidenceError):
        runtime.platform_state()
    path = tmp_path / "receipt.json"
    runtime.write_new(path, {"result": 1})
    with pytest.raises(FileExistsError):
        runtime.write_new(path, {"result": 2})


def test_measure_separate_fresh_batches_and_parity(monkeypatch, tmp_path):
    from credit_risk.monitoring.benchmark import MEASURES

    monkeypatch.setattr(benchmark, "read_json", lambda *a: {})
    monkeypatch.setattr(
        benchmark,
        "request",
        lambda *a: {"probability_of_default": 0.190382, "model_id": "catboost_fixed"},
    )
    monkeypatch.setattr(benchmark, "InferenceEngine", lambda: SimpleNamespace(config={}))
    monkeypatch.setattr(
        benchmark,
        "run_batch",
        lambda **k: SimpleNamespace(reused=False, status="completed", valid_rows=10000),
    )
    monkeypatch.setattr(benchmark, "command", lambda *a: "")
    monkeypatch.setattr(benchmark, "await_ready", lambda: 1)
    assert set(benchmark.measure(tmp_path, "trial")) == set(MEASURES)
    monkeypatch.setattr(
        benchmark,
        "run_batch",
        lambda **k: SimpleNamespace(reused=True, status="completed", valid_rows=10000),
    )
    with pytest.raises(EvidenceError, match="fresh"):
        benchmark.measure(tmp_path, "trial")
    monkeypatch.setattr(
        benchmark,
        "request",
        lambda *a: {"probability_of_default": 0.2, "model_id": "catboost_fixed"},
    )
    with pytest.raises(EvidenceError, match="parity"):
        benchmark.measure(tmp_path, "trial")


def test_benchmark_then_acceptance_frozen_targets(monkeypatch, tmp_path):
    from credit_risk.assurance import evidence

    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "clean_commit", lambda: "a" * 40)
    monkeypatch.setattr(benchmark, "code_sources", lambda: {})

    def measurement(folder, iteration):
        (folder / "synthetic.csv").write_bytes(b"same")
        return {"api_p95_ms": 10, "batch_seconds": 2, "recovery_seconds": 1}

    monkeypatch.setattr(benchmark, "measure", measurement)
    sha = benchmark.rehearse()
    result = evidence.verify(benchmark.BENCHMARK, sha, "monitor_benchmark_v1")
    assert result["targets"]["api_p95_ms"] == 20
    acceptance_sha = benchmark.acceptance(sha)
    accepted = evidence.verify(
        "reports/monitoring/acceptance_v1", acceptance_sha, "monitor_acceptance_v1"
    )
    assert accepted["status"] == "passed"
    with pytest.raises(EvidenceError, match="already"):
        benchmark.rehearse()
    with pytest.raises(EvidenceError, match="already"):
        benchmark.acceptance(sha)
    monkeypatch.setattr(benchmark, "hardware", lambda: {})
    with pytest.raises(EvidenceError, match="machine"):
        benchmark.acceptance(sha, runtime="experiment/monitoring/new")


def test_platform_rehearsal_collects_measured_not_claimed_state(monkeypatch, tmp_path):
    from credit_risk.assurance import evidence

    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(rehearsal, "ROOT", tmp_path)
    monkeypatch.setattr(rehearsal, "clean_commit", lambda: "a" * 40)
    monkeypatch.setattr(rehearsal, "source_map", lambda *a: {})
    monkeypatch.setattr(rehearsal, "platform_state", lambda *a: {"status": "ready"})
    monkeypatch.setattr(rehearsal, "await_ready", lambda: 1)
    monkeypatch.setattr(rehearsal, "read_json", lambda *a: {})

    def request(path, payload=None, port=8080):
        return {
            "/ready": {"status": "ready"},
            "/v1/predict": {"probability_of_default": 0.190382},
            "/health": "OK",
            "/_stcore/health": "ok",
        }[path]

    monkeypatch.setattr(rehearsal, "request", request)

    def command(args, timeout=900):
        if args[:3] == ["docker", "image", "inspect"]:
            return json.dumps([{"Id": "sha256:" + "b" * 64}])
        if args[:2] == ["docker", "inspect"]:
            return json.dumps([{"Image": "sha256:" + "c" * 64}])
        if "--output" in args:
            Path(args[args.index("--output") + 1]).write_bytes(b"{}")
        return ""

    monkeypatch.setattr(rehearsal, "command", command)
    with pytest.raises(EvidenceError, match=".env"):
        rehearsal.rehearse()
    (tmp_path / ".env").write_text("test")
    path = rehearsal.rehearse()
    receipt = evidence.read_json(tmp_path / path / "receipt.json")
    assert receipt["health"] == {"api": True, "mlflow": True, "ui": True}
    assert len(receipt["states"]) == 3
    with pytest.raises(EvidenceError, match="exists"):
        rehearsal.rehearse()
    monkeypatch.setattr(rehearsal, "command", lambda *a: "existing-volume")
    with pytest.raises(EvidenceError, match="volumes"):
        rehearsal.rehearse(runtime="experiment/platform/other")


def test_platform_health_waits_for_dependencies_and_times_out(monkeypatch):
    readings = iter([0, 0, 1, 2])
    monkeypatch.setattr(rehearsal.time, "perf_counter", lambda: next(readings))
    monkeypatch.setattr(rehearsal.time, "sleep", lambda _: None)
    calls = []

    def request(path, port=8080):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("restarting")
        return {"/ready": {"status": "ready"}, "/health": "OK", "/_stcore/health": "ok"}[path]

    monkeypatch.setattr(rehearsal, "request", request)
    assert all(rehearsal.wait_health().values())
    readings = iter([0, 0, 300])
    monkeypatch.setattr(rehearsal.time, "perf_counter", lambda: next(readings))
    monkeypatch.setattr(rehearsal, "request", lambda *a, **k: "not-ready")
    with pytest.raises(EvidenceError, match="healthy"):
        rehearsal.wait_health()
