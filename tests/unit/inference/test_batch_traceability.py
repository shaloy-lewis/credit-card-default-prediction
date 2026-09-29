"""Real CLI/workflow attempts retain privacy-safe identities on every exit path."""

import hashlib
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from credit_risk.inference import batch, cli
from credit_risk.inference.contracts import PHASE6_CONFIG_SHA256, load_inference_config
from credit_risk.inference.engine import InferenceError
from credit_risk.inference.logging import ALLOWED_LOG_FIELDS
from credit_risk.monitoring.workflow import summarize_events
from tests.unit.inference.test_batch import _csv, _Engine, _values


@pytest.mark.parametrize(
    ("case", "exit_code", "identified", "status", "rejections"),
    [
        ("missing_columns", 1, True, "failed", None),
        ("malformed_csv", 1, True, "failed", None),
        ("unreadable", 1, False, "failed", None),
        ("invalid_identity", 1, False, "failed", None),
        ("model_failure", 1, False, "failed", None),
        ("config_failure", 1, False, "failed", None),
        ("score_failure", 1, True, "failed", None),
        ("publication_failure", 1, True, "failed", None),
        ("partial", 3, True, "completed_with_rejections", 1),
        ("rejected", 1, True, "failed", 1),
        ("success", 0, True, "completed", 0),
        ("reuse", 0, True, "completed", 0),
        ("missing_argument", 2, False, "failed", None),
        ("unknown_option", 2, False, "failed", None),
    ],
)
def test_attempt_trace_correlation(
    tmp_path, monkeypatch, case, exit_code, identified, status, rejections
):
    config = load_inference_config()
    source = tmp_path / "input.csv"
    rows = [_values("private-customer")]
    if case == "partial":
        rows.append(_values("private-rejected", credit_limit="0"))
    elif case == "rejected":
        rows = [_values("private-rejected", credit_limit="0")]
    content = _csv(config, rows)
    if case == "missing_columns":
        content = b"account_id,wrong_column\nprivate-customer,0\n"
    elif case == "malformed_csv":
        content = b'account_id,"unterminated\n'
    if case != "unreadable":
        source.write_bytes(content)
    engine = _Engine()
    monkeypatch.setattr(cli, "InferenceEngine", lambda **kwargs: engine)
    events = []

    def emit(event, **fields):
        events.append({"event": event, **fields})

    monkeypatch.setattr(cli, "emit_event", emit)
    monkeypatch.setattr(batch, "emit_event", emit)

    def fail(*args, **kwargs):
        raise InferenceError("synthetic controlled failure")

    if case == "model_failure":
        monkeypatch.setattr(cli, "InferenceEngine", fail)
    elif case == "score_failure":
        monkeypatch.setattr(engine, "score", fail)
    elif case == "publication_failure":

        def fail_publication(*args, **kwargs):
            raise OSError("synthetic publication failure")

        monkeypatch.setattr(batch, "_publish_run", fail_publication)

    reads = []
    original_read = Path.read_bytes

    def read(path):
        if path == source:
            reads.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    args = [
        "batch",
        "--input",
        str(source),
        "--as-of-date",
        "2026-09-30",
        "--snapshot-id",
        ".." if case == "invalid_identity" else "trace-fixture",
        "--output-root",
        str(tmp_path / "batches"),
    ]
    if case == "config_failure":
        args += ["--config", str(tmp_path / "missing-config.yaml")]
    elif case == "missing_argument":
        args = ["batch"]
    elif case == "unknown_option":
        args += ["--unknown"]
    runner = CliRunner()
    result = runner.invoke(cli.inference_app, args)
    assert result.exit_code == exit_code, result.output
    terminal = [x for x in events if x["event"] == "batch_attempt_completed"]
    assert len(terminal) == 1
    attempt = terminal[0]
    assert attempt["status"] == status
    assert re.fullmatch("[0-9a-f]{32}", attempt["trace_id"])
    assert f"trace_id={attempt['trace_id']}" in result.output
    assert attempt.get("rejection_count") == rejections
    if identified:
        identity = {
            "input_sha256": hashlib.sha256(content).hexdigest(),
            "as_of_date": "2026-09-30",
            "snapshot_id": "trace-fixture",
            "config_sha256": PHASE6_CONFIG_SHA256,
            "manifest_sha256": config.bundle.manifest_sha256,
            "model_sha256": config.bundle.model_sha256,
        }
        expected_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert attempt["batch_id"] == expected_id
        assert expected_id in result.output
        assert len(reads) == 1
    else:
        assert "batch_id" not in attempt
    assert all(set(event) <= ALLOWED_LOG_FIELDS for event in events)
    assert "private-customer" not in json.dumps(events)
    assert "private-rejected" not in json.dumps(events)
    summary = summarize_events([json.dumps(event) for event in events])
    assert summary["batch_attempt_count"] == 1
    assert summary["batch_failure_count"] == int(status == "failed")
    if case == "reuse":
        folder = tmp_path / "batches/2026-09-30/trace-fixture"
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
        events.clear()
        second = runner.invoke(cli.inference_app, args)
        assert second.exit_code == 0
        assert "reused=true" in second.output
        assert len(events) == 1
        assert events[0]["batch_id"] == attempt["batch_id"]
        assert events[0]["trace_id"] != attempt["trace_id"]
        assert f"trace_id={events[0]['trace_id']}" in second.output
        assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}


def test_trace_is_assigned_before_configuration_access(monkeypatch):
    assigned = []
    original_uuid = cli.uuid4

    def assign():
        value = original_uuid()
        assigned.append(value.hex)
        return value

    def load(_path):
        assert len(assigned) == 1
        raise InferenceError("synthetic configuration failure")

    monkeypatch.setattr(cli, "uuid4", assign)
    monkeypatch.setattr(cli, "load_inference_config", load)
    events = []
    monkeypatch.setattr(cli, "emit_event", lambda event, **fields: events.append(fields))
    result = CliRunner().invoke(
        cli.inference_app,
        [
            "batch",
            "--input",
            "unused.csv",
            "--as-of-date",
            "2026-09-30",
            "--snapshot-id",
            "trace-fixture",
        ],
    )
    assert result.exit_code == 1
    assert events[0]["trace_id"] == assigned[0]
