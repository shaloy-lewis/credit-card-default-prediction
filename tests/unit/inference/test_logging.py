"""Safe inference logging tests."""

import json

import pytest

import credit_risk.inference.logging as inference_logging


def test_emit_event_contains_only_allowlisted_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(inference_logging.LOGGER, "info", messages.append)
    inference_logging.emit_event("done", status="completed", trace_id="trace-1", row_count=2)

    payload = json.loads(messages[-1])
    assert payload == {
        "event": "done",
        "row_count": 2,
        "status": "completed",
        "trace_id": "trace-1",
    }


def test_emit_event_rejects_sensitive_fields() -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        inference_logging.emit_event("unsafe", account_id="acct-1")
