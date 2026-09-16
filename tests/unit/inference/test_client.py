"""Versioned API client tests."""

from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace

import pytest

import credit_risk.inference.client as client


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.content


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "trace_id": "trace-1",
        "probability_of_default": 0.2,
        "risk_band": "standard",
        "reasons": [],
        "model_id": "catboost_fixed",
        "bundle_id": "selected_v1",
        "manifest_sha256": "a" * 64,
        "policy_id": "outreach_top_10_v1",
    }


def test_client_calls_v1_with_optional_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = SimpleNamespace(request=None, timeout=None)

    def fake_open(request: object, timeout: float) -> _Response:
        captured.request = request
        captured.timeout = timeout
        return _Response(json.dumps(_payload()).encode())

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_open)

    result = client.predict_v1(
        {"credit_limit_ntd": 1},
        base_url="http://api/",
        request_id="request-1",
        timeout=2.0,
    )

    assert result == _payload()
    assert captured.request.full_url == "http://api/v1/predict"
    assert captured.request.headers["X-request-id"] == "request-1"
    assert captured.timeout == 2.0


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (b"not-json", "not valid JSON"),
        (json.dumps({"unexpected": True}).encode(), "field allowlist"),
    ),
)
def test_client_rejects_invalid_responses(
    monkeypatch: pytest.MonkeyPatch, content: bytes, message: str
) -> None:
    monkeypatch.setattr(
        client.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(content)
    )

    with pytest.raises(client.InferenceClientError, match=message):
        client.predict_v1({"credit_limit_ntd": 1})


def test_client_wraps_transport_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )

    with pytest.raises(client.InferenceClientError, match="request failed"):
        client.predict_v1({"credit_limit_ntd": 1})
