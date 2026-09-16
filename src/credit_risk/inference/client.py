"""Small standard-library client for the versioned local demonstration API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from credit_risk.inference.contracts import CreditRiskResponse


class InferenceClientError(RuntimeError):
    """Raised when the versioned API cannot return a valid prediction."""


def predict_v1(
    payload: Mapping[str, int],
    *,
    base_url: str = "http://127.0.0.1:8080",
    request_id: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Call `/v1/predict` without adding an HTTP-client dependency."""

    headers = {"Content-Type": "application/json"}
    if request_id is not None:
        headers["X-Request-ID"] = request_id
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/predict",
        data=json.dumps(dict(payload), sort_keys=True).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
    except (OSError, urllib.error.HTTPError, urllib.error.URLError) as error:
        raise InferenceClientError(f"Versioned inference request failed: {error}") from error
    try:
        json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceClientError("Versioned inference response was not valid JSON.") from error
    try:
        result = CreditRiskResponse.model_validate_json(content)
    except ValidationError as error:
        raise InferenceClientError(
            "Versioned inference response violated its strict response contract."
        ) from error
    return result.model_dump(mode="json")
