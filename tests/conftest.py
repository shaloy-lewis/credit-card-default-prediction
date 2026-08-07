"""Shared fixtures for unit and integration tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture()
def readme_prediction_payload() -> dict[str, int | float | str]:
    """Return the documented request used to guard legacy artifact compatibility."""
    payload_path = Path(__file__).parent / "fixtures" / "prediction_request.json"
    return json.loads(payload_path.read_text(encoding="utf-8"))
