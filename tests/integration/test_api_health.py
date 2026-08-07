"""Minimal API contract tests that do not execute inference."""

import pytest
from fastapi.testclient import TestClient

from api import app

pytestmark = pytest.mark.integration


def test_health_endpoint() -> None:
    response = TestClient(app).get("/ping")

    assert response.status_code == 200
    assert response.json() == {"message": "Health check successful!"}
