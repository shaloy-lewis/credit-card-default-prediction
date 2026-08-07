"""API lifecycle, readiness, and inference contract tests."""

import json

import pytest
from fastapi.testclient import TestClient

from api import create_app
from credit_risk.artifacts import ArtifactValidationError

pytestmark = pytest.mark.integration


def test_liveness_and_readiness_endpoints() -> None:
    with TestClient(create_app()) as client:
        liveness = client.get("/ping")
        readiness = client.get("/ready")

    assert liveness.status_code == 200
    assert liveness.json() == {"message": "Health check successful!"}
    assert readiness.status_code == 200
    assert readiness.json() == {"status": "ready"}


def test_predict_endpoint_preserves_legacy_contract(
    readme_prediction_payload: dict[str, int | float | str],
) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/predict", json=readme_prediction_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["probability_of_default"] == pytest.approx(0.44088, abs=1e-5)
    assert isinstance(body["instance_feature_importance"], dict)
    assert isinstance(body["global_feature_importance"], dict)


def test_invalid_artifacts_fail_application_startup(tmp_path) -> None:
    (tmp_path / "model.pkl").write_text("not a pickle", encoding="utf-8")
    (tmp_path / "preprocessor.pkl").write_text("not a pickle", encoding="utf-8")
    thresholds = {
        "low_perc": {},
        "high_perc": {},
    }
    (tmp_path / "outlier_threshold.json").write_text(json.dumps(thresholds), encoding="utf-8")

    with pytest.raises(ArtifactValidationError):
        with TestClient(create_app(tmp_path)):
            pass


def test_readiness_and_prediction_return_503_if_pipeline_state_is_lost(
    readme_prediction_payload: dict[str, int | float | str],
) -> None:
    with TestClient(create_app()) as client:
        client.app.state.pipeline = None

        liveness = client.get("/ping")
        readiness = client.get("/ready")
        prediction = client.post("/predict", json=readme_prediction_payload)

    assert liveness.status_code == 200
    assert readiness.status_code == 503
    assert prediction.status_code == 503
