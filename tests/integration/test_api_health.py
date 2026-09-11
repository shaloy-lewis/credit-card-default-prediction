"""API lifecycle, selected-bundle readiness, and inference contract tests."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import credit_risk.modeling.selected_bundle as selected_bundle
from api import create_app
from credit_risk.modeling.selected_bundle import SelectedBundleError

pytestmark = pytest.mark.integration


def test_liveness_and_readiness_endpoints() -> None:
    with TestClient(create_app()) as client:
        liveness = client.get("/ping")
        readiness = client.get("/ready")

    assert liveness.status_code == 200
    assert liveness.json() == {"message": "Health check successful!"}
    assert readiness.status_code == 200
    assert readiness.json() == {"status": "ready"}


def test_predict_endpoint_uses_reviewed_selected_bundle(
    readme_prediction_payload: dict[str, int | float | str],
) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/predict", json=readme_prediction_payload)

    assert response.status_code == 200
    assert response.json() == {
        "probability_of_default": pytest.approx(0.190382, abs=1e-6),
        "risk_band": "standard",
        "model_id": "catboost_fixed",
        "bundle_id": "selected_v1",
    }


def test_invalid_selected_bundle_fails_application_startup(tmp_path) -> None:
    with pytest.raises(SelectedBundleError):
        with TestClient(create_app(tmp_path)):
            pass


def test_semantically_valid_manifest_edit_fails_application_startup(tmp_path: Path) -> None:
    source = Path("models/selected_v1")
    shutil.copy2(source / "model.cbm", tmp_path / "model.cbm")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["risk_band_thresholds"]["q80"] += 0.001
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SelectedBundleError, match="manifest digest mismatch"):
        with TestClient(create_app(tmp_path)):
            pass


def test_dependency_version_mismatch_fails_application_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_version = selected_bundle.importlib.metadata.version

    def mismatched_version(distribution_name: str) -> str:
        if distribution_name == "catboost":
            return "0.0.0"
        return installed_version(distribution_name)

    monkeypatch.setattr(selected_bundle.importlib.metadata, "version", mismatched_version)

    with pytest.raises(SelectedBundleError, match="'catboost' version mismatch"):
        with TestClient(create_app()):
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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("credit_limit_ntd", 0),
        ("repayment_status_lag_0", -3),
        ("repayment_status_lag_1", 1.5),
        ("repayment_status_lag_5", 10),
        ("payment_amount_ntd_lag_2", -1),
        ("payment_amount_ntd_lag_4", "1500"),
        ("bill_amount_ntd_lag_3", None),
    ),
)
def test_predict_rejects_invalid_operational_values(
    readme_prediction_payload: dict[str, int | float | str],
    field: str,
    value: int | float | str | None,
) -> None:
    payload = {**readme_prediction_payload, field: value}
    with TestClient(create_app()) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_missing_operational_field(
    readme_prediction_payload: dict[str, int | float | str],
) -> None:
    payload = dict(readme_prediction_payload)
    del payload["bill_amount_ntd_lag_0"]
    with TestClient(create_app()) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_demographics_and_unknown_fields(
    readme_prediction_payload: dict[str, int | float | str],
) -> None:
    payload = {**readme_prediction_payload, "age_years": 29, "sex_code": 2}
    with TestClient(create_app()) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
