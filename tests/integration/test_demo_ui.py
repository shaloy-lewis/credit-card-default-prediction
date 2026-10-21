"""Exercise the frozen Streamlit UI with the Pillow security compatibility override."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import credit_risk.inference.client as inference_client
from api import create_app
from credit_risk.inference.contracts import CreditRiskResponse

pytestmark = pytest.mark.integration
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_streamlit_renders_and_resizes_pillow_and_numpy_images(image_format: str) -> None:
    app = AppTest.from_string(
        f"""
import numpy as np
import streamlit as st
from PIL import Image

st.image(Image.new("RGBA", (200, 120), (30, 60, 90, 128)), width=80,
         output_format={image_format!r})
st.image(np.zeros((10, 12, 3), dtype=np.uint8), output_format={image_format!r})
""",
        default_timeout=30,
    ).run()

    assert not app.exception
    assert not app.error
    images = app.get("image")
    assert len(images) == 2
    suffix = ".jpg" if image_format == "JPEG" else ".png"
    assert all(image.proto.imgs[0].url.endswith(suffix) for image in images)


@pytest.mark.artifact
def test_predictor_renders_reviewed_synthetic_api_response(
    monkeypatch: pytest.MonkeyPatch,
    readme_prediction_payload: dict[str, int | float | str],
) -> None:
    with TestClient(create_app()) as api_client:

        def predict_via_api(payload: dict[str, int], *, base_url: str) -> dict[str, Any]:
            response = api_client.post("/v1/predict", json=payload)
            response.raise_for_status()
            return CreditRiskResponse.model_validate_json(response.content).model_dump(mode="json")

        monkeypatch.setattr(inference_client, "predict_v1", predict_via_api)
        app = AppTest.from_file(str(REPOSITORY_ROOT / "app.py"), default_timeout=30).run()
        assert not app.exception
        app.sidebar.selectbox[0].select("Predictor").run()
        app.number_input[0].set_value(readme_prediction_payload["credit_limit_ntd"])
        for lag in range(6):
            app.selectbox(key=f"status_{lag}").select(
                readme_prediction_payload[f"repayment_status_lag_{lag}"]
            )
            app.number_input(key=f"bill_{lag}").set_value(
                readme_prediction_payload[f"bill_amount_ntd_lag_{lag}"]
            )
            app.number_input(key=f"payment_{lag}").set_value(
                readme_prediction_payload[f"payment_amount_ntd_lag_{lag}"]
            )
        app.button[0].click().run()

    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "0.1904"
    displayed = [item.value for item in app.markdown]
    assert any("standard" in text for text in displayed)
    assert sum("non-causal model attribution" in text for text in displayed) == 2
