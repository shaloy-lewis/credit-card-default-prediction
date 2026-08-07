"""Compatibility checks around the committed legacy inference artifacts."""

from pathlib import Path

import pytest

from credit_risk.pipeline.prediction_pipeline import CustomData, PredictPipeline

pytestmark = pytest.mark.integration


def test_committed_artifacts_load_and_preserve_documented_prediction(
    readme_prediction_payload: dict[str, int | float | str],
) -> None:
    assert Path("artifacts/model.pkl").is_file()
    assert Path("artifacts/preprocessor.pkl").is_file()
    assert Path("artifacts/outlier_threshold.json").is_file()

    features = CustomData(**readme_prediction_payload).get_data_as_dataframe()
    probability = PredictPipeline().predict(features.copy())[0, 1]

    assert probability == pytest.approx(0.44088, abs=1e-5)


def test_custom_data_preserves_input_column_order(
    readme_prediction_payload: dict[str, int | float | str],
) -> None:
    features = CustomData(**readme_prediction_payload).get_data_as_dataframe()

    assert features.columns.tolist() == list(readme_prediction_payload)
