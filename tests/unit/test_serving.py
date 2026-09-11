"""Selected-bundle serving adapter tests."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import credit_risk.serving as serving
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.modeling.selected_bundle import SelectedBundleError


class _Model:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def predict_proba(self, _features: pd.DataFrame) -> np.ndarray:
        return self.probabilities


def test_selected_pipeline_returns_probability_and_frozen_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = SimpleNamespace(
        selected_model_id="catboost_fixed",
        risk_band_thresholds={"q80": 0.2, "q90": 0.5, "q95": 0.8},
    )
    monkeypatch.setattr(
        serving,
        "load_selected_bundle",
        lambda _root, trusted, expected_manifest_sha256, required_dependencies: (
            manifest,
            _Model(np.asarray([0.6])),
        ),
    )
    pipeline = serving.SelectedPredictPipeline("bundle")
    features = pd.DataFrame(np.zeros((1, len(PREDICTOR_COLUMNS))), columns=PREDICTOR_COLUMNS)

    assert pipeline.predict(features) == (0.6, "high")
    assert pipeline.bundle_manifest is manifest


def test_selected_pipeline_rejects_batch_online_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = SimpleNamespace(
        selected_model_id="catboost_fixed",
        risk_band_thresholds={"q80": 0.2, "q90": 0.5, "q95": 0.8},
    )
    monkeypatch.setattr(
        serving,
        "load_selected_bundle",
        lambda _root, trusted, expected_manifest_sha256, required_dependencies: (
            manifest,
            _Model(np.asarray([0.1, 0.2])),
        ),
    )
    pipeline = serving.SelectedPredictPipeline("bundle")
    features = pd.DataFrame(np.zeros((2, len(PREDICTOR_COLUMNS))), columns=PREDICTOR_COLUMNS)

    with pytest.raises(ValueError, match="exactly one account"):
        pipeline.predict(features)


def test_selected_pipeline_rejects_a_different_bundle_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = SimpleNamespace(
        selected_model_id="logistic_l2",
        risk_band_thresholds={"q80": 0.2, "q90": 0.5, "q95": 0.8},
    )
    monkeypatch.setattr(
        serving,
        "load_selected_bundle",
        lambda _root, trusted, expected_manifest_sha256, required_dependencies: (
            manifest,
            _Model(np.asarray([0.1])),
        ),
    )

    with pytest.raises(SelectedBundleError, match="requires the reviewed catboost_fixed"):
        serving.SelectedPredictPipeline("bundle")


def test_selected_pipeline_requires_only_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    manifest = SimpleNamespace(
        selected_model_id="catboost_fixed",
        risk_band_thresholds={"q80": 0.2, "q90": 0.5, "q95": 0.8},
    )

    def fake_load(_root: str, **kwargs: object) -> tuple[object, _Model]:
        captured.update(kwargs)
        return manifest, _Model(np.asarray([0.1]))

    monkeypatch.setattr(serving, "load_selected_bundle", fake_load)

    serving.SelectedPredictPipeline("bundle")

    assert captured["required_dependencies"] == serving.SERVING_DEPENDENCIES
    assert "mlflow" not in serving.SERVING_DEPENDENCIES
    assert "pandera" not in serving.SERVING_DEPENDENCIES
