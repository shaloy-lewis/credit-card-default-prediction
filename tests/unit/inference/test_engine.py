"""Shared prediction engine tests with controlled estimator doubles."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from credit_risk.inference.contracts import load_inference_config
from credit_risk.inference.engine import InferenceEngine, InferenceError, _sigmoid
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS


class _Estimator:
    def __init__(self, native: np.ndarray, raw: np.ndarray) -> None:
        self.native = native
        self.raw = raw

    def get_feature_importance(self, _pool: object, *, type: str) -> np.ndarray:
        assert type == "ShapValues"
        return self.native

    def predict(self, _pool: object, *, prediction_type: str) -> np.ndarray:
        assert prediction_type == "RawFormulaVal"
        return self.raw


class _Model:
    def __init__(self, probabilities: np.ndarray, estimator: _Estimator) -> None:
        self.probabilities = probabilities
        self.estimator = estimator

    def predict_proba(self, _frame: pd.DataFrame) -> np.ndarray:
        return self.probabilities


def _engine(probabilities: np.ndarray, native: np.ndarray, raw: np.ndarray) -> InferenceEngine:
    engine = object.__new__(InferenceEngine)
    engine.config = load_inference_config()
    engine.model = _Model(probabilities, _Estimator(native, raw))  # type: ignore[assignment]
    engine.manifest = SimpleNamespace(selected_model_id="catboost_fixed")  # type: ignore[assignment]
    return engine


def _features(rows: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((rows, len(PREDICTOR_COLUMNS)), dtype=np.int64),
        columns=PREDICTOR_COLUMNS,
        index=[f"acct-{index}" for index in range(rows)],
    ).assign(credit_limit_ntd=100000)


def test_engine_scores_and_orders_reviewed_reasons() -> None:
    contributions = np.zeros((1, 19))
    contributions[0, 0] = -0.4
    contributions[0, 1:7] = 0.1
    raw = np.asarray([0.2])
    base = raw - contributions.sum(axis=1)
    native = np.column_stack((contributions, base))
    probability = _sigmoid(raw)

    result = _engine(probability, native, raw).score(_features())

    assert result.probabilities == pytest.approx(probability)
    assert result.reasons[0][0].category == "repayment_status"
    assert result.reasons[0][0].direction == "risk_increasing"
    assert result.reasons[0][1].category == "credit_capacity"
    assert result.reasons[0][1].direction == "risk_mitigating"
    assert result.max_additivity_error == pytest.approx(0.0)
    assert result.max_probability_error == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("probabilities", "native", "raw", "message"),
    (
        (np.asarray([[0.5]]), np.zeros((1, 20)), np.zeros(1), "probability dimensions"),
        (np.asarray([1.1]), np.zeros((1, 20)), np.zeros(1), "outside"),
        (np.asarray([0.5]), np.zeros((1, 19)), np.zeros(1), "output shape"),
        (np.asarray([0.5]), np.full((1, 20), np.nan), np.zeros(1), "must be finite"),
        (
            np.asarray([0.5]),
            np.column_stack((np.ones((1, 19)), np.zeros(1))),
            np.zeros(1),
            "additivity",
        ),
        (np.asarray([0.8]), np.zeros((1, 20)), np.zeros(1), "probability parity"),
    ),
)
def test_engine_rejects_invalid_outputs(
    probabilities: np.ndarray,
    native: np.ndarray,
    raw: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(InferenceError, match=message):
        _engine(probabilities, native, raw).score(_features())


def test_engine_rejects_invalid_frames_and_prediction_failures() -> None:
    valid = _engine(np.asarray([0.5]), np.zeros((1, 20)), np.zeros(1))
    with pytest.raises(InferenceError, match="non-empty"):
        valid.score(pd.DataFrame())
    with pytest.raises(InferenceError, match="ordered operational"):
        valid.score(_features().iloc[:, ::-1])

    valid.model = SimpleNamespace(  # type: ignore[assignment]
        predict_proba=lambda _frame: (_ for _ in ()).throw(RuntimeError("secret")),
        estimator=valid.model.estimator,
    )
    with pytest.raises(InferenceError, match="prediction failed"):
        valid.score(_features())


def test_sigmoid_handles_positive_and_negative_values() -> None:
    assert _sigmoid(np.asarray([-2.0, 0.0, 2.0])) == pytest.approx([0.119202922, 0.5, 0.880797078])
