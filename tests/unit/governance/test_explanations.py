from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import credit_risk.governance.explanations as explanation_module
from credit_risk.governance.contracts import load_governance_config
from credit_risk.governance.explanations import ExplanationError, explain_validation_sample
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS, REPAYMENT_STATUS_COLUMNS
from credit_risk.modeling.selection_models import FittedSelectionModel


class _FakeEstimator:
    def get_feature_importance(self, pool: pd.DataFrame, *, type: str) -> np.ndarray:
        assert type == "ShapValues"
        raw = pool["credit_limit_ntd"].to_numpy(dtype=float)
        output = np.zeros((len(pool), 20), dtype=float)
        output[:, 0] = raw
        return output

    def predict(self, pool: pd.DataFrame, *, prediction_type: str) -> np.ndarray:
        assert prediction_type == "RawFormulaVal"
        return pool["credit_limit_ntd"].to_numpy(dtype=float)


def _inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rows = 40
    raw = np.linspace(-2.0, 2.0, rows)
    payload = {column: np.zeros(rows) for column in PREDICTOR_COLUMNS}
    payload["credit_limit_ntd"] = raw
    for column in REPAYMENT_STATUS_COLUMNS:
        payload[column] = np.zeros(rows, dtype=np.int8)
    frame = pd.DataFrame(payload, index=pd.Index(range(1, rows + 1), name="account_id"))
    target = np.asarray([0, 1] * 20, dtype=np.int8)
    probabilities = 1.0 / (1.0 + np.exp(-raw))
    bands = np.asarray(["standard", "elevated", "high", "critical"] * 10)
    return frame, target, probabilities, bands


def test_native_shap_sampling_mapping_and_additivity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explanation_module, "CatBoostClassifier", _FakeEstimator)
    monkeypatch.setattr(explanation_module, "Pool", lambda frame, cat_features: frame)
    config = load_governance_config().explanation.model_copy(update={"sample_rows": 16})
    frame, target, probabilities, bands = _inputs()

    result = explain_validation_sample(
        FittedSelectionModel("catboost_fixed", _FakeEstimator(), "native"),
        frame,
        target,
        probabilities,
        bands,
        config,
    )

    assert len(result.sampled_account_ids) == 16
    assert np.all(np.diff(result.sampled_account_ids) > 0)
    assert result.shap_values.shape == (16, 19)
    assert result.max_additivity_error == 0.0
    assert result.max_probability_error <= 1e-15
    assert {item["name"] for item in result.feature_summary} == set(PREDICTOR_COLUMNS)
    assert {item["name"] for item in result.category_summary} == {
        "credit_capacity",
        "repayment_status",
        "billing_balance",
        "payment_behaviour",
    }


def test_explanation_rejects_probability_or_model_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explanation_module, "CatBoostClassifier", _FakeEstimator)
    config = load_governance_config().explanation.model_copy(update={"sample_rows": 16})
    frame, target, probabilities, bands = _inputs()

    with pytest.raises(ExplanationError, match="reviewed CatBoost"):
        explain_validation_sample(
            FittedSelectionModel("logistic_l2", object(), "scaled"),
            frame,
            target,
            probabilities,
            bands,
            config,
        )

    bad_columns = frame.drop(columns=[PREDICTOR_COLUMNS[-1]])
    with pytest.raises(ExplanationError, match="feature contract"):
        explain_validation_sample(
            FittedSelectionModel("catboost_fixed", _FakeEstimator(), "native"),
            bad_columns,
            target,
            probabilities,
            bands,
            config,
        )
    with pytest.raises(ExplanationError, match="align"):
        explain_validation_sample(
            FittedSelectionModel("catboost_fixed", _FakeEstimator(), "native"),
            frame,
            target[:-1],
            probabilities,
            bands,
            config,
        )
    with pytest.raises(ExplanationError, match="both binary"):
        explain_validation_sample(
            FittedSelectionModel("catboost_fixed", _FakeEstimator(), "native"),
            frame,
            np.zeros(len(frame), dtype=np.int8),
            probabilities,
            bands,
            config,
        )
    with pytest.raises(ExplanationError, match="bounded"):
        explain_validation_sample(
            FittedSelectionModel("catboost_fixed", _FakeEstimator(), "native"),
            frame,
            target,
            np.full(len(frame), np.inf),
            bands,
            config,
        )


def test_explanation_rejects_non_additive_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken(_FakeEstimator):
        def predict(self, pool: pd.DataFrame, *, prediction_type: str) -> np.ndarray:
            return super().predict(pool, prediction_type=prediction_type) + 1.0

    monkeypatch.setattr(explanation_module, "CatBoostClassifier", Broken)
    monkeypatch.setattr(explanation_module, "Pool", lambda frame, cat_features: frame)
    frame, target, probabilities, bands = _inputs()
    config = load_governance_config().explanation.model_copy(update={"sample_rows": 16})

    with pytest.raises(ExplanationError, match="additivity"):
        explain_validation_sample(
            FittedSelectionModel("catboost_fixed", Broken(), "native"),
            frame,
            target,
            probabilities,
            bands,
            config,
        )


def test_explanation_rejects_sampling_native_shape_and_probability_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, target, probabilities, bands = _inputs()
    base = load_governance_config().explanation
    model = FittedSelectionModel("catboost_fixed", _FakeEstimator(), "native")
    monkeypatch.setattr(explanation_module, "CatBoostClassifier", _FakeEstimator)
    monkeypatch.setattr(explanation_module, "Pool", lambda prepared, cat_features: prepared)

    with pytest.raises(ExplanationError, match="construct the frozen"):
        explain_validation_sample(
            model,
            frame,
            target,
            probabilities,
            bands,
            base.model_copy(update={"sample_rows": len(frame) + 1}),
        )

    class Exploding(_FakeEstimator):
        def get_feature_importance(self, pool: pd.DataFrame, *, type: str) -> np.ndarray:
            raise RuntimeError("native failure")

    monkeypatch.setattr(explanation_module, "CatBoostClassifier", Exploding)
    with pytest.raises(ExplanationError, match="native SHAP calculation failed"):
        explain_validation_sample(
            FittedSelectionModel("catboost_fixed", Exploding(), "native"),
            frame,
            target,
            probabilities,
            bands,
            base.model_copy(update={"sample_rows": 16}),
        )

    class BadShape(_FakeEstimator):
        def get_feature_importance(self, pool: pd.DataFrame, *, type: str) -> np.ndarray:
            return np.zeros((len(pool), 19))

    monkeypatch.setattr(explanation_module, "CatBoostClassifier", BadShape)
    with pytest.raises(ExplanationError, match="output shape"):
        explain_validation_sample(
            FittedSelectionModel("catboost_fixed", BadShape(), "native"),
            frame,
            target,
            probabilities,
            bands,
            base.model_copy(update={"sample_rows": 16}),
        )

    class NonFinite(_FakeEstimator):
        def get_feature_importance(self, pool: pd.DataFrame, *, type: str) -> np.ndarray:
            values = super().get_feature_importance(pool, type=type)
            values[0, 0] = np.nan
            return values

    monkeypatch.setattr(explanation_module, "CatBoostClassifier", NonFinite)
    with pytest.raises(ExplanationError, match="must be finite"):
        explain_validation_sample(
            FittedSelectionModel("catboost_fixed", NonFinite(), "native"),
            frame,
            target,
            probabilities,
            bands,
            base.model_copy(update={"sample_rows": 16}),
        )

    monkeypatch.setattr(explanation_module, "CatBoostClassifier", _FakeEstimator)
    with pytest.raises(ExplanationError, match="probability parity"):
        explain_validation_sample(
            model,
            frame,
            target,
            np.full(len(frame), 0.5),
            bands,
            base.model_copy(update={"sample_rows": 16}),
        )


def test_direction_labels_cover_positive_negative_and_neutral() -> None:
    contract = load_governance_config().explanation

    assert (
        explanation_module._summary_item("p", np.asarray([1.0]), contract)["mean_direction"]
        == "risk_increasing"
    )
    assert (
        explanation_module._summary_item("n", np.asarray([-1.0]), contract)["mean_direction"]
        == "risk_mitigating"
    )
    assert (
        explanation_module._summary_item("z", np.asarray([0.0]), contract)["mean_direction"]
        == "neutral"
    )
