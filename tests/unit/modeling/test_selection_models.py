from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import credit_risk.modeling.selection_models as models
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS, REPAYMENT_STATUS_COLUMNS
from credit_risk.modeling.selection_contracts import MODEL_ORDER, load_selection_config
from credit_risk.modeling.selection_models import (
    SelectionModelError,
    fit_one_pass_models,
    prepare_features,
    validate_selection_features,
    validate_selection_target,
)


class _FakeEstimator:
    calls: list[str] = []

    def __init__(self, model_id: str, **_parameters: Any) -> None:
        self.model_id = model_id
        self.classes_ = np.asarray([0, 1])
        self.tree_count_ = 300

    def fit(self, features: pd.DataFrame, target: pd.Series, **_kwargs: Any) -> _FakeEstimator:
        assert len(features) == len(target)
        self.calls.append(self.model_id)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.column_stack((np.full(len(features), 0.6), np.full(len(features), 0.4)))


def test_fixed_workflow_performs_exactly_one_fit_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, target = _data()
    _FakeEstimator.calls = []

    def fake_logistic(features: pd.DataFrame, labels: pd.Series) -> SimpleNamespace:
        estimator = _FakeEstimator("logistic_l2")
        estimator.fit(features, labels)
        return SimpleNamespace(pipeline=estimator)

    monkeypatch.setattr(models, "fit_logistic_baseline", fake_logistic)
    monkeypatch.setattr(
        models,
        "RandomForestClassifier",
        lambda **kwargs: _FakeEstimator("random_forest", **kwargs),
    )
    monkeypatch.setattr(
        models,
        "HistGradientBoostingClassifier",
        lambda **kwargs: _FakeEstimator("hist_gradient_boosting", **kwargs),
    )
    monkeypatch.setattr(
        models,
        "CatBoostClassifier",
        lambda **kwargs: _FakeEstimator("catboost_fixed", **kwargs),
    )

    fitted = fit_one_pass_models(frame, target, load_selection_config())

    assert tuple(item.model_id for item in fitted) == MODEL_ORDER
    assert _FakeEstimator.calls == list(MODEL_ORDER)
    assert all(model.predict_proba(frame).tolist() == [0.4] * len(frame) for model in fitted)


def test_model_specific_transformations_preserve_numeric_values() -> None:
    frame, _ = _data()

    raw = prepare_features("random_forest", frame)
    catboost = prepare_features("catboost_fixed", frame)

    assert np.array_equal(raw.to_numpy(), frame.to_numpy())
    assert all(catboost[column].dtype == object for column in REPAYMENT_STATUS_COLUMNS)
    monetary = [column for column in PREDICTOR_COLUMNS if column not in REPAYMENT_STATUS_COLUMNS]
    assert np.array_equal(catboost[monetary].to_numpy(), frame[monetary].to_numpy())


def test_selection_rejects_extra_features_and_nonfinite_values() -> None:
    frame, _ = _data()
    frame["age_years"] = 40
    with pytest.raises(SelectionModelError, match="exactly the 19 operational"):
        validate_selection_features(frame)

    frame, _ = _data()
    frame.iloc[0, 0] = np.nan
    with pytest.raises(SelectionModelError, match="finite"):
        validate_selection_features(frame)


def test_selection_rejects_empty_duplicate_and_invalid_status_features() -> None:
    frame, _ = _data()
    with pytest.raises(SelectionModelError, match="non-empty"):
        validate_selection_features(frame.iloc[:0])

    duplicate = frame.copy()
    duplicate.index = pd.Index([1] * len(duplicate))
    with pytest.raises(SelectionModelError, match="indexes must be unique"):
        validate_selection_features(duplicate)

    frame.loc[frame.index[0], REPAYMENT_STATUS_COLUMNS[0]] = 10
    with pytest.raises(SelectionModelError, match="range -2..9"):
        validate_selection_features(frame)


def test_selection_rejects_invalid_targets_and_probability_contracts() -> None:
    frame, target = _data()
    with pytest.raises(SelectionModelError, match="align exactly"):
        validate_selection_target(target.reset_index(drop=True), expected_index=frame.index)
    fractional = target.astype(float)
    fractional.iloc[0] = 0.5
    with pytest.raises(SelectionModelError, match="integral labels"):
        validate_selection_target(fractional, expected_index=frame.index)
    with pytest.raises(SelectionModelError, match="both binary classes"):
        validate_selection_target(target * 0, expected_index=frame.index)
    with pytest.raises(SelectionModelError, match="Unsupported model transformation"):
        prepare_features("unknown", frame)

    class Broken:
        classes_ = np.asarray([0, 1])

        def predict_proba(self, _frame: pd.DataFrame) -> np.ndarray:
            raise ValueError("broken")

    with pytest.raises(SelectionModelError, match="scoring failed"):
        models.FittedSelectionModel("logistic_l2", Broken(), "handling").predict_proba(frame)

    broken = Broken()
    broken.predict_proba = lambda _frame: np.ones((len(frame), 1))  # type: ignore[method-assign]
    with pytest.raises(SelectionModelError, match="probability shape"):
        models.FittedSelectionModel("logistic_l2", broken, "handling").predict_proba(frame)

    broken.predict_proba = lambda _frame: np.full((len(frame), 2), np.nan)  # type: ignore[method-assign]
    with pytest.raises(SelectionModelError, match="invalid probabilities"):
        models.FittedSelectionModel("logistic_l2", broken, "handling").predict_proba(frame)

    broken.classes_ = np.asarray([1, 0])
    broken.predict_proba = lambda _frame: np.full((len(frame), 2), 0.5)  # type: ignore[method-assign]
    with pytest.raises(SelectionModelError, match="class order changed"):
        models.FittedSelectionModel("logistic_l2", broken, "handling").predict_proba(frame)


def _data() -> tuple[pd.DataFrame, pd.Series]:
    rows = 20
    values: dict[str, np.ndarray] = {}
    for column in PREDICTOR_COLUMNS:
        if column in REPAYMENT_STATUS_COLUMNS:
            values[column] = np.tile(np.asarray([-2, 0, 1, 2]), 5)
        else:
            values[column] = np.arange(rows) + 100
    frame = pd.DataFrame(values, index=pd.Index(range(1, rows + 1), name="account_id"))
    target = pd.Series(np.tile([0, 1], 10), index=frame.index, name="default_next_month")
    return frame, target
