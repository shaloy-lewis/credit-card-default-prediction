"""Fixed estimators for the four-fit one-pass selection protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from credit_risk.modeling.baselines import fit_logistic_baseline
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS, REPAYMENT_STATUS_COLUMNS
from credit_risk.modeling.selection_contracts import MODEL_ORDER, SelectionConfig


class SelectionModelError(RuntimeError):
    """Raised when a fixed selection estimator violates its contract."""


@dataclass(frozen=True, slots=True)
class FittedSelectionModel:
    """One fitted estimator and the transformations required at scoring."""

    model_id: str
    estimator: Any
    feature_handling: str

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        frame = validate_selection_features(features)
        prepared = prepare_features(self.model_id, frame)
        try:
            probabilities = np.asarray(self.estimator.predict_proba(prepared), dtype=np.float64)
        except Exception as error:
            raise SelectionModelError(f"{self.model_id} scoring failed: {error}") from error
        _validate_estimator_output(self.model_id, self.estimator, probabilities, len(frame))
        return probabilities[:, 1]


def fit_one_pass_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: SelectionConfig,
) -> tuple[FittedSelectionModel, ...]:
    """Fit each frozen model exactly once and return them without refitting."""

    frame = validate_selection_features(X_train)
    target = validate_selection_target(y_train, expected_index=frame.index)
    fitted: list[FittedSelectionModel] = []
    for model_contract in config.models:
        model_id = model_contract.model_id
        parameters = dict(model_contract.parameters)
        prepared = prepare_features(model_id, frame)
        try:
            if model_id == "logistic_l2":
                logistic = fit_logistic_baseline(frame, target)
                estimator = logistic.pipeline
            elif model_id == "random_forest":
                estimator = RandomForestClassifier(**parameters)
                estimator.fit(prepared, target)
            elif model_id == "hist_gradient_boosting":
                estimator = HistGradientBoostingClassifier(**parameters)
                estimator.fit(prepared, target)
            elif model_id == "catboost_fixed":
                estimator = CatBoostClassifier(**parameters)
                estimator.fit(
                    prepared,
                    target,
                    cat_features=list(REPAYMENT_STATUS_COLUMNS),
                )
            else:  # guarded by the strict config, retained as defence in depth
                raise SelectionModelError(f"Unsupported frozen model {model_id!r}.")
        except SelectionModelError:
            raise
        except Exception as error:
            raise SelectionModelError(f"{model_id} fitting failed: {error}") from error

        classes = np.asarray(getattr(estimator, "classes_", ()))
        if not np.array_equal(classes, np.asarray([0, 1])):
            raise SelectionModelError(
                f"{model_id} must learn ordered binary classes [0, 1], observed {classes.tolist()}."
            )
        if model_id == "catboost_fixed" and int(estimator.tree_count_) != 300:
            raise SelectionModelError("catboost_fixed must contain exactly 300 trees.")
        fitted.append(
            FittedSelectionModel(
                model_id=model_id,
                estimator=estimator,
                feature_handling=model_contract.feature_handling,
            )
        )
    if tuple(model.model_id for model in fitted) != MODEL_ORDER:
        raise SelectionModelError("The authoritative workflow must complete exactly four fits.")
    return tuple(fitted)


def validate_selection_features(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise SelectionModelError("Selection predictors must be a non-empty DataFrame.")
    if tuple(features.columns) != PREDICTOR_COLUMNS:
        raise SelectionModelError(
            "Selection predictors must contain exactly the 19 operational features in order."
        )
    if not features.index.is_unique:
        raise SelectionModelError("Selection account indexes must be unique.")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if numeric.isna().any().any() or not np.isfinite(values).all():
        raise SelectionModelError("Selection predictors must be numeric, finite, and complete.")
    statuses = numeric.loc[:, REPAYMENT_STATUS_COLUMNS].to_numpy(dtype=np.float64)
    if (
        not np.equal(statuses, np.floor(statuses)).all()
        or not np.isin(statuses, range(-2, 10)).all()
    ):
        raise SelectionModelError("Repayment-status codes must be integers in the range -2..9.")
    return numeric.loc[:, PREDICTOR_COLUMNS].copy()


def validate_selection_target(target: pd.Series, *, expected_index: pd.Index) -> pd.Series:
    if not isinstance(target, pd.Series) or not target.index.equals(expected_index):
        raise SelectionModelError("Selection target must align exactly with predictor accounts.")
    numeric = pd.to_numeric(target, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if (
        numeric.isna().any()
        or not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise SelectionModelError("Selection target must contain finite integral labels.")
    labels = set(int(value) for value in values)
    if labels != {0, 1}:
        raise SelectionModelError("Selection target must contain both binary classes 0 and 1.")
    return numeric.astype("int8")


def prepare_features(model_id: str, features: pd.DataFrame) -> pd.DataFrame:
    prepared = features.copy()
    if model_id == "catboost_fixed":
        for column in REPAYMENT_STATUS_COLUMNS:
            prepared[column] = prepared[column].astype("int64").astype(str)
    elif model_id not in MODEL_ORDER:
        raise SelectionModelError(f"Unsupported model transformation for {model_id!r}.")
    return prepared


def _validate_estimator_output(
    model_id: str,
    estimator: Any,
    probabilities: np.ndarray,
    rows: int,
) -> None:
    classes = np.asarray(getattr(estimator, "classes_", ()))
    if not np.array_equal(classes, np.asarray([0, 1])):
        raise SelectionModelError(f"{model_id} class order changed after fitting.")
    if probabilities.shape != (rows, 2):
        raise SelectionModelError(
            f"{model_id} probability shape must be {(rows, 2)}, got {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise SelectionModelError(f"{model_id} produced invalid probabilities.")
