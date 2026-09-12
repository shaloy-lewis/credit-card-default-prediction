"""Deterministic native-SHAP explanations for the reviewed CatBoost bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedShuffleSplit

from credit_risk.governance.contracts import ExplanationContract
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS, REPAYMENT_STATUS_COLUMNS
from credit_risk.modeling.selection_models import FittedSelectionModel, prepare_features


class ExplanationError(ValueError):
    """Raised when native model attribution violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class ExplanationResult:
    sampled_account_ids: np.ndarray
    sampled_target: np.ndarray
    sampled_probabilities: np.ndarray
    sampled_risk_bands: np.ndarray
    base_values: np.ndarray
    raw_scores: np.ndarray
    shap_values: np.ndarray
    feature_summary: tuple[dict[str, Any], ...]
    category_summary: tuple[dict[str, Any], ...]
    max_additivity_error: float
    max_probability_error: float


def explain_validation_sample(
    model: FittedSelectionModel,
    predictors: pd.DataFrame,
    target: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    risk_bands: np.ndarray,
    contract: ExplanationContract,
) -> ExplanationResult:
    """Explain exactly one deterministic target-by-risk-band validation sample."""

    if model.model_id != "catboost_fixed" or not isinstance(model.estimator, CatBoostClassifier):
        raise ExplanationError("Phase 5 explanations require the reviewed CatBoost bundle.")
    if tuple(predictors.columns) != PREDICTOR_COLUMNS or not predictors.index.is_unique:
        raise ExplanationError("Explanation predictors violate the operational feature contract.")
    labels = np.asarray(target, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    bands = np.asarray(risk_bands, dtype=str)
    if (
        labels.shape != (len(predictors),)
        or scores.shape != labels.shape
        or bands.shape != labels.shape
    ):
        raise ExplanationError("Explanation inputs must align with the validation population.")
    if set(np.unique(labels)) != {0, 1}:
        raise ExplanationError("Explanation labels must contain both binary classes.")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ExplanationError("Explanation probabilities must be finite and bounded.")

    strata = np.char.add(np.char.add(labels.astype(str), "|"), bands)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=contract.sample_rows,
        random_state=contract.sampling_seed,
    )
    try:
        _, positions = next(splitter.split(np.zeros(len(labels)), strata))
    except ValueError as error:
        raise ExplanationError(
            f"Unable to construct the frozen explanation sample: {error}"
        ) from error
    positions = positions[np.argsort(predictors.index.to_numpy()[positions], kind="mergesort")]
    sample = predictors.iloc[positions].copy()
    prepared = prepare_features(model.model_id, sample)
    pool = Pool(prepared, cat_features=list(REPAYMENT_STATUS_COLUMNS))
    try:
        native = np.asarray(
            model.estimator.get_feature_importance(pool, type="ShapValues"), dtype=np.float64
        )
        raw_scores = np.asarray(
            model.estimator.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64
        ).reshape(-1)
    except Exception as error:
        raise ExplanationError(f"CatBoost native SHAP calculation failed: {error}") from error

    expected_shape = (contract.sample_rows, contract.shap_output_columns)
    if native.shape != expected_shape or raw_scores.shape != (contract.sample_rows,):
        raise ExplanationError(
            f"Native SHAP output shape must be {expected_shape}, observed {native.shape}."
        )
    if not np.isfinite(native).all() or not np.isfinite(raw_scores).all():
        raise ExplanationError("Native SHAP values and raw scores must be finite.")
    contributions = native[:, :-1]
    base_values = native[:, -1]
    reconstructed = contributions.sum(axis=1) + base_values
    additivity_error = float(np.max(np.abs(reconstructed - raw_scores)))
    if additivity_error > contract.additivity_absolute_tolerance:
        raise ExplanationError(
            f"Native SHAP raw-score additivity failed: maximum_error={additivity_error:.17g}"
        )
    reconstructed_probability = _sigmoid(raw_scores)
    probability_error = float(np.max(np.abs(reconstructed_probability - scores[positions])))
    if probability_error > contract.probability_absolute_tolerance:
        raise ExplanationError(
            f"Native SHAP sigmoid/probability parity failed: maximum_error={probability_error:.17g}"
        )

    feature_summary = tuple(
        _summary_item(feature, contributions[:, index], contract)
        for index, feature in enumerate(PREDICTOR_COLUMNS)
    )
    feature_positions = {feature: index for index, feature in enumerate(PREDICTOR_COLUMNS)}
    category_summary = tuple(
        _summary_item(
            category,
            contributions[:, [feature_positions[name] for name in features]].sum(axis=1),
            contract,
        )
        for category, features in contract.reason_categories.items()
    )
    return ExplanationResult(
        sampled_account_ids=predictors.index.to_numpy(dtype=np.int64)[positions],
        sampled_target=labels[positions],
        sampled_probabilities=scores[positions],
        sampled_risk_bands=bands[positions],
        base_values=base_values,
        raw_scores=raw_scores,
        shap_values=contributions,
        feature_summary=feature_summary,
        category_summary=category_summary,
        max_additivity_error=additivity_error,
        max_probability_error=probability_error,
    )


def _summary_item(name: str, values: np.ndarray, contract: ExplanationContract) -> dict[str, Any]:
    mean_signed = float(np.mean(values))
    direction = (
        contract.direction_labels["positive"]
        if mean_signed > 0.0
        else contract.direction_labels["negative"]
        if mean_signed < 0.0
        else "neutral"
    )
    return {
        "name": name,
        "mean_absolute_contribution": float(np.mean(np.abs(values))),
        "mean_signed_contribution": mean_signed,
        "mean_direction": direction,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result
