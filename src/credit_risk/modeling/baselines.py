"""Leakage-safe statistical baselines for fold-local model evaluation.

The functions in this module deliberately accept only the governed predictor
view.  Identifier, target, and demographic columns are therefore rejected as
extra inputs rather than silently discarded.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.modeling.contracts import (
    BILL_AMOUNT_COLUMNS,
    PAYMENT_AMOUNT_COLUMNS,
    PREDICTOR_COLUMNS,
    REPAYMENT_RULE_WEIGHTS,
    REPAYMENT_STATUS_COLUMNS,
    BaselineModelsConfig,
    LogisticBaselineConfig,
)

MONETARY_COLUMNS: Final[tuple[str, ...]] = (
    "credit_limit_ntd",
    *BILL_AMOUNT_COLUMNS,
    *PAYMENT_AMOUNT_COLUMNS,
)
REPAYMENT_STATUS_CATEGORIES: Final[tuple[int, ...]] = tuple(range(-2, 10))

LOGISTIC_C: Final[float] = 1.0
LOGISTIC_MAX_ITER: Final[int] = 2_000
LOGISTIC_TOL: Final[float] = 1e-8
LOGISTIC_RANDOM_STATE: Final[int] = 42


class BaselineValidationError(ValueError):
    """Raised when a baseline cannot be trained or scored safely."""


@dataclass(frozen=True, slots=True)
class FoldBaselinePredictions:
    """Validation scores produced by the three approved baselines."""

    prevalence: np.ndarray
    repayment_rule: np.ndarray
    logistic_l2: np.ndarray


@dataclass(frozen=True, slots=True)
class LogisticBaseline:
    """A fitted fold-local preprocessing and L2-logistic pipeline."""

    pipeline: Pipeline
    transformed_feature_names: tuple[str, ...]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return finite positive-class probabilities for a predictor view."""

        frame = _validate_predictor_frame(features)
        try:
            probabilities = np.asarray(self.pipeline.predict_proba(frame), dtype=np.float64)
        except ValueError as error:
            raise BaselineValidationError(f"Logistic scoring failed: {error}") from error
        if probabilities.shape != (len(frame), 2):
            raise BaselineValidationError(
                "Logistic scoring must return one probability for each binary class."
            )
        positive = probabilities[:, 1]
        if not np.isfinite(positive).all() or np.any((positive < 0.0) | (positive > 1.0)):
            raise BaselineValidationError("Logistic scoring produced invalid probabilities.")
        return positive


@dataclass(frozen=True, slots=True)
class FittedFoldBaselines:
    """Fold-training state shared by validation scoring."""

    prevalence: float
    logistic: LogisticBaseline
    repayment_rule_weights: tuple[int, ...] = REPAYMENT_RULE_WEIGHTS

    def predict(self, features: pd.DataFrame) -> FoldBaselinePredictions:
        """Score one validation fold without refitting any training state."""

        frame = _validate_predictor_frame(features)
        return FoldBaselinePredictions(
            prevalence=np.full(len(frame), self.prevalence, dtype=np.float64),
            repayment_rule=repayment_rule_scores(
                frame,
                recency_weights=self.repayment_rule_weights,
            ),
            logistic_l2=self.logistic.predict_proba(frame),
        )


def prevalence_scores(
    train_target: Sequence[int] | np.ndarray | pd.Series,
    n_rows: int,
) -> np.ndarray:
    """Score rows with the positive-class prevalence observed in a training fold."""

    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        raise BaselineValidationError("n_rows must be a positive integer.")
    target = _validate_binary_target(train_target, require_both_classes=False)
    prevalence = float(np.mean(target, dtype=np.float64))
    if not np.isfinite(prevalence):
        raise BaselineValidationError("Training-fold prevalence is not finite.")
    return np.full(n_rows, prevalence, dtype=np.float64)


def repayment_rule_scores(
    features: pd.DataFrame,
    *,
    recency_weights: Sequence[int] = REPAYMENT_RULE_WEIGHTS,
) -> np.ndarray:
    """Apply the fixed recency-weighted positive-delinquency rule.

    Lag zero receives weight six and lag five receives weight one.  Current or
    early-payment statuses (zero or negative) contribute no risk points.
    """

    frame = _validate_predictor_frame(features)
    statuses = frame.loc[:, REPAYMENT_STATUS_COLUMNS].to_numpy(dtype=np.float64)
    weights = np.asarray(tuple(recency_weights), dtype=np.float64)
    if weights.shape != (len(REPAYMENT_STATUS_COLUMNS),) or not np.isfinite(weights).all():
        raise BaselineValidationError("Repayment-rule weights must be six finite values.")
    if not np.equal(weights, np.floor(weights)).all():
        raise BaselineValidationError("Repayment-rule weights must be integral.")
    if tuple(int(weight) for weight in weights) != REPAYMENT_RULE_WEIGHTS:
        raise BaselineValidationError("Repayment-rule weights must be exactly 6, 5, 4, 3, 2, 1.")
    scores = np.maximum(statuses, 0.0) @ weights
    if not np.isfinite(scores).all():
        raise BaselineValidationError("Repayment-rule scoring produced non-finite scores.")
    return np.asarray(scores, dtype=np.float64)


def fit_logistic_baseline(
    train_features: pd.DataFrame,
    train_target: Sequence[int] | np.ndarray | pd.Series,
    *,
    config: LogisticBaselineConfig | None = None,
) -> LogisticBaseline:
    """Fit the fixed fold-local preprocessing and L2-logistic baseline."""

    frame = _validate_predictor_frame(train_features)
    target = _validate_binary_target(
        train_target,
        expected_rows=len(frame),
        require_both_classes=True,
    )
    pipeline = _build_logistic_pipeline(config)
    max_iter = config.max_iter if config is not None else LOGISTIC_MAX_ITER
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(frame, target)
    except ConvergenceWarning as error:
        raise BaselineValidationError(
            f"Logistic baseline did not converge within {max_iter} iterations."
        ) from error
    except ValueError as error:
        raise BaselineValidationError(f"Logistic baseline fitting failed: {error}") from error

    classifier = pipeline.named_steps["classifier"]
    if not np.array_equal(classifier.classes_, np.asarray([0, 1])):
        raise BaselineValidationError("Logistic baseline must learn binary classes [0, 1].")
    if np.any(np.asarray(classifier.n_iter_) >= max_iter):
        raise BaselineValidationError(
            f"Logistic baseline did not converge within {max_iter} iterations."
        )
    if not np.isfinite(classifier.coef_).all() or not np.isfinite(classifier.intercept_).all():
        raise BaselineValidationError("Logistic baseline fitted non-finite coefficients.")

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = tuple(str(name) for name in preprocessor.get_feature_names_out())
    if len(feature_names) != classifier.coef_.shape[1]:
        raise BaselineValidationError(
            "Logistic transformed feature names do not match the fitted coefficient dimension."
        )
    return LogisticBaseline(pipeline=pipeline, transformed_feature_names=feature_names)


def fit_fold_baselines(
    train_features: pd.DataFrame,
    train_target: Sequence[int] | np.ndarray | pd.Series,
    *,
    config: BaselineModelsConfig | None = None,
) -> FittedFoldBaselines:
    """Fit all stateful baselines using one training fold only."""

    target = _validate_binary_target(
        train_target,
        expected_rows=len(train_features),
        require_both_classes=True,
    )
    logistic_config = config.logistic if config is not None else None
    repayment_weights = (
        config.repayment_rule.recency_weights if config is not None else REPAYMENT_RULE_WEIGHTS
    )
    logistic = fit_logistic_baseline(train_features, target, config=logistic_config)
    prevalence = float(np.mean(target, dtype=np.float64))
    return FittedFoldBaselines(
        prevalence=prevalence,
        logistic=logistic,
        repayment_rule_weights=tuple(repayment_weights),
    )


def _build_logistic_pipeline(config: LogisticBaselineConfig | None = None) -> Pipeline:
    status_columns = config.status_columns if config is not None else REPAYMENT_STATUS_COLUMNS
    status_categories = (
        config.status_categories if config is not None else REPAYMENT_STATUS_CATEGORIES
    )
    monetary_columns = config.monetary_columns if config is not None else MONETARY_COLUMNS
    status_encoder = OneHotEncoder(
        categories=[list(status_categories) for _ in status_columns],
        drop=config.status_drop if config is not None else "first",
        handle_unknown=config.handle_unknown if config is not None else "error",
        sparse_output=True,
        dtype=np.float64,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("repayment_status", status_encoder, list(status_columns)),
            ("monetary", StandardScaler(), list(monetary_columns)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )
    classifier = LogisticRegression(
        penalty=config.penalty if config is not None else "l2",
        C=config.c if config is not None else LOGISTIC_C,
        solver=config.solver if config is not None else "lbfgs",
        class_weight=config.class_weight if config is not None else None,
        fit_intercept=config.fit_intercept if config is not None else True,
        max_iter=config.max_iter if config is not None else LOGISTIC_MAX_ITER,
        tol=config.tolerance if config is not None else LOGISTIC_TOL,
        random_state=config.random_state if config is not None else LOGISTIC_RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _validate_predictor_frame(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise BaselineValidationError("Predictors must be provided as a pandas DataFrame.")
    if features.empty:
        raise BaselineValidationError("Predictor data must contain at least one row.")
    if features.columns.has_duplicates:
        raise BaselineValidationError("Predictor columns must be unique.")

    actual = set(features.columns)
    expected = set(PREDICTOR_COLUMNS)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise BaselineValidationError(
            "Predictor view must contain exactly the 19 governed operational columns: "
            + ", ".join(details)
        )

    frame = features.loc[:, PREDICTOR_COLUMNS].copy()
    invalid_types = [
        column
        for column in PREDICTOR_COLUMNS
        if is_bool_dtype(frame[column].dtype) or not is_numeric_dtype(frame[column].dtype)
    ]
    if invalid_types:
        raise BaselineValidationError(f"Predictors must be numeric: {invalid_types}")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise BaselineValidationError("Predictors must not contain null or non-finite values.")

    statuses = frame.loc[:, REPAYMENT_STATUS_COLUMNS].to_numpy(dtype=np.float64)
    valid_statuses = np.isin(statuses, REPAYMENT_STATUS_CATEGORIES)
    if not valid_statuses.all():
        raise BaselineValidationError("Repayment statuses must be integers in the range -2..9.")
    return frame


def _validate_binary_target(
    target: Sequence[int] | np.ndarray | pd.Series,
    *,
    expected_rows: int | None = None,
    require_both_classes: bool,
) -> np.ndarray:
    try:
        values = np.asarray(target, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise BaselineValidationError(
            "Target must be a one-dimensional numeric sequence."
        ) from error
    if values.ndim != 1 or values.size == 0:
        raise BaselineValidationError("Target must be a non-empty one-dimensional sequence.")
    if expected_rows is not None and values.size != expected_rows:
        raise BaselineValidationError(
            f"Target row count {values.size} does not match predictor row count {expected_rows}."
        )
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise BaselineValidationError("Target must contain only finite binary labels 0 and 1.")
    classes = np.unique(values)
    if require_both_classes and not np.array_equal(classes, np.asarray([0.0, 1.0])):
        raise BaselineValidationError("Training target must contain both binary classes 0 and 1.")
    return values.astype(np.int8, copy=False)
