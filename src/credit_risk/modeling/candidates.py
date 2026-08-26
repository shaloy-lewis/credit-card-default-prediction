"""Narrow deterministic CatBoost fitting boundary for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from credit_risk.modeling.candidate_contracts import (
    FixedCatBoostParameters,
    SampledParameters,
)


class CandidateModelError(RuntimeError):
    """Raised when a candidate fold cannot be fitted or scored safely."""


@dataclass(frozen=True, slots=True)
class CandidateFoldDiagnostics:
    """Deterministic non-executable evidence for one CatBoost fit."""

    train_rows: int
    validation_rows: int
    train_class_counts: tuple[int, int]
    validation_class_counts: tuple[int, int]
    predictor_count: int
    categorical_columns: tuple[str, ...]
    tree_count: int


@dataclass(frozen=True, slots=True)
class CandidateFoldResult:
    """Positive-class probabilities and deterministic fit diagnostics."""

    probabilities: np.ndarray
    diagnostics: CandidateFoldDiagnostics


def fit_candidate_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    *,
    predictor_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    fixed_parameters: FixedCatBoostParameters,
    sampled_parameters: SampledParameters,
) -> CandidateFoldResult:
    """Fit and score one reviewed fold without writing or returning an estimator."""

    train, train_labels = _validated_frame_and_target(
        X_train,
        y_train,
        predictor_columns=predictor_columns,
        categorical_columns=categorical_columns,
        description="training",
        require_both_classes=True,
    )
    validation, validation_labels = _validated_frame_and_target(
        X_validation,
        y_validation,
        predictor_columns=predictor_columns,
        categorical_columns=categorical_columns,
        description="validation",
        require_both_classes=True,
    )
    if not train.index.is_unique or not validation.index.is_unique:
        raise CandidateModelError("Candidate train and validation indexes must be unique.")
    if set(train.index) & set(validation.index):
        raise CandidateModelError("Candidate train and validation account indexes overlap.")

    try:
        model = _new_classifier(fixed_parameters, sampled_parameters)
        model.fit(train, train_labels, cat_features=list(categorical_columns))
        classes = np.asarray(model.classes_)
        tree_count = int(model.tree_count_)
        probabilities = np.asarray(model.predict_proba(validation), dtype=np.float64)
    except CandidateModelError:
        raise
    except Exception as error:
        raise CandidateModelError(f"CatBoost fold fitting failed: {error}") from error

    if not np.array_equal(classes, np.asarray([0, 1])):
        raise CandidateModelError(f"CatBoost classes must be [0, 1], observed {classes.tolist()}.")
    if tree_count != sampled_parameters.iterations:
        raise CandidateModelError(
            "CatBoost tree count differs from the frozen iteration count: "
            f"expected={sampled_parameters.iterations}, observed={tree_count}."
        )
    if probabilities.shape != (len(validation), 2):
        raise CandidateModelError(
            "CatBoost probability output has an unexpected shape: "
            f"expected={(len(validation), 2)}, observed={probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise CandidateModelError("CatBoost probabilities must be finite.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise CandidateModelError("CatBoost probabilities must be within [0, 1].")
    positive_probabilities = probabilities[:, 1]

    return CandidateFoldResult(
        probabilities=positive_probabilities,
        diagnostics=CandidateFoldDiagnostics(
            train_rows=len(train),
            validation_rows=len(validation),
            train_class_counts=_class_counts(train_labels),
            validation_class_counts=_class_counts(validation_labels),
            predictor_count=len(predictor_columns),
            categorical_columns=categorical_columns,
            tree_count=tree_count,
        ),
    )


def _new_classifier(
    fixed: FixedCatBoostParameters,
    sampled: SampledParameters,
) -> CatBoostClassifier:
    parameters = {
        **fixed.model_dump(exclude={"early_stopping_rounds"}),
        **sampled.model_dump(),
    }
    return CatBoostClassifier(**parameters)


def _validated_frame_and_target(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    predictor_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    description: str,
    require_both_classes: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    if tuple(frame.columns) != predictor_columns:
        raise CandidateModelError(
            f"Candidate {description} columns differ from the reviewed feature view."
        )
    if len(frame) != len(target) or not frame.index.equals(target.index):
        raise CandidateModelError(f"Candidate {description} features and target are not aligned.")
    if not set(categorical_columns).issubset(predictor_columns):
        raise CandidateModelError(
            f"Candidate {description} categorical columns are not a subset of predictors."
        )
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise CandidateModelError(
            f"Candidate {description} predictors must be numeric and finite before conversion."
        )
    prepared = numeric.copy()
    for column in categorical_columns:
        values = prepared[column].to_numpy(dtype=np.float64)
        if not np.equal(values, np.floor(values)).all():
            raise CandidateModelError(
                f"Candidate categorical column {column} must contain validated integer codes."
            )
        prepared[column] = prepared[column].astype("int64").astype(str)

    labels = pd.to_numeric(target, errors="coerce")
    label_values = labels.to_numpy(dtype=np.float64)
    if (
        labels.isna().any()
        or not np.isfinite(label_values).all()
        or not np.equal(label_values, np.floor(label_values)).all()
    ):
        raise CandidateModelError(f"Candidate {description} target must contain integral labels.")
    unique = set(int(value) for value in label_values)
    if not unique.issubset({0, 1}) or (require_both_classes and unique != {0, 1}):
        requirement = "both classes 0 and 1" if require_both_classes else "only labels 0 and 1"
        raise CandidateModelError(f"Candidate {description} target must contain {requirement}.")
    resolved_labels = labels.astype("int8")
    return prepared, resolved_labels


def _class_counts(target: pd.Series) -> tuple[int, int]:
    counts = target.value_counts()
    return int(counts.get(0, 0)), int(counts.get(1, 0))
