"""Tests for the narrow CatBoost fold-fitting boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from credit_risk.modeling import candidates
from credit_risk.modeling.candidate_contracts import load_candidate_config
from credit_risk.modeling.candidates import CandidateModelError, fit_candidate_fold


def _inputs():
    columns = ("repayment_status_lag_0", "credit_limit_ntd")
    train_index = pd.Index(range(1, 9), name="account_id")
    validation_index = pd.Index(range(101, 105), name="account_id")
    train = pd.DataFrame(
        {
            columns[0]: [-2, -1, 0, 1, 2, 3, 0, 1],
            columns[1]: [10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000],
        },
        index=train_index,
    )
    validation = pd.DataFrame(
        {columns[0]: [-2, 0, 1, 3], columns[1]: [15_000, 35_000, 55_000, 75_000]},
        index=validation_index,
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=train_index)
    y_validation = pd.Series([0, 1, 0, 1], index=validation_index)
    return columns, train, y_train, validation, y_validation


class _Classifier:
    def __init__(self, *, iterations: int) -> None:
        self.classes_ = np.asarray([0, 1])
        self.tree_count_ = iterations
        self.fitted = None
        self.cat_features = None
        self.output = np.asarray([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])

    def fit(self, frame, target, *, cat_features):
        self.fitted = frame.copy()
        self.cat_features = tuple(cat_features)
        assert set(target.unique()) == {0, 1}
        return self

    def predict_proba(self, frame):
        del frame
        return self.output


def test_candidate_fold_converts_only_categorical_values_and_returns_no_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns, train, y_train, validation, y_validation = _inputs()
    original_train = train.copy(deep=True)
    config = load_candidate_config()
    sampled = config.candidate.search.sampled_configurations[0].parameters
    classifier = _Classifier(iterations=sampled.iterations)
    monkeypatch.setattr(candidates, "_new_classifier", lambda *_args: classifier)

    result = fit_candidate_fold(
        train,
        y_train,
        validation,
        y_validation,
        predictor_columns=columns,
        categorical_columns=(columns[0],),
        fixed_parameters=config.candidate.fixed_parameters,
        sampled_parameters=sampled,
    )

    assert result.probabilities.tolist() == [0.2, 0.4, 0.6, 0.8]
    assert result.diagnostics.tree_count == sampled.iterations
    assert result.diagnostics.train_class_counts == (4, 4)
    assert classifier.cat_features == (columns[0],)
    assert classifier.fitted[columns[0]].tolist() == ["-2", "-1", "0", "1", "2", "3", "0", "1"]
    assert classifier.fitted[columns[1]].tolist() == train[columns[1]].tolist()
    pd.testing.assert_frame_equal(train, original_train)
    assert not hasattr(result, "model")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_column", "columns differ"),
        ("misaligned_target", "not aligned"),
        ("overlap", "overlap"),
        ("non_finite", "numeric and finite"),
        ("fractional_category", "integer codes"),
        ("duplicate_index", "unique"),
        ("unknown_category_column", "subset"),
        ("fractional_target", "integral labels"),
        ("non_finite_target", "integral labels"),
        ("non_binary_target", "both classes"),
        ("overflowing_target", "both classes"),
        ("one_class", "both classes"),
    ),
)
def test_candidate_fold_rejects_invalid_data(
    mutation: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns, train, y_train, validation, y_validation = _inputs()
    config = load_candidate_config()
    sampled = config.candidate.search.sampled_configurations[0].parameters
    monkeypatch.setattr(
        candidates,
        "_new_classifier",
        lambda *_args: _Classifier(iterations=sampled.iterations),
    )
    if mutation == "missing_column":
        validation = validation.drop(columns=columns[1])
    elif mutation == "misaligned_target":
        y_validation = y_validation.set_axis(range(201, 205))
    elif mutation == "overlap":
        validation = validation.set_axis(train.index[:4])
        y_validation = y_validation.set_axis(train.index[:4])
    elif mutation == "non_finite":
        train.loc[train.index[0], columns[1]] = np.nan
    elif mutation == "fractional_category":
        train[columns[0]] = train[columns[0]].astype(float)
        train.loc[train.index[0], columns[0]] = 0.5
    elif mutation == "duplicate_index":
        train.index = pd.Index([1, 1, 3, 4, 5, 6, 7, 8], name="account_id")
        y_train.index = train.index
    elif mutation == "unknown_category_column":
        categorical_columns = ("unknown",)
    elif mutation == "fractional_target":
        y_train = y_train.astype(float)
        y_train.iloc[0] = 0.5
    elif mutation == "non_finite_target":
        y_train = y_train.astype(float)
        y_train.iloc[0] = np.inf
    elif mutation == "non_binary_target":
        y_train.iloc[0] = 2
    elif mutation == "overflowing_target":
        y_train.iloc[0] = 256
    elif mutation == "one_class":
        y_train[:] = 0
    else:
        categorical_columns = (columns[0],)

    if mutation != "unknown_category_column":
        categorical_columns = (columns[0],)

    with pytest.raises(CandidateModelError, match=message):
        fit_candidate_fold(
            train,
            y_train,
            validation,
            y_validation,
            predictor_columns=columns,
            categorical_columns=categorical_columns,
            fixed_parameters=config.candidate.fixed_parameters,
            sampled_parameters=sampled,
        )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    (
        ("classes_", np.asarray([1, 0]), "classes"),
        ("tree_count_", 1, "tree count"),
        ("output", np.asarray([[0.5, 0.5]]), "shape"),
        (
            "output",
            np.asarray([[0.8, 0.2], [0.6, np.nan], [0.4, 0.6], [0.2, 0.8]]),
            "finite",
        ),
        (
            "output",
            np.asarray([[0.8, 0.2], [np.nan, 0.4], [0.4, 0.6], [0.2, 0.8]]),
            "finite",
        ),
        (
            "output",
            np.asarray([[0.8, 0.2], [0.6, 1.2], [0.4, 0.6], [0.2, 0.8]]),
            "within",
        ),
    ),
)
def test_candidate_fold_rejects_invalid_estimator_outputs(
    attribute: str,
    value,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns, train, y_train, validation, y_validation = _inputs()
    config = load_candidate_config()
    sampled = config.candidate.search.sampled_configurations[0].parameters
    classifier = _Classifier(iterations=sampled.iterations)
    setattr(classifier, attribute, value)
    monkeypatch.setattr(candidates, "_new_classifier", lambda *_args: classifier)

    with pytest.raises(CandidateModelError, match=message):
        fit_candidate_fold(
            train,
            y_train,
            validation,
            y_validation,
            predictor_columns=columns,
            categorical_columns=(columns[0],),
            fixed_parameters=config.candidate.fixed_parameters,
            sampled_parameters=sampled,
        )


def test_catboost_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    columns, train, y_train, validation, y_validation = _inputs()
    config = load_candidate_config()
    sampled = config.candidate.search.sampled_configurations[0].parameters
    monkeypatch.setattr(
        candidates,
        "_new_classifier",
        lambda *_args: SimpleNamespace(
            fit=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )

    with pytest.raises(CandidateModelError, match="fitting failed: boom"):
        fit_candidate_fold(
            train,
            y_train,
            validation,
            y_validation,
            predictor_columns=columns,
            categorical_columns=(columns[0],),
            fixed_parameters=config.candidate.fixed_parameters,
            sampled_parameters=sampled,
        )


def test_existing_candidate_errors_are_not_double_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns, train, y_train, validation, y_validation = _inputs()
    config = load_candidate_config()
    sampled = config.candidate.search.sampled_configurations[0].parameters
    monkeypatch.setattr(
        candidates,
        "_new_classifier",
        lambda *_args: (_ for _ in ()).throw(CandidateModelError("governed")),
    )

    with pytest.raises(CandidateModelError, match="^governed$"):
        fit_candidate_fold(
            train,
            y_train,
            validation,
            y_validation,
            predictor_columns=columns,
            categorical_columns=(columns[0],),
            fixed_parameters=config.candidate.fixed_parameters,
            sampled_parameters=sampled,
        )
