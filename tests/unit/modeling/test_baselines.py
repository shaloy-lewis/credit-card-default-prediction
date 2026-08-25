"""Unit tests for the governed statistical baselines."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from credit_risk.modeling.baselines import (
    BILL_AMOUNT_COLUMNS,
    LOGISTIC_C,
    LOGISTIC_MAX_ITER,
    LOGISTIC_RANDOM_STATE,
    LOGISTIC_TOL,
    MONETARY_COLUMNS,
    PAYMENT_AMOUNT_COLUMNS,
    PREDICTOR_COLUMNS,
    REPAYMENT_RULE_WEIGHTS,
    REPAYMENT_STATUS_CATEGORIES,
    REPAYMENT_STATUS_COLUMNS,
    BaselineValidationError,
    LogisticBaseline,
    fit_fold_baselines,
    fit_logistic_baseline,
    prevalence_scores,
    repayment_rule_scores,
)
from credit_risk.modeling.contracts import load_baseline_config


def _predictors(rows: int = 240, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    payload: dict[str, np.ndarray] = {
        "credit_limit_ntd": rng.integers(10_000, 1_000_001, size=rows),
    }
    for column in REPAYMENT_STATUS_COLUMNS:
        payload[column] = rng.integers(-2, 10, size=rows)
    for column in BILL_AMOUNT_COLUMNS:
        payload[column] = rng.integers(-50_000, 800_001, size=rows)
    for column in PAYMENT_AMOUNT_COLUMNS:
        payload[column] = rng.integers(0, 300_001, size=rows)
    return pd.DataFrame(payload, columns=PREDICTOR_COLUMNS)


def _target(features: pd.DataFrame, *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = (
        features["repayment_status_lag_0"].to_numpy()
        + 0.5 * features["repayment_status_lag_1"].to_numpy()
        + rng.normal(0.0, 3.0, len(features))
    )
    return (signal > np.median(signal)).astype(np.int8)


def test_predictor_contract_contains_only_the_19_operational_features() -> None:
    assert PREDICTOR_COLUMNS == (
        "credit_limit_ntd",
        *REPAYMENT_STATUS_COLUMNS,
        *BILL_AMOUNT_COLUMNS,
        *PAYMENT_AMOUNT_COLUMNS,
    )
    assert len(PREDICTOR_COLUMNS) == 19
    assert len(MONETARY_COLUMNS) == 13
    assert REPAYMENT_STATUS_CATEGORIES == tuple(range(-2, 10))
    assert REPAYMENT_RULE_WEIGHTS == (6, 5, 4, 3, 2, 1)


def test_prevalence_scores_use_only_the_training_fold_prior() -> None:
    scores = prevalence_scores(np.asarray([0, 0, 1, 1, 1]), n_rows=3)

    np.testing.assert_array_equal(scores, np.asarray([0.6, 0.6, 0.6]))


@pytest.mark.parametrize("n_rows", [0, -1, True, 1.5])
def test_prevalence_rejects_invalid_output_sizes(n_rows: object) -> None:
    with pytest.raises(BaselineValidationError, match="positive integer"):
        prevalence_scores([0, 1], n_rows)  # type: ignore[arg-type]


def test_repayment_rule_applies_recency_weights_and_clips_nonpositive_statuses() -> None:
    features = _predictors(rows=2)
    features.loc[0, REPAYMENT_STATUS_COLUMNS] = [-2, -1, 0, 1, 2, 3]
    features.loc[1, REPAYMENT_STATUS_COLUMNS] = [1, 1, 1, 1, 1, 1]

    scores = repayment_rule_scores(features)

    np.testing.assert_array_equal(scores, np.asarray([10.0, 21.0]))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda frame: frame.drop(columns=["credit_limit_ntd"]), "missing"),
        (lambda frame: frame.assign(age_years=30), "unexpected"),
        (
            lambda frame: frame.assign(repayment_status_lag_0=10),
            "range -2..9",
        ),
        (
            lambda frame: frame.assign(payment_amount_ntd_lag_0=np.nan),
            "null or non-finite",
        ),
        (
            lambda frame: frame.assign(credit_limit_ntd="not numeric"),
            "must be numeric",
        ),
    ],
)
def test_baselines_reject_invalid_predictor_views(mutate, message: str) -> None:
    features = mutate(_predictors(rows=8))

    with pytest.raises(BaselineValidationError, match=message):
        repayment_rule_scores(features)


def test_logistic_baseline_has_the_fixed_fold_local_pipeline() -> None:
    features = _predictors()
    target = _target(features)

    fitted = fit_logistic_baseline(features, target)
    classifier = fitted.pipeline.named_steps["classifier"]
    preprocessor = fitted.pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["repayment_status"]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.penalty == "l2"
    assert classifier.C == LOGISTIC_C
    assert classifier.solver == "lbfgs"
    assert classifier.class_weight is None
    assert classifier.fit_intercept is True
    assert classifier.max_iter == LOGISTIC_MAX_ITER
    assert classifier.tol == LOGISTIC_TOL
    assert classifier.random_state == LOGISTIC_RANDOM_STATE
    assert len(fitted.transformed_feature_names) == 79
    assert classifier.coef_.shape == (1, 79)
    for categories in encoder.categories_:
        np.testing.assert_array_equal(categories, REPAYMENT_STATUS_CATEGORIES)
    np.testing.assert_array_equal(encoder.drop_idx_, np.zeros(6, dtype=object))


def test_fold_baselines_execute_the_reviewed_machine_configuration() -> None:
    features = _predictors()
    target = _target(features)
    config = load_baseline_config().baselines

    fitted = fit_fold_baselines(features, target, config=config)
    classifier = fitted.logistic.pipeline.named_steps["classifier"]

    assert classifier.C == config.logistic.c
    assert classifier.max_iter == config.logistic.max_iter
    assert classifier.tol == config.logistic.tolerance
    assert classifier.random_state == config.logistic.random_state
    assert fitted.repayment_rule_weights == config.repayment_rule.recency_weights


def test_fold_baselines_are_deterministic_and_reuse_training_state() -> None:
    train_features = _predictors(rows=240, seed=1)
    validation_features = _predictors(rows=31, seed=2)
    train_target = _target(train_features)

    first = fit_fold_baselines(train_features, train_target)
    second = fit_fold_baselines(train_features, train_target)
    first_predictions = first.predict(validation_features)
    second_predictions = second.predict(validation_features.loc[:, reversed(PREDICTOR_COLUMNS)])

    assert first.prevalence == pytest.approx(float(np.mean(train_target)))
    np.testing.assert_array_equal(
        first_predictions.prevalence,
        np.full(len(validation_features), first.prevalence),
    )
    np.testing.assert_array_equal(
        first_predictions.repayment_rule,
        repayment_rule_scores(validation_features),
    )
    np.testing.assert_allclose(
        first_predictions.logistic_l2,
        second_predictions.logistic_l2,
        rtol=0.0,
        atol=0.0,
    )
    assert np.isfinite(first_predictions.logistic_l2).all()
    assert np.all((first_predictions.logistic_l2 >= 0.0) & (first_predictions.logistic_l2 <= 1.0))


@pytest.mark.parametrize(
    "target, message",
    [
        (np.zeros(8, dtype=int), "both binary classes"),
        (np.asarray([0, 1, 0, 1, 0, 1, 0]), "row count"),
        (np.asarray([0, 1, 0, 1, 0, 1, 0, np.nan]), "finite binary"),
        (np.asarray([0, 1, 0, 1, 0, 1, 0, 2]), "finite binary"),
    ],
)
def test_logistic_rejects_invalid_training_targets(target: np.ndarray, message: str) -> None:
    with pytest.raises(BaselineValidationError, match=message):
        fit_logistic_baseline(_predictors(rows=8), target)


def test_logistic_fails_fast_on_nonconvergence(monkeypatch) -> None:
    def warn_on_fit(self, features, target, sample_weight=None):  # noqa: ARG001
        warnings.warn("iteration limit", ConvergenceWarning, stacklevel=2)
        return self

    monkeypatch.setattr(LogisticRegression, "fit", warn_on_fit)
    features = _predictors(rows=20)

    with pytest.raises(BaselineValidationError, match="did not converge"):
        fit_logistic_baseline(features, _target(features))


@pytest.mark.parametrize(
    "features, message",
    [
        (np.zeros((2, 19)), "pandas DataFrame"),
        (pd.DataFrame(columns=PREDICTOR_COLUMNS), "at least one row"),
    ],
)
def test_predictor_view_rejects_nonframes_and_empty_frames(features, message: str) -> None:
    with pytest.raises(BaselineValidationError, match=message):
        repayment_rule_scores(features)  # type: ignore[arg-type]


def test_predictor_view_rejects_duplicate_columns() -> None:
    features = _predictors(rows=3)
    features.columns = (*PREDICTOR_COLUMNS[:-1], PREDICTOR_COLUMNS[-2])

    with pytest.raises(BaselineValidationError, match="must be unique"):
        repayment_rule_scores(features)


@pytest.mark.parametrize(
    "target, message",
    [
        (["invalid"], "numeric sequence"),
        ([[0, 1]], "one-dimensional"),
        ([], "one-dimensional"),
    ],
)
def test_prevalence_rejects_malformed_targets(target, message: str) -> None:
    with pytest.raises(BaselineValidationError, match=message):
        prevalence_scores(target, n_rows=1)  # type: ignore[arg-type]


def test_logistic_wraps_sklearn_fit_errors(monkeypatch) -> None:
    def fail_fit(self, features, target, **kwargs):  # noqa: ARG001
        raise ValueError("solver rejected input")

    monkeypatch.setattr("sklearn.pipeline.Pipeline.fit", fail_fit)
    features = _predictors(rows=20)

    with pytest.raises(BaselineValidationError, match="fitting failed: solver rejected input"):
        fit_logistic_baseline(features, _target(features))


class _ScoringPipeline:
    def __init__(self, result) -> None:
        self.result = result

    def predict_proba(self, features):  # noqa: ARG002
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize(
    "result, message",
    [
        (ValueError("bad transform"), "scoring failed"),
        (np.asarray([[0.5, 0.5, 0.0]]), "each binary class"),
        (np.asarray([[np.nan, np.nan]]), "invalid probabilities"),
        (np.asarray([[1.1, -0.1]]), "invalid probabilities"),
    ],
)
def test_logistic_scoring_rejects_pipeline_failures_or_invalid_outputs(
    result, message: str
) -> None:
    fitted = LogisticBaseline(
        pipeline=_ScoringPipeline(result),  # type: ignore[arg-type]
        transformed_feature_names=(),
    )

    with pytest.raises(BaselineValidationError, match=message):
        fitted.predict_proba(_predictors(rows=1))
