"""Unit tests for discrimination, probability, capacity, and repeat metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from credit_risk.modeling.metrics import (
    DEFAULT_CAPACITIES,
    MetricValidationError,
    capacity_metrics,
    discrimination_metrics,
    evaluate_predictions,
    probability_metrics,
    summarize_repeat_values,
)


def test_discrimination_metrics_use_average_precision_and_standard_roc_derivatives() -> None:
    target = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.4, 0.35, 0.8])

    metrics = discrimination_metrics(target, scores)

    assert metrics.average_precision == pytest.approx(5.0 / 6.0)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.ks == pytest.approx(0.5)
    assert metrics.gini == pytest.approx(0.5)


def test_probability_metrics_are_computed_only_from_bounded_probabilities() -> None:
    target = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.35, 0.8])

    metrics = probability_metrics(target, probabilities)
    expected_log_loss = -np.mean(
        target * np.log(probabilities) + (1 - target) * np.log(1 - probabilities)
    )

    assert metrics.brier_score == pytest.approx(0.158125)
    assert metrics.log_loss == pytest.approx(expected_log_loss)


@pytest.mark.parametrize("probabilities", [[-0.1, 0.2], [0.2, 1.1]])
def test_probability_metrics_reject_scores_outside_probability_bounds(probabilities) -> None:
    with pytest.raises(MetricValidationError, match="closed interval"):
        probability_metrics([0, 1], probabilities)


def test_ranking_scores_are_not_misclassified_as_probabilities() -> None:
    evaluated = evaluate_predictions(
        [0, 0, 1, 1],
        [-10.0, 2.0, 8.0, 20.0],
        capacities=(0.5,),
    )

    assert evaluated.probability is None
    assert evaluated.discrimination.roc_auc == pytest.approx(1.0)


def test_explicit_probabilities_add_proper_scoring_rules() -> None:
    probabilities = [0.1, 0.2, 0.8, 0.9]

    evaluated = evaluate_predictions(
        [0, 0, 1, 1],
        probabilities,
        probabilities=probabilities,
        capacities=(0.5,),
    )

    assert evaluated.probability is not None
    assert evaluated.probability.brier_score == pytest.approx(0.025)


def test_capacity_metrics_fractionally_allocate_cutoff_ties() -> None:
    target = np.asarray([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.asarray([0.9, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    (metric,) = capacity_metrics(target, scores, capacities=(0.20,))

    assert metric.selected_count == 2
    assert metric.cutoff_score == pytest.approx(0.8)
    assert metric.cutoff_tie_count == 3
    assert metric.cutoff_fraction == pytest.approx(1.0 / 3.0)
    assert metric.expected_true_positives == pytest.approx(4.0 / 3.0)
    assert metric.precision == pytest.approx(2.0 / 3.0)
    assert metric.recall == pytest.approx(2.0 / 3.0)
    assert metric.lift == pytest.approx(10.0 / 3.0)


def test_all_equal_scores_have_prevalence_precision_and_unit_lift() -> None:
    target = np.asarray([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.ones(10)

    (metric,) = capacity_metrics(target, scores, capacities=(0.20,))

    assert metric.selected_count == 2
    assert metric.cutoff_tie_count == 10
    assert metric.cutoff_fraction == pytest.approx(0.2)
    assert metric.expected_true_positives == pytest.approx(0.4)
    assert metric.precision == pytest.approx(0.2)
    assert metric.recall == pytest.approx(0.2)
    assert metric.lift == pytest.approx(1.0)


def test_default_capacities_use_ceiling_selected_counts() -> None:
    target = np.asarray([0, 1, 0, 1, 0, 1, 0])
    scores = np.arange(7, dtype=float)

    metrics = capacity_metrics(target, scores)

    assert tuple(metric.capacity for metric in metrics) == DEFAULT_CAPACITIES
    assert tuple(metric.selected_count for metric in metrics) == (1, 1, 2)


@pytest.mark.parametrize(
    "capacities",
    [(), (0.0,), (-0.1,), (1.1,), (math.nan,), (0.1, 0.1), "invalid"],
)
def test_capacity_metrics_reject_invalid_capacities(capacities) -> None:
    with pytest.raises(MetricValidationError, match="Capacit|capacity"):
        capacity_metrics([0, 1], [0.1, 0.9], capacities=capacities)


@pytest.mark.parametrize(
    "target, scores, message",
    [
        ([0, 0], [0.1, 0.2], "both binary classes"),
        ([0, 2], [0.1, 0.2], "binary labels"),
        ([0, 1], [0.1], "row count"),
        ([0, 1], [0.1, np.inf], "finite"),
        ([0, np.nan], [0.1, 0.2], "finite"),
        ([[0, 1]], [0.1, 0.2], "one-dimensional"),
        ([0, 1], ["invalid", 0.2], "numeric one-dimensional"),
        ([0, 1], [], "one-dimensional"),
    ],
)
def test_metrics_reject_invalid_targets_and_scores(target, scores, message: str) -> None:
    with pytest.raises(MetricValidationError, match=message):
        discrimination_metrics(target, scores)


def test_repeat_summary_is_descriptive_population_variation_without_ci() -> None:
    summary = summarize_repeat_values([1.0, 2.0, 3.0])

    assert summary.n_repeats == 3
    assert summary.mean == pytest.approx(2.0)
    assert summary.standard_deviation == pytest.approx(math.sqrt(2.0 / 3.0))
    assert summary.minimum == pytest.approx(1.0)
    assert summary.maximum == pytest.approx(3.0)
    assert not hasattr(summary, "confidence_interval")


@pytest.mark.parametrize("values", [[], [1.0, np.nan], [[1.0, 2.0]]])
def test_repeat_summary_rejects_empty_nonfinite_or_nondimensional_values(values) -> None:
    with pytest.raises(MetricValidationError):
        summarize_repeat_values(values)


def test_capacity_metrics_reject_nonnumeric_capacity_values() -> None:
    with pytest.raises(MetricValidationError, match="numeric sequence"):
        capacity_metrics([0, 1], [0.1, 0.9], capacities=(object(),))
