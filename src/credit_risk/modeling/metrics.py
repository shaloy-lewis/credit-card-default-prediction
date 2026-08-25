"""Deterministic baseline metrics for ranking and probability evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

DEFAULT_CAPACITIES: Final[tuple[float, ...]] = (0.05, 0.10, 0.20)


class MetricValidationError(ValueError):
    """Raised when prediction metrics cannot be computed safely."""


@dataclass(frozen=True, slots=True)
class DiscriminationMetrics:
    """Threshold-independent ranking metrics."""

    average_precision: float
    roc_auc: float
    ks: float
    gini: float


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    """Proper scoring rules that are valid only for probabilities."""

    brier_score: float
    log_loss: float


@dataclass(frozen=True, slots=True)
class CapacityMetrics:
    """Expected outcome at a capacity with fractional allocation across cutoff ties."""

    capacity: float
    selected_count: int
    cutoff_score: float
    cutoff_tie_count: int
    cutoff_fraction: float
    expected_true_positives: float
    precision: float
    recall: float
    lift: float


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Metrics for one model on one complete out-of-fold repeat."""

    discrimination: DiscriminationMetrics
    probability: ProbabilityMetrics | None
    capacities: tuple[CapacityMetrics, ...]


@dataclass(frozen=True, slots=True)
class RepeatSummary:
    """Descriptive variation across complete repeated-CV evaluations."""

    n_repeats: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


def discrimination_metrics(
    target: Sequence[int] | np.ndarray | pd.Series,
    ranking_scores: Sequence[float] | np.ndarray | pd.Series,
) -> DiscriminationMetrics:
    """Compute non-interpolated average precision, ROC-AUC, KS, and Gini."""

    labels, scores = _validated_target_and_scores(target, ranking_scores)
    average_precision = float(average_precision_score(labels, scores))
    roc_auc = float(roc_auc_score(labels, scores))
    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels,
        scores,
        drop_intermediate=False,
    )
    ks = float(np.max(true_positive_rate - false_positive_rate))
    values = np.asarray((average_precision, roc_auc, ks), dtype=np.float64)
    if not np.isfinite(values).all():
        raise MetricValidationError("Discrimination metrics must be finite.")
    return DiscriminationMetrics(
        average_precision=average_precision,
        roc_auc=roc_auc,
        ks=ks,
        gini=float(2.0 * roc_auc - 1.0),
    )


def probability_metrics(
    target: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
) -> ProbabilityMetrics:
    """Compute Brier score and log loss after enforcing probability bounds."""

    labels, values = _validated_target_and_scores(target, probabilities)
    if np.any((values < 0.0) | (values > 1.0)):
        raise MetricValidationError("Probabilities must be within the closed interval [0, 1].")
    brier = float(brier_score_loss(labels, values))
    cross_entropy = float(log_loss(labels, values, labels=[0, 1]))
    if not np.isfinite((brier, cross_entropy)).all():
        raise MetricValidationError("Probability metrics must be finite.")
    return ProbabilityMetrics(brier_score=brier, log_loss=cross_entropy)


def capacity_metrics(
    target: Sequence[int] | np.ndarray | pd.Series,
    ranking_scores: Sequence[float] | np.ndarray | pd.Series,
    *,
    capacities: Sequence[float] = DEFAULT_CAPACITIES,
) -> tuple[CapacityMetrics, ...]:
    """Evaluate expected precision, recall, and lift at fixed review capacities.

    The selected count is ``ceil(n * capacity)``.  When the cutoff score is
    tied, the remaining slots are allocated fractionally across every account
    at that score.  This reports the expected result without using account ID
    or row order to break a model-score tie.
    """

    labels, scores = _validated_target_and_scores(target, ranking_scores)
    resolved_capacities = _validate_capacities(capacities)
    n_rows = labels.size
    positives = float(np.sum(labels))
    prevalence = positives / n_rows
    results: list[CapacityMetrics] = []

    descending = np.sort(scores)[::-1]
    for capacity in resolved_capacities:
        selected_count = int(math.ceil(n_rows * capacity))
        cutoff = float(descending[selected_count - 1])
        above = scores > cutoff
        tied = scores == cutoff
        above_count = int(np.sum(above))
        tie_count = int(np.sum(tied))
        remaining = selected_count - above_count
        if tie_count < 1 or remaining < 1 or remaining > tie_count:
            raise MetricValidationError("Capacity tie allocation is internally inconsistent.")
        cutoff_fraction = remaining / tie_count
        expected_true_positives = float(
            np.sum(labels[above]) + cutoff_fraction * np.sum(labels[tied])
        )
        precision = expected_true_positives / selected_count
        recall = expected_true_positives / positives
        lift = precision / prevalence
        values = np.asarray(
            (cutoff, cutoff_fraction, expected_true_positives, precision, recall, lift),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise MetricValidationError("Capacity metrics must be finite.")
        results.append(
            CapacityMetrics(
                capacity=capacity,
                selected_count=selected_count,
                cutoff_score=cutoff,
                cutoff_tie_count=tie_count,
                cutoff_fraction=float(cutoff_fraction),
                expected_true_positives=expected_true_positives,
                precision=float(precision),
                recall=float(recall),
                lift=float(lift),
            )
        )
    return tuple(results)


def evaluate_predictions(
    target: Sequence[int] | np.ndarray | pd.Series,
    ranking_scores: Sequence[float] | np.ndarray | pd.Series,
    *,
    probabilities: Sequence[float] | np.ndarray | pd.Series | None = None,
    capacities: Sequence[float] = DEFAULT_CAPACITIES,
) -> PredictionMetrics:
    """Evaluate rankings, adding proper scoring rules only for explicit probabilities."""

    return PredictionMetrics(
        discrimination=discrimination_metrics(target, ranking_scores),
        probability=(
            probability_metrics(target, probabilities) if probabilities is not None else None
        ),
        capacities=capacity_metrics(target, ranking_scores, capacities=capacities),
    )


def summarize_repeat_values(
    values: Sequence[float] | np.ndarray | pd.Series,
) -> RepeatSummary:
    """Return descriptive mean/std/min/max across complete repeat-level values.

    The standard deviation is the population statistic (``ddof=0``).  No
    confidence interval is inferred from the three deterministic repeats.
    """

    resolved = _as_finite_vector("Repeat values", values)
    return RepeatSummary(
        n_repeats=int(resolved.size),
        mean=float(np.mean(resolved)),
        standard_deviation=float(np.std(resolved, ddof=0)),
        minimum=float(np.min(resolved)),
        maximum=float(np.max(resolved)),
    )


def _validated_target_and_scores(
    target: Sequence[int] | np.ndarray | pd.Series,
    scores: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    labels = _as_finite_vector("Target", target)
    if not np.isin(labels, (0.0, 1.0)).all():
        raise MetricValidationError("Target must contain only binary labels 0 and 1.")
    if not np.array_equal(np.unique(labels), np.asarray([0.0, 1.0])):
        raise MetricValidationError("Target must contain both binary classes 0 and 1.")
    resolved_scores = _as_finite_vector("Scores", scores)
    if resolved_scores.size != labels.size:
        raise MetricValidationError(
            f"Score row count {resolved_scores.size} does not match target row count {labels.size}."
        )
    return labels.astype(np.int8, copy=False), resolved_scores


def _as_finite_vector(
    name: str,
    values: Sequence[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    try:
        resolved = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MetricValidationError(
            f"{name} must be a numeric one-dimensional sequence."
        ) from error
    if resolved.ndim != 1 or resolved.size == 0:
        raise MetricValidationError(f"{name} must be a non-empty one-dimensional sequence.")
    if not np.isfinite(resolved).all():
        raise MetricValidationError(f"{name} must contain only finite values.")
    return resolved


def _validate_capacities(capacities: Sequence[float]) -> tuple[float, ...]:
    if isinstance(capacities, (str, bytes)):
        raise MetricValidationError("Capacities must be a numeric sequence.")
    try:
        values = tuple(float(capacity) for capacity in capacities)
    except (TypeError, ValueError) as error:
        raise MetricValidationError("Capacities must be a numeric sequence.") from error
    if not values:
        raise MetricValidationError("At least one capacity is required.")
    if any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in values):
        raise MetricValidationError("Each capacity must be finite and within (0, 1].")
    if len(set(values)) != len(values):
        raise MetricValidationError("Capacities must be unique.")
    return values
