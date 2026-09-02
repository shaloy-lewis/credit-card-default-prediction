"""Deterministic validation analysis for one-pass model selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from credit_risk.modeling.metrics import PredictionMetrics, evaluate_predictions
from credit_risk.modeling.selection_contracts import MODEL_ORDER, SIMPLICITY_ORDER, SelectionConfig


class SelectionAnalysisError(RuntimeError):
    """Raised when stored validation predictions cannot support selection."""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    model_id: str
    probabilities: np.ndarray
    metrics: PredictionMetrics


@dataclass(frozen=True, slots=True)
class ModelDecision:
    model_id: str
    eligible: bool
    brier_guardrail_passed: bool
    lift_guardrail_passed: bool
    within_equivalence_band: bool


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    selected_model_id: str
    results: tuple[ValidationResult, ...]
    decisions: tuple[ModelDecision, ...]
    best_eligible_average_precision: float


def select_validation_winner(
    results: tuple[ValidationResult, ...],
    config: SelectionConfig,
) -> SelectionDecision:
    """Apply inclusive guardrails, AP equivalence, and the frozen simplicity rule."""

    by_id = {result.model_id: result for result in results}
    if set(by_id) != set(SIMPLICITY_ORDER) or len(by_id) != len(results):
        raise SelectionAnalysisError("Selection requires one unique result for each frozen model.")
    logistic = by_id["logistic_l2"]
    logistic_brier = _brier(logistic.metrics)
    logistic_lift = _lift_at_ten(logistic.metrics)
    eligibility: dict[str, tuple[bool, bool]] = {}
    for model_id, result in by_id.items():
        brier_passed = _brier(result.metrics) <= (
            logistic_brier + config.selection.brier_guardrail_relative_to_logistic
        )
        lift_passed = _lift_at_ten(result.metrics) >= (
            logistic_lift - config.selection.lift_at_0_1_guardrail_relative_to_logistic
        )
        eligibility[model_id] = (brier_passed, lift_passed)
    eligible = [result for result in by_id.values() if all(eligibility[result.model_id])]
    if not eligible:  # logistic is its own reference, but keep an explicit governed fallback
        eligible = [logistic]
    best_ap = max(result.metrics.discrimination.average_precision for result in eligible)
    equivalent = {
        result.model_id
        for result in eligible
        if result.metrics.discrimination.average_precision
        >= best_ap - config.selection.average_precision_equivalence_band
    }
    selected = next(model_id for model_id in SIMPLICITY_ORDER if model_id in equivalent)
    decisions = tuple(
        ModelDecision(
            model_id=model_id,
            eligible=all(eligibility[model_id]),
            brier_guardrail_passed=eligibility[model_id][0],
            lift_guardrail_passed=eligibility[model_id][1],
            within_equivalence_band=model_id in equivalent,
        )
        for model_id in MODEL_ORDER
    )
    ordered_results = tuple(by_id[model_id] for model_id in MODEL_ORDER)
    return SelectionDecision(selected, ordered_results, decisions, best_ap)


def bootstrap_validation_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    resamples: int,
    random_state: int,
) -> dict[str, Any]:
    """Compute deterministic prediction-only stratified bootstrap intervals."""

    labels = np.asarray(target, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise SelectionAnalysisError("Bootstrap labels and probabilities must be aligned vectors.")
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if not len(negative) or not len(positive):
        raise SelectionAnalysisError("Bootstrap evidence requires both target classes.")
    generator = np.random.default_rng(random_state)
    values: dict[str, list[float]] = {
        "average_precision": [],
        "brier_score": [],
        "lift_at_0_1": [],
    }
    for _ in range(resamples):
        sampled = np.concatenate(
            (
                generator.choice(negative, size=len(negative), replace=True),
                generator.choice(positive, size=len(positive), replace=True),
            )
        )
        metrics = evaluate_predictions(
            labels[sampled], scores[sampled], probabilities=scores[sampled]
        )
        values["average_precision"].append(metrics.discrimination.average_precision)
        values["brier_score"].append(_brier(metrics))
        values["lift_at_0_1"].append(_lift_at_ten(metrics))
    point = evaluate_predictions(labels, scores, probabilities=scores)
    point_values = {
        "average_precision": point.discrimination.average_precision,
        "brier_score": _brier(point),
        "lift_at_0_1": _lift_at_ten(point),
    }
    return {
        "method": "stratified_prediction_only_percentile_bootstrap",
        "resamples": resamples,
        "random_state": random_state,
        "confidence_level": 0.95,
        "metrics": {
            name: {
                "point": point_values[name],
                "lower": float(np.quantile(samples, 0.025)),
                "upper": float(np.quantile(samples, 0.975)),
            }
            for name, samples in values.items()
        },
    }


def calibration_diagnostics(target: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Describe identity calibration without fitting a calibrator."""

    labels = np.asarray(target, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    bins = np.array_split(order, 10)
    reliability = []
    weighted_error = 0.0
    for index, members in enumerate(bins):
        mean_score = float(np.mean(scores[members]))
        event_rate = float(np.mean(labels[members]))
        weight = len(members) / len(labels)
        weighted_error += weight * abs(mean_score - event_rate)
        reliability.append(
            {
                "bin": index + 1,
                "rows": len(members),
                "mean_probability": mean_score,
                "observed_event_rate": event_rate,
            }
        )
    return {
        "method": "identity",
        "calibrator_fitted": False,
        "mean_probability": float(np.mean(scores)),
        "observed_prevalence": float(np.mean(labels)),
        "expected_calibration_error_10_equal_count_bins": float(weighted_error),
        "reliability_bins": reliability,
    }


def risk_band_thresholds(
    probabilities: np.ndarray, quantiles: tuple[float, ...]
) -> dict[str, float]:
    scores = np.asarray(probabilities, dtype=np.float64)
    return {
        f"q{int(quantile * 100)}": float(np.quantile(scores, quantile, method="higher"))
        for quantile in quantiles
    }


def metrics_payload(metrics: PredictionMetrics) -> dict[str, Any]:
    return asdict(metrics)


def flat_tracking_metrics(metrics: PredictionMetrics) -> dict[str, float]:
    return {
        "average_precision": metrics.discrimination.average_precision,
        "roc_auc": metrics.discrimination.roc_auc,
        "brier_score": _brier(metrics),
        "log_loss": metrics.probability.log_loss
        if metrics.probability is not None
        else float("nan"),
        "lift_at_0_1": _lift_at_ten(metrics),
    }


def _brier(metrics: PredictionMetrics) -> float:
    if metrics.probability is None:
        raise SelectionAnalysisError("Selection requires probability metrics.")
    return metrics.probability.brier_score


def _lift_at_ten(metrics: PredictionMetrics) -> float:
    for capacity in metrics.capacities:
        if capacity.capacity == 0.1:
            return capacity.lift
    raise SelectionAnalysisError("Selection metrics are missing lift at 10% capacity.")
