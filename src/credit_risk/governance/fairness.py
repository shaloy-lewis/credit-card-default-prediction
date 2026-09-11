"""Validation-only subgroup review with deterministic support rules."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.governance.contracts import FairnessContract


class FairnessAnalysisError(ValueError):
    """Raised when subgroup evidence cannot satisfy its frozen contract."""


_MEASURE_NAMES = (
    "target_prevalence",
    "mean_probability",
    "calibration_in_the_large",
    "brier_score",
    "selection_rate_at_q90",
    "true_positive_rate_at_q90",
    "false_positive_rate_at_q90",
)


@dataclass(frozen=True, slots=True)
class FairnessResult:
    overall: dict[str, float | int]
    groups: tuple[dict[str, Any], ...]
    triggers: tuple[dict[str, Any], ...]
    bootstrap: dict[str, Any]


def analyse_subgroups(
    audit: pd.DataFrame,
    target: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    contract: FairnessContract,
) -> FairnessResult:
    """Compute predeclared group metrics and within-group bootstrap intervals."""

    labels = np.asarray(target, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if not isinstance(audit, pd.DataFrame) or len(audit) != len(labels):
        raise FairnessAnalysisError("Audit fields must align with validation labels.")
    if labels.ndim != 1 or scores.shape != labels.shape or set(np.unique(labels)) != {0, 1}:
        raise FairnessAnalysisError("Validation labels and probabilities are invalid.")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise FairnessAnalysisError("Validation probabilities must be finite and bounded.")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise FairnessAnalysisError("The q90 policy threshold must be a finite probability.")
    required_columns = {axis.source_column for axis in contract.axes.values()}
    missing = sorted(required_columns - set(audit.columns))
    if missing:
        raise FairnessAnalysisError(f"Audit frame is missing required columns: {missing}")

    selected = scores >= threshold
    overall = _metrics(labels, scores, selected)
    group_results: list[dict[str, Any]] = []
    bootstrap_evidence: dict[str, Any] = {
        "confidence_level": contract.bootstrap.confidence_level,
        "method": contract.bootstrap.method,
        "resamples": contract.bootstrap.resamples,
        "seed": contract.bootstrap.seed,
        "groups": {},
    }
    trigger_results: list[dict[str, Any]] = []

    for axis_name, axis in contract.axes.items():
        mapped = _map_axis(audit[axis.source_column], axis_name, axis)
        bootstrap_evidence["groups"][axis_name] = {}
        expected_groups = (
            tuple(axis.groups) if axis.groups is not None else tuple(axis.labels or ())
        )
        for group_name in expected_groups:
            mask = mapped.eq(group_name).to_numpy()
            group_labels = labels[mask]
            group_scores = scores[mask]
            counts = {
                "rows": int(mask.sum()),
                "positive_labels": int(group_labels.sum()),
                "negative_labels": int(len(group_labels) - group_labels.sum()),
            }
            supported = (
                counts["rows"] >= contract.support.minimum_rows
                and counts["positive_labels"] >= contract.support.minimum_positive_labels
                and counts["negative_labels"] >= contract.support.minimum_negative_labels
            )
            if not supported:
                group_results.append(
                    {
                        "axis": axis_name,
                        "group": group_name,
                        "status": contract.support.unsupported_status,
                        **counts,
                    }
                )
                continue

            measures = _metrics(group_labels, group_scores, group_scores >= threshold)
            comparisons = _comparisons(measures, overall)
            intervals, distributions = _bootstrap_group(
                group_labels,
                group_scores,
                threshold=threshold,
                resamples=contract.bootstrap.resamples,
                confidence=contract.bootstrap.confidence_level,
                seed=_stable_seed(contract.bootstrap.seed, axis_name, group_name),
            )
            group_triggers = _triggers(axis_name, group_name, measures, comparisons, contract)
            trigger_results.extend(group_triggers)
            group_results.append(
                {
                    "axis": axis_name,
                    "group": group_name,
                    "status": "reviewed",
                    **counts,
                    "metrics": measures,
                    "comparisons_to_overall": comparisons,
                    "confidence_intervals": intervals,
                    "review_triggers": group_triggers,
                }
            )
            bootstrap_evidence["groups"][axis_name][group_name] = distributions

    return FairnessResult(
        overall=overall,
        groups=tuple(group_results),
        triggers=tuple(trigger_results),
        bootstrap=bootstrap_evidence,
    )


def _map_axis(series: pd.Series, axis_name: str, axis: Any) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise FairnessAnalysisError(f"Audit axis {axis_name!r} contains invalid values.")
    mapped = pd.Series(index=series.index, dtype="object")
    if axis.groups is not None:
        for label, values in axis.groups.items():
            mapped.loc[numeric.isin(values)] = label
    else:
        assert axis.boundaries is not None and axis.labels is not None
        mapped = pd.cut(
            numeric,
            bins=list(axis.boundaries),
            labels=list(axis.labels),
            right=False,
            include_lowest=True,
        ).astype("object")
    if mapped.isna().any():
        unknown = sorted(set(int(value) for value in numeric[mapped.isna()].unique()))
        raise FairnessAnalysisError(f"Audit axis {axis_name!r} has unmapped values: {unknown}")
    return mapped.astype(str)


def _metrics(
    labels: np.ndarray, scores: np.ndarray, selected: np.ndarray
) -> dict[str, float | int]:
    positives = labels == 1
    negatives = ~positives
    return {
        "rows": int(len(labels)),
        "target_prevalence": float(np.mean(labels)),
        "mean_probability": float(np.mean(scores)),
        "calibration_in_the_large": float(np.mean(scores) - np.mean(labels)),
        "brier_score": float(np.mean(np.square(scores - labels))),
        "selection_rate_at_q90": float(np.mean(selected)),
        "true_positive_rate_at_q90": float(np.mean(selected[positives])),
        "false_positive_rate_at_q90": float(np.mean(selected[negatives])),
    }


def _comparisons(
    group: dict[str, float | int], overall: dict[str, float | int]
) -> dict[str, float]:
    overall_selection = float(overall["selection_rate_at_q90"])
    return {
        "selection_rate_ratio": float(group["selection_rate_at_q90"]) / overall_selection,
        "true_positive_rate_gap": float(group["true_positive_rate_at_q90"])
        - float(overall["true_positive_rate_at_q90"]),
        "false_positive_rate_gap": float(group["false_positive_rate_at_q90"])
        - float(overall["false_positive_rate_at_q90"]),
        "brier_score_degradation": float(group["brier_score"]) - float(overall["brier_score"]),
    }


def _bootstrap_group(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]]]:
    rng = np.random.default_rng(seed)
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    distributions: dict[str, list[float]] = {name: [] for name in _MEASURE_NAMES}
    for _ in range(resamples):
        sampled = np.concatenate(
            (
                rng.choice(positive_indices, len(positive_indices), replace=True),
                rng.choice(negative_indices, len(negative_indices), replace=True),
            )
        )
        values = _metrics(labels[sampled], scores[sampled], scores[sampled] >= threshold)
        for name in _MEASURE_NAMES:
            distributions[name].append(float(values[name]))
    alpha = (1.0 - confidence) / 2.0
    intervals = {
        name: {
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }
        for name, values in distributions.items()
    }
    return intervals, distributions


def _triggers(
    axis: str,
    group: str,
    metrics: dict[str, float | int],
    comparisons: dict[str, float],
    contract: FairnessContract,
) -> list[dict[str, Any]]:
    thresholds = contract.triggers
    checks = (
        (
            "selection_rate_ratio",
            comparisons["selection_rate_ratio"],
            comparisons["selection_rate_ratio"] < thresholds.selection_rate_ratio_lower,
            "below_lower_bound",
            thresholds.selection_rate_ratio_lower,
        ),
        (
            "selection_rate_ratio",
            comparisons["selection_rate_ratio"],
            comparisons["selection_rate_ratio"] > thresholds.selection_rate_ratio_upper,
            "above_upper_bound",
            thresholds.selection_rate_ratio_upper,
        ),
        (
            "true_positive_rate_gap",
            comparisons["true_positive_rate_gap"],
            abs(comparisons["true_positive_rate_gap"]) > thresholds.absolute_true_positive_rate_gap,
            "absolute_gap_above_limit",
            thresholds.absolute_true_positive_rate_gap,
        ),
        (
            "false_positive_rate_gap",
            comparisons["false_positive_rate_gap"],
            abs(comparisons["false_positive_rate_gap"])
            > thresholds.absolute_false_positive_rate_gap,
            "absolute_gap_above_limit",
            thresholds.absolute_false_positive_rate_gap,
        ),
        (
            "brier_score_degradation",
            comparisons["brier_score_degradation"],
            comparisons["brier_score_degradation"] > thresholds.maximum_brier_degradation,
            "above_upper_bound",
            thresholds.maximum_brier_degradation,
        ),
        (
            "calibration_in_the_large",
            float(metrics["calibration_in_the_large"]),
            abs(float(metrics["calibration_in_the_large"]))
            > thresholds.absolute_calibration_in_the_large,
            "absolute_value_above_limit",
            thresholds.absolute_calibration_in_the_large,
        ),
    )
    return [
        {
            "axis": axis,
            "group": group,
            "metric": metric,
            "direction": direction,
            "observed": observed,
            "threshold": limit,
            "disposition": "documented_human_review_required",
        }
        for metric, observed, fired, direction, limit in checks
        if fired
    ]


def _stable_seed(seed: int, axis: str, group: str) -> int:
    digest = hashlib.sha256(f"{seed}|{axis}|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "little")
