"""Deterministic advancement gate and selection for Phase 3 candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

from credit_risk.modeling.candidate_contracts import (
    AdvancementGate,
    SampledConfiguration,
)


class CandidateSelectionError(ValueError):
    """Raised when candidate metrics cannot support a governed decision."""


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Protocol-level metrics for one full-view sampled configuration."""

    configuration: SampledConfiguration
    average_precision_mean: float
    average_precision_standard_deviation: float
    brier_score_mean: float
    lift_at_0_1_mean: float


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """Individual and combined advancement conditions for one configuration."""

    average_precision_passed: bool
    brier_score_passed: bool
    lift_at_0_1_passed: bool
    repeat_stability_passed: bool
    eligible: bool


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """Deterministic full-view selection and fallback decision."""

    selected_configuration: SampledConfiguration
    selected_model_id: str
    catboost_advances: bool
    gate_outcomes: tuple[tuple[str, GateOutcome], ...]
    equivalent_configuration_ids: tuple[str, ...]


def apply_advancement_gate(score: CandidateScore, gate: AdvancementGate) -> GateOutcome:
    """Apply every inclusive balanced-gate condition."""

    _validate_score(score)
    thresholds = gate.derived_absolute_thresholds
    conditions = (
        score.average_precision_mean >= thresholds.minimum_average_precision_mean,
        score.brier_score_mean <= thresholds.maximum_brier_score_mean,
        score.lift_at_0_1_mean >= thresholds.minimum_lift_at_0_1_mean,
        score.average_precision_standard_deviation
        <= thresholds.maximum_average_precision_repeat_standard_deviation,
    )
    return GateOutcome(
        average_precision_passed=conditions[0],
        brier_score_passed=conditions[1],
        lift_at_0_1_passed=conditions[2],
        repeat_stability_passed=conditions[3],
        eligible=all(conditions),
    )


def select_candidate(
    scores: tuple[CandidateScore, ...],
    gate: AdvancementGate,
) -> CandidateSelection:
    """Choose the full-view configuration and explicit CatBoost/fallback outcome."""

    if not scores:
        raise CandidateSelectionError("Candidate selection requires at least one configuration.")
    ids = tuple(score.configuration.configuration_id for score in scores)
    if len(set(ids)) != len(ids):
        raise CandidateSelectionError("Candidate selection configuration IDs must be unique.")
    outcomes = tuple(
        (score.configuration.configuration_id, apply_advancement_gate(score, gate))
        for score in scores
    )
    outcome_by_id = dict(outcomes)
    eligible = tuple(
        score for score in scores if outcome_by_id[score.configuration.configuration_id].eligible
    )
    considered = eligible if eligible else scores
    best_average_precision = max(score.average_precision_mean for score in considered)
    equivalent = tuple(
        score
        for score in considered
        if best_average_precision - score.average_precision_mean
        <= gate.equivalence_band_average_precision + 1e-15
    )
    selected = min(equivalent, key=_complexity_key)
    advances = bool(eligible)
    return CandidateSelection(
        selected_configuration=selected.configuration,
        selected_model_id=("catboost_v1" if advances else gate.fallback_model_id),
        catboost_advances=advances,
        gate_outcomes=outcomes,
        equivalent_configuration_ids=tuple(
            score.configuration.configuration_id
            for score in sorted(equivalent, key=lambda item: item.configuration.configuration_id)
        ),
    )


def _validate_score(score: CandidateScore) -> None:
    values = (
        score.average_precision_mean,
        score.average_precision_standard_deviation,
        score.brier_score_mean,
        score.lift_at_0_1_mean,
    )
    if not all(math.isfinite(value) for value in values):
        raise CandidateSelectionError("Candidate gate metrics must be finite.")
    if not 0.0 <= score.average_precision_mean <= 1.0:
        raise CandidateSelectionError("Candidate average precision must be within [0, 1].")
    if score.average_precision_standard_deviation < 0.0:
        raise CandidateSelectionError("Candidate repeat standard deviation must be non-negative.")
    if not 0.0 <= score.brier_score_mean <= 1.0:
        raise CandidateSelectionError("Candidate Brier score must be within [0, 1].")
    if score.lift_at_0_1_mean <= 0.0:
        raise CandidateSelectionError("Candidate lift at 10% must be positive.")


def _complexity_key(score: CandidateScore) -> tuple[float | int | str, ...]:
    parameters = score.configuration.parameters
    return (
        parameters.depth,
        parameters.iterations,
        -parameters.l2_leaf_reg,
        parameters.learning_rate,
        parameters.random_strength,
        parameters.bagging_temperature,
        score.configuration.configuration_id,
    )
