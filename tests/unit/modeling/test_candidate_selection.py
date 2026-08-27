"""Tests for the deterministic Phase 3 advancement decision."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from credit_risk.modeling.candidate_contracts import load_candidate_config
from credit_risk.modeling.candidate_selection import (
    CandidateScore,
    CandidateSelectionError,
    apply_advancement_gate,
    select_candidate,
)


@pytest.fixture(scope="module")
def config():
    return load_candidate_config()


def _score(config, index: int = 0, **overrides: float) -> CandidateScore:
    thresholds = config.advancement_gate.derived_absolute_thresholds
    values = {
        "average_precision_mean": thresholds.minimum_average_precision_mean,
        "average_precision_standard_deviation": (
            thresholds.maximum_average_precision_repeat_standard_deviation
        ),
        "brier_score_mean": thresholds.maximum_brier_score_mean,
        "lift_at_0_1_mean": thresholds.minimum_lift_at_0_1_mean,
    }
    values.update(overrides)
    return CandidateScore(
        configuration=config.candidate.search.sampled_configurations[index],
        **values,
    )


def test_gate_boundaries_are_inclusive(config) -> None:
    outcome = apply_advancement_gate(_score(config), config.advancement_gate)

    assert outcome.eligible is True
    assert all(
        (
            outcome.average_precision_passed,
            outcome.brier_score_passed,
            outcome.lift_at_0_1_passed,
            outcome.repeat_stability_passed,
        )
    )


@pytest.mark.parametrize(
    ("overrides", "field"),
    (
        ({"average_precision_mean": 0.55}, "average_precision_passed"),
        ({"brier_score_mean": 0.15}, "brier_score_passed"),
        ({"lift_at_0_1_mean": 3.0}, "lift_at_0_1_passed"),
        ({"average_precision_standard_deviation": 0.02}, "repeat_stability_passed"),
    ),
)
def test_each_gate_condition_can_block_advancement(config, overrides, field: str) -> None:
    outcome = apply_advancement_gate(_score(config, **overrides), config.advancement_gate)

    assert getattr(outcome, field) is False
    assert outcome.eligible is False


def test_selection_uses_equivalence_band_then_complexity_order(config) -> None:
    complex_score = _score(config, 0, average_precision_mean=0.60)
    simpler_score = _score(config, 3, average_precision_mean=0.598)

    decision = select_candidate((complex_score, simpler_score), config.advancement_gate)

    assert decision.selected_configuration.configuration_id == "cb_cfg_004"
    assert decision.catboost_advances is True
    assert decision.selected_model_id == "catboost_v1"
    assert decision.equivalent_configuration_ids == ("cb_cfg_001", "cb_cfg_004")


def test_selection_is_input_order_invariant(config) -> None:
    scores = (
        _score(config, 0, average_precision_mean=0.60),
        _score(config, 3, average_precision_mean=0.599),
        _score(config, 4, average_precision_mean=0.58),
    )

    first = select_candidate(scores, config.advancement_gate)
    second = select_candidate(tuple(reversed(scores)), config.advancement_gate)

    assert first.selected_configuration == second.selected_configuration
    assert dict(first.gate_outcomes) == dict(second.gate_outcomes)


def test_no_eligible_configuration_uses_diagnostic_winner_and_logistic_fallback(config) -> None:
    first = _score(config, 0, average_precision_mean=0.50)
    second = _score(config, 3, average_precision_mean=0.499)

    decision = select_candidate((first, second), config.advancement_gate)

    assert decision.selected_configuration.configuration_id == "cb_cfg_004"
    assert decision.catboost_advances is False
    assert decision.selected_model_id == "logistic_l2"


def test_invalid_inputs_are_rejected(config) -> None:
    valid = _score(config)
    with pytest.raises(CandidateSelectionError, match="at least one"):
        select_candidate((), config.advancement_gate)
    with pytest.raises(CandidateSelectionError, match="unique"):
        select_candidate((valid, valid), config.advancement_gate)
    with pytest.raises(CandidateSelectionError, match="finite"):
        apply_advancement_gate(
            replace(valid, average_precision_mean=math.nan),
            config.advancement_gate,
        )
    with pytest.raises(CandidateSelectionError, match="within"):
        apply_advancement_gate(
            replace(valid, brier_score_mean=1.1),
            config.advancement_gate,
        )
    with pytest.raises(CandidateSelectionError, match="within"):
        apply_advancement_gate(
            replace(valid, average_precision_mean=1.1),
            config.advancement_gate,
        )
    with pytest.raises(CandidateSelectionError, match="non-negative"):
        apply_advancement_gate(
            replace(valid, average_precision_standard_deviation=-0.1),
            config.advancement_gate,
        )
    with pytest.raises(CandidateSelectionError, match="positive"):
        apply_advancement_gate(
            replace(valid, lift_at_0_1_mean=0.0),
            config.advancement_gate,
        )
