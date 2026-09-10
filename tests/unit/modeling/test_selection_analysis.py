from __future__ import annotations

import numpy as np
import pytest

from credit_risk.modeling.metrics import (
    CapacityMetrics,
    DiscriminationMetrics,
    PredictionMetrics,
    ProbabilityMetrics,
)
from credit_risk.modeling.selection_analysis import (
    SelectionAnalysisError,
    ValidationResult,
    bootstrap_validation_metrics,
    calibration_diagnostics,
    risk_band_thresholds,
    select_validation_winner,
)
from credit_risk.modeling.selection_contracts import load_selection_config


def test_selection_applies_guardrails_and_simplicity_equivalence_independent_of_order() -> None:
    config = load_selection_config()
    results = (
        _result("catboost_fixed", ap=0.5600, brier=0.135, lift=3.1),
        _result("random_forest", ap=0.5590, brier=0.134, lift=3.1),
        _result("logistic_l2", ap=0.5400, brier=0.140, lift=3.0),
        _result("hist_gradient_boosting", ap=0.5581, brier=0.142, lift=3.0),
    )

    decision = select_validation_winner(results, config)

    assert decision.selected_model_id == "hist_gradient_boosting"
    by_id = {item.model_id: item for item in decision.decisions}
    assert by_id["hist_gradient_boosting"].within_equivalence_band
    assert by_id["catboost_fixed"].within_equivalence_band
    assert by_id["random_forest"].within_equivalence_band


@pytest.mark.parametrize(
    ("brier", "lift", "eligible"),
    ((0.145, 2.9, True), (0.1450001, 2.9, False), (0.145, 2.899999, False)),
)
def test_guardrail_boundaries_are_inclusive(brier: float, lift: float, eligible: bool) -> None:
    config = load_selection_config()
    results = (
        _result("logistic_l2", ap=0.5, brier=0.14, lift=3.0),
        _result("hist_gradient_boosting", ap=0.6, brier=brier, lift=lift),
        _result("random_forest", ap=0.49, brier=0.14, lift=3.0),
        _result("catboost_fixed", ap=0.48, brier=0.14, lift=3.0),
    )

    decision = select_validation_winner(results, config)

    observed = next(
        item.eligible for item in decision.decisions if item.model_id == "hist_gradient_boosting"
    )
    assert observed is eligible


def test_prediction_only_analysis_is_deterministic_and_never_fits() -> None:
    target = np.tile(np.asarray([0, 0, 0, 1], dtype=np.int8), 20)
    probabilities = np.linspace(0.01, 0.8, len(target))

    first = bootstrap_validation_metrics(target, probabilities, resamples=20, random_state=42)
    second = bootstrap_validation_metrics(target, probabilities, resamples=20, random_state=42)

    assert first == second
    diagnostics = calibration_diagnostics(target, probabilities)
    assert diagnostics["calibrator_fitted"] is False
    thresholds = risk_band_thresholds(probabilities, (0.8, 0.9, 0.95))
    assert thresholds["q80"] <= thresholds["q90"] <= thresholds["q95"]


def test_analysis_rejects_incomplete_results_and_invalid_bootstrap() -> None:
    config = load_selection_config()
    duplicate = tuple(_result("logistic_l2", ap=0.5, brier=0.1, lift=2.0) for _ in range(4))
    with pytest.raises(SelectionAnalysisError, match="one unique result"):
        select_validation_winner(duplicate, config)

    with pytest.raises(SelectionAnalysisError, match="aligned vectors"):
        bootstrap_validation_metrics(
            np.asarray([0, 1]), np.asarray([0.2]), resamples=2, random_state=42
        )
    with pytest.raises(SelectionAnalysisError, match="both target classes"):
        bootstrap_validation_metrics(
            np.asarray([0, 0]), np.asarray([0.2, 0.3]), resamples=2, random_state=42
        )

    missing_probability = _result("logistic_l2", ap=0.5, brier=0.1, lift=2.0)
    missing_probability = ValidationResult(
        missing_probability.model_id,
        missing_probability.probabilities,
        PredictionMetrics(
            missing_probability.metrics.discrimination,
            None,
            missing_probability.metrics.capacities,
        ),
    )
    remaining = tuple(
        _result(model_id, ap=0.5, brier=0.1, lift=2.0)
        for model_id in ("hist_gradient_boosting", "random_forest", "catboost_fixed")
    )
    with pytest.raises(SelectionAnalysisError, match="probability metrics"):
        select_validation_winner((missing_probability, *remaining), config)


def _result(model_id: str, *, ap: float, brier: float, lift: float) -> ValidationResult:
    capacity = CapacityMetrics(0.1, 10, 0.5, 1, 1.0, 5.0, 0.5, 0.5, lift)
    metrics = PredictionMetrics(
        discrimination=DiscriminationMetrics(ap, 0.7, 0.3, 0.4),
        probability=ProbabilityMetrics(brier, 0.5),
        capacities=(capacity,),
    )
    return ValidationResult(model_id, np.asarray([0.1, 0.9]), metrics)
