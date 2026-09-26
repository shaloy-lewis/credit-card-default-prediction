"""The only real-row boundary used by Release B diagnostics."""

from __future__ import annotations

from typing import Any

from credit_risk.assurance.evidence import ROOT, EvidenceError


def validation_cohort(data_root: str = "data") -> tuple[Any, Any]:
    # Reuse the already-reviewed checks; never call the final-test loader.
    from credit_risk.governance.contracts import load_governance_config
    from credit_risk.governance.workflow import (
        _validate_data_lineage,
        _validate_development_boundary,
        _validation_slice,
    )
    from credit_risk.modeling.dataset import load_governed_development_data

    config = load_governance_config(ROOT / "configs/governance/phase5_v1.json")
    governed = load_governed_development_data(data_root=data_root)
    _validate_development_boundary(governed, config)
    _validate_data_lineage(governed, config)
    features, target, _ = _validation_slice(governed, config)
    if len(features) != 4800:
        raise EvidenceError("Release B requires exactly the reviewed validation cohort.")
    return features, target


def metrics(target: Any, probabilities: Any) -> dict[str, Any]:
    from credit_risk.modeling.metrics import evaluate_predictions

    if len(set(target.tolist())) < 2 or len(target) < 10:
        return {"status": "insufficient_support", "rows": len(target)}
    result = evaluate_predictions(target, probabilities, probabilities=probabilities)
    assert result.probability is not None
    return {
        "status": "measured",
        "rows": len(target),
        "average_precision": result.discrimination.average_precision,
        "brier_score": result.probability.brier_score,
        "lift_at_0_1": next(x.lift for x in result.capacities if x.capacity == 0.1),
    }


def check_baseline(target: Any, probabilities: Any) -> dict[str, Any]:
    from credit_risk.governance.contracts import load_governance_config

    config = load_governance_config(ROOT / "configs/governance/phase5_v1.json")
    measured = metrics(target, probabilities)
    if any(
        abs(measured[name] - expected) > config.prediction.metric_absolute_tolerance
        for name, expected in config.prediction.expected_validation_metrics.items()
    ):
        raise EvidenceError("Baseline metrics do not match reviewed validation evidence.")
    return measured
