from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.governance.contracts import load_governance_config
from credit_risk.governance.fairness import (
    FairnessAnalysisError,
    _map_axis,
    _triggers,
    analyse_subgroups,
)


def _contract():
    base = load_governance_config().fairness
    return base.model_copy(
        update={
            "axes": {"sex_code": base.axes["sex_code"]},
            "support": base.support.model_copy(
                update={
                    "minimum_rows": 4,
                    "minimum_positive_labels": 2,
                    "minimum_negative_labels": 2,
                }
            ),
            "bootstrap": base.bootstrap.model_copy(update={"resamples": 20}),
        }
    )


def test_subgroup_metrics_bootstrap_and_order_are_deterministic() -> None:
    audit = pd.DataFrame({"sex_code": [1] * 6 + [2] * 6})
    target = np.asarray([0, 0, 0, 1, 1, 1] * 2, dtype=np.int8)
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.6, 0.7, 0.9] * 2)

    first = analyse_subgroups(audit, target, probabilities, threshold=0.5, contract=_contract())
    second = analyse_subgroups(audit, target, probabilities, threshold=0.5, contract=_contract())

    assert first == second
    assert first.overall["rows"] == 12
    assert [group["group"] for group in first.groups] == ["1", "2"]
    assert all(group["status"] == "reviewed" for group in first.groups)
    assert len(first.bootstrap["groups"]["sex_code"]["1"]["brier_score"]) == 20


def test_support_suppression_reports_counts_only() -> None:
    contract = _contract()
    audit = pd.DataFrame({"sex_code": [1] * 8 + [2]})
    target = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int8)
    probabilities = np.linspace(0.1, 0.9, 9)

    result = analyse_subgroups(audit, target, probabilities, threshold=0.5, contract=contract)
    unsupported = next(group for group in result.groups if group["group"] == "2")

    assert unsupported == {
        "axis": "sex_code",
        "group": "2",
        "status": "insufficient_support",
        "rows": 1,
        "positive_labels": 0,
        "negative_labels": 1,
    }


def test_selection_ratio_trigger_boundaries_are_inclusive() -> None:
    contract = _contract()
    audit = pd.DataFrame({"sex_code": [1] * 10 + [2] * 10})
    target = np.asarray([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5, dtype=np.int8)
    # Overall selection is 50%; group ratios are exactly 0.8 and 1.2.
    probabilities = np.asarray([0.9] * 4 + [0.1] * 6 + [0.9] * 6 + [0.1] * 4, dtype=np.float64)

    result = analyse_subgroups(audit, target, probabilities, threshold=0.5, contract=contract)

    assert not any(trigger["metric"] == "selection_rate_ratio" for trigger in result.triggers)


def test_invalid_audit_or_probabilities_fail() -> None:
    target = np.asarray([0, 1, 0, 1])
    with pytest.raises(FairnessAnalysisError, match="missing"):
        analyse_subgroups(
            pd.DataFrame({"wrong": [1] * 4}),
            target,
            np.asarray([0.1, 0.8, 0.2, 0.9]),
            threshold=0.5,
            contract=_contract(),
        )
    with pytest.raises(FairnessAnalysisError, match="labels and probabilities"):
        analyse_subgroups(
            pd.DataFrame({"sex_code": [1, 1, 2, 2]}),
            np.asarray([0, 0, 0, 0]),
            np.asarray([0.1, 0.8, 0.2, 0.9]),
            threshold=0.5,
            contract=_contract(),
        )
    with pytest.raises(FairnessAnalysisError, match="threshold"):
        analyse_subgroups(
            pd.DataFrame({"sex_code": [1, 1, 2, 2]}),
            target,
            np.asarray([0.1, 0.8, 0.2, 0.9]),
            threshold=2.0,
            contract=_contract(),
        )
    with pytest.raises(FairnessAnalysisError, match="bounded"):
        analyse_subgroups(
            pd.DataFrame({"sex_code": [1, 1, 2, 2]}),
            target,
            np.asarray([0.1, 1.8, 0.2, 0.9]),
            threshold=0.5,
            contract=_contract(),
        )


def test_age_mapping_and_unknown_group_values() -> None:
    config = load_governance_config().fairness
    ages = pd.Series([18, 29, 30, 59, 60, 100])

    mapped = _map_axis(ages, "age_band", config.axes["age_band"])

    assert mapped.tolist() == ["18_29", "18_29", "30_39", "50_59", "60_100", "60_100"]
    with pytest.raises(FairnessAnalysisError, match="unmapped"):
        _map_axis(pd.Series([9]), "age_band", config.axes["age_band"])
    with pytest.raises(FairnessAnalysisError, match="invalid"):
        _map_axis(pd.Series([np.nan]), "age_band", config.axes["age_band"])


def test_all_predeclared_review_trigger_formulas() -> None:
    contract = load_governance_config().fairness
    metrics = {
        "calibration_in_the_large": 0.06,
        "brier_score": 0.2,
    }
    comparisons = {
        "selection_rate_ratio": 0.7,
        "true_positive_rate_gap": 0.11,
        "false_positive_rate_gap": -0.11,
        "brier_score_degradation": 0.03,
    }

    observed = _triggers("axis", "group", metrics, comparisons, contract)

    assert {item["metric"] for item in observed} == {
        "selection_rate_ratio",
        "true_positive_rate_gap",
        "false_positive_rate_gap",
        "brier_score_degradation",
        "calibration_in_the_large",
    }
