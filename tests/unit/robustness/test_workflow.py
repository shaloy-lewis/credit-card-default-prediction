"""The frozen grid preserves the reviewed feature domains and outcome boundary."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.inference.contracts import OperationalFeatures
from credit_risk.inference.engine import InferenceResult, ReasonAttribution
from credit_risk.robustness.workflow import material, scenarios, selected_ids, sensitivity

FIXTURE = "tests/fixtures/inference_batch_v1.csv"


def test_scenario_grid_and_constraints():
    frame = pd.read_csv(FIXTURE).set_index("account_id")
    original = frame.copy()
    grid = scenarios(frame)
    assert len(grid) == 14
    pd.testing.assert_frame_equal(frame, original)
    for changed in grid.values():
        assert changed.index.equals(frame.index)
        assert changed.columns.equals(frame.columns)
        for row in changed.to_dict("records"):
            OperationalFeatures.model_validate(row)
    negative = frame.repayment_status_lag_0 < 0
    assert (
        grid["repayment_plus_2"]
        .loc[negative, "repayment_status_lag_0"]
        .equals(frame.loc[negative, "repayment_status_lag_0"])
    )


def result(probabilities, band="standard", direction="neutral"):
    n = len(probabilities)
    reasons = tuple((ReasonAttribution("credit_capacity", direction, 0.0),) for _ in range(n))
    return InferenceResult(
        np.array(probabilities), (band,) * n, reasons, np.zeros((n, 4)), 0.0, 0.0
    )


def test_exact_queue_and_sensitivity():
    frame = pd.read_csv(FIXTURE).set_index("account_id")
    baseline = result(np.arange(20) / 20)
    assert selected_ids(frame, baseline.probabilities) == {"acct-019", "acct-020"}
    values = sensitivity(frame, baseline, baseline)
    assert values["queue_turnover"] == values["mean_absolute_probability_change"] == 0
    assert material(values) is False
    changed = result(np.arange(20)[::-1] / 20, "high", "risk_increasing")
    values = sensitivity(frame, baseline, changed)
    assert values["queue_turnover"] == values["reason_change_fraction"] == 1
    assert material(values)


@pytest.mark.parametrize(
    "field,threshold",
    [
        ("mean_absolute_probability_change", 0.05),
        ("queue_turnover", 0.2),
        ("risk_band_change_fraction", 0.2),
        ("reason_change_fraction", 0.2),
    ],
)
def test_review_triggers(field, threshold):
    values = {
        key: 0.0
        for key in [
            "mean_absolute_probability_change",
            "queue_turnover",
            "risk_band_change_fraction",
            "reason_change_fraction",
        ]
    }
    values[field] = threshold
    assert material(values)
