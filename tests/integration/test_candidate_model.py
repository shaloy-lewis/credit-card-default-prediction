"""Small real-CatBoost smoke test for the governed fold boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.modeling.candidate_contracts import load_candidate_config
from credit_risk.modeling.candidates import fit_candidate_fold
from credit_risk.modeling.contracts import (
    MONETARY_COLUMNS,
    PREDICTOR_COLUMNS,
    REPAYMENT_STATUS_COLUMNS,
)


@pytest.mark.integration
def test_real_catboost_fold_is_deterministic_and_writes_no_files(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_candidate_config()
    sampled = config.candidate.search.sampled_configurations[1].parameters.model_copy(
        update={"iterations": 5, "depth": 4}
    )
    train = _features(80, start=1)
    validation = _features(20, start=1_001)
    y_train = pd.Series(np.tile([0, 1], 40), index=train.index, name="default_next_month")
    y_validation = pd.Series(np.tile([0, 1], 10), index=validation.index, name="default_next_month")
    monkeypatch.chdir(tmp_path)

    first = fit_candidate_fold(
        train,
        y_train,
        validation,
        y_validation,
        predictor_columns=PREDICTOR_COLUMNS,
        categorical_columns=REPAYMENT_STATUS_COLUMNS,
        fixed_parameters=config.candidate.fixed_parameters,
        sampled_parameters=sampled,
    )
    second = fit_candidate_fold(
        train,
        y_train,
        validation,
        y_validation,
        predictor_columns=PREDICTOR_COLUMNS,
        categorical_columns=REPAYMENT_STATUS_COLUMNS,
        fixed_parameters=config.candidate.fixed_parameters,
        sampled_parameters=sampled,
    )

    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    assert first.diagnostics.tree_count == sampled.iterations
    assert not (tmp_path / "catboost_info").exists()
    assert list(tmp_path.iterdir()) == []


def _features(rows: int, *, start: int) -> pd.DataFrame:
    indexes = pd.Index(range(start, start + rows), name="account_id")
    values: dict[str, np.ndarray] = {}
    row_numbers = np.arange(rows)
    for offset, column in enumerate(REPAYMENT_STATUS_COLUMNS):
        values[column] = ((row_numbers + offset) % 12) - 2
    for offset, column in enumerate(MONETARY_COLUMNS):
        values[column] = 10_000 + (offset + 1) * 1_000 + row_numbers * 50
    return pd.DataFrame(values, index=indexes).loc[:, list(PREDICTOR_COLUMNS)]
