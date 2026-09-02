"""Manually opt-in smoke test for the four real fixed estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.modeling.contracts import PREDICTOR_COLUMNS, REPAYMENT_STATUS_COLUMNS
from credit_risk.modeling.selection_contracts import load_selection_config
from credit_risk.modeling.selection_models import fit_one_pass_models

pytestmark = [pytest.mark.integration, pytest.mark.training]


def test_real_fixed_models_fit_once_and_return_valid_probabilities(tmp_path, monkeypatch) -> None:
    rows = 240
    values = {}
    for column in PREDICTOR_COLUMNS:
        if column in REPAYMENT_STATUS_COLUMNS:
            values[column] = np.resize(np.arange(-2, 10), rows)
        else:
            values[column] = np.arange(rows, dtype=np.int64) * 100 + 10_000
    features = pd.DataFrame(
        values,
        index=pd.Index(range(1, rows + 1), name="account_id"),
    )
    target = pd.Series(
        np.resize(np.asarray([0, 0, 0, 1]), rows),
        index=features.index,
        name="default_next_month",
    )
    config = load_selection_config()
    monkeypatch.chdir(tmp_path)

    fitted = fit_one_pass_models(features, target, config)

    assert len(fitted) == 4
    for model in fitted:
        probabilities = model.predict_proba(features.iloc[:20])
        assert probabilities.shape == (20,)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert not (tmp_path / "catboost_info").exists()
