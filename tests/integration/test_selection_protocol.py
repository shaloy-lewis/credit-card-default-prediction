"""Complete-file and semantic integrity checks for the one-pass protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path

from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.modeling.selection_contracts import (
    MODEL_ORDER,
    SIMPLICITY_ORDER,
    load_selection_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "modeling" / "selection_v1.json"

# Change only after explicit protocol review, before any replacement selection metrics are seen.
EXPECTED_CONFIG_SHA256 = "2c85c6c0c07fa875256f2d861e2ded24a96532395c93f975d40886b0d6dc8c09"


def test_one_pass_selection_protocol_is_frozen_and_holdout_blind() -> None:
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == EXPECTED_CONFIG_SHA256
    config = load_selection_config(CONFIG_PATH)

    assert tuple(model.model_id for model in config.models) == MODEL_ORDER
    assert config.selection.fit_budget == 4
    assert config.selection.fits_per_model == 1
    assert config.selection.winner_refit is False
    assert config.governance.parameter_tuning == "prohibited"
    assert config.governance.cross_validation_iteration == "prohibited"
    assert config.data.holdout_access == "prohibited"
    assert config.data.training_rows == 19200
    assert config.data.validation_rows == 4800
    assert config.data.sealed_test_rows == 6000
    assert config.predictor_columns == PREDICTOR_COLUMNS
    assert config.features.demographic_predictors == "prohibited"
    assert config.selection.simplicity_order == SIMPLICITY_ORDER
    assert config.selection.bootstrap_resamples == 500
    assert config.selection.calibration == "identity"


def test_fixed_model_budget_contains_no_search_or_early_stopping() -> None:
    config = load_selection_config(CONFIG_PATH)
    parameters = {model.model_id: model.parameters for model in config.models}

    assert parameters["random_forest"]["n_estimators"] == 100
    assert parameters["hist_gradient_boosting"]["early_stopping"] is False
    assert parameters["catboost_fixed"]["iterations"] == 300
    assert parameters["catboost_fixed"]["depth"] == 4
    assert parameters["catboost_fixed"]["use_best_model"] is False
    assert all("parameter_grid" not in model.parameters for model in config.models)
