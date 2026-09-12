"""Small real-bundle smoke test; prediction and SHAP only, never fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.governance.contracts import load_governance_config
from credit_risk.governance.explanations import explain_validation_sample
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.modeling.risk_policy import risk_band
from credit_risk.modeling.selected_bundle import load_selected_bundle


@pytest.mark.integration
def test_reviewed_bundle_native_shap_smoke(
    readme_prediction_payload: dict[str, int | float | str],
) -> None:
    config = load_governance_config()
    manifest, model = load_selected_bundle(
        "models/selected_v1",
        trusted=True,
        expected_manifest_sha256=config.lineage["bundle_manifest_sha256"],
        required_dependencies=tuple(config.dependencies),
    )
    rows = []
    for index in range(16):
        row = {name: readme_prediction_payload[name] for name in PREDICTOR_COLUMNS}
        row["credit_limit_ntd"] = int(row["credit_limit_ntd"]) + index * 1000
        rows.append(row)
    frame = pd.DataFrame(rows, columns=PREDICTOR_COLUMNS, index=range(1, 17))
    target = np.asarray([0, 1] * 8, dtype=np.int8)
    probabilities = model.predict_proba(frame)
    bands = np.asarray(
        [risk_band(float(value), manifest.risk_band_thresholds) for value in probabilities]
    )
    smoke_contract = config.explanation.model_copy(update={"sample_rows": 8})

    result = explain_validation_sample(
        model,
        frame,
        target,
        probabilities,
        bands,
        smoke_contract,
    )

    assert result.shap_values.shape == (8, 19)
    assert result.max_additivity_error <= smoke_contract.additivity_absolute_tolerance
    assert result.max_probability_error <= smoke_contract.probability_absolute_tolerance
