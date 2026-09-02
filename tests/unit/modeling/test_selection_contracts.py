from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_risk.modeling.selection_contracts import (
    SelectionConfig,
    SelectionContractError,
    load_selection_config,
)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("selection", "fit_budget"), 5, "Input should be 4"),
        (("selection", "winner_refit"), True, "Input should be False"),
        (("data", "holdout_access"), "allowed", "Input should be 'prohibited'"),
        (("governance", "parameter_tuning"), "allowed", "Input should be 'prohibited'"),
        (("models", 0, "parameters", "C"), 2.0, "fixed model parameters"),
        (("models", 1, "feature_handling"), "scaled", "feature handling"),
        (("selection", "simplicity_order"), ["catboost_fixed"], "simplicity order"),
        (("selection", "risk_band_quantiles"), [0.5], "risk-band quantiles"),
        (("selection", "brier_guardrail_relative_to_logistic"), 0.5, "guardrails"),
        (("dependencies", "joblib"), "0.0.0", "dependency versions"),
        (("test_gate_deltas", "maximum_brier_score_delta"), 1.0, "test-gate deltas"),
        (("evidence", "runtime_outputs"), ["rows.csv"], "runtime evidence allowlist"),
        (("data", "training_target_counts"), {"0": 1, "1": 1}, "training target counts"),
    ),
)
def test_selection_contract_rejects_governance_mutations(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    payload = json.loads(Path("configs/modeling/selection_v1.json").read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        SelectionConfig.model_validate_json(json.dumps(payload))


def test_selection_loader_rejects_unsafe_or_mismatched_references(tmp_path: Path) -> None:
    payload = json.loads(Path("configs/modeling/selection_v1.json").read_text(encoding="utf-8"))
    payload["data"]["feature_contract_path"] = "../feature.json"
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SelectionContractError, match="safe repository-relative"):
        load_selection_config(path)
