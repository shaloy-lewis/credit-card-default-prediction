from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from credit_risk.governance.contracts import (
    OFFICIAL_PHASE5_CONFIG_SHA256,
    AxisContract,
    GovernanceConfig,
    GovernanceContractError,
    governance_config_sha256,
    load_governance_config,
)
from credit_risk.modeling.contracts import AUDIT_COLUMNS, PREDICTOR_COLUMNS


def test_official_contract_is_complete_and_digest_protected() -> None:
    path = Path("configs/governance/phase5_v1.json")
    config = load_governance_config(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == OFFICIAL_PHASE5_CONFIG_SHA256
    assert config.features.predictor_columns == PREDICTOR_COLUMNS
    assert config.features.audit_columns == AUDIT_COLUMNS
    assert config.population.rows == 4800
    assert config.population.target_counts == {"0": 3738, "1": 1062}
    assert config.explanation.sample_rows == 1000
    assert config.fairness.bootstrap.resamples == 500
    assert config.review.g3_result == "closed_with_conditions"
    assert "test_partition_loading" in config.prohibitions
    assert "training" in config.prohibitions


def test_contract_rejects_digest_and_semantic_changes(tmp_path: Path) -> None:
    payload = json.loads(Path("configs/governance/phase5_v1.json").read_text(encoding="utf-8"))
    payload["population"]["rows"] = 4799
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GovernanceContractError, match="digest differs"):
        load_governance_config(altered)
    with pytest.raises(GovernanceContractError, match="4800"):
        load_governance_config(altered, require_official_digest=False)


def test_contract_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GovernanceContractError, match="Unable to read"):
        load_governance_config(tmp_path / "missing.json")
    with pytest.raises(GovernanceContractError, match="Unable to hash"):
        governance_config_sha256(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda p: p["features"]["predictor_columns"].pop(), "predictors differ"),
        (lambda p: p["features"]["audit_columns"].pop(), "audit columns differ"),
        (lambda p: p["features"]["forbidden_predictor_columns"].pop(), "forbidden predictor"),
        (lambda p: p["population"]["target_counts"].update({"0": 1}), "target counts differ"),
        (lambda p: p["dependencies"].pop("numpy"), "dependency boundary"),
        (
            lambda p: p["explanation"]["reason_categories"]["credit_capacity"].clear(),
            "categories must cover",
        ),
        (
            lambda p: p["explanation"]["direction_labels"].update({"positive": "higher"}),
            "direction labels differ",
        ),
        (lambda p: p["explanation"].update({"stratification": ["target"]}), "must stratify"),
        (lambda p: p["fairness"]["axes"].pop("sex_code"), "audit axes differ"),
        (lambda p: p["prohibitions"].remove("training"), "prohibitions are missing"),
        (lambda p: p["outputs"]["committed"].pop(), "evidence allowlist differs"),
    ),
)
def test_semantic_contract_rejects_governance_drift(mutator, message: str) -> None:
    payload = json.loads(Path("configs/governance/phase5_v1.json").read_text(encoding="utf-8"))
    mutator(payload)

    with pytest.raises(ValueError, match=message):
        GovernanceConfig.model_validate(payload)


def test_axis_contract_requires_one_valid_mapping_form() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        AxisContract(source_column="age", groups={"a": (1,)}, boundaries=(0, 2), labels=("a",))
    with pytest.raises(ValueError, match="boundaries"):
        AxisContract(source_column="age", boundaries=(0, 2), labels=("a", "b"))
