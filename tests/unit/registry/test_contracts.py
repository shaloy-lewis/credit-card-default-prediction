from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from credit_risk.registry.contracts import (
    EXPECTED_CONFIG_SHA256,
    REQUIRED_CHECKS,
    ArtifactReference,
    RegistryConfig,
    RegistryContractError,
    config_sha256,
    load_approval,
    load_registry_config,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "configs/registry/phase7_v1.json"


def test_registry_config_is_complete_and_digest_protected() -> None:
    config = load_registry_config(CONFIG_PATH)

    assert config_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256
    assert config.registry.registered_model_name == "credit-risk-default"
    assert tuple(item.release_revision for item in config.registry.revisions) == (
        "phase7_rev_001",
        "phase7_rev_002",
    )
    assert config.approvals.required_checks == REQUIRED_CHECKS
    assert config.bundle.revisions_share_model_bytes is True
    assert config.image_scan.severity == ("HIGH", "CRITICAL")
    assert config.image_scan.ignore_unfixed is True
    assert config.image_scan.waiver_supported is False
    assert config.smoke_test.expected_probability_six_decimals == 0.190382
    assert config.smoke_test.revisions == ("phase7_rev_001", "phase7_rev_002")
    assert config.smoke_test.prediction_only is True
    assert config.smoke_test.sealed_test_fixture is False
    assert {
        "model_fitting",
        "final_test_loading",
        "sealed_test_scoring",
        "automatic_promotion",
    }.issubset(config.prohibitions)


def test_registry_config_rejects_any_byte_change(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_bytes())
    payload["registry"]["registered_model_name"] = "changed"
    changed = tmp_path / "phase7.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RegistryContractError, match="digest mismatch"):
        load_registry_config(changed)


def test_approval_requires_external_digest_and_exact_transition(tmp_path: Path) -> None:
    approval = {
        "schema_version": "1.0.0",
        "protocol_id": "phase7_v1",
        "decision_id": "phase7_promotion_approval",
        "action": "promote",
        "decision": "approved",
        "scope": "local_portfolio_demo",
        "approved_by_role": "project_owner",
        "implementation_git_commit": "a" * 40,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "registered_model_name": "credit-risk-default",
        "source_revision": "phase7_rev_001",
        "target_revision": "phase7_rev_002",
        "checks": [{"name": name, "conclusion": "success"} for name in REQUIRED_CHECKS],
        "model_bytes_unchanged": True,
        "no_training_or_test_access": True,
        "limitations_acknowledged": True,
    }
    path = tmp_path / "approval.json"
    content = (json.dumps(approval, sort_keys=True) + "\n").encode()
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    assert load_approval(path, digest).action == "promote"
    with pytest.raises(RegistryContractError, match="digest mismatch"):
        load_approval(path, "0" * 64)

    approval["source_revision"] = "phase7_rev_002"
    changed = (json.dumps(approval, sort_keys=True) + "\n").encode()
    path.write_bytes(changed)
    with pytest.raises(RegistryContractError, match="reviewed transition"):
        load_approval(path, hashlib.sha256(changed).hexdigest())


@pytest.mark.parametrize("digest", ["", "G" * 64, "a" * 63])
def test_approval_rejects_malformed_external_digest(tmp_path: Path, digest: str) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(RegistryContractError, match="lowercase SHA-256"):
        load_approval(path, digest)


def test_contract_validator_boundaries_and_missing_files(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_bytes())
    cases = (
        ("registry", "revisions", list(reversed(payload["registry"]["revisions"]))),
        ("approvals", "required_checks", list(reversed(payload["approvals"]["required_checks"]))),
        ("evidence", "published_files", ["summary.json"]),
        ("smoke_test", "expected_probability_six_decimals", 0.2),
        ("smoke_test", "probability_absolute_tolerance", 0.01),
        ("smoke_test", "revisions", list(reversed(payload["smoke_test"]["revisions"]))),
        ("source_evidence", None, {}),
        ("prohibitions", None, ["model_fitting"]),
    )
    for section, key, value in cases:
        changed = json.loads(CONFIG_PATH.read_bytes())
        if key is None:
            changed[section] = value
        else:
            changed[section][key] = value
        with pytest.raises(ValueError):
            RegistryConfig.model_validate(changed, strict=True)

    with pytest.raises(ValueError, match="safe repository-relative"):
        ArtifactReference(path="../outside", sha256="a" * 64)
    with pytest.raises(RegistryContractError, match="Invalid Phase 7"):
        load_registry_config(tmp_path / "missing.json")
    with pytest.raises(RegistryContractError, match="Unable to read release approval"):
        load_approval(tmp_path / "missing.json", "a" * 64)
    with pytest.raises(RegistryContractError, match="Unable to hash"):
        config_sha256(tmp_path / "missing.json")


def test_approval_validator_rejects_id_and_check_changes(tmp_path: Path) -> None:
    base = {
        "schema_version": "1.0.0",
        "protocol_id": "phase7_v1",
        "decision_id": "phase7_rollback_approval",
        "action": "rollback",
        "decision": "approved",
        "scope": "local_portfolio_demo",
        "approved_by_role": "project_owner",
        "implementation_git_commit": "a" * 40,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "registered_model_name": "credit-risk-default",
        "source_revision": "phase7_rev_002",
        "target_revision": "phase7_rev_001",
        "checks": [{"name": name, "conclusion": "success"} for name in REQUIRED_CHECKS],
        "model_bytes_unchanged": True,
        "no_training_or_test_access": True,
        "limitations_acknowledged": True,
    }
    for field, value in (
        ("decision_id", "phase7_promotion_approval"),
        ("checks", list(reversed(base["checks"]))),
    ):
        changed = dict(base)
        changed[field] = value
        content = (json.dumps(changed, sort_keys=True) + "\n").encode()
        path = tmp_path / f"{field}.json"
        path.write_bytes(content)
        with pytest.raises(RegistryContractError, match="Invalid release approval"):
            load_approval(path, hashlib.sha256(content).hexdigest())
