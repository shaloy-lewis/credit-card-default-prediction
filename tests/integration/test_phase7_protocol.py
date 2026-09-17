from __future__ import annotations

import hashlib
from pathlib import Path

from credit_risk.registry.contracts import (
    EXPECTED_CONFIG_SHA256,
    REQUIRED_CHECKS,
    load_approval,
    load_registry_config,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs/registry/phase7_v1.json"
PROMOTION_PATH = REPOSITORY_ROOT / "configs/registry/phase7_promotion_approval.json"
ROLLBACK_PATH = REPOSITORY_ROOT / "configs/registry/phase7_rollback_approval.json"
PROMOTION_SHA256 = "0e0a056963365b395751fe778b1b4c0cef89787d2a68d6d9f41d339732068df9"
ROLLBACK_SHA256 = "db4041c2b7902ecbaba4fdea30cdbbabbac4c5e98776f53437ba63e975c1b5af"
IMPLEMENTATION_COMMIT = "f7a5f1d79c7dec2d661ad5b1b4dcafb1111b1456"


def test_phase7_protocol_is_frozen_before_release_operations() -> None:
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == EXPECTED_CONFIG_SHA256
    config = load_registry_config(CONFIG_PATH)
    assert config.status == "frozen_before_implementation"
    assert config.governance.training == "prohibited"
    assert config.governance.sealed_test_access == "prohibited"
    assert config.registry.backend == "sqlite"
    assert config.registry.artifact_store == "content_addressed_filesystem"
    assert config.deployment.runtime_mlflow_dependency is False
    assert config.bundle.revisions_share_model_bytes is True


def test_phase7_manual_approvals_bind_the_green_implementation() -> None:
    promotion = load_approval(PROMOTION_PATH, PROMOTION_SHA256)
    rollback = load_approval(ROLLBACK_PATH, ROLLBACK_SHA256)

    assert promotion.implementation_git_commit == IMPLEMENTATION_COMMIT
    assert rollback.implementation_git_commit == IMPLEMENTATION_COMMIT
    assert promotion.config_sha256 == EXPECTED_CONFIG_SHA256
    assert rollback.config_sha256 == EXPECTED_CONFIG_SHA256
    assert tuple(check.name for check in promotion.checks) == REQUIRED_CHECKS
    assert tuple(check.name for check in rollback.checks) == REQUIRED_CHECKS
    assert promotion.action == "promote"
    assert rollback.action == "rollback"
    assert promotion.no_training_or_test_access is True
    assert rollback.no_training_or_test_access is True
