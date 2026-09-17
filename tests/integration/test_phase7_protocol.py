from __future__ import annotations

import hashlib
from pathlib import Path

from credit_risk.registry.contracts import EXPECTED_CONFIG_SHA256, load_registry_config

REPOSITORY_ROOT = Path(__file__).parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs/registry/phase7_v1.json"


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
