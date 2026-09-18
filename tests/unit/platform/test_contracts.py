from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from credit_risk.platform.contracts import (
    EXPECTED_CONFIG_SHA256,
    PlatformConfig,
    PlatformContractError,
    config_sha256,
    load_platform_config,
)

CONFIG = Path("configs/platform/phase8_v1.json")


def test_phase8_config_is_complete_and_digest_protected() -> None:
    content = CONFIG.read_bytes()
    assert hashlib.sha256(content).hexdigest() == EXPECTED_CONFIG_SHA256
    config = load_platform_config(CONFIG)

    assert config.governance.bootstrap_is_not_database_migration is True
    assert config.governance.training == "prohibited"
    assert config.governance.sealed_test_access == "prohibited"
    assert config.services.api.runtime_mlflow_dependency is False
    assert config.services.ui.direct_model_access is False
    assert config.registry_bootstrap.aliases == {"champion": "1", "rollback": "2"}
    assert config.registry_bootstrap.writer_lock.kind == "postgres_advisory"
    assert config.registry_bootstrap.writer_lock.key == -4653285090134190835
    assert config.registry_bootstrap.writer_lock.contention_policy == "fail_fast"
    assert config.security.database_not_host_published is True
    assert config.security.object_api_not_host_published is True


def test_phase8_config_rejects_any_byte_change(tmp_path: Path) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["services"]["api"]["host_port"] = 8081
    altered = tmp_path / "phase8.json"
    altered.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(PlatformContractError, match="digest mismatch"):
        load_platform_config(altered)


def test_all_bound_sources_and_bundle_files_match() -> None:
    config = load_platform_config(CONFIG)
    for reference in config.source_evidence.values():
        assert hashlib.sha256(Path(reference.path).read_bytes()).hexdigest() == reference.sha256
    assert (
        hashlib.sha256(Path(config.bundle.manifest_path).read_bytes()).hexdigest()
        == config.bundle.manifest_sha256
    )
    assert (
        hashlib.sha256(Path(config.bundle.model_path).read_bytes()).hexdigest()
        == config.bundle.model_sha256
    )


def _config_value() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["source_evidence"].pop("phase7_config"),
            "source-evidence allowlist",
        ),
        (
            lambda value: value["bundle"].__setitem__("model_sha256", "0" * 64),
            "unreviewed model",
        ),
        (
            lambda value: value["persistence"].__setitem__("named_volumes", ["foreign"]),
            "volume allowlist",
        ),
        (
            lambda value: value.__setitem__("prohibitions", ["model_fitting"]),
            "prohibitions",
        ),
        (
            lambda value: value["images"]["postgres"].__setitem__("digest", f"sha256:{'0' * 64}"),
            "PostgreSQL image",
        ),
        (
            lambda value: value["services"]["postgres"].__setitem__("internal_port", 5433),
            "storage ports",
        ),
        (
            lambda value: value["registry_bootstrap"].__setitem__(
                "artifact_prefix", "registry/foreign/bundle"
            ),
            "content addressed",
        ),
        (
            lambda value: value["registry_bootstrap"].__setitem__(
                "aliases", {"champion": "2", "rollback": "1"}
            ),
            "approved Phase 7 final state",
        ),
        (
            lambda value: value["registry_bootstrap"]["writer_lock"].__setitem__("key", 1),
            "literal_error",
        ),
        (
            lambda value: value["source_evidence"]["phase7_config"].__setitem__(
                "path", "../unsafe.json"
            ),
            "safe repository-relative paths",
        ),
    ],
)
def test_semantic_contract_rejects_frozen_boundary_changes(mutate, message: str) -> None:
    value = deepcopy(_config_value())
    mutate(value)

    with pytest.raises(ValueError, match=message):
        PlatformConfig.model_validate_json(json.dumps(value), strict=True)


def test_config_helpers_report_missing_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(PlatformContractError, match="Invalid Phase 8"):
        load_platform_config(missing)
    with pytest.raises(PlatformContractError, match="Unable to hash"):
        config_sha256(missing)
