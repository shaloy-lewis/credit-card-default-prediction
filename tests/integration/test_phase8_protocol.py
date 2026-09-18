"""Static integrity checks for the frozen Phase 8 platform prerequisite layer."""

from __future__ import annotations

import hashlib
from pathlib import Path

from credit_risk.platform.contracts import EXPECTED_CONFIG_SHA256, load_platform_config

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_phase8_contract_and_compose_preserve_security_boundaries() -> None:
    config_path = REPOSITORY_ROOT / "configs/platform/phase8_v1.json"
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == EXPECTED_CONFIG_SHA256
    config = load_platform_config(config_path)
    compose = (REPOSITORY_ROOT / "docker-compose.platform.yml").read_text(encoding="utf-8")

    assert config.security.database_not_host_published is True
    assert config.security.object_api_not_host_published is True
    assert config.registry_bootstrap.writer_lock.kind == "postgres_advisory"
    assert config.registry_bootstrap.writer_lock.key == -4653285090134190835
    assert config.registry_bootstrap.writer_lock.contention_policy == "fail_fast"
    assert "127.0.0.1:5432:5432" not in compose
    assert "127.0.0.1:9000:9000" not in compose
    assert "phase8_deployment:/app/deployment:ro" in compose
    assert 'user: "0:0"' in compose
    assert "read_only: true" in compose
    assert "phase8_postgres_data:" in compose
    assert "phase8_minio_data:" in compose
    assert "phase8_deployment:" in compose
    assert f"{config.images.postgres.reference}@{config.images.postgres.digest}" in compose
    assert f"{config.images.minio.reference}@{config.images.minio.digest}" in compose
    assert "credit_risk.platform.mlflow_server" in compose
    assert "--backend-store-uri" not in compose
    assert "postgresql+psycopg2://${POSTGRES_USER}" not in compose


def test_platform_images_and_environment_do_not_leak_runtime_state() -> None:
    dockerfile = (REPOSITORY_ROOT / "Dockerfile.platform").read_text(encoding="utf-8")
    ui_dockerfile = (REPOSITORY_ROOT / "Dockerfile.demo").read_text(encoding="utf-8")
    api_dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    example = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
    ignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
    docker_ignore = (REPOSITORY_ROOT / ".dockerignore").read_text(encoding="utf-8")
    lockfile = (REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8")

    assert "--extra platform" in dockerfile
    assert "libpcre2-8-0=10.42-1+deb12u1" in dockerfile
    assert "libpcre2-8-0=10.42-1+deb12u1" in ui_dockerfile
    assert "USER app" in dockerfile
    assert "--extra platform" not in api_dockerfile
    assert "mlflow" not in api_dockerfile.lower()
    assert ".env" in ignore
    assert ".env" in docker_ignore.splitlines()
    assert "change-me" in example
    assert "MLFLOW_ARTIFACT_BUCKET=credit-risk-mlflow" in example
    assert 'name = "cryptography"\nversion = "50.0.0"' in lockfile
    assert 'name = "gitpython"\nversion = "3.1.59"' in lockfile


def test_phase7_and_selected_bundle_sources_remain_byte_identical() -> None:
    config = load_platform_config(REPOSITORY_ROOT / "configs/platform/phase8_v1.json")
    for reference in config.source_evidence.values():
        source = REPOSITORY_ROOT / reference.path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == reference.sha256
    assert (
        hashlib.sha256((REPOSITORY_ROOT / config.bundle.manifest_path).read_bytes()).hexdigest()
        == config.bundle.manifest_sha256
    )
    assert (
        hashlib.sha256((REPOSITORY_ROOT / config.bundle.model_path).read_bytes()).hexdigest()
        == config.bundle.model_sha256
    )
