"""Strict Phase 8 local-platform configuration contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_PLATFORM_CONFIG_PATH = Path("configs/platform/phase8_v1.json")
EXPECTED_CONFIG_SHA256 = "5de84bdb4f176b49f965f4983d69eb7b34e5ce763f04a855f200eb899a6529a6"
EXPECTED_POSTGRES_IMAGE = (
    "postgres:16.15-bookworm@"
    "sha256:bb3e1a57e5407e0a5280b4211980a5e537f4abd234a87014ac979849a78dd825"
)
EXPECTED_MINIO_IMAGE = (
    "quay.io/minio/minio:RELEASE.2025-07-23T15-54-02Z@"
    "sha256:d249d1fb6966de4d8ad26c04754b545205ff15a62e4fd19ebd0f26fa5baacbc0"
)


class PlatformContractError(RuntimeError):
    """Raised when the frozen Phase 8 platform contract is altered or unsafe."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())


class ArtifactReference(_FrozenModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def safe_path(self) -> ArtifactReference:
        _safe_relative_path(self.path)
        return self


class GovernanceContract(_FrozenModel):
    protocol_base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    phase7_state_is_immutable: Literal[True]
    bootstrap_is_not_database_migration: Literal[True]
    training: Literal["prohibited"]
    refitting: Literal["prohibited"]
    sealed_test_access: Literal["prohibited"]


class BundleContract(_FrozenModel):
    bundle_id: Literal["selected_v1"]
    model_id: Literal["catboost_fixed"]
    manifest_path: Literal["models/selected_v1/manifest.json"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_path: Literal["models/selected_v1/model.cbm"]
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ImageContract(_FrozenModel):
    reference: str
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ImageContracts(_FrozenModel):
    python: ImageContract
    postgres: ImageContract
    minio: ImageContract

    @model_validator(mode="after")
    def exact_external_images(self) -> ImageContracts:
        if f"{self.postgres.reference}@{self.postgres.digest}" != EXPECTED_POSTGRES_IMAGE:
            raise ValueError("PostgreSQL image differs from the reviewed digest")
        if f"{self.minio.reference}@{self.minio.digest}" != EXPECTED_MINIO_IMAGE:
            raise ValueError("MinIO image differs from the reviewed digest")
        return self


class PortVolumeService(_FrozenModel):
    internal_port: int = Field(gt=0, le=65535)
    persistent_volume: str


class MinioService(PortVolumeService):
    console_port: Literal[9001]


class MlflowService(_FrozenModel):
    version: Literal["3.15.0"]
    host_port: Literal[5000]
    workers: Literal[1]


class ApiService(_FrozenModel):
    host_port: Literal[8080]
    runtime_mlflow_dependency: Literal[False]


class UiService(_FrozenModel):
    host_port: Literal[8501]
    direct_model_access: Literal[False]


class ServiceContracts(_FrozenModel):
    postgres: PortVolumeService
    minio: MinioService
    mlflow: MlflowService
    api: ApiService
    ui: UiService

    @model_validator(mode="after")
    def exact_internal_ports(self) -> ServiceContracts:
        if self.postgres.internal_port != 5432 or self.minio.internal_port != 9000:
            raise ValueError("Platform storage ports differ from the reviewed contract")
        return self


class RegistryVersion(_FrozenModel):
    registry_version: Literal["1", "2"]
    release_revision: Literal["phase7_rev_001", "phase7_rev_002"]


class WriterLockContract(_FrozenModel):
    kind: Literal["postgres_advisory"]
    key: Literal[-4653285090134190835]
    contention_policy: Literal["fail_fast"]


class RegistryBootstrapContract(_FrozenModel):
    registered_model_name: Literal["credit-risk-default"]
    artifact_bucket: Literal["credit-risk-mlflow"]
    artifact_prefix: str
    versions: tuple[RegistryVersion, RegistryVersion]
    aliases: dict[str, str]
    active_revision: Literal["phase7_rev_001"]
    active_registry_version: Literal[1]
    rollback_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rollback_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    writer_lock: WriterLockContract
    idempotent_only_after_complete_verification: Literal[True]

    @model_validator(mode="after")
    def exact_release_state(self) -> RegistryBootstrapContract:
        expected_versions = (("1", "phase7_rev_001"), ("2", "phase7_rev_002"))
        observed = tuple((item.registry_version, item.release_revision) for item in self.versions)
        if observed != expected_versions or self.aliases != {"champion": "1", "rollback": "2"}:
            raise ValueError(
                "Platform registry state differs from the approved Phase 7 final state"
            )
        expected_prefix = f"registry/{EXPECTED_MODEL_SHA256}/bundle"
        if self.artifact_prefix != expected_prefix:
            raise ValueError("Platform artifact prefix is not content addressed")
        return self


class EnvironmentContract(_FrozenModel):
    required_secret_variables: tuple[str, ...]
    required_identity_variables: tuple[str, ...]
    example_file: Literal[".env.example"]
    committed_real_secrets: Literal[False]


class PersistenceContract(_FrozenModel):
    named_volumes: tuple[str, ...]
    destructive_reset_command_documented: Literal[False]
    restart_must_preserve_registry_aliases: Literal[True]
    restart_must_preserve_artifact_bytes: Literal[True]


class SecurityContract(_FrozenModel):
    pin_external_images_by_digest: Literal[True]
    api_has_no_mlflow_dependency: Literal[True]
    deployment_mount_read_only: Literal[True]
    database_not_host_published: Literal[True]
    object_api_not_host_published: Literal[True]
    fixable_high_critical_scan_blocking: Literal[True]
    sbom_required: Literal[True]


class PlatformConfig(_FrozenModel):
    schema_version: Literal["1.0.0"]
    protocol_id: Literal["phase8_v1"]
    status: Literal["frozen_before_platform_execution"]
    purpose: Literal["zero_cost_persistent_local_mlops_platform"]
    governance: GovernanceContract
    source_evidence: dict[str, ArtifactReference]
    bundle: BundleContract
    images: ImageContracts
    services: ServiceContracts
    registry_bootstrap: RegistryBootstrapContract
    environment: EnvironmentContract
    persistence: PersistenceContract
    security: SecurityContract
    prohibitions: tuple[str, ...]

    @model_validator(mode="after")
    def exact_contract(self) -> PlatformConfig:
        if set(self.source_evidence) != {
            "phase7_config",
            "phase7_evidence_manifest",
            "phase7_rollback_approval",
        }:
            raise ValueError("Platform source-evidence allowlist differs from the protocol")
        if self.bundle.model_sha256 != EXPECTED_MODEL_SHA256:
            raise ValueError("Platform contract references an unreviewed model")
        expected_volumes = (
            "phase8_postgres_data",
            "phase8_minio_data",
            "phase8_deployment",
        )
        if self.persistence.named_volumes != expected_volumes:
            raise ValueError("Platform volume allowlist differs from the protocol")
        expected_prohibitions = {
            "phase7_database_mutation",
            "model_fitting",
            "model_refitting",
            "parameter_tuning",
            "sealed_test_loading",
            "unversioned_external_images",
            "committed_runtime_secrets",
            "mlflow_dependency_in_api_image",
        }
        if set(self.prohibitions) != expected_prohibitions:
            raise ValueError("Platform prohibitions differ from the reviewed contract")
        return self


EXPECTED_MODEL_SHA256 = "844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c"


def load_platform_config(path: str | Path = DEFAULT_PLATFORM_CONFIG_PATH) -> PlatformConfig:
    """Authenticate and load the immutable Phase 8 contract."""

    candidate = Path(path)
    try:
        content = candidate.read_bytes()
        observed = hashlib.sha256(content).hexdigest()
        if observed != EXPECTED_CONFIG_SHA256:
            raise PlatformContractError(
                "Phase 8 configuration digest mismatch: "
                f"expected={EXPECTED_CONFIG_SHA256}, observed={observed}"
            )
        return PlatformConfig.model_validate_json(content, strict=True)
    except PlatformContractError:
        raise
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise PlatformContractError(f"Invalid Phase 8 platform configuration: {error}") from error


def config_sha256(path: str | Path = DEFAULT_PLATFORM_CONFIG_PATH) -> str:
    """Return the byte-level Phase 8 configuration digest."""

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise PlatformContractError(f"Unable to hash Phase 8 configuration: {error}") from error


def _safe_relative_path(value: str) -> None:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or "\\" in value
    ):
        raise ValueError("platform paths must be safe repository-relative paths")
