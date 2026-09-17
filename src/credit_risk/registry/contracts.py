"""Strict Phase 7 registry, approval, and evidence contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_REGISTRY_CONFIG_PATH = Path("configs/registry/phase7_v1.json")
DEFAULT_REGISTRY_ROOT = Path("experiment/registry/phase7_v1")
DEFAULT_DEPLOYMENT_ROOT = Path("experiment/deployments/phase7_v1")
DEFAULT_EVIDENCE_ROOT = Path("reports/registry/phase7_v1")
EXPECTED_CONFIG_SHA256 = "83a29f8e927e91336bfc39779f27c5bb27de91194b5e1b8889acfd95dafe925b"
EXPECTED_BUNDLE_MANIFEST_SHA256 = "df5ce6ce07b268f57fa3bf72c97cd32f8ebb66695d7157139942c91e46d7cd88"
EXPECTED_MODEL_SHA256 = "844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c"
PUBLISHED_FILES = (
    "summary.json",
    "registry-release-report.md",
    "promotion-checklist.md",
    "rollback-runbook.md",
    "evidence-manifest.json",
)
REQUIRED_CHECKS = (
    "Lint, type-check, and test",
    "Build, test, and scan API container",
)
RELEASE_REVISIONS = ("phase7_rev_001", "phase7_rev_002")
ALIASES = ("candidate", "champion", "rollback")


class RegistryContractError(RuntimeError):
    """Raised when a Phase 7 contract is missing, altered, or unsafe."""


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
    official_evidence_requires_clean_commit: Literal[True]
    manual_promotion_only: Literal[True]
    training: Literal["prohibited"]
    refitting: Literal["prohibited"]
    parameter_tuning: Literal["prohibited"]
    calibration_fitting: Literal["prohibited"]
    sealed_test_access: Literal["prohibited"]


class BundleContract(_FrozenModel):
    bundle_id: Literal["selected_v1"]
    model_id: Literal["catboost_fixed"]
    manifest_path: Literal["models/selected_v1/manifest.json"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_path: Literal["models/selected_v1/model.cbm"]
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    revisions_share_model_bytes: Literal[True]


class AliasContract(_FrozenModel):
    candidate: Literal["candidate"]
    champion: Literal["champion"]
    rollback: Literal["rollback"]


class RevisionContract(_FrozenModel):
    release_revision: Literal["phase7_rev_001", "phase7_rev_002"]
    initial_alias: Literal["champion", "candidate"]


class RegistryBackendContract(_FrozenModel):
    backend: Literal["sqlite"]
    artifact_store: Literal["content_addressed_filesystem"]
    mlflow_version: Literal["3.15.0"]
    registered_model_name: Literal["credit-risk-default"]
    aliases: AliasContract
    revisions: tuple[RevisionContract, RevisionContract]
    promotion_transition: Literal["candidate_to_champion_previous_champion_to_rollback"]
    rollback_transition: Literal["rollback_to_champion_displaced_champion_to_rollback"]
    single_writer: Literal[True]
    automatic_promotion: Literal[False]

    @model_validator(mode="after")
    def exact_revisions(self) -> RegistryBackendContract:
        pairs = tuple((item.release_revision, item.initial_alias) for item in self.revisions)
        if pairs != (("phase7_rev_001", "champion"), ("phase7_rev_002", "candidate")):
            raise ValueError("registry revisions differ from the reviewed release drill")
        return self


class ApprovalPolicy(_FrozenModel):
    digest_authentication_required: Literal[True]
    scope: Literal["local_portfolio_demo"]
    approver_role: Literal["project_owner"]
    required_checks: tuple[str, str]
    required_conclusion: Literal["success"]
    force_or_bypass: Literal[False]

    @model_validator(mode="after")
    def exact_checks(self) -> ApprovalPolicy:
        if self.required_checks != REQUIRED_CHECKS:
            raise ValueError("approval checks differ from the reviewed allowlist")
        return self


class DeploymentContract(_FrozenModel):
    environment_variable: Literal["CREDIT_RISK_DEPLOYMENT_ROOT"]
    layout: Literal["deployment_root/releases/release_revision/bundle_and_active_json"]
    activation: Literal["atomic_pointer"]
    api_restart_required: Literal[True]
    runtime_mlflow_dependency: Literal[False]
    allowed_root: Literal["experiment/deployments/phase7_v1"]


class SmokeTestContract(_FrozenModel):
    fixture_path: Literal["tests/fixtures/prediction_request.json"]
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_probability_six_decimals: float
    probability_absolute_tolerance: float
    expected_risk_band: Literal["standard"]
    revisions: tuple[Literal["phase7_rev_001"], Literal["phase7_rev_002"]]
    prediction_only: Literal[True]
    sealed_test_fixture: Literal[False]

    @model_validator(mode="after")
    def exact_smoke_contract(self) -> SmokeTestContract:
        if self.expected_probability_six_decimals != 0.190382:
            raise ValueError("registry smoke probability differs from the reviewed fixture")
        if self.probability_absolute_tolerance != 1e-6:
            raise ValueError("registry smoke tolerance differs from the reviewed contract")
        if self.revisions != RELEASE_REVISIONS:
            raise ValueError("registry smoke revisions differ from the release drill")
        return self


class PathContract(_FrozenModel):
    registry_root: Literal["experiment/registry/phase7_v1"]
    deployment_root: Literal["experiment/deployments/phase7_v1"]
    evidence_root: Literal["reports/registry/phase7_v1"]
    repository_relative_only: Literal[True]
    reject_symlinks: Literal[True]
    reject_overlaps: Literal[True]


class ImageScanContract(_FrozenModel):
    scanner: Literal["trivy"]
    severity: tuple[Literal["HIGH"], Literal["CRITICAL"]]
    ignore_unfixed: Literal[True]
    exit_code_on_finding: Literal[1]
    waiver_supported: Literal[False]
    sbom_format: Literal["cyclonedx"]


class EvidenceContract(_FrozenModel):
    published_files: tuple[str, ...]
    timestamps: Literal["prohibited"]
    local_paths: Literal["prohibited"]
    row_level_data: Literal["prohibited"]

    @model_validator(mode="after")
    def exact_outputs(self) -> EvidenceContract:
        if self.published_files != PUBLISHED_FILES:
            raise ValueError("registry evidence outputs differ from the reviewed allowlist")
        return self


class RegistryConfig(_FrozenModel):
    schema_version: Literal["1.0.0"]
    protocol_id: Literal["phase7_v1"]
    status: Literal["frozen_before_implementation"]
    purpose: Literal["governed_local_registry_promotion_deployment_and_rollback"]
    governance: GovernanceContract
    source_evidence: dict[str, ArtifactReference]
    bundle: BundleContract
    registry: RegistryBackendContract
    approvals: ApprovalPolicy
    deployment: DeploymentContract
    smoke_test: SmokeTestContract
    paths: PathContract
    image_scan: ImageScanContract
    evidence: EvidenceContract
    prohibitions: tuple[str, ...]

    @model_validator(mode="after")
    def exact_contract(self) -> RegistryConfig:
        if set(self.source_evidence) != {
            "release_a_manifest",
            "phase5_manifest",
            "phase6_config",
            "phase6_manifest",
        }:
            raise ValueError("registry source evidence differs from the reviewed allowlist")
        required = {
            "model_fitting",
            "parameter_tuning",
            "calibration_fitting",
            "bootstrap_generation",
            "final_test_loading",
            "sealed_test_scoring",
            "automatic_promotion",
            "arbitrary_model_registration",
            "scan_waiver",
            "production_readiness_claim",
        }
        if set(self.prohibitions) != required or len(self.prohibitions) != len(required):
            raise ValueError("registry prohibitions differ from the reviewed contract")
        return self


class CheckResult(_FrozenModel):
    name: str = Field(min_length=1)
    conclusion: Literal["success"]


class ReleaseApproval(_FrozenModel):
    schema_version: Literal["1.0.0"]
    protocol_id: Literal["phase7_v1"]
    decision_id: Literal["phase7_promotion_approval", "phase7_rollback_approval"]
    action: Literal["promote", "rollback"]
    decision: Literal["approved"]
    scope: Literal["local_portfolio_demo"]
    approved_by_role: Literal["project_owner"]
    implementation_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registered_model_name: Literal["credit-risk-default"]
    source_revision: Literal["phase7_rev_001", "phase7_rev_002"]
    target_revision: Literal["phase7_rev_001", "phase7_rev_002"]
    checks: tuple[CheckResult, CheckResult]
    model_bytes_unchanged: Literal[True]
    no_training_or_test_access: Literal[True]
    limitations_acknowledged: Literal[True]

    @model_validator(mode="after")
    def coherent_decision(self) -> ReleaseApproval:
        expected_id = f"phase7_{'promotion' if self.action == 'promote' else 'rollback'}_approval"
        if self.decision_id != expected_id:
            raise ValueError("approval decision ID does not match its action")
        if tuple(check.name for check in self.checks) != REQUIRED_CHECKS:
            raise ValueError("approval check names differ from the reviewed allowlist")
        expected = (
            ("phase7_rev_001", "phase7_rev_002")
            if self.action == "promote"
            else ("phase7_rev_002", "phase7_rev_001")
        )
        if (self.source_revision, self.target_revision) != expected:
            raise ValueError("approval revisions differ from the reviewed transition")
        return self


def load_registry_config(path: str | Path = DEFAULT_REGISTRY_CONFIG_PATH) -> RegistryConfig:
    """Load the immutable Phase 7 registry protocol."""

    candidate = Path(path)
    try:
        content = candidate.read_bytes()
        observed = hashlib.sha256(content).hexdigest()
        if observed != EXPECTED_CONFIG_SHA256:
            raise RegistryContractError(
                "Phase 7 configuration digest mismatch: "
                f"expected={EXPECTED_CONFIG_SHA256}, observed={observed}"
            )
        return RegistryConfig.model_validate_json(content, strict=True)
    except RegistryContractError:
        raise
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise RegistryContractError(f"Invalid Phase 7 registry configuration: {error}") from error


def load_approval(path: str | Path, expected_sha256: str) -> ReleaseApproval:
    """Authenticate and parse a reviewed release decision."""

    _validate_sha256(expected_sha256, "Expected approval digest")
    candidate = Path(path)
    try:
        content = candidate.read_bytes()
    except OSError as error:
        raise RegistryContractError(f"Unable to read release approval: {error}") from error
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected_sha256:
        raise RegistryContractError(
            f"Release approval digest mismatch: expected={expected_sha256}, observed={observed}"
        )
    try:
        return ReleaseApproval.model_validate_json(content, strict=True)
    except (UnicodeError, ValidationError, ValueError) as error:
        raise RegistryContractError(f"Invalid release approval: {error}") from error


def config_sha256(path: str | Path = DEFAULT_REGISTRY_CONFIG_PATH) -> str:
    """Return the byte-level Phase 7 configuration digest."""

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise RegistryContractError(f"Unable to hash Phase 7 configuration: {error}") from error


def _safe_relative_path(value: str) -> str:
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
        raise ValueError("registry paths must be safe repository-relative paths")
    return value


def _validate_sha256(value: str, description: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RegistryContractError(f"{description} must be a lowercase SHA-256 digest.")
