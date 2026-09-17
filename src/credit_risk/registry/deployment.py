"""Runtime-safe deployment pointer validation with no MLflow dependency."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from credit_risk.registry.contracts import (
    EXPECTED_BUNDLE_MANIFEST_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_MODEL_SHA256,
    RELEASE_REVISIONS,
)


class DeploymentResolutionError(RuntimeError):
    """Raised when an active deployment pointer is unsafe or incompatible."""


class ActiveDeployment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())

    schema_version: Literal["1.0.0"]
    protocol_id: Literal["phase7_v1"]
    registered_model_name: Literal["credit-risk-default"]
    alias: Literal["champion"]
    registry_version: int = Field(ge=1)
    release_revision: Literal["phase7_rev_001", "phase7_rev_002"]
    bundle_relative_path: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def safe_bundle_path(self) -> ActiveDeployment:
        expected = f"releases/{self.release_revision}/bundle"
        if self.bundle_relative_path != expected:
            raise ValueError("active deployment bundle path differs from its release revision")
        expected_version = RELEASE_REVISIONS.index(self.release_revision) + 1
        if self.registry_version != expected_version:
            raise ValueError("active deployment version differs from its release revision")
        if self.config_sha256 != EXPECTED_CONFIG_SHA256:
            raise ValueError("active deployment configuration digest is not reviewed")
        if self.bundle_manifest_sha256 != EXPECTED_BUNDLE_MANIFEST_SHA256:
            raise ValueError("active deployment manifest digest is not reviewed")
        if self.model_sha256 != EXPECTED_MODEL_SHA256:
            raise ValueError("active deployment model digest is not reviewed")
        return self


def resolve_active_bundle(deployment_root: str | Path) -> Path:
    """Resolve and authenticate the exact bundle selected by ``active.json``."""

    _, bundle = _load_active_deployment(deployment_root)
    return bundle


def load_active_deployment(deployment_root: str | Path) -> ActiveDeployment:
    """Validate the complete deployment tree and return its active pointer."""

    pointer, _ = _load_active_deployment(deployment_root)
    return pointer


def _load_active_deployment(deployment_root: str | Path) -> tuple[ActiveDeployment, Path]:
    """Resolve the pointer and enforce the complete deployment allowlist."""

    root = Path(deployment_root)
    if root.is_symlink():
        raise DeploymentResolutionError("Deployment root must not be a symlink.")
    try:
        root_resolved = root.resolve(strict=True)
        pointer_path = root_resolved / "active.json"
        if pointer_path.is_symlink() or not pointer_path.is_file():
            raise DeploymentResolutionError("Active deployment pointer is missing or unsafe.")
        pointer = ActiveDeployment.model_validate_json(pointer_path.read_bytes(), strict=True)
        bundle = (root_resolved / pointer.bundle_relative_path).resolve(strict=True)
    except DeploymentResolutionError:
        raise
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise DeploymentResolutionError(f"Invalid active deployment pointer: {error}") from error
    if root_resolved not in bundle.parents or bundle.is_symlink() or not bundle.is_dir():
        raise DeploymentResolutionError("Active deployment bundle escapes its governed root.")
    _validate_deployment_tree(root_resolved, pointer)
    return pointer, bundle


def _validate_deployment_tree(root: Path, pointer: ActiveDeployment) -> None:
    expected_root = {"active.json", "releases"}
    try:
        root_entries = tuple(root.iterdir())
    except OSError as error:
        raise DeploymentResolutionError(
            f"Unable to inspect active deployment root: {error}"
        ) from error
    if {entry.name for entry in root_entries} != expected_root or any(
        entry.is_symlink() for entry in root_entries
    ):
        raise DeploymentResolutionError("Active deployment root violates its file allowlist.")
    releases = root / "releases"
    if not releases.is_dir():
        raise DeploymentResolutionError("Active deployment releases directory is missing.")
    try:
        release_entries = tuple(releases.iterdir())
    except OSError as error:
        raise DeploymentResolutionError(
            f"Unable to inspect deployment releases: {error}"
        ) from error
    allowed_revisions = set(RELEASE_REVISIONS)
    observed_revisions = {entry.name for entry in release_entries}
    if (
        not observed_revisions
        or pointer.release_revision not in observed_revisions
        or not observed_revisions.issubset(allowed_revisions)
        or any(entry.is_symlink() or not entry.is_dir() for entry in release_entries)
    ):
        raise DeploymentResolutionError("Deployment releases violate the reviewed allowlist.")
    for revision in observed_revisions:
        release = releases / revision
        contents = tuple(release.iterdir())
        if (
            {entry.name for entry in contents} != {"bundle"}
            or contents[0].is_symlink()
            or not contents[0].is_dir()
        ):
            raise DeploymentResolutionError(
                f"Deployment release {revision} violates its directory allowlist."
            )
        _validate_bundle_files(contents[0])


def _validate_bundle_files(bundle: Path) -> None:
    expected_files = {"manifest.json", "model.cbm"}
    try:
        entries = tuple(bundle.iterdir())
    except OSError as error:
        raise DeploymentResolutionError(
            f"Unable to inspect active deployment bundle: {error}"
        ) from error
    if {entry.name for entry in entries} != expected_files or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise DeploymentResolutionError("Active deployment bundle violates its file allowlist.")
    expected = {
        "manifest.json": EXPECTED_BUNDLE_MANIFEST_SHA256,
        "model.cbm": EXPECTED_MODEL_SHA256,
    }
    for filename, digest in expected.items():
        try:
            observed = hashlib.sha256((bundle / filename).read_bytes()).hexdigest()
        except OSError as error:
            raise DeploymentResolutionError(f"Unable to hash active bundle: {error}") from error
        if observed != digest:
            raise DeploymentResolutionError(
                f"Active deployment digest mismatch for {filename}: "
                f"expected={digest}, observed={observed}"
            )
