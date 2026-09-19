"""Strict contracts for external binary-artifact distribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_DISTRIBUTION_LOCK = Path("configs/artifacts/hf_distribution_v2.lock.json")
SELECTED_MANIFEST = Path("models/selected_v1/manifest.json")
EXPECTED_DISTRIBUTION_LOCK_SHA256 = (
    "d14be69fd454a0461a5baf1ee51a587f2a35821d808d0f81565c5afb6311586f"
)


class ArtifactContractError(RuntimeError):
    """Raised when distribution metadata is missing, unsafe, or inconsistent."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DigestReference(StrictModel):
    manifest_path: str
    json_pointer: str = Field(pattern=r"^/(?:[^/~]|~[01])+(?:/(?:[^/~]|~[01])+)*$")

    @model_validator(mode="after")
    def safe_manifest_path(self) -> DigestReference:
        _validate_relative_path(self.manifest_path, "manifest path")
        return self


class ArtifactRecord(StrictModel):
    artifact_id: Literal["selected_model"]
    remote_path: str
    local_path: str
    size_bytes: int = Field(gt=0)
    serialization: Literal["catboost_cbm"]
    trust_classification: Literal["digest_authenticated"]
    digest_reference: DigestReference

    @model_validator(mode="after")
    def safe_and_compatible(self) -> ArtifactRecord:
        _validate_remote_path(self.remote_path)
        _validate_relative_path(self.local_path, "local path")
        expected = (
            "models/selected_v1/model.cbm",
            "selected_v1/model.cbm",
            "catboost_cbm",
            "digest_authenticated",
            SELECTED_MANIFEST.as_posix(),
            "/model_sha256",
        )
        observed = (
            self.local_path,
            self.remote_path,
            self.serialization,
            self.trust_classification,
            self.digest_reference.manifest_path,
            self.digest_reference.json_pointer,
        )
        if observed != expected:
            raise ValueError(f"artifact {self.artifact_id!r} differs from its approved mapping")
        return self


class RepositoryContract(StrictModel):
    provider: Literal["huggingface_hub"]
    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
    repo_type: Literal["model"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    public: Literal[True]


class DistributionLock(StrictModel):
    schema_version: Literal["2.0.0"]
    distribution_id: Literal["hf_distribution_v2"]
    repository: RepositoryContract
    artifacts: tuple[ArtifactRecord, ...]

    @model_validator(mode="after")
    def exact_inventory(self) -> DistributionLock:
        if len(self.artifacts) != 1 or self.artifacts[0].artifact_id != "selected_model":
            raise ValueError("distribution lock must contain exactly the selected model")
        return self


def load_distribution_lock(path: str | Path) -> DistributionLock:
    return _load_json_contract(path, DistributionLock, "artifact distribution lock")


def resolve_digest_reference(repository: Path, reference: DigestReference) -> str:
    manifest = safe_repository_path(repository, reference.manifest_path, must_exist=True)
    try:
        payload: object = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactContractError(
            f"Unable to read digest authority {reference.manifest_path!r}: {error}"
        ) from error
    current = payload
    for raw_token in reference.json_pointer.removeprefix("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            raise ArtifactContractError(
                f"Digest pointer {reference.json_pointer!r} does not exist in "
                f"{reference.manifest_path!r}."
            )
        current = current[token]
    if (
        not isinstance(current, str)
        or len(current) != 64
        or any(character not in "0123456789abcdef" for character in current)
    ):
        raise ArtifactContractError(
            f"Digest pointer {reference.json_pointer!r} does not resolve to lowercase SHA-256."
        )
    return current


def safe_repository_path(repository: Path, relative: str | Path, *, must_exist: bool) -> Path:
    root = repository.resolve(strict=True)
    candidate_value = str(relative).replace("\\", "/")
    _validate_relative_path(candidate_value, "repository path")
    candidate = root.joinpath(*PurePosixPath(candidate_value).parts)
    _reject_symlink_components(root, candidate)
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as error:
        raise ArtifactContractError(
            f"Unable to resolve repository path {relative!r}: {error}"
        ) from error
    if not resolved.is_relative_to(root):
        raise ArtifactContractError(f"Repository path escapes the project root: {relative!r}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ArtifactContractError(f"Unable to hash artifact {path}: {error}") from error
    return digest.hexdigest()


def _load_json_contract(path: str | Path, model: type[StrictModel], label: str):
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise ArtifactContractError(f"Unable to read {label} {path!s}: {error}") from error
    try:
        return model.model_validate_json(payload)
    except ValidationError as error:
        raise ArtifactContractError(f"Invalid {label}: {error}") from error


def _validate_remote_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"remote path must be a safe relative POSIX path: {value!r}")


def _validate_relative_path(value: str, label: str) -> None:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or len(path.parts) == 0
        or any(part in {"", ".", ".."} for part in path.parts)
        or (len(normalized) >= 2 and normalized[1] == ":")
    ):
        raise ValueError(f"{label} must be repository-relative and traversal-free: {value!r}")


def _reject_symlink_components(root: Path, candidate: Path) -> None:
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ArtifactContractError(f"Artifact path contains a symlink: {current}")
