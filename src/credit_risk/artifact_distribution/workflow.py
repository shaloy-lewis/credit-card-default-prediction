"""Fail-closed artifact acquisition, verification, and publication."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from credit_risk.artifact_distribution.contracts import (
    SELECTED_MANIFEST,
    ArtifactContractError,
    ArtifactRecord,
    DigestReference,
    DistributionLock,
    RepositoryContract,
    load_distribution_lock,
    resolve_digest_reference,
    safe_repository_path,
    sha256_file,
)
from credit_risk.artifact_distribution.transport import (
    ArtifactTransportError,
    HuggingFaceTransport,
)


class ArtifactDistributionError(RuntimeError):
    """Raised when a distribution operation cannot safely complete."""


class DownloadTransport(Protocol):
    def download(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        remote_path: str,
        cache_dir: Path | None,
        offline: bool,
    ) -> Path: ...


class PublishTransport(DownloadTransport, Protocol):
    def publish(self, *, repo_id: str, files: dict[str, Path], commit_message: str) -> str: ...


@dataclass(frozen=True, slots=True)
class ArtifactOperationResult:
    materialized: tuple[Path, ...]
    reused: tuple[Path, ...]
    revision: str | None = None


def pull_artifacts(
    *,
    config_path: str | Path,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    repository_root: str | Path = ".",
    transport: DownloadTransport | None = None,
) -> ArtifactOperationResult:
    repository, config = _context(repository_root, config_path)
    records = config.artifacts
    client = transport or HuggingFaceTransport()
    materialized: list[Path] = []
    reused: list[Path] = []
    for record in records:
        destination, expected = _artifact_location(repository, record, must_exist=False)
        if destination.exists():
            if _matches(destination, record.size_bytes, expected):
                reused.append(destination)
                continue
            try:
                _quarantine(repository, destination)
            except (ArtifactContractError, OSError) as error:
                raise ArtifactDistributionError(
                    f"Unable to quarantine invalid artifact {record.local_path!r}: {error}"
                ) from error
            raise ArtifactDistributionError(
                f"Existing artifact {record.local_path!r} failed validation and was quarantined; "
                "rerun the pull command to retrieve the reviewed bytes."
            )
        try:
            _quarantine_partials(repository, destination, record.artifact_id)
        except (ArtifactContractError, OSError) as error:
            raise ArtifactDistributionError(
                f"Unable to prepare artifact destination {record.local_path!r}: {error}"
            ) from error
        try:
            downloaded = client.download(
                repo_id=config.repository.repo_id,
                repo_type=config.repository.repo_type,
                revision=config.repository.revision,
                remote_path=record.remote_path,
                cache_dir=Path(cache_dir) if cache_dir is not None else None,
                offline=offline,
            )
        except ArtifactTransportError as error:
            raise ArtifactDistributionError(str(error)) from error
        if not _matches_source(downloaded, record.size_bytes, expected):
            raise ArtifactDistributionError(
                f"Downloaded artifact {record.remote_path!r} does not match its Git-tracked "
                "size and SHA-256 contract."
            )
        try:
            _atomic_materialize(downloaded, destination, record, expected)
        except ArtifactDistributionError:
            raise
        except OSError as error:
            raise ArtifactDistributionError(
                f"Unable to materialize artifact {record.local_path!r}: {error}"
            ) from error
        materialized.append(destination)
    verify_artifacts(
        config_path=config_path,
        repository_root=repository,
    )
    return ArtifactOperationResult(
        materialized=tuple(materialized),
        reused=tuple(reused),
        revision=config.repository.revision,
    )


def verify_artifacts(
    *,
    config_path: str | Path,
    repository_root: str | Path = ".",
) -> ArtifactOperationResult:
    repository, config = _context(repository_root, config_path)
    verified: list[Path] = []
    for record in config.artifacts:
        path, expected = _artifact_location(repository, record, must_exist=True)
        if not _matches(path, record.size_bytes, expected):
            raise ArtifactDistributionError(
                f"Artifact {record.local_path!r} does not match its reviewed contract."
            )
        verified.append(path)
    selected_root = repository / "models" / "selected_v1"
    _require_allowlist(selected_root, {"manifest.json", "model.cbm"}, "selected bundle")
    return ArtifactOperationResult(materialized=(), reused=tuple(verified))


def publish_artifacts(
    *,
    repo_id: str,
    source_root: str | Path = ".",
    lock_output: str | Path = "experiment/artifacts/hf_distribution_v2.candidate.json",
    transport: PublishTransport | None = None,
) -> ArtifactOperationResult:
    try:
        repository = Path(source_root).resolve(strict=True)
        records = _publication_records(repository)
        card = safe_repository_path(
            repository, "docs/artifacts/hugging-face-repository-card.md", must_exist=True
        )
    except (ArtifactContractError, OSError, ValueError) as error:
        raise ArtifactDistributionError(
            f"Artifact publication preflight failed: {error}"
        ) from error
    try:
        output = _preflight_candidate_lock_output(repository, lock_output)
    except (ArtifactContractError, OSError, ValueError) as error:
        raise ArtifactDistributionError(
            f"Candidate lock output preflight failed: {error}"
        ) from error
    files: dict[str, Path] = {}
    expected_digests: dict[str, str] = {}
    for record in records:
        artifact_path, expected = _artifact_location(repository, record, must_exist=True)
        files[record.remote_path] = artifact_path
        expected_digests[record.artifact_id] = expected
    files["README.md"] = card
    for record in records:
        expected = expected_digests[record.artifact_id]
        if not _matches(files[record.remote_path], record.size_bytes, expected):
            raise ArtifactDistributionError(
                f"Local artifact {record.local_path!r} does not match reviewed metadata."
            )
    client = transport or HuggingFaceTransport()
    try:
        revision = client.publish(
            repo_id=repo_id,
            files=files,
            commit_message="Publish reviewed credit-risk selected artifact v2",
        )
    except ArtifactTransportError as error:
        raise ArtifactDistributionError(str(error)) from error
    try:
        lock = DistributionLock(
            schema_version="2.0.0",
            distribution_id="hf_distribution_v2",
            repository=RepositoryContract(
                provider="huggingface_hub",
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                public=True,
            ),
            artifacts=_publication_records(repository),
        )
    except (ArtifactContractError, OSError, ValueError) as error:
        raise ArtifactDistributionError(
            f"Published revision failed its distribution contract: {error}"
        ) from error
    for record in lock.artifacts:
        _, expected = _artifact_location(repository, record, must_exist=True)
        try:
            downloaded = client.download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                remote_path=record.remote_path,
                cache_dir=None,
                offline=False,
            )
        except ArtifactTransportError as error:
            raise ArtifactDistributionError(
                f"Published artifact could not be anonymously verified: {error}"
            ) from error
        if not _matches_source(downloaded, record.size_bytes, expected):
            raise ArtifactDistributionError(
                f"Published artifact {record.remote_path!r} failed anonymous verification."
            )
    try:
        rechecked = safe_repository_path(
            repository, output.relative_to(repository), must_exist=False
        )
    except (ArtifactContractError, OSError, ValueError) as error:
        raise ArtifactDistributionError(
            f"Candidate lock destination became unsafe during publication: {error}"
        ) from error
    if rechecked != output or output.exists() or output.is_symlink():
        raise ArtifactDistributionError(
            "Candidate lock destination changed during publication; refusing to overwrite it."
        )
    try:
        _atomic_write(output, _json_bytes(lock.model_dump(mode="json")))
    except OSError as error:
        raise ArtifactDistributionError(
            f"Unable to publish candidate lock {output.relative_to(repository)!s}: {error}"
        ) from error
    return ArtifactOperationResult(
        materialized=(output,),
        reused=(),
        revision=revision,
    )


def _publication_records(repository: Path) -> tuple[ArtifactRecord, ...]:
    selected_model = safe_repository_path(
        repository, "models/selected_v1/model.cbm", must_exist=True
    )
    selected_size = selected_model.stat().st_size
    return (
        ArtifactRecord(
            artifact_id="selected_model",
            remote_path="selected_v1/model.cbm",
            local_path="models/selected_v1/model.cbm",
            size_bytes=selected_size,
            serialization="catboost_cbm",
            trust_classification="digest_authenticated",
            digest_reference=DigestReference(
                manifest_path=SELECTED_MANIFEST.as_posix(), json_pointer="/model_sha256"
            ),
        ),
    )


def _context(repository_root: str | Path, config_path: str | Path) -> tuple[Path, DistributionLock]:
    try:
        repository = Path(repository_root).resolve(strict=True)
        config_file = safe_repository_path(repository, config_path, must_exist=True)
        return repository, load_distribution_lock(config_file)
    except (ArtifactContractError, OSError, ValueError) as error:
        raise ArtifactDistributionError(str(error)) from error


def _artifact_location(
    repository: Path, record: ArtifactRecord, *, must_exist: bool
) -> tuple[Path, str]:
    try:
        path = safe_repository_path(repository, record.local_path, must_exist=must_exist)
        digest = resolve_digest_reference(repository, record.digest_reference)
    except (ArtifactContractError, OSError, ValueError) as error:
        raise ArtifactDistributionError(
            f"Artifact contract failed for {record.artifact_id!r}: {error}"
        ) from error
    return path, digest


def _matches(path: Path, expected_size: int, expected_sha256: str) -> bool:
    try:
        return (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == expected_size
            and sha256_file(path) == expected_sha256
        )
    except (OSError, ArtifactContractError):
        return False


def _matches_source(path: Path, expected_size: int, expected_sha256: str) -> bool:
    """Validate downloaded bytes while permitting an HF cache symlink to its blob store."""

    try:
        resolved = path.resolve(strict=True)
        return (
            resolved.is_file()
            and resolved.stat().st_size == expected_size
            and sha256_file(resolved) == expected_sha256
        )
    except (OSError, ArtifactContractError):
        return False


def _atomic_materialize(
    source: Path, destination: Path, record: ArtifactRecord, expected_sha256: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or destination.exists():
        raise ArtifactDistributionError(
            f"Artifact destination changed during materialization: {record.local_path!r}."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{record.artifact_id}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if not _matches(temporary, record.size_bytes, expected_sha256):
            raise ArtifactDistributionError("Staged artifact failed validation before promotion.")
        os.replace(temporary, destination)
        if not _matches(destination, record.size_bytes, expected_sha256):
            raise ArtifactDistributionError(
                "Materialized artifact failed post-promotion validation."
            )
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_candidate_lock_output(repository: Path, lock_output: str | Path) -> Path:
    relative_output = Path(lock_output)
    if relative_output.is_absolute():
        raise ArtifactDistributionError("Candidate lock output must be repository-relative.")
    output = safe_repository_path(repository, relative_output, must_exist=False)
    _reject_candidate_symlink_components(repository, relative_output)
    if output.exists():
        raise ArtifactDistributionError("Refusing to overwrite an existing candidate lock.")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_candidate_symlink_components(repository, relative_output)
    relative_parent = output.parent.relative_to(repository)
    safe_repository_path(repository, relative_parent, must_exist=True)
    descriptor, probe_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".preflight", dir=output.parent
    )
    probe = Path(probe_name)
    try:
        os.close(descriptor)
    finally:
        probe.unlink(missing_ok=True)
    return output


def _reject_candidate_symlink_components(repository: Path, relative_output: Path) -> None:
    current = repository
    for component in relative_output.parts:
        current /= component
        if current.is_symlink():
            raise ArtifactDistributionError(
                f"Candidate lock output contains a symlinked component: {current}"
            )


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ArtifactDistributionError(
                "Candidate lock appeared during final publication; refusing to overwrite it."
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _quarantine(repository: Path, path: Path) -> None:
    observed = sha256_file(path) if path.is_file() else "unreadable"
    destination = repository / "experiment" / "artifacts" / "quarantine" / observed / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination = destination.with_name(f"{destination.stem}-{os.getpid()}{destination.suffix}")
    os.replace(path, destination)


def _quarantine_partials(repository: Path, destination: Path, artifact_id: str) -> None:
    if not destination.parent.exists():
        return
    for partial in destination.parent.glob(f".{artifact_id}.*.partial"):
        if partial.is_file() and not partial.is_symlink():
            _quarantine(repository, partial)


def _require_allowlist(root: Path, expected: set[str], label: str) -> None:
    try:
        entries = list(root.iterdir())
    except OSError as error:
        raise ArtifactDistributionError(f"Unable to inspect {label}: {error}") from error
    if {entry.name for entry in entries} != expected or any(
        not entry.is_file() or entry.is_symlink() for entry in entries
    ):
        raise ArtifactDistributionError(
            f"{label.capitalize()} violates its file allowlist: expected={sorted(expected)}."
        )


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
