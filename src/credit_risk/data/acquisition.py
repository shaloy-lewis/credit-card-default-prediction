"""Checksum-first, idempotent acquisition of immutable source data."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from types import TracebackType
from typing import Protocol, Self, cast
from urllib.request import Request, urlopen

from credit_risk.data.manifest import DatasetManifest

DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
RETRY_DELAYS_SECONDS = (0.25, 0.5)


class DataAcquisitionError(RuntimeError):
    """Base class for source-data acquisition failures."""


class SourceIntegrityError(DataAcquisitionError):
    """Raised when source bytes differ from the pinned manifest."""


class ExistingRawDataError(DataAcquisitionError):
    """Raised after an invalid existing raw file is quarantined."""


class DownloadFailedError(DataAcquisitionError):
    """Raised after all bounded acquisition attempts fail."""


class ReadableResponse(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class UrlOpener(Protocol):
    def __call__(self, request: Request, *, timeout: float) -> ReadableResponse: ...


@dataclass(frozen=True, slots=True)
class SourceVerification:
    """Observed identity of one verified raw source file."""

    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Outcome of an idempotent fetch operation."""

    path: Path
    verification: SourceVerification
    downloaded: bool
    attempts: int


def resolve_raw_data_path(
    manifest: DatasetManifest,
    data_root: str | Path = Path("data"),
) -> Path:
    """Return the content-addressed immutable location for the source snapshot."""

    return (
        Path(data_root)
        / "raw"
        / manifest.dataset_id
        / manifest.dataset_version
        / manifest.source.sha256
        / manifest.source.filename
    )


def verify_source_file(path: str | Path, manifest: DatasetManifest) -> SourceVerification:
    """Verify file size and SHA-256 against the checked-in source manifest."""

    source_path = Path(path)
    if not source_path.is_file():
        raise SourceIntegrityError(
            f"Source file is missing or is not a regular file: {source_path}"
        )

    observed_sha256, observed_size = _hash_file(source_path)
    failures: list[str] = []
    if observed_size != manifest.source.size_bytes:
        failures.append(
            f"size is {observed_size} bytes; expected {manifest.source.size_bytes} bytes"
        )
    if observed_sha256 != manifest.source.sha256:
        failures.append(f"SHA-256 is {observed_sha256}; expected {manifest.source.sha256}")
    if failures:
        raise SourceIntegrityError(
            f"Source integrity check failed for {source_path}: {'; '.join(failures)}"
        )

    return SourceVerification(
        path=source_path,
        size_bytes=observed_size,
        sha256=observed_sha256,
    )


def fetch_source(
    manifest: DatasetManifest,
    data_root: str | Path = Path("data"),
    *,
    opener: UrlOpener | None = None,
    sleeper: Callable[[float], None] = sleep,
) -> AcquisitionResult:
    """Fetch and atomically promote the pinned source, without overwriting raw data."""

    destination = resolve_raw_data_path(manifest, data_root)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ExistingRawDataError(
                f"Existing raw path is not a regular file and was not modified: {destination}"
            )
        try:
            verification = verify_source_file(destination, manifest)
        except SourceIntegrityError as error:
            quarantine_path = _quarantine_file(destination, manifest, Path(data_root))
            raise ExistingRawDataError(
                "Existing immutable raw data failed verification and was quarantined at "
                f"{quarantine_path}. Re-run the fetch to acquire a clean copy. Cause: {error}"
            ) from error
        return AcquisitionResult(
            path=destination,
            verification=verification,
            downloaded=False,
            attempts=0,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    open_url = opener or cast(UrlOpener, urlopen)
    last_error: Exception | None = None

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        partial_path = _new_partial_path(destination)
        try:
            _download_once(manifest, partial_path, open_url)
            verification = verify_source_file(partial_path, manifest)
            published_verification, published = _publish_without_overwrite(
                partial_path,
                destination,
                manifest,
                Path(data_root),
                verification,
            )
            return AcquisitionResult(
                path=destination,
                verification=published_verification,
                downloaded=published,
                attempts=attempt,
            )
        except ExistingRawDataError:
            partial_path.unlink(missing_ok=True)
            raise
        except Exception as error:
            last_error = error
            _preserve_or_remove_failed_partial(partial_path, manifest, Path(data_root))
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                sleeper(RETRY_DELAYS_SECONDS[attempt - 1])

    raise DownloadFailedError(
        f"Failed to acquire {manifest.dataset_id}/{manifest.dataset_version} after "
        f"{MAX_DOWNLOAD_ATTEMPTS} attempts. Last error: {last_error}"
    ) from last_error


def _new_partial_path(destination: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def _download_once(manifest: DatasetManifest, partial_path: Path, opener: UrlOpener) -> None:
    request = Request(
        manifest.source.url,
        headers={"User-Agent": "credit-risk-early-warning/0.1 data-acquisition"},
    )
    observed_size = 0
    with opener(request, timeout=float(DOWNLOAD_TIMEOUT_SECONDS)) as response:
        with partial_path.open("wb") as output:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                observed_size += len(chunk)
                output.write(chunk)
                if observed_size > manifest.source.size_bytes:
                    raise SourceIntegrityError(
                        "Downloaded source exceeds the byte size pinned in the manifest"
                    )
            output.flush()
            os.fsync(output.fileno())


def _publish_without_overwrite(
    partial_path: Path,
    destination: Path,
    manifest: DatasetManifest,
    data_root: Path,
    partial_verification: SourceVerification,
) -> tuple[SourceVerification, bool]:
    """Atomically publish with create-if-absent semantics across concurrent fetches."""

    try:
        os.link(partial_path, destination)
    except FileExistsError as error:
        if destination.is_symlink() or not destination.is_file():
            raise ExistingRawDataError(
                "A conflicting raw path appeared during publication and was not modified: "
                f"{destination}"
            ) from error
        try:
            existing_verification = verify_source_file(destination, manifest)
        except SourceIntegrityError as integrity_error:
            quarantine_path = _quarantine_file(destination, manifest, data_root)
            raise ExistingRawDataError(
                "Conflicting raw data appeared during publication, failed verification, "
                f"and was quarantined at {quarantine_path}. Re-run fetch. Cause: "
                f"{integrity_error}"
            ) from integrity_error
        partial_path.unlink()
        return existing_verification, False

    partial_path.unlink()
    return (
        SourceVerification(
            path=destination,
            size_bytes=partial_verification.size_bytes,
            sha256=partial_verification.sha256,
        ),
        True,
    )


def _preserve_or_remove_failed_partial(
    partial_path: Path,
    manifest: DatasetManifest,
    data_root: Path,
) -> None:
    if not partial_path.exists():
        return
    if partial_path.is_file() and partial_path.stat().st_size > 0:
        _quarantine_file(partial_path, manifest, data_root)
        return
    partial_path.unlink(missing_ok=True)


def _quarantine_file(path: Path, manifest: DatasetManifest, data_root: Path) -> Path:
    observed_sha256, _ = _hash_file(path)
    quarantine_path = (
        data_root
        / "quarantine"
        / manifest.dataset_id
        / manifest.dataset_version
        / observed_sha256
        / manifest.source.filename
    )
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    if quarantine_path.exists():
        existing_sha256, _ = _hash_file(quarantine_path)
        if existing_sha256 != observed_sha256:
            raise DataAcquisitionError(
                f"Quarantine path contains conflicting bytes: {quarantine_path}"
            )
        path.unlink()
        return quarantine_path
    os.replace(path, quarantine_path)
    return quarantine_path


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes
