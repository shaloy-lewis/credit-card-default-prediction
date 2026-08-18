"""Tests for network-independent, checksum-first source acquisition."""

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import Request

import pytest

from credit_risk.data.acquisition import (
    DOWNLOAD_TIMEOUT_SECONDS,
    DataAcquisitionError,
    DownloadFailedError,
    ExistingRawDataError,
    SourceIntegrityError,
    fetch_source,
    resolve_raw_data_path,
    verify_source_file,
)
from credit_risk.data.manifest import DatasetManifest, load_dataset_manifest


class RecordingOpener:
    def __init__(self, responses: list[bytes | Exception]) -> None:
        self.responses = responses.copy()
        self.calls: list[tuple[str, float]] = []

    def __call__(self, request: Request, *, timeout: float) -> BytesIO:
        self.calls.append((request.full_url, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return BytesIO(response)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_for(payload: bytes, *, expected_sha256: str | None = None) -> DatasetManifest:
    source = load_dataset_manifest()
    return source.model_copy(
        update={
            "dataset_id": "synthetic_acquisition_test",
            "dataset_version": "v1",
            "source": source.source.model_copy(
                update={
                    "url": "https://example.test/data.csv",
                    "size_bytes": len(payload),
                    "sha256": expected_sha256 or _sha256(payload),
                }
            ),
        }
    )


def test_fetch_streams_verifies_and_atomically_promotes(tmp_path) -> None:
    payload = b"ID,X1,Y\n1,100,0\n"
    manifest = _manifest_for(payload)
    opener = RecordingOpener([payload])

    result = fetch_source(manifest, tmp_path, opener=opener, sleeper=lambda _: None)

    assert result.downloaded is True
    assert result.attempts == 1
    assert result.path == resolve_raw_data_path(manifest, tmp_path)
    assert result.path.read_bytes() == payload
    assert result.verification.sha256 == _sha256(payload)
    assert opener.calls == [(manifest.source.url, float(DOWNLOAD_TIMEOUT_SECONDS))]
    assert list(result.path.parent.glob("*.part")) == []


def test_fetch_is_a_zero_network_noop_for_valid_existing_data(tmp_path) -> None:
    payload = b"ID,X1,Y\n1,100,0\n"
    manifest = _manifest_for(payload)
    first = fetch_source(
        manifest,
        tmp_path,
        opener=RecordingOpener([payload]),
        sleeper=lambda _: None,
    )
    unused_opener = RecordingOpener([AssertionError("network must not be called")])

    second = fetch_source(
        manifest,
        tmp_path,
        opener=unused_opener,
        sleeper=lambda _: None,
    )

    assert second.path == first.path
    assert second.downloaded is False
    assert second.attempts == 0
    assert unused_opener.calls == []


def test_transient_failures_are_bounded_and_retry_until_success(tmp_path) -> None:
    payload = b"ID,X1,Y\n1,100,0\n"
    manifest = _manifest_for(payload)
    opener = RecordingOpener([OSError("one"), TimeoutError("two"), payload])
    delays: list[float] = []

    result = fetch_source(manifest, tmp_path, opener=opener, sleeper=delays.append)

    assert result.downloaded is True
    assert result.attempts == 3
    assert len(opener.calls) == 3
    assert delays == [0.25, 0.5]


def test_checksum_mismatch_retries_three_times_and_quarantines_bytes(tmp_path) -> None:
    payload = b"wrong bytes"
    expected_sha256 = "0" * 64
    manifest = _manifest_for(payload, expected_sha256=expected_sha256)
    opener = RecordingOpener([payload, payload, payload])

    with pytest.raises(DownloadFailedError, match="after 3 attempts"):
        fetch_source(manifest, tmp_path, opener=opener, sleeper=lambda _: None)

    destination = resolve_raw_data_path(manifest, tmp_path)
    quarantine = (
        tmp_path
        / "quarantine"
        / manifest.dataset_id
        / manifest.dataset_version
        / _sha256(payload)
        / "data.csv"
    )
    assert not destination.exists()
    assert quarantine.read_bytes() == payload
    assert len(opener.calls) == 3


def test_corrupt_existing_raw_data_is_quarantined_without_redownload(tmp_path) -> None:
    valid_payload = b"ID,X1,Y\n1,100,0\n"
    corrupt_payload = b"corrupt existing"
    manifest = _manifest_for(valid_payload)
    destination = resolve_raw_data_path(manifest, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(corrupt_payload)
    unused_opener = RecordingOpener([valid_payload])

    with pytest.raises(ExistingRawDataError, match="Re-run the fetch"):
        fetch_source(
            manifest,
            tmp_path,
            opener=unused_opener,
            sleeper=lambda _: None,
        )

    quarantine = (
        tmp_path
        / "quarantine"
        / manifest.dataset_id
        / manifest.dataset_version
        / _sha256(corrupt_payload)
        / "data.csv"
    )
    assert not destination.exists()
    assert quarantine.read_bytes() == corrupt_payload
    assert unused_opener.calls == []


def test_offline_verification_reports_observed_identity(tmp_path) -> None:
    payload = b"ID,X1,Y\n1,100,0\n"
    manifest = _manifest_for(payload)
    source_path = tmp_path / "data.csv"
    source_path.write_bytes(payload)

    verification = verify_source_file(source_path, manifest)

    assert verification.path == source_path
    assert verification.size_bytes == len(payload)
    assert verification.sha256 == _sha256(payload)


def test_content_addressed_path_uses_dataset_version_and_checksum(tmp_path) -> None:
    payload = b"example"
    manifest = _manifest_for(payload)

    path = resolve_raw_data_path(manifest, tmp_path)

    assert path == (
        Path(tmp_path) / "raw" / "synthetic_acquisition_test" / "v1" / _sha256(payload) / "data.csv"
    )


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_verify_rejects_missing_or_non_file_sources(tmp_path, kind: str) -> None:
    payload = b"expected"
    manifest = _manifest_for(payload)
    source = tmp_path / "source.csv"
    if kind == "directory":
        source.mkdir()

    with pytest.raises(SourceIntegrityError, match="missing or is not a regular file"):
        verify_source_file(source, manifest)


def test_verify_reports_size_and_checksum_drift_together(tmp_path) -> None:
    expected = b"expected bytes"
    actual = b"bad"
    manifest = _manifest_for(expected)
    source = tmp_path / "source.csv"
    source.write_bytes(actual)

    with pytest.raises(SourceIntegrityError) as caught:
        verify_source_file(source, manifest)

    assert f"size is {len(actual)} bytes" in str(caught.value)
    assert f"SHA-256 is {_sha256(actual)}" in str(caught.value)


def test_existing_raw_directory_is_never_replaced(tmp_path) -> None:
    payload = b"expected"
    manifest = _manifest_for(payload)
    destination = resolve_raw_data_path(manifest, tmp_path)
    destination.mkdir(parents=True)
    opener = RecordingOpener([payload])

    with pytest.raises(ExistingRawDataError, match="not a regular file"):
        fetch_source(manifest, tmp_path, opener=opener, sleeper=lambda _: None)

    assert destination.is_dir()
    assert opener.calls == []


def test_download_larger_than_manifest_is_stopped_and_quarantined(tmp_path) -> None:
    expected = b"short"
    oversized = expected + b"!"
    manifest = _manifest_for(expected)
    opener = RecordingOpener([oversized, oversized, oversized])

    with pytest.raises(DownloadFailedError, match="exceeds the byte size"):
        fetch_source(manifest, tmp_path, opener=opener, sleeper=lambda _: None)

    quarantine = (
        tmp_path
        / "quarantine"
        / manifest.dataset_id
        / manifest.dataset_version
        / _sha256(oversized)
        / manifest.source.filename
    )
    assert quarantine.read_bytes() == oversized
    assert not resolve_raw_data_path(manifest, tmp_path).exists()


def test_network_failure_removes_empty_partial_files(tmp_path) -> None:
    payload = b"expected"
    manifest = _manifest_for(payload)
    opener = RecordingOpener([OSError("offline")] * 3)

    with pytest.raises(DownloadFailedError, match="offline"):
        fetch_source(manifest, tmp_path, opener=opener, sleeper=lambda _: None)

    destination = resolve_raw_data_path(manifest, tmp_path)
    assert list(destination.parent.glob("*.part")) == []
    assert not (tmp_path / "quarantine").exists()


class _PartialFailureResponse(BytesIO):
    def __init__(self, partial: bytes) -> None:
        super().__init__(partial)
        self._raised = False

    def read(self, size: int = -1) -> bytes:
        chunk = super().read(size)
        if chunk:
            return chunk
        if not self._raised:
            self._raised = True
            raise OSError("connection interrupted")
        return b""


def test_interrupted_partial_download_is_preserved_in_quarantine(tmp_path) -> None:
    expected = b"complete payload"
    partial = b"partial"
    manifest = _manifest_for(expected)

    def opener(request: Request, *, timeout: float) -> _PartialFailureResponse:
        assert request.full_url == manifest.source.url
        assert timeout == float(DOWNLOAD_TIMEOUT_SECONDS)
        return _PartialFailureResponse(partial)

    with pytest.raises(DownloadFailedError, match="connection interrupted"):
        fetch_source(manifest, tmp_path, opener=opener, sleeper=lambda _: None)

    quarantine = (
        tmp_path
        / "quarantine"
        / manifest.dataset_id
        / manifest.dataset_version
        / _sha256(partial)
        / manifest.source.filename
    )
    assert quarantine.read_bytes() == partial


def test_identical_failed_downloads_deduplicate_quarantine_content(tmp_path) -> None:
    bad = b"same bad bytes"
    manifest = _manifest_for(b"expected bytes")

    with pytest.raises(DownloadFailedError):
        fetch_source(
            manifest,
            tmp_path,
            opener=RecordingOpener([bad, bad, bad]),
            sleeper=lambda _: None,
        )

    quarantine_files = list((tmp_path / "quarantine").rglob("data.csv"))
    assert len(quarantine_files) == 1
    assert quarantine_files[0].read_bytes() == bad


def test_conflicting_preexisting_quarantine_file_fails_without_overwrite(tmp_path) -> None:
    expected = b"valid bytes"
    corrupt = b"corrupt raw bytes"
    conflict = b"different quarantine bytes"
    manifest = _manifest_for(expected)
    destination = resolve_raw_data_path(manifest, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(corrupt)
    quarantine = (
        tmp_path
        / "quarantine"
        / manifest.dataset_id
        / manifest.dataset_version
        / _sha256(corrupt)
        / manifest.source.filename
    )
    quarantine.parent.mkdir(parents=True)
    quarantine.write_bytes(conflict)

    with pytest.raises(DataAcquisitionError, match="conflicting bytes"):
        fetch_source(manifest, tmp_path, opener=RecordingOpener([expected]))

    assert destination.read_bytes() == corrupt
    assert quarantine.read_bytes() == conflict


def test_valid_destination_winning_publish_race_is_verified_without_overwrite(
    tmp_path, monkeypatch
) -> None:
    payload = b"ID,X1,Y\n1,100,0\n"
    manifest = _manifest_for(payload)
    destination = resolve_raw_data_path(manifest, tmp_path)
    opener = RecordingOpener([payload])

    def competing_publish(partial_path: Path, published_path: Path) -> None:
        assert partial_path.read_bytes() == payload
        assert published_path == destination
        published_path.write_bytes(payload)
        raise FileExistsError("a concurrent fetch published first")

    monkeypatch.setattr("credit_risk.data.acquisition.os.link", competing_publish)

    result = fetch_source(manifest, tmp_path, opener=opener, sleeper=lambda _: None)

    assert result.path == destination
    assert result.verification.path == destination
    assert result.verification.sha256 == _sha256(payload)
    assert result.downloaded is False
    assert result.attempts == 1
    assert destination.read_bytes() == payload
    assert list(destination.parent.glob("*.part")) == []
    assert opener.calls == [(manifest.source.url, float(DOWNLOAD_TIMEOUT_SECONDS))]


def test_corrupt_destination_winning_publish_race_is_quarantined_and_fails(
    tmp_path, monkeypatch
) -> None:
    payload = b"ID,X1,Y\n1,100,0\n"
    corrupt = b"concurrent corrupt bytes"
    manifest = _manifest_for(payload)
    destination = resolve_raw_data_path(manifest, tmp_path)
    opener = RecordingOpener([payload])

    def competing_publish(partial_path: Path, published_path: Path) -> None:
        assert partial_path.read_bytes() == payload
        assert published_path == destination
        published_path.write_bytes(corrupt)
        raise FileExistsError("a concurrent fetch published first")

    monkeypatch.setattr("credit_risk.data.acquisition.os.link", competing_publish)

    with pytest.raises(ExistingRawDataError, match="appeared during publication"):
        fetch_source(manifest, tmp_path, opener=opener, sleeper=lambda _: None)

    quarantine = (
        tmp_path
        / "quarantine"
        / manifest.dataset_id
        / manifest.dataset_version
        / _sha256(corrupt)
        / manifest.source.filename
    )
    assert not destination.exists()
    assert quarantine.read_bytes() == corrupt
    assert list(destination.parent.glob("*.part")) == []
    assert opener.calls == [(manifest.source.url, float(DOWNLOAD_TIMEOUT_SECONDS))]
