from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import credit_risk.artifact_distribution.workflow as workflow
from credit_risk.artifact_distribution.transport import ArtifactTransportError
from credit_risk.artifact_distribution.workflow import (
    ArtifactDistributionError,
    publish_artifacts,
    pull_artifacts,
    verify_artifacts,
)


class FakeTransport:
    def __init__(self, files: dict[str, Path], revision: str = "b" * 40) -> None:
        self.files = files
        self.revision = revision
        self.downloads: list[tuple[str, bool]] = []
        self.published: dict[str, Path] | None = None

    def download(self, **kwargs) -> Path:
        remote_path = str(kwargs["remote_path"])
        self.downloads.append((remote_path, bool(kwargs["offline"])))
        try:
            return self.files[remote_path]
        except KeyError as error:
            raise RuntimeError("missing remote") from error

    def publish(self, *, repo_id: str, files: dict[str, Path], commit_message: str) -> str:
        assert repo_id == "owner/repository"
        assert commit_message
        self.published = files
        return self.revision


class FailingTransport(FakeTransport):
    def download(self, **kwargs) -> Path:
        raise ArtifactTransportError("offline cache miss")

    def publish(self, *, repo_id: str, files: dict[str, Path], commit_message: str) -> str:
        raise ArtifactTransportError("publication denied")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture()
def distribution_repository(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    payloads = {
        "selected_v1/model.cbm": b"reviewed-selected",
    }
    (tmp_path / "models/selected_v1").mkdir(parents=True)
    (tmp_path / "configs/artifacts").mkdir(parents=True)
    (tmp_path / "docs/artifacts").mkdir(parents=True)
    (tmp_path / "docs/artifacts/hugging-face-repository-card.md").write_text(
        "# Card\n", encoding="utf-8"
    )
    (tmp_path / "models/selected_v1/manifest.json").write_text(
        json.dumps({"model_sha256": _sha(payloads["selected_v1/model.cbm"])}),
        encoding="utf-8",
    )
    records = [
        {
            "artifact_id": "selected_model",
            "remote_path": "selected_v1/model.cbm",
            "local_path": "models/selected_v1/model.cbm",
            "size_bytes": len(payloads["selected_v1/model.cbm"]),
            "serialization": "catboost_cbm",
            "trust_classification": "digest_authenticated",
            "digest_reference": {
                "manifest_path": "models/selected_v1/manifest.json",
                "json_pointer": "/model_sha256",
            },
        },
    ]
    lock = {
        "schema_version": "2.0.0",
        "distribution_id": "hf_distribution_v2",
        "repository": {
            "provider": "huggingface_hub",
            "repo_id": "owner/repository",
            "repo_type": "model",
            "revision": "a" * 40,
            "public": True,
        },
        "artifacts": records,
    }
    lock_path = tmp_path / "configs/artifacts/hf_distribution_v2.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return tmp_path, lock_path.relative_to(tmp_path), payloads


def _remote_files(tmp_path: Path, payloads: dict[str, bytes]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for remote, payload in payloads.items():
        path = tmp_path / "remote" / remote
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        result[remote] = path
    return result


def test_pull_materializes_and_reuses_without_network(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, config, payloads = distribution_repository
    transport = FakeTransport(_remote_files(root, payloads))
    first = pull_artifacts(
        config_path=config,
        repository_root=root,
        transport=transport,
    )
    assert len(first.materialized) == 1
    assert len(transport.downloads) == 1
    second = pull_artifacts(
        config_path=config,
        repository_root=root,
        transport=FakeTransport({}),
    )
    assert len(second.reused) == 1


def test_pull_propagates_offline_and_rejects_bad_download(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, config, payloads = distribution_repository
    files = _remote_files(root, payloads)
    files["selected_v1/model.cbm"].write_bytes(b"wrong")
    transport = FakeTransport(files)
    with pytest.raises(ArtifactDistributionError, match="does not match"):
        pull_artifacts(
            config_path=config,
            repository_root=root,
            transport=transport,
            offline=True,
        )
    assert transport.downloads == [("selected_v1/model.cbm", True)]
    assert not (root / "models/selected_v1/model.cbm").exists()


def test_pull_normalizes_transport_failure(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, config, _ = distribution_repository
    with pytest.raises(ArtifactDistributionError, match="offline cache miss"):
        pull_artifacts(
            config_path=config,
            repository_root=root,
            transport=FailingTransport({}),
        )


def test_pull_normalizes_partial_cleanup_failure(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config, payloads = distribution_repository

    def fail(*_: object) -> None:
        raise PermissionError("partial cleanup denied")

    monkeypatch.setattr(workflow, "_quarantine_partials", fail)

    with pytest.raises(ArtifactDistributionError, match="Unable to prepare artifact destination"):
        pull_artifacts(
            config_path=config,
            repository_root=root,
            transport=FakeTransport(_remote_files(root, payloads)),
        )


@pytest.mark.parametrize(
    "failure_point", ["mkdir", "mkstemp", "copy", "fsync", "replace", "cleanup"]
)
def test_pull_normalizes_materialization_filesystem_failures(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    root, config, payloads = distribution_repository
    files = _remote_files(root, payloads)

    if failure_point == "mkdir":
        original_mkdir = Path.mkdir

        def fail_mkdir(path: Path, *args: object, **kwargs: object) -> None:
            if path == root / "models/selected_v1":
                raise PermissionError("directory creation denied")
            original_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    elif failure_point == "mkstemp":
        monkeypatch.setattr(
            workflow.tempfile,
            "mkstemp",
            lambda **_: (_ for _ in ()).throw(PermissionError("staging denied")),
        )
    elif failure_point == "copy":
        monkeypatch.setattr(
            workflow.shutil,
            "copyfileobj",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")),
        )
    elif failure_point == "fsync":
        monkeypatch.setattr(
            workflow.os,
            "fsync",
            lambda *_: (_ for _ in ()).throw(OSError("fsync failed")),
        )
    elif failure_point == "replace":
        monkeypatch.setattr(
            workflow.os,
            "replace",
            lambda *_: (_ for _ in ()).throw(OSError("replace failed")),
        )
    else:
        original_unlink = Path.unlink

        def fail_cleanup(path: Path, *args: object, **kwargs: object) -> None:
            if path.name.endswith(".partial"):
                raise PermissionError("cleanup denied")
            original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(ArtifactDistributionError, match="Unable to materialize artifact"):
        pull_artifacts(
            config_path=config,
            repository_root=root,
            transport=FakeTransport(files),
        )


def test_pull_accepts_hugging_face_cache_symlink(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, config, payloads = distribution_repository
    files = _remote_files(root, payloads)
    blob = files["selected_v1/model.cbm"]
    link = blob.with_name("model-link.cbm")
    try:
        link.symlink_to(blob)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    files["selected_v1/model.cbm"] = link

    result = pull_artifacts(
        config_path=config,
        repository_root=root,
        transport=FakeTransport(files),
    )

    destination = root / "models/selected_v1/model.cbm"
    assert result.materialized == (destination,)
    assert destination.read_bytes() == payloads["selected_v1/model.cbm"]
    assert not destination.is_symlink()


def test_corrupt_existing_artifact_is_quarantined(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, config, _ = distribution_repository
    target = root / "models/selected_v1/model.cbm"
    target.write_bytes(b"wrong")
    with pytest.raises(ArtifactDistributionError, match="quarantined"):
        pull_artifacts(
            config_path=config,
            repository_root=root,
            transport=FakeTransport({}),
        )
    assert not target.exists()
    assert list((root / "experiment/artifacts/quarantine").rglob("model.cbm"))


def test_verify_rejects_extra_file(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, config, payloads = distribution_repository
    pull_artifacts(
        config_path=config,
        repository_root=root,
        transport=FakeTransport(_remote_files(root, payloads)),
    )
    (root / "models/selected_v1/foreign.bin").write_bytes(b"foreign")
    with pytest.raises(ArtifactDistributionError, match="allowlist"):
        verify_artifacts(config_path=config, repository_root=root)


def test_verify_normalizes_missing_artifact(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, config, _ = distribution_repository
    with pytest.raises(ArtifactDistributionError, match="Artifact contract failed"):
        verify_artifacts(config_path=config, repository_root=root)


def test_publish_verifies_remote_and_writes_candidate_lock(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    transport = FakeTransport(_remote_files(root, payloads))
    result = publish_artifacts(
        repo_id="owner/repository",
        source_root=root,
        lock_output="experiment/artifacts/candidate.json",
        transport=transport,
    )
    assert result.revision == "b" * 40
    assert transport.published is not None
    assert set(transport.published) == {
        "README.md",
        "selected_v1/model.cbm",
    }
    candidate = json.loads((root / "experiment/artifacts/candidate.json").read_text())
    assert candidate["repository"]["revision"] == "b" * 40
    assert candidate["distribution_id"] == "hf_distribution_v2"


def test_publish_refuses_to_replace_candidate_lock(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    output = root / "experiment/artifacts/candidate.json"
    output.parent.mkdir(parents=True)
    output.write_text("preserve", encoding="utf-8")
    transport = FakeTransport(_remote_files(root, payloads))

    with pytest.raises(ArtifactDistributionError, match="overwrite"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            lock_output=output.relative_to(root),
            transport=transport,
        )

    assert output.read_text(encoding="utf-8") == "preserve"
    assert transport.published is None
    assert transport.downloads == []


def test_publish_normalizes_transport_and_preflight_failures(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    with pytest.raises(ArtifactDistributionError, match="preflight"):
        publish_artifacts(repo_id="owner/repository", source_root=root / "missing")

    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    with pytest.raises(ArtifactDistributionError, match="publication denied"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            transport=FailingTransport({}),
        )


def test_publish_rejects_changed_local_bytes_before_remote_mutation(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(b"changed-selected")
    transport = FakeTransport(_remote_files(root, payloads))

    with pytest.raises(ArtifactDistributionError, match="reviewed metadata"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            transport=transport,
        )

    assert transport.published is None


def test_publish_normalizes_anonymous_verification_failure(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])

    class VerificationFailureTransport(FakeTransport):
        def download(self, **kwargs) -> Path:
            raise ArtifactTransportError("anonymous retrieval denied")

    with pytest.raises(ArtifactDistributionError, match="anonymously verified"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            transport=VerificationFailureTransport({}),
        )


def test_publish_rejects_nonimmutable_returned_revision(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])

    with pytest.raises(ArtifactDistributionError, match="Published revision failed"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            transport=FakeTransport(_remote_files(root, payloads), revision="short"),
        )


def test_publish_rejects_absolute_lock_output(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    transport = FakeTransport(_remote_files(root, payloads))

    with pytest.raises(ArtifactDistributionError, match="repository-relative"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            lock_output=root / "candidate.json",
            transport=transport,
        )

    assert transport.published is None
    assert transport.downloads == []


@pytest.mark.parametrize(
    ("lock_output", "blocked_parent"),
    [("../candidate.json", False), ("blocked/candidate.json", True)],
)
def test_publish_preflights_unsafe_or_unwritable_output_before_remote_mutation(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
    lock_output: str,
    blocked_parent: bool,
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    if blocked_parent:
        (root / "blocked").write_text("not a directory", encoding="utf-8")
    transport = FakeTransport(_remote_files(root, payloads))

    with pytest.raises(ArtifactDistributionError, match="preflight"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            lock_output=lock_output,
            transport=transport,
        )

    assert transport.published is None
    assert transport.downloads == []


def test_publish_probes_candidate_parent_before_remote_mutation(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    transport = FakeTransport(_remote_files(root, payloads))
    monkeypatch.setattr(
        workflow.tempfile,
        "mkstemp",
        lambda **_: (_ for _ in ()).throw(PermissionError("candidate directory is read-only")),
    )

    with pytest.raises(ArtifactDistributionError, match="preflight"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            lock_output="experiment/artifacts/candidate.json",
            transport=transport,
        )

    assert transport.published is None
    assert transport.downloads == []


def test_publish_rejects_symlinked_candidate_before_remote_mutation(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    output = root / "experiment/artifacts/candidate.json"
    output.parent.mkdir(parents=True)
    try:
        output.symlink_to(root / "future-candidate.json")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    transport = FakeTransport(_remote_files(root, payloads))

    with pytest.raises(ArtifactDistributionError, match="symlinked component"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            lock_output=output.relative_to(root),
            transport=transport,
        )

    assert transport.published is None
    assert transport.downloads == []


def test_publish_rechecks_candidate_destination_without_overwrite(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    output = root / "experiment/artifacts/candidate.json"

    class RacingTransport(FakeTransport):
        def publish(self, *, repo_id: str, files: dict[str, Path], commit_message: str) -> str:
            revision = super().publish(repo_id=repo_id, files=files, commit_message=commit_message)
            output.write_text("concurrent", encoding="utf-8")
            return revision

    transport = RacingTransport(_remote_files(root, payloads))

    with pytest.raises(ArtifactDistributionError, match="changed during publication"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            lock_output=output.relative_to(root),
            transport=transport,
        )

    assert output.read_text(encoding="utf-8") == "concurrent"


def test_publish_uses_atomic_no_clobber_for_candidate_lock(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    output = root / "experiment/artifacts/candidate.json"
    transport = FakeTransport(_remote_files(root, payloads))
    real_link = workflow.os.link

    def race_link(source: str | Path, destination: str | Path) -> None:
        Path(destination).write_text("concurrent", encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(workflow.os, "link", race_link)

    with pytest.raises(ArtifactDistributionError, match="refusing to overwrite"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            lock_output=output.relative_to(root),
            transport=transport,
        )

    assert output.read_text(encoding="utf-8") == "concurrent"
