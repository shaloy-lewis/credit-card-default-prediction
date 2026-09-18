from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

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
        "legacy_v1/model.pkl": b"reviewed-model-pickle",
        "legacy_v1/preprocessor.pkl": b"reviewed-preprocessor-pickle",
    }
    (tmp_path / "models/selected_v1").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "configs/artifacts").mkdir(parents=True)
    (tmp_path / "docs/artifacts").mkdir(parents=True)
    (tmp_path / "docs/artifacts/hugging-face-repository-card.md").write_text(
        "# Card\n", encoding="utf-8"
    )
    (tmp_path / "models/selected_v1/manifest.json").write_text(
        json.dumps({"model_sha256": _sha(payloads["selected_v1/model.cbm"])}),
        encoding="utf-8",
    )
    threshold = b'{"high_perc":{},"low_perc":{}}\n'
    (tmp_path / "artifacts/outlier_threshold.json").write_bytes(threshold)
    legacy = {
        "schema_version": "1.0.0",
        "bundle_id": "legacy_v1",
        "trust_classification": "trusted_pickle_explicit_only",
        "warning": "Pickle files are trusted only after exact digest verification.",
        "files": {
            "model.pkl": {
                "size_bytes": len(payloads["legacy_v1/model.pkl"]),
                "sha256": _sha(payloads["legacy_v1/model.pkl"]),
                "serialization": "python_pickle",
            },
            "preprocessor.pkl": {
                "size_bytes": len(payloads["legacy_v1/preprocessor.pkl"]),
                "sha256": _sha(payloads["legacy_v1/preprocessor.pkl"]),
                "serialization": "python_pickle",
            },
            "outlier_threshold.json": {
                "size_bytes": len(threshold),
                "sha256": _sha(threshold),
                "serialization": "json",
            },
        },
    }
    (tmp_path / "configs/artifacts/legacy_v1.json").write_text(json.dumps(legacy), encoding="utf-8")
    records = [
        {
            "artifact_id": "selected_model",
            "group": "selected",
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
        {
            "artifact_id": "legacy_model",
            "group": "legacy",
            "remote_path": "legacy_v1/model.pkl",
            "local_path": "artifacts/model.pkl",
            "size_bytes": len(payloads["legacy_v1/model.pkl"]),
            "serialization": "python_pickle",
            "trust_classification": "trusted_pickle_explicit_only",
            "digest_reference": {
                "manifest_path": "configs/artifacts/legacy_v1.json",
                "json_pointer": "/files/model.pkl/sha256",
            },
        },
        {
            "artifact_id": "legacy_preprocessor",
            "group": "legacy",
            "remote_path": "legacy_v1/preprocessor.pkl",
            "local_path": "artifacts/preprocessor.pkl",
            "size_bytes": len(payloads["legacy_v1/preprocessor.pkl"]),
            "serialization": "python_pickle",
            "trust_classification": "trusted_pickle_explicit_only",
            "digest_reference": {
                "manifest_path": "configs/artifacts/legacy_v1.json",
                "json_pointer": "/files/preprocessor.pkl/sha256",
            },
        },
    ]
    lock = {
        "schema_version": "1.0.0",
        "distribution_id": "hf_distribution_v1",
        "repository": {
            "provider": "huggingface_hub",
            "repo_id": "owner/repository",
            "repo_type": "model",
            "revision": "a" * 40,
            "public": True,
        },
        "artifacts": records,
    }
    lock_path = tmp_path / "configs/artifacts/hf_distribution_v1.lock.json"
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
        group="all",
        repository_root=root,
        transport=transport,
    )
    assert len(first.materialized) == 3
    assert len(transport.downloads) == 3
    second = pull_artifacts(
        config_path=config,
        group="all",
        repository_root=root,
        transport=FakeTransport({}),
    )
    assert len(second.reused) == 3


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


def test_selected_only_lock_rejects_legacy_request(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, config, _ = distribution_repository
    lock_path = root / config
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["artifacts"] = payload["artifacts"][:1]
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactDistributionError, match="does not contain"):
        verify_artifacts(config_path=config, group="legacy", repository_root=root)


def test_verify_normalizes_missing_artifact(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, config, _ = distribution_repository
    with pytest.raises(ArtifactDistributionError, match="Artifact contract failed"):
        verify_artifacts(config_path=config, repository_root=root)


def test_workflow_rejects_unknown_group(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, config, _ = distribution_repository
    with pytest.raises(ArtifactDistributionError, match="Unsupported artifact group"):
        verify_artifacts(config_path=config, group="unknown", repository_root=root)  # type: ignore[arg-type]


def test_publish_verifies_remote_and_writes_candidate_lock(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    (root / "artifacts/model.pkl").write_bytes(payloads["legacy_v1/model.pkl"])
    (root / "artifacts/preprocessor.pkl").write_bytes(payloads["legacy_v1/preprocessor.pkl"])
    transport = FakeTransport(_remote_files(root, payloads))
    result = publish_artifacts(
        repo_id="owner/repository",
        source_root=root,
        lock_output="experiment/artifacts/candidate.json",
        include_legacy=True,
        transport=transport,
    )
    assert result.revision == "b" * 40
    assert transport.published is not None
    assert set(transport.published) == {
        "README.md",
        "selected_v1/model.cbm",
        "legacy_v1/model.pkl",
        "legacy_v1/preprocessor.pkl",
    }
    candidate = json.loads((root / "experiment/artifacts/candidate.json").read_text())
    assert candidate["repository"]["revision"] == "b" * 40


def test_publish_refuses_to_replace_candidate_lock(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])
    output = root / "experiment/artifacts/candidate.json"
    output.parent.mkdir(parents=True)
    output.write_text("preserve", encoding="utf-8")

    with pytest.raises(ArtifactDistributionError, match="overwrite"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            lock_output=output.relative_to(root),
            transport=FakeTransport(_remote_files(root, payloads)),
        )

    assert output.read_text(encoding="utf-8") == "preserve"


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


def test_publish_rejects_absolute_lock_output(
    distribution_repository: tuple[Path, Path, dict[str, bytes]],
) -> None:
    root, _, payloads = distribution_repository
    (root / "models/selected_v1/model.cbm").write_bytes(payloads["selected_v1/model.cbm"])

    with pytest.raises(ArtifactDistributionError, match="repository-relative"):
        publish_artifacts(
            repo_id="owner/repository",
            source_root=root,
            lock_output=root / "candidate.json",
            transport=FakeTransport(_remote_files(root, payloads)),
        )
