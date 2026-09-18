from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from credit_risk.artifact_distribution.transport import (
    ArtifactTransportError,
    HuggingFaceTransport,
)


def test_download_is_anonymous_and_revision_pinned(monkeypatch, tmp_path: Path) -> None:
    artifact = tmp_path / "model.cbm"
    artifact.write_bytes(b"model")
    observed: dict[str, object] = {}

    def fake_download(**kwargs) -> str:
        observed.update(kwargs)
        return str(artifact)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download)
    )

    result = HuggingFaceTransport().download(
        repo_id="owner/repository",
        repo_type="model",
        revision="a" * 40,
        remote_path="selected_v1/model.cbm",
        cache_dir=tmp_path / "cache",
        offline=True,
    )

    assert result == artifact
    assert observed["revision"] == "a" * 40
    assert observed["token"] is False
    assert observed["local_files_only"] is True


def test_download_normalizes_transport_failure(monkeypatch) -> None:
    def fail(**_) -> str:
        raise RuntimeError("service secret")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fail))

    with pytest.raises(ArtifactTransportError, match="pinned public"):
        HuggingFaceTransport().download(
            repo_id="owner/repository",
            repo_type="model",
            revision="a" * 40,
            remote_path="selected_v1/model.cbm",
            cache_dir=None,
            offline=False,
        )


def test_publish_reuses_identical_remote_paths(monkeypatch, tmp_path: Path) -> None:
    local = tmp_path / "local.bin"
    remote = tmp_path / "remote.bin"
    local.write_bytes(b"same")
    remote.write_bytes(b"same")

    class FakeApi:
        def __init__(self, token: str | None) -> None:
            assert token == "maintainer-token"

        def create_repo(self, **kwargs) -> None:
            assert kwargs["private"] is False

        def repo_info(self, **_) -> SimpleNamespace:
            return SimpleNamespace(private=False, sha="c" * 40)

        def list_repo_files(self, **_) -> list[str]:
            return ["selected_v1/model.cbm"]

        def create_commit(self, **_) -> None:
            raise AssertionError("identical publication must not create a commit")

    module = SimpleNamespace(CommitOperationAdd=object, HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("HF_TOKEN", "maintainer-token")
    monkeypatch.setattr(HuggingFaceTransport, "download", lambda self, **_: remote)

    revision = HuggingFaceTransport().publish(
        repo_id="owner/repository",
        files={"selected_v1/model.cbm": local},
        commit_message="publish",
    )

    assert revision == "c" * 40


def test_publish_refuses_conflicting_remote_path(monkeypatch, tmp_path: Path) -> None:
    local = tmp_path / "local.bin"
    remote = tmp_path / "remote.bin"
    local.write_bytes(b"local")
    remote.write_bytes(b"remote")

    class FakeApi:
        def __init__(self, token: str | None) -> None:
            pass

        def create_repo(self, **_) -> None:
            pass

        def repo_info(self, **_) -> SimpleNamespace:
            return SimpleNamespace(private=False, sha="c" * 40)

        def list_repo_files(self, **_) -> list[str]:
            return ["selected_v1/model.cbm"]

    module = SimpleNamespace(CommitOperationAdd=object, HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setattr(HuggingFaceTransport, "download", lambda self, **_: remote)

    with pytest.raises(ArtifactTransportError, match="Refusing to overwrite"):
        HuggingFaceTransport().publish(
            repo_id="owner/repository",
            files={"selected_v1/model.cbm": local},
            commit_message="publish",
        )


def test_publish_creates_only_missing_paths_with_parent_protection(
    monkeypatch, tmp_path: Path
) -> None:
    local = tmp_path / "local.bin"
    local.write_bytes(b"local")
    observed: dict[str, object] = {}

    class Operation:
        def __init__(self, **kwargs) -> None:
            observed["operation"] = kwargs

    class FakeApi:
        def __init__(self, token: str | None) -> None:
            pass

        def create_repo(self, **_) -> None:
            pass

        def repo_info(self, **_) -> SimpleNamespace:
            return SimpleNamespace(private=False, sha="c" * 40)

        def list_repo_files(self, **_) -> list[str]:
            return []

        def create_commit(self, **kwargs) -> SimpleNamespace:
            observed.update(kwargs)
            return SimpleNamespace(oid="d" * 40)

    module = SimpleNamespace(CommitOperationAdd=Operation, HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    revision = HuggingFaceTransport().publish(
        repo_id="owner/repository",
        files={"selected_v1/model.cbm": local},
        commit_message="publish",
    )

    assert revision == "d" * 40
    assert observed["parent_commit"] == "c" * 40
    assert observed["operation"] == {
        "path_in_repo": "selected_v1/model.cbm",
        "path_or_fileobj": str(local),
    }


@pytest.mark.parametrize(
    ("private", "revision", "message"),
    [
        (True, "c" * 40, "must be public"),
        (False, "short", "parent commit SHA"),
    ],
)
def test_publish_rejects_unsafe_repository_state(
    monkeypatch, private: bool, revision: str, message: str
) -> None:
    class FakeApi:
        def __init__(self, token: str | None) -> None:
            pass

        def create_repo(self, **_) -> None:
            pass

        def repo_info(self, **_) -> SimpleNamespace:
            return SimpleNamespace(private=private, sha=revision)

        def list_repo_files(self, **_) -> list[str]:
            return []

    module = SimpleNamespace(CommitOperationAdd=object, HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(ArtifactTransportError, match=message):
        HuggingFaceTransport().publish(
            repo_id="owner/repository",
            files={},
            commit_message="publish",
        )


def test_publish_normalizes_client_construction_failure(monkeypatch) -> None:
    class FailingApi:
        def __init__(self, token: str | None) -> None:
            raise RuntimeError("credential failure")

    module = SimpleNamespace(CommitOperationAdd=object, HfApi=FailingApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(ArtifactTransportError, match="Unable to publish"):
        HuggingFaceTransport().publish(
            repo_id="owner/repository",
            files={},
            commit_message="publish",
        )
