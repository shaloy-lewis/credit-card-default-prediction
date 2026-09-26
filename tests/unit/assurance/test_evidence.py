"""Externally anchored integrity, publication boundaries and approval-independent checks."""

import json

import pytest

from credit_risk.assurance import evidence as ev


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/source.json").write_bytes(b"{}")
    return tmp_path


def package(root):
    return ev.publish(
        "reports/test/v1",
        kind="test",
        summary={"passed": True},
        sources=ev.source_map(["configs/source.json"]),
        commit="a" * 40,
    )


def test_publish_authenticate_no_overwrite(repository):
    sha = package(repository)
    assert ev.verify("reports/test/v1", sha, "test") == {"passed": True}
    with pytest.raises(ev.EvidenceError, match="overwrite"):
        package(repository)
    with pytest.raises(ev.EvidenceError, match="trust anchor"):
        ev.verify("reports/test/v1", "b" * 64, "test")


@pytest.mark.parametrize("target", ["summary.json", "evidence-manifest.json"])
def test_tampered_output_or_manifest_is_rejected(repository, target):
    sha = package(repository)
    (repository / "reports/test/v1" / target).write_bytes(b"{}")
    with pytest.raises(ev.EvidenceError):
        ev.verify("reports/test/v1", sha, "test")


def test_changed_source_and_extra_output_are_rejected(repository):
    sha = package(repository)
    source = repository / "configs/source.json"
    source.write_bytes(b"changed")
    with pytest.raises(ev.EvidenceError, match="Source"):
        ev.verify("reports/test/v1", sha, "test")
    source.write_bytes(b"{}")
    (repository / "reports/test/v1/unapproved").write_bytes(b"")
    with pytest.raises(ev.EvidenceError, match="unapproved"):
        ev.verify("reports/test/v1", sha, "test")


@pytest.mark.parametrize(
    "path,subtree",
    [
        ("../outside", None),
        ("configs/source.json", "reports"),
        ("reports", "reports"),
        ("/outside", None),
    ],
)
def test_safe_path_rejects_escape(repository, path, subtree):
    with pytest.raises(ev.EvidenceError):
        ev.safe_path(path, subtree)


@pytest.mark.parametrize("content", [b"[]", b"bad", b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_invalid_json(repository, content):
    path = repository / "invalid.json"
    path.write_bytes(content)
    with pytest.raises(ev.EvidenceError):
        ev.read_json(path)


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), -1])
def test_invalid_measurement(value):
    with pytest.raises(ev.EvidenceError):
        ev.finite_number(value)


def test_missing_and_invalid_sources(repository):
    with pytest.raises(ev.EvidenceError):
        ev.hash_file("missing")
    with pytest.raises(ev.EvidenceError):
        ev.read_json(repository / "missing")
    with pytest.raises(ev.EvidenceError):
        ev.require_sha(None)
    with pytest.raises(ev.EvidenceError):
        ev.publish("reports/test/v1", kind="test", summary={}, sources={}, commit="bad")
    with pytest.raises(ev.EvidenceError):
        ev.publish(
            "reports/test/v1",
            kind="test",
            summary={},
            sources={"configs/source.json": "b" * 64},
            commit="a" * 40,
        )
    with pytest.raises(ev.EvidenceError):
        ev.publish(
            "reports/test/v1",
            kind="test",
            summary={},
            sources={},
            commit="a" * 40,
            extra={"../outside": b""},
        )


def test_clean_commit_and_dirty_refusal(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        ev, "collect_git_evidence", lambda root: SimpleNamespace(dirty=False, commit_sha="a" * 40)
    )
    assert ev.clean_commit() == "a" * 40
    monkeypatch.setattr(
        ev, "collect_git_evidence", lambda root: SimpleNamespace(dirty=True, commit_sha="a" * 40)
    )
    with pytest.raises(ev.EvidenceError, match="clean"):
        ev.clean_commit()


@pytest.mark.parametrize(
    "change",
    [
        {"boundary": {}},
        {"implementation_commit": "bad"},
        {"kind": "wrong"},
        {"outputs": {}},
        {"outputs": {"../bad": "a" * 64}},
    ],
)
def test_semantic_manifest_tampering(repository, change):
    package(repository)
    path = repository / "reports/test/v1/evidence-manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest.update(change)
    path.write_bytes(ev.encode(manifest))
    with pytest.raises(ev.EvidenceError):
        ev.verify("reports/test/v1", ev.hash_file(path), "test")
