from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from credit_risk.cli import app
from credit_risk.registry import cli

runner = CliRunner()


def _operation(**overrides):
    values = {
        "status": "ok",
        "registered_model_name": "credit-risk-default",
        "aliases": {"champion": "1"},
        "receipt_sha256": "a" * 64,
        "active_revision": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_register_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_register", lambda **_kwargs: _operation())
    result = runner.invoke(app, ["registry", "register"])
    assert result.exit_code == 0
    assert "status=ok" in result.stdout


def test_registry_promote_deploy_and_rollback_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_promote", lambda **_kwargs: _operation(status="promoted"))
    promoted = runner.invoke(
        app,
        [
            "registry",
            "promote",
            "--approval",
            "configs/registry/approval.json",
            "--expected-approval-sha256",
            "a" * 64,
        ],
    )
    assert promoted.exit_code == 0
    assert "status=promoted" in promoted.stdout

    monkeypatch.setattr(
        cli,
        "_deploy",
        lambda **_kwargs: _operation(status="deployed", active_revision="phase7_rev_002"),
    )
    deployed = runner.invoke(app, ["registry", "deploy"])
    assert deployed.exit_code == 0
    assert "active_revision=phase7_rev_002" in deployed.stdout

    monkeypatch.setattr(
        cli,
        "_rollback",
        lambda **_kwargs: _operation(status="rolled_back", active_revision="phase7_rev_001"),
    )
    rolled_back = runner.invoke(
        app,
        [
            "registry",
            "rollback",
            "--approval",
            "configs/registry/approval.json",
            "--expected-approval-sha256",
            "b" * 64,
        ],
    )
    assert rolled_back.exit_code == 0
    assert "status=rolled_back" in rolled_back.stdout


def test_registry_status_and_evidence_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_status", lambda **_kwargs: {"active_revision": "phase7_rev_001"})
    status = runner.invoke(app, ["registry", "status"])
    assert status.exit_code == 0
    assert '"active_revision": "phase7_rev_001"' in status.stdout

    evidence = SimpleNamespace(
        status="registry_release_control_complete",
        evidence_manifest_sha256="c" * 64,
    )
    monkeypatch.setattr(cli, "_publish", lambda **_kwargs: evidence)
    published = runner.invoke(app, ["registry", "publish-evidence"])
    assert published.exit_code == 0
    assert "Registry evidence published" in published.stdout

    monkeypatch.setattr(cli, "_verify", lambda **_kwargs: evidence)
    verified = runner.invoke(
        app,
        ["registry", "verify-evidence", "--expected-manifest-sha256", "c" * 64],
    )
    assert verified.exit_code == 0
    assert "Registry evidence verified" in verified.stdout


def test_registry_cli_converts_expected_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli, "_register", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    result = runner.invoke(app, ["registry", "register"])
    assert result.exit_code == 1
    assert "Registry registration failed: bad" in result.output
    assert "Traceback" not in result.output


def test_registry_cli_exposes_no_force_or_bypass() -> None:
    result = runner.invoke(app, ["registry", "promote", "--help"])
    assert result.exit_code == 0
    assert "--force" not in result.stdout
    assert "--bypass" not in result.stdout


def test_registry_cli_lazy_import_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    from credit_risk.registry import workflow

    sentinels = {
        "register_release_revisions": "register",
        "promote_candidate": "promote",
        "deploy_champion": "deploy",
        "rollback_release": "rollback",
        "registry_status": "status",
        "publish_registry_evidence": "publish",
        "verify_registry_evidence": "verify",
    }
    wrappers = (
        cli._register,
        cli._promote,
        cli._deploy,
        cli._rollback,
        cli._status,
        cli._publish,
        cli._verify,
    )
    for (name, sentinel), wrapper in zip(sentinels.items(), wrappers, strict=True):
        monkeypatch.setattr(workflow, name, lambda marker=sentinel, **_kwargs: marker)
        assert wrapper() == sentinel
