from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from credit_risk.cli import app
from credit_risk.platform import cli
from credit_risk.platform.bootstrap import PlatformBootstrapError

runner = CliRunner()


def _result(status: str) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        registered_model_name="credit-risk-default",
        aliases={"champion": "1", "rollback": "2"},
        object_sha256={"manifest.json": "a" * 64, "model.cbm": "b" * 64},
        active_revision="phase7_rev_001",
        fit_count=0,
        sealed_test_accessed=False,
    )


def test_platform_bootstrap_and_verify_commands(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_bootstrap", lambda **kwargs: _result("ready"))
    bootstrap = runner.invoke(app, ["platform", "bootstrap"])
    assert bootstrap.exit_code == 0
    assert '"status": "ready"' in bootstrap.stdout

    monkeypatch.setattr(cli, "_verify", lambda **kwargs: _result("verified"))
    verify = runner.invoke(app, ["platform", "verify"])
    assert verify.exit_code == 0
    assert '"status": "verified"' in verify.stdout


def test_platform_command_returns_controlled_error(monkeypatch) -> None:
    def fail(**kwargs):
        del kwargs
        raise PlatformBootstrapError("foreign registry state")

    monkeypatch.setattr(cli, "_bootstrap", fail)
    result = runner.invoke(app, ["platform", "bootstrap"])
    assert result.exit_code == 1
    assert "Platform operation failed: foreign registry state" in result.stdout
    assert "Traceback" not in result.stdout


def test_platform_commands_normalize_missing_and_invalid_paths(tmp_path: Path) -> None:
    missing_config = runner.invoke(
        app,
        ["platform", "verify", "--config", str(tmp_path / "missing.json")],
    )
    assert missing_config.exit_code == 1
    assert "Invalid Phase 8 platform configuration" in missing_config.stdout
    assert "Traceback" not in missing_config.stdout

    missing_bundle = runner.invoke(
        app,
        ["platform", "verify", "--bundle-root", "models/missing"],
    )
    assert missing_bundle.exit_code == 1
    assert "Unable to resolve bundle root" in missing_bundle.stdout
    assert "Traceback" not in missing_bundle.stdout

    deployment_file = tmp_path / "deployment-file"
    deployment_file.write_text("not a directory", encoding="utf-8")
    invalid_deployment = runner.invoke(
        app,
        ["platform", "bootstrap", "--deployment-root", str(deployment_file)],
    )
    assert invalid_deployment.exit_code == 1
    assert "Deployment root must be a directory" in invalid_deployment.stdout
    assert "Traceback" not in invalid_deployment.stdout
