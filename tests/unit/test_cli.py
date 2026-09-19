"""Tests for the cross-platform project CLI."""

from importlib.metadata import version as distribution_version

from typer.testing import CliRunner

from credit_risk import __version__
from credit_risk.cli import app

runner = CliRunner()


def test_version_reports_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ == distribution_version("credit-risk-early-warning")
    assert result.stdout.strip() == __version__


def test_removed_legacy_commands_are_not_exposed() -> None:
    for command in ("doctor", "train"):
        result = runner.invoke(app, [command])
        assert result.exit_code != 0
        assert "No such command" in result.output
