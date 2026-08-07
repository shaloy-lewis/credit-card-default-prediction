"""Tests for the cross-platform project CLI."""

from typer.testing import CliRunner

from credit_risk import __version__
from credit_risk.cli import app

runner = CliRunner()


def test_version_reports_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_doctor_accepts_complete_artifact_directory() -> None:
    result = runner.invoke(
        app,
        ["doctor", "--artifact-dir", "tests/fixtures/complete_artifacts"],
    )

    assert result.exit_code == 0
    assert "Environment ready" in result.stdout


def test_doctor_fails_with_actionable_missing_artifacts() -> None:
    result = runner.invoke(
        app,
        ["doctor", "--artifact-dir", "tests/fixtures/empty_artifacts"],
    )

    assert result.exit_code == 1
    assert "model.pkl" in result.output
    assert "preprocessor.pkl" in result.output
