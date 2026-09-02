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


def test_doctor_accepts_complete_artifact_directory() -> None:
    result = runner.invoke(
        app,
        ["doctor", "--artifact-dir", "artifacts"],
    )

    assert result.exit_code == 0
    assert "Inference artifacts validated" in result.stdout


def test_doctor_fails_with_actionable_missing_artifacts() -> None:
    result = runner.invoke(
        app,
        ["doctor", "--artifact-dir", "tests/fixtures/empty_artifacts"],
    )

    assert result.exit_code == 1
    assert "model.pkl" in result.output
    assert "preprocessor.pkl" in result.output


def test_doctor_rejects_corrupt_artifacts() -> None:
    result = runner.invoke(
        app,
        ["doctor", "--artifact-dir", "tests/fixtures/corrupt_artifacts"],
    )

    assert result.exit_code == 1
    assert "Artifact validation failed" in result.output


def test_legacy_train_is_retired_without_importing_training_code() -> None:
    result = runner.invoke(app, ["train"])

    assert result.exit_code == 1
    assert "Legacy training is retired" in result.output
    assert "credit-risk model select" in result.output
    assert "Traceback" not in result.output
