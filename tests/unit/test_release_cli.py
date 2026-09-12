"""Tests for Release A CLI behavior and stable option contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.main import get_command
from typer.testing import CliRunner

import credit_risk.release.cli as cli
from credit_risk.release.workflow import ReleaseWorkflowError, ReleaseWorkflowResult

runner = CliRunner()
EXPECTED_MANIFEST = "a" * 64


def _result() -> ReleaseWorkflowResult:
    return ReleaseWorkflowResult(
        evidence_root=Path("reports/releases/release_a_v1"),
        summary_sha256="b" * 64,
        evidence_manifest_sha256=EXPECTED_MANIFEST,
        status="complete",
    )


def test_build_and_verify_forward_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    build_arguments: dict[str, Any] = {}
    verify_arguments: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> ReleaseWorkflowResult:
        build_arguments.update(kwargs)
        return _result()

    def fake_verify(**kwargs: Any) -> ReleaseWorkflowResult:
        verify_arguments.update(kwargs)
        return _result()

    monkeypatch.setattr(cli, "_run_build", fake_build)
    monkeypatch.setattr(cli, "_run_verify", fake_verify)
    built = runner.invoke(cli.release_app, ["build"])
    verified = runner.invoke(
        cli.release_app,
        ["verify", "--expected-manifest-sha256", EXPECTED_MANIFEST],
    )

    assert built.exit_code == 0
    assert verified.exit_code == 0
    assert build_arguments["uncertainty_source"] == cli.DEFAULT_UNCERTAINTY_SOURCE
    assert build_arguments["output_root"] == cli.DEFAULT_OUTPUT_ROOT
    assert verify_arguments["expected_manifest_sha256"] == EXPECTED_MANIFEST
    assert verify_arguments["evidence_root"] == cli.DEFAULT_OUTPUT_ROOT


def test_cli_returns_actionable_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "_run_build",
        lambda **_kwargs: (_ for _ in ()).throw(ReleaseWorkflowError("source mismatch")),
    )
    failed = runner.invoke(cli.release_app, ["build"])
    assert failed.exit_code == 1
    assert "source mismatch" in failed.output
    assert "Traceback" not in failed.output

    monkeypatch.setattr(
        cli,
        "_run_verify",
        lambda **_kwargs: (_ for _ in ()).throw(ReleaseWorkflowError("manifest mismatch")),
    )
    failed = runner.invoke(
        cli.release_app,
        ["verify", "--expected-manifest-sha256", EXPECTED_MANIFEST],
    )
    assert failed.exit_code == 1
    assert "manifest mismatch" in failed.output
    assert "Traceback" not in failed.output


def test_cli_metadata_exposes_only_evidence_paths() -> None:
    command = get_command(cli.release_app)
    build_options = {
        option
        for parameter in command.commands["build"].params  # type: ignore[attr-defined]
        for option in getattr(parameter, "opts", ())
    }
    verify_options = {
        option
        for parameter in command.commands["verify"].params  # type: ignore[attr-defined]
        for option in getattr(parameter, "opts", ())
    }

    assert build_options == {
        "--data-root",
        "--config",
        "--uncertainty-source",
        "--output-root",
    }
    assert verify_options == {
        "--expected-manifest-sha256",
        "--config",
        "--evidence-root",
    }
    for forbidden in ("--force", "--allow-dirty", "--tracking-root"):
        assert forbidden not in build_options | verify_options


def test_lazy_wrappers_import_the_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    import credit_risk.release.workflow as workflow

    monkeypatch.setattr(workflow, "run_release_build", lambda **kwargs: kwargs)
    monkeypatch.setattr(workflow, "verify_release_evidence", lambda **kwargs: kwargs)

    assert cli._run_build(value=1) == {"value": 1}
    assert cli._run_verify(value=2) == {"value": 2}
