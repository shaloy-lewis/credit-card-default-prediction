from __future__ import annotations

from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

import credit_risk.governance.cli as cli
import credit_risk.governance.workflow as workflow
from credit_risk.governance.workflow import GovernanceWorkflowError, GovernanceWorkflowResult

runner = CliRunner()
EXPECTED_MANIFEST_SHA256 = "a" * 64


def _result(tmp_path: Path) -> GovernanceWorkflowResult:
    return GovernanceWorkflowResult(
        evidence_root=tmp_path / "reports",
        runtime_root=tmp_path / "runtime",
        evidence_manifest_sha256="a" * 64,
        summary_sha256="b" * 64,
        g3_result="closed_with_conditions",
        review_trigger_count=2,
    )


def test_build_and_verify_forward_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_arguments = {}
    verify_arguments = {}

    def fake_build(**kwargs):
        build_arguments.update(kwargs)
        return _result(tmp_path)

    def fake_verify(**kwargs):
        verify_arguments.update(kwargs)
        return _result(tmp_path)

    monkeypatch.setattr(cli, "_run_build", fake_build)
    monkeypatch.setattr(cli, "_run_verify", fake_verify)

    built = runner.invoke(cli.governance_app, ["build"])
    verified = runner.invoke(
        cli.governance_app,
        ["verify", "--expected-manifest-sha256", EXPECTED_MANIFEST_SHA256],
    )

    assert built.exit_code == 0
    assert "closed_with_conditions" in built.output
    assert verify_arguments["evidence_root"] == Path("reports/governance/phase5_v1")
    assert verify_arguments["runtime_root"] == Path("experiment/governance/phase5_v1")
    assert verify_arguments["expected_manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert verify_arguments["aggregate_only"] is False
    assert build_arguments["runtime_root"] == Path("experiment/governance/phase5_v1")
    assert verified.exit_code == 0


def test_cli_returns_actionable_errors_without_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "_run_build",
        lambda **_kwargs: (_ for _ in ()).throw(GovernanceWorkflowError("dirty worktree")),
    )
    result = runner.invoke(cli.governance_app, ["build"])

    assert result.exit_code == 1
    assert "dirty worktree" in result.output
    assert "Traceback" not in result.output

    monkeypatch.setattr(
        cli,
        "_run_verify",
        lambda **_kwargs: (_ for _ in ()).throw(GovernanceWorkflowError("altered evidence")),
    )
    verified = runner.invoke(
        cli.governance_app,
        ["verify", "--expected-manifest-sha256", EXPECTED_MANIFEST_SHA256],
    )
    assert verified.exit_code == 1
    assert "altered evidence" in verified.output
    assert "Traceback" not in verified.output


def test_cli_explains_missing_data_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "_run_build",
        lambda **_kwargs: (_ for _ in ()).throw(ModuleNotFoundError("pandera")),
    )
    monkeypatch.setattr(
        cli,
        "_run_verify",
        lambda **_kwargs: (_ for _ in ()).throw(ModuleNotFoundError("pandera")),
    )

    built = runner.invoke(cli.governance_app, ["build"])
    verified = runner.invoke(
        cli.governance_app,
        ["verify", "--expected-manifest-sha256", EXPECTED_MANIFEST_SHA256],
    )

    assert built.exit_code == 1
    assert "install the project with the 'data' extra" in built.output
    assert verified.exit_code == 1
    assert "install the project with the 'data' extra" in verified.output


def test_cli_metadata_exposes_only_governed_paths() -> None:
    command = get_command(cli.governance_app)
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
        "--bundle-root",
        "--runtime-root",
        "--output-root",
    }
    assert verify_options == {
        "--expected-manifest-sha256",
        "--data-root",
        "--config",
        "--bundle-root",
        "--runtime-root",
        "--evidence-root",
        "--aggregate-only",
    }
    assert "--force" not in build_options
    assert "--allow-dirty" not in build_options


def test_lazy_wrappers_import_the_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow, "run_governance_build", lambda **kwargs: kwargs)
    monkeypatch.setattr(workflow, "verify_governance_evidence", lambda **kwargs: kwargs)

    assert cli._run_build(value=1) == {"value": 1}
    assert cli._run_verify(value=2) == {"value": 2}
