"""CLI gates for the authoritative one-pass modelling workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from credit_risk.modeling import cli, selection_workflow
from credit_risk.modeling.selection_workflow import (
    SelectionWorkflowError,
    SelectionWorkflowResult,
)
from credit_risk.modeling.tracking import TrackingRunResult

runner = CliRunner()


@pytest.mark.parametrize("command", ("baseline", "candidate", "candidate-evidence"))
def test_historical_commands_fail_before_importing_training(command: str) -> None:
    result = runner.invoke(cli.model_app, [command])

    assert result.exit_code == 1
    assert "historical and cannot be rerun" in result.output
    assert "credit-risk model select" in result.output
    assert "Traceback" not in result.output


def test_select_forwards_defaults_and_reports_governance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> SelectionWorkflowResult:
        captured.update(kwargs)
        return _result(tmp_path)

    monkeypatch.setattr(selection_workflow, "run_model_selection", fake_run)
    result = runner.invoke(cli.model_app, ["select"])

    assert result.exit_code == 0
    assert captured == {
        "data_root": Path("data"),
        "config_path": Path("configs/modeling/selection_v1.json"),
        "tracking_root": Path("experiment/mlflow"),
        "output_root": Path("reports/modeling/selection_v1"),
        "bundle_root": Path("models/selected_v1"),
    }
    assert "fit_count=4" in result.output
    assert "winner_refitted=false" in result.output


def test_select_returns_actionable_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_workflow,
        "run_model_selection",
        lambda **_kwargs: (_ for _ in ()).throw(
            SelectionWorkflowError("official selection requires a clean worktree")
        ),
    )
    result = runner.invoke(cli.model_app, ["select"])

    assert result.exit_code == 1
    assert "clean worktree" in result.output
    assert "Traceback" not in result.output


def test_final_test_remains_unimplemented_and_sealed() -> None:
    result = runner.invoke(cli.model_app, ["final-test"])

    assert result.exit_code == 1
    assert "separate explicit request" in result.output
    assert "sealed test remains untouched" in result.output


def test_select_help_exposes_release_destinations() -> None:
    result = runner.invoke(cli.model_app, ["select", "--help"])

    assert result.exit_code == 0
    assert "--bundle-root" in result.output
    assert "--output-root" in result.output
    assert "--tracking-root" in result.output


def _result(tmp_path: Path) -> SelectionWorkflowResult:
    return SelectionWorkflowResult(
        selected_model_id="logistic_l2",
        summary_path=tmp_path / "summary.json",
        report_path=tmp_path / "selection-report.md",
        manifest_path=tmp_path / "manifest.json",
        model_path=tmp_path / "model.joblib",
        validation_predictions_path=tmp_path / "validation_predictions.csv",
        bootstrap_path=tmp_path / "bootstrap_intervals.json",
        summary_sha256="a" * 64,
        report_sha256="b" * 64,
        manifest_sha256="c" * 64,
        model_sha256="d" * 64,
        tracking=TrackingRunResult(
            tracking_uri="sqlite:///experiment/mlflow/mlflow.db",
            experiment_name="credit-risk-selection-v1",
            parent_run_id="parent",
            child_run_ids=(("logistic_l2", "child"),),
        ),
    )
