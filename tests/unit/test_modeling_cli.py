"""Command-line tests for the governed baseline workflow."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from credit_risk.modeling import cli, tracking, workflow
from credit_risk.modeling.tracking import TrackingDependencyError, TrackingRunResult
from credit_risk.modeling.workflow import BaselineExperimentResult, BaselineWorkflowError

runner = CliRunner()


def test_baseline_command_uses_defaults_and_reports_operational_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> BaselineExperimentResult:
        captured.update(kwargs)
        return _result(tmp_path)

    monkeypatch.setattr(workflow, "run_baseline_experiment", fake_run)
    result = runner.invoke(cli.model_app, ["baseline"])

    assert result.exit_code == 0
    assert captured == {
        "data_root": Path("data"),
        "config_path": Path("configs/modeling/baseline_v1.json"),
        "tracking_root": Path("experiment/mlflow"),
        "output_root": Path("reports/modeling/baseline_v1"),
        "allow_dirty": False,
    }
    assert "Baseline experiment passed" in result.stdout
    assert "parent-run" in result.stdout
    assert "summary_sha256=" + "a" * 64 in result.stdout
    assert "Runtime OOF evidence:" in result.stdout
    assert "sqlite:///experiment/mlflow/mlflow.db" in result.stdout


def test_baseline_command_forwards_custom_roots_and_dirty_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> BaselineExperimentResult:
        captured.update(kwargs)
        return _result(tmp_path)

    monkeypatch.setattr(workflow, "run_baseline_experiment", fake_run)
    result = runner.invoke(
        cli.model_app,
        [
            "baseline",
            "--data-root",
            "custom-data",
            "--config",
            "custom-config.json",
            "--tracking-root",
            "custom-tracking",
            "--output-root",
            "custom-output",
            "--allow-dirty",
        ],
    )

    assert result.exit_code == 0
    assert captured["data_root"] == Path("custom-data")
    assert captured["config_path"] == Path("custom-config.json")
    assert captured["tracking_root"] == Path("custom-tracking")
    assert captured["output_root"] == Path("custom-output")
    assert captured["allow_dirty"] is True


def test_baseline_command_returns_actionable_workflow_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow,
        "run_baseline_experiment",
        lambda **_kwargs: (_ for _ in ()).throw(
            BaselineWorkflowError("Git worktree is dirty; pass --allow-dirty")
        ),
    )
    result = runner.invoke(cli.model_app, ["baseline"])

    assert result.exit_code == 1
    assert "Model baseline failed: Git worktree is dirty" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("missing_name", "expected"),
    (
        ("pandera", "'data' extra"),
        ("mlflow", "'modeling' extra"),
    ),
)
def test_baseline_command_explains_missing_optional_extras(
    missing_name: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "credit_risk.modeling.workflow":
            raise ModuleNotFoundError(
                f"No module named {missing_name!r}",
                name=missing_name,
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    result = runner.invoke(cli.model_app, ["baseline"])

    assert result.exit_code == 1
    assert expected in result.output
    assert "Traceback" not in result.output


def test_modeling_preflight_reports_missing_mlflow_before_workflow_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracking,
        "ensure_mlflow_available",
        lambda: (_ for _ in ()).throw(
            TrackingDependencyError(
                "MLflow dependency 'mlflow' is unavailable; "
                "install the project with the 'modeling' extra."
            )
        ),
    )
    result = runner.invoke(cli.model_app, ["baseline"])

    assert result.exit_code == 1
    assert "install the project with the 'modeling' extra" in result.output
    assert "Traceback" not in result.output


def test_model_group_help_lists_the_baseline_interface() -> None:
    result = runner.invoke(cli.model_app, ["baseline", "--help"])

    assert result.exit_code == 0
    assert "--data-root" in result.stdout
    assert "--tracking-root" in result.stdout
    assert "--output-root" in result.stdout
    assert "--allow-dirty" in result.stdout


def _result(tmp_path: Path) -> BaselineExperimentResult:
    return BaselineExperimentResult(
        summary_path=tmp_path / "summary.json",
        report_path=tmp_path / "baseline-report.md",
        oof_predictions_path=tmp_path / "oof_predictions.csv",
        logistic_diagnostics_path=tmp_path / "logistic_fold_diagnostics.json",
        summary_sha256="a" * 64,
        report_sha256="b" * 64,
        oof_predictions_sha256="c" * 64,
        logistic_diagnostics_sha256="d" * 64,
        tracking=TrackingRunResult(
            tracking_uri="sqlite:///experiment/mlflow/mlflow.db",
            experiment_name="credit-risk-baseline-v1",
            parent_run_id="parent-run",
            child_run_ids=(
                ("fold_prevalence", "child-1"),
                ("repayment_burden_rule", "child-2"),
                ("logistic_l2", "child-3"),
            ),
        ),
    )
