"""Command-line entry points for governed model-development workflows."""

from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

model_app = typer.Typer(
    name="model",
    help="Run governed development-only modelling experiments.",
    no_args_is_help=True,
)


@model_app.callback()
def model() -> None:
    """Operate the governed model-development command group."""


def _fail(error: Exception) -> NoReturn:
    typer.echo(f"Model baseline failed: {error}", err=True)
    raise typer.Exit(code=1)


def _candidate_fail(error: Exception) -> NoReturn:
    typer.echo(f"Model candidate failed: {error}", err=True)
    raise typer.Exit(code=1)


def _candidate_progress(progress: Any, *, run_name: str | None = None) -> None:
    prefix = f"Candidate {run_name}" if run_name is not None else "Candidate"
    typer.echo(
        f"{prefix} progress: {progress.completed_folds}/{progress.total_folds} folds; "
        f"resumed={progress.resumed_folds}; "
        f"quarantined={progress.quarantined_checkpoints}; "
        f"current={progress.variant_id}/r{progress.repeat_index}/f{progress.fold_index}"
    )


@model_app.command()
def baseline(
    data_root: Annotated[
        Path,
        typer.Option(help="Root containing verified canonical data and sealed assignments."),
    ] = Path("data"),
    config: Annotated[
        Path,
        typer.Option(help="Versioned baseline experiment configuration."),
    ] = Path("configs/modeling/baseline_v1.json"),
    tracking_root: Annotated[
        Path,
        typer.Option(help="Ignored root for the SQLite MLflow store and runtime artifacts."),
    ] = Path("experiment/mlflow"),
    output_root: Annotated[
        Path,
        typer.Option(help="Destination for deterministic baseline evidence."),
    ] = Path("reports/modeling/baseline_v1"),
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty",
            help=(
                "Permit a dirty worktree only with an exploratory output root and record "
                "its content-sensitive diff hash."
            ),
        ),
    ] = False,
) -> None:
    """Evaluate baselines without exposing holdout rows to fitting or evaluation."""

    try:
        from credit_risk.modeling.tracking import (
            TrackingDependencyError,
            ensure_mlflow_available,
        )

        ensure_mlflow_available()
    except TrackingDependencyError as error:
        _fail(error)

    try:
        from credit_risk.modeling.workflow import (
            BaselineWorkflowError,
            run_baseline_experiment,
        )

        result = run_baseline_experiment(
            data_root=data_root,
            config_path=config,
            tracking_root=tracking_root,
            output_root=output_root,
            allow_dirty=allow_dirty,
        )
    except ModuleNotFoundError as error:
        if error.name == "pandera":
            _fail(
                ModuleNotFoundError(
                    "Pandera is unavailable; install the project with the 'data' extra."
                )
            )
        if error.name == "mlflow":
            _fail(
                ModuleNotFoundError(
                    "MLflow is unavailable; install the project with the 'modeling' extra."
                )
            )
        raise
    except BaselineWorkflowError as error:
        _fail(error)

    typer.echo(
        "Baseline experiment passed: "
        f"parent_run_id={result.tracking.parent_run_id}, "
        f"summary_sha256={result.summary_sha256}"
    )
    typer.echo(f"Summary: {result.summary_path.resolve()}")
    typer.echo(f"Report: {result.report_path.resolve()}")
    typer.echo(f"Runtime OOF evidence: {result.oof_predictions_path.resolve()}")
    typer.echo(f"Logistic fold diagnostics: {result.logistic_diagnostics_path.resolve()}")
    typer.echo(f"MLflow tracking URI: {result.tracking.tracking_uri}")


@model_app.command()
def candidate(
    data_root: Annotated[
        Path,
        typer.Option(help="Root containing verified canonical data and sealed assignments."),
    ] = Path("data"),
    config: Annotated[
        Path,
        typer.Option(help="Frozen Phase 3 candidate experiment configuration."),
    ] = Path("configs/modeling/candidate_v1.json"),
    tracking_root: Annotated[
        Path,
        typer.Option(help="Ignored root for the SQLite MLflow store and runtime artifacts."),
    ] = Path("experiment/mlflow"),
    output_root: Annotated[
        Path,
        typer.Option(help="Ignored destination for one-run candidate evidence."),
    ] = Path("experiment/provisional/candidate_v1"),
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty",
            help=(
                "Permit a dirty worktree only with the candidate provisional output root and "
                "record its content-sensitive diff hash."
            ),
        ),
    ] = False,
) -> None:
    """Run the frozen CatBoost search without exposing the sealed holdout."""

    try:
        from credit_risk.modeling.tracking import (
            TrackingDependencyError,
            ensure_mlflow_available,
        )

        ensure_mlflow_available()
    except TrackingDependencyError as error:
        _candidate_fail(error)

    try:
        from credit_risk.modeling.candidate_workflow import (
            CandidateWorkflowError,
            run_candidate_experiment,
        )

        result = run_candidate_experiment(
            data_root=data_root,
            config_path=config,
            tracking_root=tracking_root,
            output_root=output_root,
            allow_dirty=allow_dirty,
            progress_callback=_candidate_progress,
        )
    except ModuleNotFoundError as error:
        if error.name == "pandera":
            _candidate_fail(
                ModuleNotFoundError(
                    "Pandera is unavailable; install the project with the 'data' extra."
                )
            )
        if error.name == "mlflow":
            _candidate_fail(
                ModuleNotFoundError(
                    "MLflow is unavailable; install the project with the 'modeling' extra."
                )
            )
        raise
    except CandidateWorkflowError as error:
        _candidate_fail(error)

    typer.echo(
        "Candidate experiment passed: "
        f"selected_model_id={result.selected_model_id}, "
        f"selected_configuration_id={result.selected_configuration_id}, "
        f"catboost_advances={str(result.catboost_advances).lower()}, "
        f"summary_sha256={result.summary_sha256}"
    )
    typer.echo(f"Summary: {result.summary_path.resolve()}")
    typer.echo(f"Report: {result.report_path.resolve()}")
    typer.echo(f"Runtime OOF evidence: {result.oof_predictions_path.resolve()}")
    typer.echo(f"Fold diagnostics: {result.fold_diagnostics_path.resolve()}")
    typer.echo(f"MLflow tracking URI: {result.tracking.tracking_uri}")


@model_app.command("candidate-evidence")
def candidate_evidence(
    data_root: Annotated[
        Path,
        typer.Option(help="Root containing verified canonical data and sealed assignments."),
    ] = Path("data"),
    config: Annotated[
        Path,
        typer.Option(help="Frozen Phase 3 candidate experiment configuration."),
    ] = Path("configs/modeling/candidate_v1.json"),
    tracking_root: Annotated[
        Path,
        typer.Option(help="Ignored primary SQLite MLflow and checkpoint root."),
    ] = Path("experiment/mlflow"),
    verification_root: Annotated[
        Path,
        typer.Option(help="Ignored independent execution and comparison root."),
    ] = Path("experiment/phase3-verification"),
    output_root: Annotated[
        Path,
        typer.Option(help="Official destination for verified aggregate candidate evidence."),
    ] = Path("reports/modeling/candidate_v1"),
) -> None:
    """Publish one of two byte-identical, independently checkpointed executions."""

    try:
        from credit_risk.modeling.tracking import (
            TrackingDependencyError,
            ensure_mlflow_available,
        )

        ensure_mlflow_available()
    except TrackingDependencyError as error:
        _candidate_fail(error)

    try:
        from credit_risk.modeling.candidate_evidence import (
            CandidateEvidenceError,
            run_candidate_evidence,
        )

        result = run_candidate_evidence(
            data_root=data_root,
            config_path=config,
            tracking_root=tracking_root,
            verification_root=verification_root,
            output_root=output_root,
            progress_callback=lambda run_name, progress: _candidate_progress(
                progress, run_name=run_name
            ),
        )
    except ModuleNotFoundError as error:
        if error.name == "pandera":
            _candidate_fail(
                ModuleNotFoundError(
                    "Pandera is unavailable; install the project with the 'data' extra."
                )
            )
        if error.name == "mlflow":
            _candidate_fail(
                ModuleNotFoundError(
                    "MLflow is unavailable; install the project with the 'modeling' extra."
                )
            )
        raise
    except CandidateEvidenceError as error:
        _candidate_fail(error)

    typer.echo(
        "Candidate evidence passed: independent_executions=2, "
        f"selected_model_id={result.selected_model_id}, "
        f"selected_configuration_id={result.selected_configuration_id}, "
        f"catboost_advances={str(result.catboost_advances).lower()}, "
        f"summary_sha256={result.summary_sha256}"
    )
    typer.echo(f"Official summary: {result.summary_path.resolve()}")
    typer.echo(f"Official report: {result.report_path.resolve()}")
    typer.echo(f"Primary MLflow tracking URI: {result.primary.tracking.tracking_uri}")
    typer.echo(f"Verification MLflow tracking URI: {result.verification.tracking.tracking_uri}")
