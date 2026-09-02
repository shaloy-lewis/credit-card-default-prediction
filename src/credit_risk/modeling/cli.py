"""Command-line interfaces for governed model selection and test authorization."""

from pathlib import Path
from typing import Annotated, NoReturn

import typer

model_app = typer.Typer(
    name="model",
    help="Operate governed model-selection and release gates.",
    no_args_is_help=True,
)


@model_app.callback()
def model() -> None:
    """Operate the governed model command group."""


def _retired(name: str) -> NoReturn:
    typer.echo(
        f"Model {name} is historical and cannot be rerun. Use 'credit-risk model select' "
        "for the governed four-fit workflow.",
        err=True,
    )
    raise typer.Exit(code=1)


@model_app.command()
def baseline(
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    config: Annotated[Path, typer.Option()] = Path("configs/modeling/baseline_v1.json"),
    tracking_root: Annotated[Path, typer.Option()] = Path("experiment/mlflow"),
    output_root: Annotated[Path, typer.Option()] = Path("reports/modeling/baseline_v1"),
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> None:
    """Report the retired Phase 2 baseline interface without importing training code."""

    del data_root, config, tracking_root, output_root, allow_dirty
    _retired("baseline")


@model_app.command()
def candidate(
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    config: Annotated[Path, typer.Option()] = Path("configs/modeling/candidate_v1.json"),
    tracking_root: Annotated[Path, typer.Option()] = Path("experiment/mlflow"),
    output_root: Annotated[Path, typer.Option()] = Path("experiment/provisional/candidate_v1"),
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> None:
    """Report the retired Phase 3 candidate interface without importing training code."""

    del data_root, config, tracking_root, output_root, allow_dirty
    _retired("candidate")


@model_app.command("candidate-evidence")
def candidate_evidence(
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    config: Annotated[Path, typer.Option()] = Path("configs/modeling/candidate_v1.json"),
    tracking_root: Annotated[Path, typer.Option()] = Path("experiment/mlflow"),
    verification_root: Annotated[Path, typer.Option()] = Path("experiment/phase3-verification"),
    output_root: Annotated[Path, typer.Option()] = Path("reports/modeling/candidate_v1"),
) -> None:
    """Report the retired Phase 3 evidence interface without importing training code."""

    del data_root, config, tracking_root, verification_root, output_root
    _retired("candidate-evidence")


@model_app.command("select")
def select_models(
    data_root: Annotated[
        Path, typer.Option(help="Root containing verified data and sealed assignments.")
    ] = Path("data"),
    config: Annotated[Path, typer.Option(help="Frozen one-pass selection configuration.")] = Path(
        "configs/modeling/selection_v1.json"
    ),
    tracking_root: Annotated[
        Path, typer.Option(help="Ignored MLflow and row-level evidence root.")
    ] = Path("experiment/mlflow"),
    output_root: Annotated[
        Path, typer.Option(help="Destination for aggregate selection evidence.")
    ] = Path("reports/modeling/selection_v1"),
    bundle_root: Annotated[
        Path, typer.Option(help="Destination for the exact selected estimator bundle.")
    ] = Path("models/selected_v1"),
) -> None:
    """Fit four fixed models once and publish the exact validation winner."""

    try:
        from credit_risk.modeling.selection_workflow import (
            SelectionWorkflowError,
            run_model_selection,
        )

        result = run_model_selection(
            data_root=data_root,
            config_path=config,
            tracking_root=tracking_root,
            output_root=output_root,
            bundle_root=bundle_root,
        )
    except ModuleNotFoundError as error:
        extra = "data" if error.name == "pandera" else "modeling"
        typer.echo(
            f"Model selection failed: dependency {error.name!r} is unavailable; install the "
            f"project with the '{extra}' extra.",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except SelectionWorkflowError as error:
        typer.echo(f"Model selection failed: {error}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(
        "Model selection passed: "
        f"selected_model_id={result.selected_model_id}, fit_count=4, winner_refitted=false, "
        f"summary_sha256={result.summary_sha256}"
    )
    typer.echo(f"Summary: {result.summary_path.resolve()}")
    typer.echo(f"Report: {result.report_path.resolve()}")
    typer.echo(f"Bundle manifest: {result.manifest_path.resolve()}")
    typer.echo(f"MLflow tracking URI: {result.tracking.tracking_uri}")


@model_app.command("freeze-test")
def freeze_test(
    selection_root: Annotated[
        Path, typer.Option(help="Reviewed aggregate selection evidence root.")
    ] = Path("reports/modeling/selection_v1"),
    bundle_root: Annotated[Path, typer.Option(help="Reviewed selected-model bundle root.")] = Path(
        "models/selected_v1"
    ),
    output: Annotated[
        Path, typer.Option(help="Destination for immutable final-test authorization.")
    ] = Path("configs/modeling/final_test_v1.json"),
) -> None:
    """Freeze validation-derived test gates without loading the test partition."""

    from credit_risk.modeling.final_test import (
        FinalTestFreezeError,
        freeze_final_test_authorization,
    )

    try:
        path = freeze_final_test_authorization(
            selection_root=selection_root, bundle_root=bundle_root, output=output
        )
    except FinalTestFreezeError as error:
        typer.echo(f"Final-test freeze failed: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Final-test authorization frozen without test access: {path.resolve()}")


@model_app.command("final-test")
def final_test(
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    authorization: Annotated[Path, typer.Option()] = Path("configs/modeling/final_test_v1.json"),
    bundle_root: Annotated[Path, typer.Option()] = Path("models/selected_v1"),
    output_root: Annotated[Path, typer.Option()] = Path("reports/modeling/final_test_v1"),
) -> None:
    """Keep the sealed test closed until a separate explicit implementation request."""

    del data_root, authorization, bundle_root, output_root
    typer.echo(
        "Final-test execution is not implemented or authorized in this delivery. The sealed "
        "test remains untouched and requires a separate explicit request.",
        err=True,
    )
    raise typer.Exit(code=1)
