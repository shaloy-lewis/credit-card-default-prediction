"""Cross-platform command-line entry points for project workflows."""

from pathlib import Path
from typing import Annotated

import typer

from credit_risk import __version__
from credit_risk.artifacts import ArtifactValidationError, load_artifact_bundle
from credit_risk.data.cli import data_app
from credit_risk.modeling.cli import model_app

app = typer.Typer(
    name="credit-risk",
    help="Operate the credit-risk early-warning project.",
    no_args_is_help=True,
)
app.add_typer(data_app)
app.add_typer(model_app)


@app.command()
def version() -> None:
    """Print the installed project version."""
    typer.echo(__version__)


@app.command()
def doctor(
    artifact_dir: Annotated[
        Path,
        typer.Option(help="Directory containing trusted legacy inference artifacts."),
    ] = Path("artifacts"),
) -> None:
    """Load trusted artifacts and validate the complete inference contract."""
    try:
        load_artifact_bundle(artifact_dir)
    except ArtifactValidationError as error:
        typer.echo(f"Artifact validation failed: {error}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Inference artifacts validated: {artifact_dir.resolve()}")


@app.command()
def train() -> None:
    """Reject the retired legacy fitting path before importing training code."""
    typer.echo(
        "Legacy training is retired and cannot be rerun. Use 'credit-risk model select' for "
        "the governed four-fit workflow.",
        err=True,
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
