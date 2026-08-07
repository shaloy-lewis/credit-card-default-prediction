"""Cross-platform command-line entry points for project workflows."""

from pathlib import Path
from typing import Annotated

import typer

from credit_risk import __version__
from credit_risk.artifacts import ArtifactValidationError, load_artifact_bundle

app = typer.Typer(
    name="credit-risk",
    help="Operate the credit-risk early-warning project.",
    no_args_is_help=True,
)


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
    """Run the existing end-to-end training workflow."""
    from credit_risk.pipeline.training_pipeline import run_training

    run_training()


if __name__ == "__main__":
    app()
