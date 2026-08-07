"""Cross-platform command-line entry points for project workflows."""

from pathlib import Path
from typing import Annotated

import typer

from credit_risk import __version__

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
        typer.Option(help="Directory containing the legacy inference artifacts."),
    ] = Path("artifacts"),
) -> None:
    """Check whether the current checkout has the artifacts needed for inference."""
    required = ("model.pkl", "preprocessor.pkl", "outlier_threshold.json")
    missing = [name for name in required if not (artifact_dir / name).is_file()]

    if missing:
        typer.echo(f"Missing inference artifacts: {', '.join(missing)}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Environment ready: {artifact_dir.resolve()}")


@app.command()
def train() -> None:
    """Run the existing end-to-end training workflow."""
    from credit_risk.pipeline.training_pipeline import run_training

    run_training()


if __name__ == "__main__":
    app()
