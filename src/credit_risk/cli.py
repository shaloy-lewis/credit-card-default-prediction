"""Cross-platform command-line entry points for project workflows."""

import typer

from credit_risk import __version__
from credit_risk.artifact_distribution.cli import artifact_app
from credit_risk.data.cli import data_app
from credit_risk.governance.cli import governance_app
from credit_risk.inference.cli import inference_app
from credit_risk.modeling.cli import model_app
from credit_risk.platform.cli import platform_app
from credit_risk.registry.cli import registry_app
from credit_risk.release.cli import release_app

app = typer.Typer(
    name="credit-risk",
    help="Operate the credit-risk early-warning project.",
    no_args_is_help=True,
)
app.add_typer(data_app)
app.add_typer(model_app)
app.add_typer(governance_app)
app.add_typer(release_app)
app.add_typer(inference_app)
app.add_typer(registry_app)
app.add_typer(platform_app)
app.add_typer(artifact_app)


@app.command()
def version() -> None:
    """Print the installed project version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
