"""Validation-only stress evidence commands."""

from typing import Annotated

import typer

from credit_risk.assurance.cli import invoke
from credit_risk.robustness import workflow

robustness_app = typer.Typer(name="robustness", no_args_is_help=True)


@robustness_app.command("build")
def build(
    data_root: Annotated[str, typer.Option()] = "data",
    output: Annotated[str, typer.Option()] = workflow.DEFAULT_OUTPUT,
    runtime: Annotated[str, typer.Option()] = "experiment/robustness/release_b_v1",
) -> None:
    """Publish all frozen prediction-only scenarios from clean committed code."""
    invoke(workflow.build, data_root=data_root, output=output, runtime=runtime)


@robustness_app.command("verify")
def verify(
    expected_manifest_sha256: Annotated[str, typer.Option()],
    evidence_root: Annotated[str, typer.Option()] = workflow.DEFAULT_OUTPUT,
) -> None:
    """Authenticate robustness evidence without loading real rows or a model."""
    invoke(workflow.verify_evidence, root=evidence_root, expected=expected_manifest_sha256)
