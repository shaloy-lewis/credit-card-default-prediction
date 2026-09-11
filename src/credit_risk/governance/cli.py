"""CLI entry points for validation-only Phase 5 governance."""

from pathlib import Path
from typing import Annotated, Any

import typer

from credit_risk.governance.contracts import DEFAULT_GOVERNANCE_CONFIG_PATH

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_BUNDLE_ROOT = Path("models/selected_v1")
DEFAULT_RUNTIME_ROOT = Path("experiment/governance/phase5_v1")
DEFAULT_OUTPUT_ROOT = Path("reports/governance/phase5_v1")

governance_app = typer.Typer(
    name="governance",
    help="Build or verify prediction-only model-governance evidence.",
    no_args_is_help=True,
)


@governance_app.command("build")
def build(
    data_root: Annotated[
        Path, typer.Option(help="Verified Phase 1 data root.")
    ] = DEFAULT_DATA_ROOT,
    config: Annotated[
        Path, typer.Option(help="Frozen Phase 5 configuration.")
    ] = DEFAULT_GOVERNANCE_CONFIG_PATH,
    bundle_root: Annotated[
        Path, typer.Option(help="Reviewed selected-model bundle.")
    ] = DEFAULT_BUNDLE_ROOT,
    runtime_root: Annotated[
        Path, typer.Option(help="Ignored row-level runtime evidence.")
    ] = DEFAULT_RUNTIME_ROOT,
    output_root: Annotated[
        Path, typer.Option(help="Aggregate governance evidence root.")
    ] = DEFAULT_OUTPUT_ROOT,
) -> None:
    """Publish validation-only subgroup and native-SHAP evidence."""

    try:
        result = _run_build(
            data_root=data_root,
            config_path=config,
            bundle_root=bundle_root,
            runtime_root=runtime_root,
            output_root=output_root,
        )
    except (RuntimeError, ModuleNotFoundError) as error:
        message = str(error)
        if isinstance(error, ModuleNotFoundError):
            message = "Governance build requires the data extra; install the project with the 'data' extra."
        typer.echo(f"Governance build failed: {message}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Phase 5 evidence published: g3={result.g3_result}, "
        f"review_triggers={result.review_trigger_count}, summary_sha256={result.summary_sha256}"
    )


@governance_app.command("verify")
def verify(
    data_root: Annotated[
        Path, typer.Option(help="Verified Phase 1 data root.")
    ] = DEFAULT_DATA_ROOT,
    config: Annotated[
        Path, typer.Option(help="Frozen Phase 5 configuration.")
    ] = DEFAULT_GOVERNANCE_CONFIG_PATH,
    bundle_root: Annotated[
        Path, typer.Option(help="Reviewed selected-model bundle.")
    ] = DEFAULT_BUNDLE_ROOT,
    evidence_root: Annotated[
        Path, typer.Option(help="Aggregate governance evidence root.")
    ] = DEFAULT_OUTPUT_ROOT,
) -> None:
    """Verify Phase 5 evidence without rescoring validation rows."""

    try:
        result = _run_verify(
            data_root=data_root,
            config_path=config,
            bundle_root=bundle_root,
            evidence_root=evidence_root,
        )
    except (RuntimeError, ModuleNotFoundError) as error:
        message = str(error)
        if isinstance(error, ModuleNotFoundError):
            message = "Governance verify requires the data extra; install the project with the 'data' extra."
        typer.echo(f"Governance verification failed: {message}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Phase 5 evidence verified: g3={result.g3_result}, "
        f"manifest_sha256={result.evidence_manifest_sha256}"
    )


def _run_build(**kwargs: Any) -> Any:
    from credit_risk.governance.workflow import run_governance_build

    return run_governance_build(**kwargs)


def _run_verify(**kwargs: Any) -> Any:
    from credit_risk.governance.workflow import verify_governance_evidence

    return verify_governance_evidence(**kwargs)
