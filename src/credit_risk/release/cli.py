"""CLI for non-computational release evidence."""

from pathlib import Path
from typing import Annotated, Any

import typer

from credit_risk.release.contracts import (
    DEFAULT_RELEASE_CONFIG_PATH,
    DEFAULT_UNCERTAINTY_SOURCE,
)

DEFAULT_OUTPUT_ROOT = Path("reports/releases/release_a_v1")

release_app = typer.Typer(
    name="release",
    help="Build or verify authenticated portfolio-release evidence.",
    no_args_is_help=True,
)


@release_app.command("build")
def build(
    data_root: Annotated[
        Path, typer.Option(help="Verified Phase 1 data root used for integrity checks only.")
    ] = Path("data"),
    config: Annotated[
        Path, typer.Option(help="Frozen Release A configuration.")
    ] = DEFAULT_RELEASE_CONFIG_PATH,
    uncertainty_source: Annotated[
        Path, typer.Option(help="Reviewed aggregate validation uncertainty JSON.")
    ] = DEFAULT_UNCERTAINTY_SOURCE,
    output_root: Annotated[
        Path, typer.Option(help="Repository-relative child beneath reports/releases/.")
    ] = DEFAULT_OUTPUT_ROOT,
) -> None:
    """Publish Release A evidence without loading a model or scoring any row."""

    try:
        result = _run_build(
            data_root=data_root,
            config_path=config,
            uncertainty_source=uncertainty_source,
            output_root=output_root,
        )
    except (RuntimeError, ModuleNotFoundError) as error:
        message = str(error)
        if isinstance(error, ModuleNotFoundError):
            message = (
                "Release build requires the data extra; install the project with the 'data' extra."
            )
        typer.echo(f"Release build failed: {message}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Release A dossier published: status={result.status}, "
        f"summary_sha256={result.summary_sha256}, output={result.evidence_root}"
    )


@release_app.command("verify")
def verify(
    expected_manifest_sha256: Annotated[
        str, typer.Option(help="Externally reviewed SHA-256 of evidence-manifest.json.")
    ],
    config: Annotated[
        Path, typer.Option(help="Frozen Release A configuration.")
    ] = DEFAULT_RELEASE_CONFIG_PATH,
    evidence_root: Annotated[
        Path, typer.Option(help="Committed Release A evidence directory.")
    ] = DEFAULT_OUTPUT_ROOT,
) -> None:
    """Verify the authenticated Release A dossier without runtime data."""

    try:
        result = _run_verify(
            expected_manifest_sha256=expected_manifest_sha256,
            config_path=config,
            evidence_root=evidence_root,
        )
    except RuntimeError as error:
        typer.echo(f"Release verification failed: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Release A dossier verified: status={result.status}, "
        f"manifest_sha256={result.evidence_manifest_sha256}"
    )


def _run_build(**kwargs: Any) -> Any:
    from credit_risk.release.workflow import run_release_build

    return run_release_build(**kwargs)


def _run_verify(**kwargs: Any) -> Any:
    from credit_risk.release.workflow import verify_release_evidence

    return verify_release_evidence(**kwargs)


@release_app.command("build-b")
def build_b_command(
    expected_ci_sha256: Annotated[str, typer.Option()],
    ci_receipt: Annotated[str, typer.Option()] = "experiment/release_b/ci-receipt.json",
    config: Annotated[str, typer.Option()] = "configs/releases/release_b_v1.json",
    output: Annotated[str, typer.Option()] = "reports/releases/release_b_v1",
) -> None:
    """Build a zero-scoring Release B review candidate; never grant approval."""
    from credit_risk.assurance.cli import invoke
    from credit_risk.release.release_b import build

    invoke(
        build,
        config=config,
        output=output,
        ci_receipt=ci_receipt,
        expected_ci_sha256=expected_ci_sha256,
    )


@release_app.command("verify-b")
def verify_b_command(
    expected_manifest_sha256: Annotated[str, typer.Option()],
    evidence_root: Annotated[str, typer.Option()] = "reports/releases/release_b_v1",
    approval: Annotated[str | None, typer.Option()] = None,
    approval_sha256: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Verify evidence and, when supplied, externally authenticated owner approval."""
    from credit_risk.assurance.cli import invoke
    from credit_risk.release.release_b import verify_b

    invoke(
        verify_b,
        root=evidence_root,
        expected=expected_manifest_sha256,
        approval=approval,
        approval_sha256=approval_sha256,
    )


@release_app.command("capture-ci")
def capture_ci_command(
    output: Annotated[str, typer.Option()] = "experiment/release_b/ci-receipt.json",
) -> None:
    """Fetch successful exact-commit GitHub CI results without granting release approval."""
    from credit_risk.assurance.cli import invoke
    from credit_risk.release.ci_evidence import capture

    invoke(capture, output=output)
