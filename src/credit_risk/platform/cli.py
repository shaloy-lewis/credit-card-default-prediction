"""CLI for deterministic Phase 8 bootstrap and verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from credit_risk.platform.contracts import DEFAULT_PLATFORM_CONFIG_PATH

platform_app = typer.Typer(
    name="platform",
    help="Bootstrap and verify the persistent local Phase 8 platform.",
    no_args_is_help=True,
)


@platform_app.command("bootstrap")
def bootstrap(
    config: Annotated[Path, typer.Option(help="Frozen Phase 8 platform contract.")] = (
        DEFAULT_PLATFORM_CONFIG_PATH
    ),
    bundle_root: Annotated[Path, typer.Option(help="Reviewed selected-model bundle.")] = Path(
        "models/selected_v1"
    ),
    deployment_root: Annotated[
        Path, typer.Option(help="Persistent deployment volume mount.")
    ] = Path("/deployment"),
) -> None:
    """Create the exact platform state or verify an identical existing state."""

    result = _invoke(
        _bootstrap,
        config_path=config,
        bundle_root=bundle_root,
        deployment_root=deployment_root,
    )
    typer.echo(json.dumps(_result_payload(result), sort_keys=True))


@platform_app.command("verify")
def verify(
    config: Annotated[Path, typer.Option(help="Frozen Phase 8 platform contract.")] = (
        DEFAULT_PLATFORM_CONFIG_PATH
    ),
    bundle_root: Annotated[Path, typer.Option(help="Reviewed selected-model bundle.")] = Path(
        "models/selected_v1"
    ),
    deployment_root: Annotated[
        Path, typer.Option(help="Persistent deployment volume mount.")
    ] = Path("/deployment"),
) -> None:
    """Verify platform state without creating or changing it."""

    result = _invoke(
        _verify,
        config_path=config,
        bundle_root=bundle_root,
        deployment_root=deployment_root,
    )
    typer.echo(json.dumps(_result_payload(result), sort_keys=True))


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "registered_model_name": result.registered_model_name,
        "aliases": result.aliases,
        "object_sha256": result.object_sha256,
        "active_revision": result.active_revision,
        "fit_count": result.fit_count,
        "sealed_test_accessed": result.sealed_test_accessed,
    }


def _invoke(function: Any, **kwargs: Any) -> Any:
    try:
        return function(**kwargs)
    except (RuntimeError, ModuleNotFoundError, OSError) as error:
        typer.echo(f"Platform operation failed: {error}", err=True)
        raise typer.Exit(code=1) from None


def _bootstrap(**kwargs: Any) -> Any:
    from credit_risk.platform.bootstrap import bootstrap_platform

    return bootstrap_platform(**kwargs)


def _verify(**kwargs: Any) -> Any:
    from credit_risk.platform.bootstrap import verify_platform

    return verify_platform(**kwargs)


@platform_app.command("rehearse")
def rehearse_command(
    runtime: Annotated[str, typer.Option()] = "experiment/platform/release_b_v1",
    trivy: Annotated[str, typer.Option()] = "trivy",
) -> None:
    """Measure isolated startup, persistence, scans and smoke parity."""
    from credit_risk.assurance.cli import invoke
    from credit_risk.platform.rehearsal import rehearse

    invoke(rehearse, runtime=runtime, trivy=trivy)


@platform_app.command("publish-evidence")
def publish_evidence_command(
    runtime: Annotated[str, typer.Option()] = "experiment/platform/release_b_v1",
    output: Annotated[str, typer.Option()] = "reports/platform/phase8_v1",
) -> None:
    """Authenticate measured runtime receipts and publish aggregate evidence."""
    from credit_risk.assurance.cli import invoke
    from credit_risk.platform.evidence import publish_evidence

    invoke(publish_evidence, runtime=runtime, output=output)


@platform_app.command("verify-evidence")
def verify_evidence_command(
    expected_manifest_sha256: Annotated[str, typer.Option()],
    evidence_root: Annotated[str, typer.Option()] = "reports/platform/phase8_v1",
) -> None:
    """Verify the platform package without live services."""
    from credit_risk.assurance.cli import invoke
    from credit_risk.platform.evidence import verify_evidence

    invoke(verify_evidence, root=evidence_root, expected=expected_manifest_sha256)
