"""Command-line interface for governed registry release control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from credit_risk.registry.contracts import (
    DEFAULT_DEPLOYMENT_ROOT,
    DEFAULT_EVIDENCE_ROOT,
    DEFAULT_REGISTRY_CONFIG_PATH,
    DEFAULT_REGISTRY_ROOT,
)

registry_app = typer.Typer(
    name="registry",
    help="Register, promote, deploy, and roll back reviewed model releases.",
    no_args_is_help=True,
)


@registry_app.command("register")
def register(
    config: Annotated[Path, typer.Option(help="Frozen Phase 7 registry contract.")] = (
        DEFAULT_REGISTRY_CONFIG_PATH
    ),
    bundle_root: Annotated[Path, typer.Option(help="Exact reviewed selected-model bundle.")] = Path(
        "models/selected_v1"
    ),
    registry_root: Annotated[
        Path, typer.Option(help="Ignored repository-relative MLflow registry root.")
    ] = DEFAULT_REGISTRY_ROOT,
) -> None:
    """Register both immutable release revisions and their initial aliases."""

    result = _invoke(
        "Registry registration",
        _register,
        config_path=config,
        bundle_root=bundle_root,
        registry_root=registry_root,
    )
    typer.echo(_result_message(result))


@registry_app.command("promote")
def promote(
    approval: Annotated[Path, typer.Option(help="Reviewed promotion approval JSON.")],
    expected_approval_sha256: Annotated[
        str, typer.Option(help="External SHA-256 trust anchor for the approval.")
    ],
    config: Annotated[Path, typer.Option(help="Frozen Phase 7 registry contract.")] = (
        DEFAULT_REGISTRY_CONFIG_PATH
    ),
    registry_root: Annotated[
        Path, typer.Option(help="Ignored repository-relative MLflow registry root.")
    ] = DEFAULT_REGISTRY_ROOT,
) -> None:
    """Promote the approved candidate and preserve the prior champion."""

    result = _invoke(
        "Registry promotion",
        _promote,
        approval_path=approval,
        expected_approval_sha256=expected_approval_sha256,
        config_path=config,
        registry_root=registry_root,
    )
    typer.echo(_result_message(result))


@registry_app.command("deploy")
def deploy(
    alias: Annotated[
        str, typer.Option(help="Reviewed registry alias; only champion is allowed.")
    ] = ("champion"),
    config: Annotated[Path, typer.Option(help="Frozen Phase 7 registry contract.")] = (
        DEFAULT_REGISTRY_CONFIG_PATH
    ),
    registry_root: Annotated[
        Path, typer.Option(help="Ignored repository-relative MLflow registry root.")
    ] = DEFAULT_REGISTRY_ROOT,
    deployment_root: Annotated[
        Path, typer.Option(help="Ignored repository-relative deployment root.")
    ] = DEFAULT_DEPLOYMENT_ROOT,
) -> None:
    """Materialise and atomically activate the current champion."""

    result = _invoke(
        "Registry deployment",
        _deploy,
        alias=alias,
        config_path=config,
        registry_root=registry_root,
        deployment_root=deployment_root,
    )
    typer.echo(_result_message(result))


@registry_app.command("rollback")
def rollback(
    approval: Annotated[Path, typer.Option(help="Reviewed rollback approval JSON.")],
    expected_approval_sha256: Annotated[
        str, typer.Option(help="External SHA-256 trust anchor for the approval.")
    ],
    config: Annotated[Path, typer.Option(help="Frozen Phase 7 registry contract.")] = (
        DEFAULT_REGISTRY_CONFIG_PATH
    ),
    registry_root: Annotated[
        Path, typer.Option(help="Ignored repository-relative MLflow registry root.")
    ] = DEFAULT_REGISTRY_ROOT,
    deployment_root: Annotated[
        Path, typer.Option(help="Ignored repository-relative deployment root.")
    ] = DEFAULT_DEPLOYMENT_ROOT,
) -> None:
    """Restore the reviewed rollback revision and active deployment pointer."""

    result = _invoke(
        "Registry rollback",
        _rollback,
        approval_path=approval,
        expected_approval_sha256=expected_approval_sha256,
        config_path=config,
        registry_root=registry_root,
        deployment_root=deployment_root,
    )
    typer.echo(_result_message(result))


@registry_app.command("status")
def status(
    config: Annotated[Path, typer.Option(help="Frozen Phase 7 registry contract.")] = (
        DEFAULT_REGISTRY_CONFIG_PATH
    ),
    registry_root: Annotated[
        Path, typer.Option(help="Ignored repository-relative MLflow registry root.")
    ] = DEFAULT_REGISTRY_ROOT,
    deployment_root: Annotated[
        Path, typer.Option(help="Ignored repository-relative deployment root.")
    ] = DEFAULT_DEPLOYMENT_ROOT,
) -> None:
    """Validate and display live registry/deployment state."""

    result = _invoke(
        "Registry status",
        _status,
        config_path=config,
        registry_root=registry_root,
        deployment_root=deployment_root,
    )
    typer.echo(json.dumps(result, sort_keys=True))


@registry_app.command("publish-evidence")
def publish_evidence(
    config: Annotated[Path, typer.Option(help="Frozen Phase 7 registry contract.")] = (
        DEFAULT_REGISTRY_CONFIG_PATH
    ),
    registry_root: Annotated[
        Path, typer.Option(help="Ignored repository-relative MLflow registry root.")
    ] = DEFAULT_REGISTRY_ROOT,
    deployment_root: Annotated[
        Path, typer.Option(help="Ignored repository-relative deployment root.")
    ] = DEFAULT_DEPLOYMENT_ROOT,
    output_root: Annotated[
        Path, typer.Option(help="Repository-relative child beneath reports/registry/.")
    ] = DEFAULT_EVIDENCE_ROOT,
) -> None:
    """Publish deterministic aggregate evidence after the rollback drill."""

    result = _invoke(
        "Registry evidence publication",
        _publish,
        config_path=config,
        registry_root=registry_root,
        deployment_root=deployment_root,
        output_root=output_root,
    )
    typer.echo(
        f"Registry evidence published: status={result.status}, "
        f"manifest_sha256={result.evidence_manifest_sha256}"
    )


@registry_app.command("verify-evidence")
def verify_evidence(
    expected_manifest_sha256: Annotated[
        str, typer.Option(help="Externally reviewed SHA-256 of evidence-manifest.json.")
    ],
    config: Annotated[Path, typer.Option(help="Frozen Phase 7 registry contract.")] = (
        DEFAULT_REGISTRY_CONFIG_PATH
    ),
    evidence_root: Annotated[
        Path, typer.Option(help="Committed Phase 7 registry evidence directory.")
    ] = DEFAULT_EVIDENCE_ROOT,
) -> None:
    """Authenticate committed Phase 7 evidence without live registry state."""

    result = _invoke(
        "Registry evidence verification",
        _verify,
        expected_manifest_sha256=expected_manifest_sha256,
        config_path=config,
        evidence_root=evidence_root,
    )
    typer.echo(
        f"Registry evidence verified: status={result.status}, "
        f"manifest_sha256={result.evidence_manifest_sha256}"
    )


def _invoke(description: str, function: Any, **kwargs: Any) -> Any:
    try:
        return function(**kwargs)
    except (RuntimeError, ModuleNotFoundError) as error:
        typer.echo(f"{description} failed: {error}", err=True)
        raise typer.Exit(code=1) from None


def _result_message(result: Any) -> str:
    active = f", active_revision={result.active_revision}" if result.active_revision else ""
    return (
        f"Registry operation complete: status={result.status}, aliases={result.aliases}, "
        f"receipt_sha256={result.receipt_sha256}{active}"
    )


def _register(**kwargs: Any) -> Any:
    from credit_risk.registry.workflow import register_release_revisions

    return register_release_revisions(**kwargs)


def _promote(**kwargs: Any) -> Any:
    from credit_risk.registry.workflow import promote_candidate

    return promote_candidate(**kwargs)


def _deploy(**kwargs: Any) -> Any:
    from credit_risk.registry.workflow import deploy_champion

    return deploy_champion(**kwargs)


def _rollback(**kwargs: Any) -> Any:
    from credit_risk.registry.workflow import rollback_release

    return rollback_release(**kwargs)


def _status(**kwargs: Any) -> Any:
    from credit_risk.registry.workflow import registry_status

    return registry_status(**kwargs)


def _publish(**kwargs: Any) -> Any:
    from credit_risk.registry.workflow import publish_registry_evidence

    return publish_registry_evidence(**kwargs)


def _verify(**kwargs: Any) -> Any:
    from credit_risk.registry.workflow import verify_registry_evidence

    return verify_registry_evidence(**kwargs)
