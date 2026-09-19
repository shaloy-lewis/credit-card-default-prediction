"""Command-line interface for explicit artifact distribution operations."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from credit_risk.artifact_distribution.contracts import DEFAULT_DISTRIBUTION_LOCK
from credit_risk.artifact_distribution.workflow import (
    ArtifactDistributionError,
    publish_artifacts,
    pull_artifacts,
    verify_artifacts,
)

artifact_app = typer.Typer(
    name="artifacts",
    help="Pull, verify, or publish checksum-authenticated binary artifacts.",
    no_args_is_help=True,
)


class ArtifactGroupOption(StrEnum):
    selected = "selected"
    legacy = "legacy"
    all = "all"


def _fail(action: str, error: Exception) -> NoReturn:
    typer.echo(f"Artifact {action} failed: {error}", err=True)
    raise typer.Exit(code=1)


@artifact_app.command("pull")
def pull_command(
    config: Annotated[Path, typer.Option(help="Reviewed Hugging Face distribution lock.")] = (
        DEFAULT_DISTRIBUTION_LOCK
    ),
    group: Annotated[
        ArtifactGroupOption,
        typer.Option(help="Artifact group to materialize."),
    ] = ArtifactGroupOption.selected,
    cache_dir: Annotated[
        Path | None,
        typer.Option(help="Optional Hugging Face cache directory."),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option(help="Use only an already populated local Hugging Face cache."),
    ] = False,
) -> None:
    """Materialize reviewed artifacts from an immutable public revision."""

    if group in {ArtifactGroupOption.legacy, ArtifactGroupOption.all}:
        typer.echo(
            "Warning: legacy pickle artifacts can execute code when deserialized; "
            "only load bytes authenticated by the reviewed Git manifest.",
            err=True,
        )
    try:
        result = pull_artifacts(
            config_path=config,
            group=group.value,
            cache_dir=cache_dir,
            offline=offline,
        )
    except ArtifactDistributionError as error:
        _fail("pull", error)
    typer.echo(
        f"Artifact pull passed: materialized={len(result.materialized)}, "
        f"reused={len(result.reused)}, revision={result.revision}"
    )


@artifact_app.command("verify")
def verify_command(
    config: Annotated[Path, typer.Option(help="Reviewed Hugging Face distribution lock.")] = (
        DEFAULT_DISTRIBUTION_LOCK
    ),
    group: Annotated[
        ArtifactGroupOption,
        typer.Option(help="Artifact group to verify locally."),
    ] = ArtifactGroupOption.selected,
) -> None:
    """Verify local artifacts without network access or deserialization."""

    try:
        result = verify_artifacts(config_path=config, group=group.value)
    except ArtifactDistributionError as error:
        _fail("verification", error)
    typer.echo(f"Artifact verification passed: files={len(result.reused)}, group={group.value}")


@artifact_app.command("publish")
def publish_command(
    repo_id: Annotated[str, typer.Option(help="Public Hugging Face owner/repository ID.")],
    source_root: Annotated[
        Path, typer.Option(help="Repository root containing reviewed bytes.")
    ] = (Path(".")),
    lock_output: Annotated[
        Path,
        typer.Option(help="Ignored candidate distribution-lock destination."),
    ] = Path("experiment/artifacts/hf_distribution_v1.candidate.json"),
    include_legacy: Annotated[
        bool,
        typer.Option(help="Also publish the two reviewed trusted-pickle compatibility files."),
    ] = False,
) -> None:
    """Publish exact reviewed bytes and verify the resulting immutable revision."""

    try:
        result = publish_artifacts(
            repo_id=repo_id,
            source_root=source_root,
            lock_output=lock_output,
            include_legacy=include_legacy,
        )
    except (ArtifactDistributionError, OSError, ValueError) as error:
        _fail("publication", error)
    typer.echo(f"Artifact publication passed: revision={result.revision}")
    typer.echo(f"Candidate lock: {result.materialized[0]}")
