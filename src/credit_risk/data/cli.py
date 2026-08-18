"""Command-line interface for reproducible data workflows.

The heavy data-validation dependency is imported inside command handlers so the
base inference installation can continue to run without the optional ``data``
extra.
"""

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from credit_risk.data.acquisition import DataAcquisitionError, fetch_source
from credit_risk.data.manifest import (
    DEFAULT_DATASET_MANIFEST_PATH,
    DEFAULT_SPLIT_CONFIG_PATH,
    ManifestLoadError,
    load_dataset_manifest,
)

data_app = typer.Typer(
    name="data",
    help="Acquire, build, and verify the governed UCI data snapshot.",
    no_args_is_help=True,
)


def _fail(action: str, error: Exception) -> NoReturn:
    typer.echo(f"Data {action} failed: {error}", err=True)
    raise typer.Exit(code=1)


@data_app.command()
def fetch(
    data_root: Annotated[
        Path,
        typer.Option(help="Root for ignored raw, processed, split, and quarantine data."),
    ] = Path("data"),
    manifest: Annotated[
        Path,
        typer.Option(help="Versioned source manifest to acquire and verify."),
    ] = DEFAULT_DATASET_MANIFEST_PATH,
) -> None:
    """Acquire the pinned source bytes, or verify an existing immutable copy."""

    try:
        result = fetch_source(load_dataset_manifest(manifest), data_root)
    except (DataAcquisitionError, ManifestLoadError, OSError) as error:
        _fail("fetch", error)

    outcome = "downloaded" if result.downloaded else "verified existing"
    typer.echo(
        f"Source {outcome}: {result.path.resolve()} "
        f"(sha256={result.verification.sha256}, attempts={result.attempts})"
    )


@data_app.command()
def build(
    data_root: Annotated[
        Path,
        typer.Option(help="Root for ignored raw, processed, split, and quarantine data."),
    ] = Path("data"),
    manifest: Annotated[
        Path,
        typer.Option(help="Versioned source manifest to build."),
    ] = DEFAULT_DATASET_MANIFEST_PATH,
    split_config: Annotated[
        Path,
        typer.Option(help="Sealed holdout and cross-validation configuration."),
    ] = DEFAULT_SPLIT_CONFIG_PATH,
    offline: Annotated[
        bool,
        typer.Option(help="Require an already downloaded source and make no network request."),
    ] = False,
) -> None:
    """Build canonical data, its quality report, and sealed split assignments."""

    try:
        from credit_risk.data.workflow import DataWorkflowError, build_dataset

        result = build_dataset(
            data_root=data_root,
            manifest_path=manifest,
            split_config_path=split_config,
            offline=offline,
        )
    except ModuleNotFoundError as error:
        if error.name == "pandera":
            _fail("build", ModuleNotFoundError("install the project with the 'data' extra"))
        raise
    except (DataWorkflowError, DataAcquisitionError, ManifestLoadError, OSError) as error:
        _fail("build", error)

    typer.echo(
        "Data build passed: "
        f"canonical_sha256={result.canonical_sha256}, "
        f"assignment_sha256={result.assignment_sha256}"
    )
    typer.echo(f"Canonical data: {result.paths.canonical.resolve()}")
    typer.echo(f"Runtime split manifest: {result.paths.split_manifest.resolve()}")
    lock_status = "verified" if result.reviewed_lock_verified else "not yet reviewed"
    typer.echo(f"Reviewed split lock ({lock_status}): {result.paths.reviewed_split_lock.resolve()}")


@data_app.command()
def verify(
    data_root: Annotated[
        Path,
        typer.Option(help="Root containing the generated data products to verify offline."),
    ] = Path("data"),
    manifest: Annotated[
        Path,
        typer.Option(help="Versioned source manifest to verify against."),
    ] = DEFAULT_DATASET_MANIFEST_PATH,
    split_config: Annotated[
        Path,
        typer.Option(help="Sealed holdout and cross-validation configuration."),
    ] = DEFAULT_SPLIT_CONFIG_PATH,
) -> None:
    """Verify the complete raw-to-split lineage without network access."""

    try:
        from credit_risk.data.workflow import DataWorkflowError, verify_dataset

        result = verify_dataset(
            data_root=data_root,
            manifest_path=manifest,
            split_config_path=split_config,
        )
    except ModuleNotFoundError as error:
        if error.name == "pandera":
            _fail("verification", ModuleNotFoundError("install the project with the 'data' extra"))
        raise
    except (DataWorkflowError, DataAcquisitionError, ManifestLoadError, OSError) as error:
        _fail("verification", error)

    typer.echo(
        "Offline verification passed: "
        f"source_sha256={result.source_sha256}, "
        f"canonical_sha256={result.canonical_sha256}, "
        f"assignment_sha256={result.assignment_sha256}"
    )
