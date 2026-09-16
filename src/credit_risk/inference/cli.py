"""CLI commands for idempotent Phase 6 batch inference."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from credit_risk.inference.batch import (
    BatchInferenceError,
    run_batch,
    validate_batch_identity,
    verify_batch_run,
)
from credit_risk.inference.contracts import (
    DEFAULT_INFERENCE_CONFIG_PATH,
    InferenceContractError,
    load_inference_config,
)
from credit_risk.inference.engine import InferenceEngine, InferenceError
from credit_risk.inference.evidence import (
    DEFAULT_EVIDENCE_ROOT,
    DEFAULT_FIXTURE,
    DEFAULT_RUNTIME_ROOT,
    InferenceEvidenceError,
    build_inference_evidence,
    verify_inference_evidence,
)
from credit_risk.modeling.selected_bundle import SelectedBundleError

inference_app = typer.Typer(
    name="inference",
    help="Run and verify governed prediction-only inference.",
    no_args_is_help=True,
)


@inference_app.command("batch")
def batch_command(
    input_path: Annotated[Path, typer.Option("--input", help="Strict operational CSV snapshot.")],
    as_of_date: Annotated[str, typer.Option(help="Monthly scoring date in YYYY-MM-DD form.")],
    snapshot_id: Annotated[str, typer.Option(help="Opaque identifier for this input snapshot.")],
    config_path: Annotated[
        Path, typer.Option("--config", help="Frozen Phase 6 inference configuration.")
    ] = DEFAULT_INFERENCE_CONFIG_PATH,
    bundle_root: Annotated[
        Path, typer.Option(help="Reviewed selected-model bundle directory.")
    ] = Path("models/selected_v1"),
    output_root: Annotated[
        Path, typer.Option(help="Ignored root for deterministic batch outputs.")
    ] = Path("experiment/inference/batches"),
) -> None:
    """Score valid rows, publish rejections, and enforce idempotent reuse."""

    try:
        config = load_inference_config(config_path)
        validate_batch_identity(
            as_of_date=as_of_date,
            snapshot_id=snapshot_id,
            config=config,
        )
        engine = InferenceEngine(bundle_root=bundle_root, config_path=config_path)
        result = run_batch(
            input_path=input_path,
            as_of_date=as_of_date,
            snapshot_id=snapshot_id,
            output_root=output_root,
            config=config,
            engine=engine,
        )
    except (
        BatchInferenceError,
        InferenceContractError,
        InferenceError,
        SelectedBundleError,
    ) as error:
        typer.echo(f"Inference batch failed: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Batch {result.batch_id}: status={result.status}, valid={result.valid_rows}, "
        f"rejected={result.rejected_rows}, reused={str(result.reused).lower()}, "
        f"run_root={result.run_root}"
    )
    if result.exit_code:
        raise typer.Exit(code=result.exit_code)


@inference_app.command("verify")
def verify_command(
    run_root: Annotated[Path, typer.Option(help="Existing batch run directory.")],
    config_path: Annotated[
        Path, typer.Option("--config", help="Frozen Phase 6 inference configuration.")
    ] = DEFAULT_INFERENCE_CONFIG_PATH,
    bundle_root: Annotated[
        Path, typer.Option(help="Reviewed selected-model bundle directory.")
    ] = Path("models/selected_v1"),
) -> None:
    """Verify aggregate batch lineage and output digests without rescoring."""

    try:
        config = load_inference_config(config_path)
        InferenceEngine(bundle_root=bundle_root, config_path=config_path)
        manifest = verify_batch_run(run_root, config=config)
    except (
        BatchInferenceError,
        InferenceContractError,
        InferenceError,
        SelectedBundleError,
    ) as error:
        typer.echo(f"Inference verification failed: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Batch verified: batch_id={manifest['batch_id']}, status={manifest['status']}, "
        f"run_root={run_root}"
    )


@inference_app.command("evidence")
def evidence_command(
    fixture_path: Annotated[
        Path, typer.Option("--fixture", help="Reviewed synthetic inference fixture.")
    ] = DEFAULT_FIXTURE,
    config_path: Annotated[
        Path, typer.Option("--config", help="Frozen Phase 6 inference configuration.")
    ] = DEFAULT_INFERENCE_CONFIG_PATH,
    bundle_root: Annotated[
        Path, typer.Option(help="Reviewed selected-model bundle directory.")
    ] = Path("models/selected_v1"),
    runtime_root: Annotated[
        Path, typer.Option(help="Ignored row-level parity runtime directory.")
    ] = DEFAULT_RUNTIME_ROOT,
    output_root: Annotated[
        Path, typer.Option(help="Aggregate Phase 6 evidence directory.")
    ] = DEFAULT_EVIDENCE_ROOT,
) -> None:
    """Publish authenticated synthetic parity evidence from a clean commit."""

    try:
        result = build_inference_evidence(
            fixture_path=fixture_path,
            config_path=config_path,
            bundle_root=bundle_root,
            runtime_root=runtime_root,
            output_root=output_root,
        )
    except (
        BatchInferenceError,
        InferenceContractError,
        InferenceError,
        InferenceEvidenceError,
        SelectedBundleError,
    ) as error:
        typer.echo(f"Inference evidence failed: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Phase 6 evidence published: batch_id={result.batch_id}, "
        f"summary_sha256={result.summary_sha256}, "
        f"manifest_sha256={result.evidence_manifest_sha256}"
    )


@inference_app.command("verify-evidence")
def verify_evidence_command(
    expected_manifest_sha256: Annotated[
        str,
        typer.Option(help="Externally reviewed SHA-256 of evidence-manifest.json."),
    ],
    evidence_root: Annotated[
        Path, typer.Option(help="Published Phase 6 evidence directory.")
    ] = DEFAULT_EVIDENCE_ROOT,
    config_path: Annotated[
        Path, typer.Option("--config", help="Frozen Phase 6 inference configuration.")
    ] = DEFAULT_INFERENCE_CONFIG_PATH,
    bundle_root: Annotated[
        Path, typer.Option(help="Reviewed selected-model bundle directory.")
    ] = Path("models/selected_v1"),
    fixture_path: Annotated[
        Path, typer.Option("--fixture", help="Reviewed synthetic inference fixture.")
    ] = DEFAULT_FIXTURE,
) -> None:
    """Authenticate aggregate evidence without model loading or runtime rows."""

    try:
        result = verify_inference_evidence(
            evidence_root=evidence_root,
            expected_manifest_sha256=expected_manifest_sha256,
            config_path=config_path,
            bundle_root=bundle_root,
            fixture_path=fixture_path,
        )
    except InferenceEvidenceError as error:
        typer.echo(f"Inference evidence verification failed: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Phase 6 evidence verified: batch_id={result.batch_id}, "
        f"summary_sha256={result.summary_sha256}"
    )
