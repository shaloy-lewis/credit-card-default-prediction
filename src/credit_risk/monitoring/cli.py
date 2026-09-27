"""Explicit monthly monitoring commands; no scheduler or automatic interventions."""

from typing import Annotated

import typer

from credit_risk.assurance.cli import invoke
from credit_risk.monitoring import workflow

monitor_app = typer.Typer(name="monitor", no_args_is_help=True)


@monitor_app.command("reference")
def reference(
    data_root: Annotated[str, typer.Option()] = "data",
    output: Annotated[str, typer.Option()] = workflow.REFERENCE,
) -> None:
    """Publish a validation-only reference profile."""
    invoke(workflow.reference, data_root=data_root, output=output)


@monitor_app.command("batch")
def batch(
    input_path: Annotated[str, typer.Option("--input")],
    run_root: Annotated[str, typer.Option()],
    expected_reference_sha256: Annotated[str, typer.Option()],
    output: Annotated[str, typer.Option()],
    reference_root: Annotated[str, typer.Option()] = workflow.REFERENCE,
) -> None:
    """Compare a verified batch against an externally authenticated reference."""
    invoke(
        workflow.batch,
        input_path=input_path,
        run_root=run_root,
        reference_root=reference_root,
        expected_reference_sha256=expected_reference_sha256,
        output=output,
    )


@monitor_app.command("service")
def service(
    log_path: Annotated[str, typer.Option()],
    output: Annotated[str, typer.Option()],
) -> None:
    """Publish aggregate counters and latency from allowlisted runtime events."""
    invoke(workflow.service, log_path=log_path, output=output)


@monitor_app.command("verify")
def verify(
    evidence_root: Annotated[str, typer.Option()],
    expected_manifest_sha256: Annotated[str, typer.Option()],
    kind: Annotated[str, typer.Option()],
) -> None:
    """Authenticate reference, batch or service evidence without model loading."""
    invoke(workflow.verify_report, root=evidence_root, expected=expected_manifest_sha256, kind=kind)


@monitor_app.command("benchmark")
def benchmark(
    output: Annotated[str, typer.Option()] = "reports/monitoring/benchmark_v1",
    runtime: Annotated[str, typer.Option()] = "experiment/monitoring/benchmark_v1",
    frozen_targets: Annotated[str | None, typer.Option()] = None,
    expected_origin_sha256: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Measure three rehearsals and freeze service targets."""
    from credit_risk.monitoring.benchmark import rehearse

    if frozen_targets is not None:
        from credit_risk.monitoring.benchmark import retain_targets

        invoke(
            retain_targets,
            expected_manifest_sha256=expected_origin_sha256,
            config=frozen_targets,
            output=output,
        )
    else:
        invoke(rehearse, output=output, runtime=runtime)


@monitor_app.command("acceptance")
def acceptance(
    expected_benchmark_sha256: Annotated[str, typer.Option()],
    benchmark_root: Annotated[str, typer.Option()] = "reports/monitoring/benchmark_v1",
    output: Annotated[str, typer.Option()] = "reports/monitoring/acceptance_v1",
    runtime: Annotated[str, typer.Option()] = "experiment/monitoring/acceptance_v1",
) -> None:
    """Measure a separate acceptance run against already-frozen targets."""
    from credit_risk.monitoring.benchmark import acceptance as run

    invoke(
        run,
        expected_benchmark_sha256=expected_benchmark_sha256,
        benchmark_root=benchmark_root,
        output=output,
        runtime=runtime,
    )


@monitor_app.command("drill")
def drill(
    expected_benchmark_sha256: Annotated[str, typer.Option()],
    benchmark_root: Annotated[str, typer.Option()] = "reports/monitoring/benchmark_v1",
    output: Annotated[str, typer.Option()] = "reports/incidents/release_b_v1",
    runtime: Annotated[str, typer.Option()] = "experiment/incidents/release_b_v1",
) -> None:
    """Run isolated schema, drift, artifact, outage and SQLite rollback drills."""
    from credit_risk.incidents.workflow import build

    invoke(
        build,
        expected_benchmark_sha256=expected_benchmark_sha256,
        benchmark_root=benchmark_root,
        output=output,
        runtime=runtime,
    )
