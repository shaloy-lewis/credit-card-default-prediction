"""Repeatable synthetic failures; no mutation of historical platform/registry state."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from credit_risk.assurance.evidence import (
    ROOT,
    EvidenceError,
    clean_commit,
    digest,
    encode,
    publish,
    read_json,
    safe_path,
    source_map,
    verify,
)
from credit_risk.assurance.runtime import COMPOSE, await_ready, command, platform_state, request
from credit_risk.inference.batch import (
    BatchInferenceError,
    _batch_id,
    _batch_identity,
    parse_batch_csv,
    run_batch,
    verify_batch_run,
)
from credit_risk.inference.contracts import PHASE6_CONFIG_SHA256
from credit_risk.inference.engine import InferenceEngine, InferenceError
from credit_risk.inference.logging import emit_event
from credit_risk.monitoring.drift import compare, profile
from credit_risk.registry.workflow import (
    deploy_champion,
    promote_candidate,
    register_release_revisions,
    registry_status,
    rollback_release,
)

KIND = "release_b_incidents_v1"
OUTPUT = "reports/incidents/release_b_v1"


def data_drills(engine: InferenceEngine, runtime: Path | None = None) -> list[dict[str, Any]]:
    content = (ROOT / "tests/fixtures/inference_batch_v1.csv").read_bytes()
    clean = parse_batch_csv(content, engine.config)
    lines = content.decode().splitlines()
    outcomes: list[dict[str, Any]] = []
    try:
        parse_batch_csv(
            ("\n".join([",".join(lines[0].split(",")[:-1]), *lines[1:]])).encode(), engine.config
        )
    except BatchInferenceError:
        outcomes.append({"drill": "missing_columns", "detected": True, "contained": True})
    else:
        raise EvidenceError("Missing-column failure was not detected.")
    invalid = lines.copy()
    parts = invalid[1].split(",")
    parts[1] = "0"
    invalid[1] = ",".join(parts)
    parsed = parse_batch_csv("\n".join(invalid).encode(), engine.config)
    if len(parsed.rejections) != 1 or len(parsed.account_ids) != len(clean.account_ids) - 1:
        raise EvidenceError("Invalid-value quarantine changed.")
    outcomes.append({"drill": "invalid_values", "detected": True, "contained": True})
    duplicated = parse_batch_csv("\n".join([*lines, lines[1]]).encode(), engine.config)
    if len(duplicated.rejections) != 2:
        raise EvidenceError("Duplicate-account quarantine changed.")
    outcomes.append({"drill": "duplicate_ids", "detected": True, "contained": True})
    control = np.tile(np.arange(100, dtype=float), 4)
    reference = profile(control)
    if compare(reference, control)["status"] != "clear":
        raise EvidenceError("Unchanged monitoring control alerted.")
    for name, shifted in (
        ("population_shift", control + 10000),
        ("prediction_shift", np.ones(400)),
    ):
        ref = reference if name == "population_shift" else profile(control / 100)
        if compare(ref, shifted)["status"] != "investigate":
            raise EvidenceError("Injected shift was not detected.")
        outcomes.append({"drill": name, "detected": True, "contained": True})
    if runtime is not None:
        payloads = {
            "missing_columns": (
                "\n".join([",".join(lines[0].split(",")[:-1]), *lines[1:]])
            ).encode(),
            "invalid_values": "\n".join(invalid).encode(),
            "duplicate_ids": "\n".join([*lines, lines[1]]).encode(),
        }
        for outcome in outcomes[:3]:
            name = outcome["drill"]
            payload = payloads[name]
            source = runtime / f"{name}.csv"
            source.write_bytes(payload)
            try:
                run = run_batch(
                    input_path=source,
                    as_of_date="2026-09-30",
                    snapshot_id=name,
                    output_root=runtime / "batches",
                    config=engine.config,
                    engine=engine,
                )
            except BatchInferenceError:
                if name != "missing_columns":
                    raise
                attempt = _batch_id(
                    _batch_identity(
                        input_sha256=digest(payload),
                        as_of_date="2026-09-30",
                        snapshot_id=name,
                        config_sha256=PHASE6_CONFIG_SHA256,
                        manifest_sha256=engine.config.bundle.manifest_sha256,
                        model_sha256=engine.config.bundle.model_sha256,
                    )
                )
                emit_event(
                    "batch_attempt_completed", operation="batch", status="failed", batch_id=attempt
                )
                outcome["batch_id"] = attempt
                outcome["batch_status"] = "failed"
                outcome["input_sha256"] = digest(payload)
            else:
                expected_rejections = 1 if name == "invalid_values" else 2
                if (
                    name == "missing_columns"
                    or run.status != "completed_with_rejections"
                    or run.rejected_rows != expected_rejections
                ):
                    raise EvidenceError(
                        "Injected input failure did not produce the expected batch evidence."
                    )
                manifest = verify_batch_run(run.run_root, config=engine.config)
                emit_event(
                    "batch_attempt_completed",
                    operation="batch",
                    status=run.status,
                    batch_id=run.batch_id,
                    rejection_count=run.rejected_rows,
                )
                outcome.update(
                    batch_status=run.status,
                    batch_id=run.batch_id,
                    rejected_rows=run.rejected_rows,
                    input_sha256=manifest["input_sha256"],
                )
            restored = runtime / f"restored-{name}.csv"
            restored.write_bytes(content)
            recovery = run_batch(
                input_path=restored,
                as_of_date="2026-09-30",
                snapshot_id=f"restored-{name}",
                output_root=runtime / "batches",
                config=engine.config,
                engine=engine,
            )
            recovered_manifest = verify_batch_run(recovery.run_root, config=engine.config)
            if recovery.status != "completed" or recovery.rejected_rows != 0:
                raise EvidenceError("Corrected batch did not recover cleanly.")
            outcome.update(
                recovery_batch_id=recovery.batch_id,
                recovery_manifest_sha256=digest(encode(recovered_manifest)),
            )
            emit_event(
                "batch_attempt_completed",
                operation="batch",
                status=recovery.status,
                batch_id=recovery.batch_id,
                rejection_count=0,
            )
    # Recovery restores the original valid fixture/distribution; it never alters the model.
    if len(clean.rejections) or compare(reference, control)["status"] != "clear":
        raise EvidenceError("Incident control fixture did not recover.")
    for outcome in outcomes:
        outcome.update(recovered=True, verified=True)
    return outcomes


def artifact_drill(folder: Path, engine: InferenceEngine) -> dict[str, Any]:
    bundle = folder / "corrupt-bundle"
    bundle.mkdir()
    source = ROOT / "models/selected_v1"
    shutil.copyfile(source / "manifest.json", bundle / "manifest.json")
    # Missing model first, then altered bytes: never touch the reviewed bundle.
    for content in (None, b"synthetic-corruption"):
        if content is not None:
            (bundle / "model.cbm").write_bytes(content)
        try:
            InferenceEngine(bundle_root=bundle)
        except InferenceError:
            pass
        else:
            raise EvidenceError("Unsafe artifact was accepted.")
    shutil.copyfile(source / "model.cbm", bundle / "model.cbm")
    recovered = InferenceEngine(bundle_root=bundle)
    fixture = parse_batch_csv(
        (ROOT / "tests/fixtures/inference_batch_v1.csv").read_bytes(), engine.config
    ).features
    if not np.array_equal(
        engine.score(fixture).probabilities, recovered.score(fixture).probabilities
    ):
        raise EvidenceError("Artifact recovery changed predictions.")
    return {
        "drill": "artifact_integrity",
        "detected": True,
        "contained": True,
        "recovered": True,
        "verified": True,
    }


def build(
    expected_benchmark_sha256: str,
    benchmark_root: str = "reports/monitoring/benchmark_v1",
    output: str = OUTPUT,
    runtime: str = "experiment/incidents/release_b_v1",
) -> str:
    commit = clean_commit()
    frozen = verify(benchmark_root, expected_benchmark_sha256, "monitor_benchmark_v1")
    folder = safe_path(runtime, "experiment/incidents")
    if folder.exists():
        raise EvidenceError("Incident runtime already exists.")
    folder.mkdir(parents=True)
    engine = InferenceEngine()
    outcomes = data_drills(engine, folder)
    outcomes.append(artifact_drill(folder, engine))
    # Phase 7 permits only its approved runtime subtrees. Isolate new paths within them.
    registry_folder = safe_path("experiment/registry/release_b_drill", "experiment/registry")
    deployment_folder = safe_path(
        "experiment/deployments/release_b_drill", "experiment/deployments"
    )
    if registry_folder.exists() or deployment_folder.exists():
        raise EvidenceError("Isolated rollback rehearsal paths already exist.")
    outcomes.append(rollback_paths(registry_folder, deployment_folder))
    before = platform_state()
    command([*COMPOSE, "stop", "api"])
    try:
        try:
            request("/ready")
        except OSError:
            detected = True
        else:
            detected = False
        emit_event(
            "service_health_probe", route="/ready", status="unavailable" if detected else "ready"
        )
    finally:
        started = time.perf_counter()
        command([*COMPOSE, "start", "api"])
        await_ready()
        recovery = time.perf_counter() - started
    if (
        not detected
        or recovery > frozen["targets"]["recovery_seconds"]
        or platform_state() != before
    ):
        raise EvidenceError("Service interruption detection, recovery or persistence failed.")
    if (
        request("/v1/predict", read_json(ROOT / "tests/fixtures/prediction_request.json"))[
            "probability_of_default"
        ]
        != 0.190382
    ):
        raise EvidenceError("Recovered platform prediction changed.")
    outcomes.append(
        {
            "drill": "service_interruption",
            "detected": True,
            "contained": True,
            "recovered": True,
            "recovery_seconds": recovery,
            "verified": True,
        }
    )
    for outcome in outcomes:
        outcome["disposition"] = "pending_owner_review"
        outcome["runbook"] = outcome["drill"]
    return publish(
        safe_path(output, "reports/incidents"),
        kind=KIND,
        commit=commit,
        sources=source_map(
            [
                str(Path(benchmark_root) / "evidence-manifest.json"),
                "docs/operations/release-b-runbooks.md",
            ]
        ),
        summary={
            "status": "pending_owner_review",
            "drills": outcomes,
            "historical_state_modified": False,
            "postgres_rollback_claimed": False,
        },
    )


def rollback_paths(registry: Path, deployment: Path) -> dict[str, Any]:
    # Phase 7 deliberately accepts repository-relative governed paths only.
    registry = registry.relative_to(ROOT) if registry.is_absolute() else registry
    deployment = deployment.relative_to(ROOT) if deployment.is_absolute() else deployment
    register_release_revisions(registry_root=registry)
    promote = Path("configs/registry/phase7_promotion_approval.json")
    rollback = Path("configs/registry/phase7_rollback_approval.json")
    promote_candidate(
        registry_root=registry,
        approval_path=promote,
        expected_approval_sha256="8c7b9fbd4bdecbf1e632d5f307e706adb5c2a7fd272a41d2471a087e6915cad4",
    )
    deploy_champion(registry_root=registry, deployment_root=deployment)
    rollback_release(
        registry_root=registry,
        deployment_root=deployment,
        approval_path=rollback,
        expected_approval_sha256="466f3b7f98480603035d907c6ff7a713c43febe0e7f3447e0d1233a3cd7fce43",
    )
    state = registry_status(registry_root=registry, deployment_root=deployment)
    if state["aliases"] != {"champion": "1", "rollback": "2"}:
        raise EvidenceError("Rollback aliases are invalid.")
    from fastapi.testclient import TestClient

    from credit_risk.inference.api import create_app
    from credit_risk.registry.deployment import DeploymentResolutionError, resolve_active_bundle

    # Faults are injected only into the newly created isolated deployment tree.
    pointer = ROOT / deployment / "active.json"
    original = pointer.read_bytes()
    try:
        for corrupt in (None, b"{}"):
            if corrupt is None:
                pointer.unlink()
            else:
                pointer.write_bytes(corrupt)
            try:
                resolve_active_bundle(ROOT / deployment)
            except DeploymentResolutionError:
                pass
            else:
                raise EvidenceError("Missing/corrupt deployment pointer was accepted.")
    finally:
        pointer.write_bytes(original)
    with TestClient(create_app(bundle_root=resolve_active_bundle(ROOT / deployment))) as client:
        response = client.post(
            "/v1/predict", json=read_json(ROOT / "tests/fixtures/prediction_request.json")
        )
        if (
            client.get("/ready").status_code != 200
            or response.status_code != 200
            or response.json()["probability_of_default"] != 0.190382
        ):
            raise EvidenceError("Rolled-back API readiness or parity failed.")
    return {
        "drill": "phase7_sqlite_rollback",
        "detected": True,
        "contained": True,
        "recovered": True,
        "prediction": 0.190382,
        "verified": True,
        "deployment_pointer_failures_refused": 2,
        "restored_pointer_sha256": digest(original),
    }
