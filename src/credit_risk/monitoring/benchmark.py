"""Separate measured rehearsals and acceptance; targets are frozen before acceptance."""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.assurance.evidence import (
    ROOT,
    EvidenceError,
    clean_commit,
    finite_number,
    hash_file,
    publish,
    read_json,
    safe_path,
    source_map,
    verify,
)
from credit_risk.assurance.runtime import COMPOSE, await_ready, command, request
from credit_risk.inference.batch import run_batch
from credit_risk.inference.engine import InferenceEngine

BENCHMARK = "reports/monitoring/benchmark_v1"
MEASURES = ("api_p95_ms", "batch_seconds", "recovery_seconds")


def hardware() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }


def synthetic_batch(path: Path, rows: int = 10000) -> None:
    original = pd.read_csv(ROOT / "tests/fixtures/inference_batch_v1.csv")
    frame = original.iloc[np.arange(rows) % len(original)].copy()
    frame["account_id"] = [f"synthetic-{i:06d}" for i in range(rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def measure(runtime: Path, iteration: str) -> dict[str, float]:
    payload = read_json(ROOT / "tests/fixtures/prediction_request.json")
    samples = []
    for index in range(220):
        started = time.perf_counter()
        result = request("/v1/predict", payload)
        elapsed = (time.perf_counter() - started) * 1000
        if result["probability_of_default"] != 0.190382 or result["model_id"] != "catboost_fixed":
            raise EvidenceError("Valid-request failure or synthetic parity drift.")
        if index >= 20:
            samples.append(elapsed)
    engine = InferenceEngine()
    input_path = runtime / "synthetic.csv"
    if not input_path.exists():
        synthetic_batch(input_path)
    started = time.perf_counter()
    result = run_batch(
        input_path=input_path,
        as_of_date="2026-09-30",
        snapshot_id=iteration,
        output_root=runtime / "batches",
        config=engine.config,
        engine=engine,
    )
    duration = time.perf_counter() - started
    if result.reused or result.status != "completed" or result.valid_rows != 10000:
        raise EvidenceError("Benchmark must perform one fresh complete synthetic batch.")
    started = time.perf_counter()
    command([*COMPOSE, "restart", "api"])
    await_ready()
    recovery = time.perf_counter() - started
    if request("/v1/predict", payload)["probability_of_default"] != 0.190382:
        raise EvidenceError("Recovery changed the smoke prediction.")
    return {
        "api_p95_ms": float(np.percentile(samples, 95)),
        "batch_seconds": duration,
        "recovery_seconds": recovery,
    }


def targets(rehearsals: list[dict[str, Any]]) -> dict[str, float]:
    if len(rehearsals) != 3:
        raise EvidenceError("Exactly three rehearsals are required.")
    return {
        key: 2 * max(finite_number(row[key], minimum=0.000001) for row in rehearsals)
        for key in MEASURES
    }


def check_acceptance(measured: dict[str, Any], limits: dict[str, Any]) -> None:
    if set(measured) != set(MEASURES) or set(limits) != set(MEASURES):
        raise EvidenceError("Benchmark measurement contract changed.")
    if any(
        finite_number(measured[key]) > finite_number(limits[key], minimum=0.000001)
        for key in MEASURES
    ):
        raise EvidenceError("Measured service performance exceeds a frozen target.")


def code_sources() -> dict[str, str]:
    files = sorted((ROOT / "src/credit_risk").rglob("*.py"))
    return source_map(files + [ROOT / "uv.lock", ROOT / "docker-compose.platform.yml"])


def rehearse(output: str = BENCHMARK, runtime: str = "experiment/monitoring/benchmark_v1") -> str:
    commit = clean_commit()
    folder = safe_path(runtime, "experiment/monitoring")
    if folder.exists():
        raise EvidenceError("Benchmark runtime already exists.")
    folder.mkdir(parents=True)
    trials = [measure(folder, f"rehearsal-{i}") for i in range(3)]
    summary = {
        "status": "targets_frozen",
        "hardware": hardware(),
        "rehearsals": trials,
        "targets": targets(trials),
        "valid_request_failures": 0,
        "synthetic_input_sha256": hash_file(folder / "synthetic.csv"),
    }
    return publish(
        safe_path(output, "reports/monitoring"),
        kind="monitor_benchmark_v1",
        summary=summary,
        sources=code_sources(),
        commit=commit,
    )


def acceptance(
    expected_benchmark_sha256: str,
    benchmark_root: str = BENCHMARK,
    output: str = "reports/monitoring/acceptance_v1",
    runtime: str = "experiment/monitoring/acceptance_v1",
) -> str:
    commit = clean_commit()
    frozen = verify(benchmark_root, expected_benchmark_sha256, "monitor_benchmark_v1")
    if frozen["hardware"] != hardware() or frozen["targets"] != targets(frozen["rehearsals"]):
        raise EvidenceError("Acceptance machine or frozen targets differ from rehearsal.")
    folder = safe_path(runtime, "experiment/monitoring")
    if folder.exists():
        raise EvidenceError("Acceptance runtime already exists.")
    folder.mkdir(parents=True)
    measured = measure(folder, "acceptance")
    check_acceptance(measured, frozen["targets"])
    if hash_file(folder / "synthetic.csv") != frozen["synthetic_input_sha256"]:
        raise EvidenceError("Acceptance fixture changed.")
    return publish(
        safe_path(output, "reports/monitoring"),
        kind="monitor_acceptance_v1",
        summary={
            "status": "passed",
            "measurements": measured,
            "targets": frozen["targets"],
            "hardware": hardware(),
            "benchmark_sha256": expected_benchmark_sha256,
            "valid_request_failures": 0,
            "prediction": 0.190382,
        },
        sources=source_map([str(Path(benchmark_root) / "evidence-manifest.json")]),
        commit=commit,
    )
