"""Measured isolated Compose startup/restart, scans and SBOM capture."""

from __future__ import annotations

import json
import time
from typing import Any

from credit_risk.assurance.evidence import (
    ROOT,
    EvidenceError,
    clean_commit,
    hash_file,
    read_json,
    safe_path,
    source_map,
)
from credit_risk.assurance.runtime import (
    COMPOSE,
    await_ready,
    command,
    platform_state,
    request,
    write_new,
)
from credit_risk.platform.evidence import SERVICES, SOURCE_FILES


def rehearse(runtime: str = "experiment/platform/release_b_v1", trivy: str = "trivy") -> str:
    commit = clean_commit()
    folder = safe_path(runtime, "experiment/platform")
    if folder.exists():
        raise EvidenceError("Rehearsal destination already exists.")
    if not (ROOT / ".env").is_file():
        raise EvidenceError("Create an ignored .env with local rehearsal credentials first.")
    # A new project may never adopt existing volumes or alter a historical deployment.
    if command(
        [
            "docker",
            "volume",
            "ls",
            "-q",
            "--filter",
            "label=com.docker.compose.project=credit-risk-release-b",
        ]
    ):
        raise EvidenceError("Isolated rehearsal volumes already exist; use the recorded state.")
    folder.mkdir(parents=True)
    command([*COMPOSE, "build"], timeout=1800)
    images: dict[str, Any] = {}
    scan_hashes = {}
    sbom_hashes = {}
    scanner_version = command([trivy, "--version"])
    for service in SERVICES:
        tag = {
            "api": "credit-risk-api:phase8",
            "mlflow": "credit-risk-mlflow:phase8",
            "ui": "credit-risk-ui:phase8",
        }[service]
        image_id = json.loads(command(["docker", "image", "inspect", tag]))[0]["Id"]
        images[service] = image_id
        scan = folder / f"{service}-scan.json"
        sbom = folder / f"{service}-sbom.json"
        command(
            [
                trivy,
                "image",
                "--scanners",
                "vuln",
                "--severity",
                "HIGH,CRITICAL",
                "--ignore-unfixed",
                "--exit-code",
                "1",
                "--format",
                "json",
                "--output",
                str(scan),
                image_id,
            ],
            timeout=1800,
        )
        command(
            [
                trivy,
                "image",
                "--scanners",
                "vuln",
                "--format",
                "cyclonedx",
                "--output",
                str(sbom),
                image_id,
            ],
            timeout=1800,
        )
        scan_hashes[service] = hash_file(scan)
        sbom_hashes[service] = hash_file(sbom)
    command([*COMPOSE, "up", "--detach", "--wait", "--wait-timeout", "240"])
    await_ready()
    first = platform_state()
    repeat = platform_state("bootstrap")
    payload = read_json(ROOT / "tests/fixtures/prediction_request.json")
    before = request("/v1/predict", payload)["probability_of_default"]
    command([*COMPOSE, "restart", "postgres", "minio", "mlflow", "api", "ui"])
    await_ready()
    wait_health()
    after = platform_state()
    final = request("/v1/predict", payload)["probability_of_default"]
    health = wait_health()
    infrastructure = {}
    for service in ("minio", "postgres"):
        container = command([*COMPOSE, "ps", "-q", service])
        infrastructure[service] = json.loads(command(["docker", "inspect", container]))[0]["Image"]
    receipt = {
        "implementation_commit": commit,
        "sources": source_map(SOURCE_FILES),
        "states": [first, repeat, after],
        "health": health,
        "images": images,
        "infrastructure_images": infrastructure,
        "smoke_probabilities": [before, final],
        "scanner_version": scanner_version,
        "scan_sha256": scan_hashes,
        "sbom_sha256": sbom_hashes,
    }
    write_new(folder / "receipt.json", receipt)
    return runtime


def wait_health(timeout: float = 240) -> dict[str, bool]:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            health = {
                "api": request("/ready") == {"status": "ready"},
                "mlflow": request("/health", port=5000) in {"OK", "ok"},
                "ui": request("/_stcore/health", port=8501) == "ok",
            }
            if all(health.values()):
                return health
        except OSError:
            pass
        time.sleep(0.25)
    raise EvidenceError("Platform services did not become healthy after restart.")
