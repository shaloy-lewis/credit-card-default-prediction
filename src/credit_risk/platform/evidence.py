"""Authenticated Phase 8 evidence from measured platform rehearsals."""

from __future__ import annotations

from typing import Any

from credit_risk.assurance.evidence import (
    ROOT,
    EvidenceError,
    clean_commit,
    encode,
    hash_file,
    publish,
    read_json,
    safe_path,
    source_map,
    verify,
)
from credit_risk.platform.contracts import load_platform_config

KIND = "phase8_platform_evidence_v1"
OUTPUT = "reports/platform/phase8_v1"
SERVICES = ("api", "mlflow", "ui")
SOURCE_FILES = [
    "configs/platform/phase8_v1.json",
    "docker-compose.platform.yml",
    "Dockerfile",
    "Dockerfile.platform",
    "Dockerfile.demo",
    "uv.lock",
    "reports/registry/phase7_v1/evidence-manifest.json",
    "models/selected_v1/manifest.json",
]


def validate_receipt(receipt: dict[str, Any], commit: str) -> None:
    config = load_platform_config(ROOT / SOURCE_FILES[0])
    if receipt.get("implementation_commit") != commit:
        raise EvidenceError("Platform rehearsal must match the clean implementation commit.")
    if receipt.get("sources") != source_map(SOURCE_FILES):
        raise EvidenceError("Platform rehearsal source identities changed.")
    expected = {
        "status": "ready",
        "registered_model_name": "credit-risk-default",
        "aliases": {"champion": "1", "rollback": "2"},
        "object_sha256": {
            "manifest.json": config.bundle.manifest_sha256,
            "model.cbm": config.bundle.model_sha256,
        },
        "active_revision": "phase7_rev_001",
        "fit_count": 0,
        "sealed_test_accessed": False,
    }
    states = receipt.get("states")
    if not isinstance(states, list) or len(states) != 3 or any(x != expected for x in states):
        raise EvidenceError("Startup, idempotency and restart states must agree exactly.")
    if receipt.get("health") != {"api": True, "mlflow": True, "ui": True}:
        raise EvidenceError("All platform services must be healthy after recovery.")
    if receipt.get("smoke_probabilities") != [0.190382, 0.190382]:
        raise EvidenceError("Synthetic prediction parity failed.")
    images = receipt.get("images", {})
    if set(images) != set(SERVICES) or any(
        not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71
        for value in images.values()
    ):
        raise EvidenceError("Missing built-image identities.")


def publish_evidence(runtime: str, output: str = OUTPUT) -> str:
    commit = clean_commit()
    folder = safe_path(runtime, "experiment/platform")
    receipt = read_json(folder / "receipt.json")
    validate_receipt(receipt, commit)
    extra = {"receipt.json": encode(receipt)}
    scan_summary = {}
    for service in SERVICES:
        scan = read_json(folder / f"{service}-scan.json")
        sbom = read_json(folder / f"{service}-sbom.json")
        if scan.get("Metadata", {}).get("ImageID") != receipt["images"][service]:
            raise EvidenceError("Scan is not bound to the measured image identity.")
        findings = [
            v
            for result in scan.get("Results", [])
            for v in result.get("Vulnerabilities", [])
            if v.get("Severity") in {"HIGH", "CRITICAL"} and v.get("FixedVersion")
        ]
        if findings:
            raise EvidenceError("Fixable HIGH/CRITICAL findings block publication.")
        if sbom.get("bomFormat") != "CycloneDX" or not sbom.get("components"):
            raise EvidenceError("Missing CycloneDX software inventory.")
        if receipt["scan_sha256"].get(service) != hash_file(folder / f"{service}-scan.json"):
            raise EvidenceError("Scan receipt hash mismatch.")
        if receipt["sbom_sha256"].get(service) != hash_file(folder / f"{service}-sbom.json"):
            raise EvidenceError("SBOM receipt hash mismatch.")
        extra[f"{service}-scan.json"] = encode(scan)
        extra[f"{service}-sbom.json"] = encode(sbom)
        scan_summary[service] = {"image_id": receipt["images"][service], "fixable_high_critical": 0}
    return publish(
        safe_path(output, "reports/platform"),
        kind=KIND,
        commit=commit,
        sources=source_map(SOURCE_FILES),
        extra=extra,
        summary={
            "status": "platform_verified",
            "states_verified": 3,
            "scans": scan_summary,
            "prediction": 0.190382,
            "runtime_receipt_sha256": hash_file(folder / "receipt.json"),
        },
    )


def verify_evidence(root: str, expected: str) -> dict[str, Any]:
    summary = verify(root, expected, KIND)
    folder = safe_path(root, "reports/platform")
    manifest = read_json(folder / "evidence-manifest.json")
    receipt = read_json(folder / "receipt.json")
    validate_receipt(receipt, manifest["implementation_commit"])
    if summary.get("status") != "platform_verified" or summary.get("states_verified") != 3:
        raise EvidenceError("Platform evidence is incomplete.")
    return summary
