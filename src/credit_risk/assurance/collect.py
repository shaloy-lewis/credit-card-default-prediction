"""Sequential Linux evidence collection for the approved Release B protocol.

This command never publishes a release approval or modifies historical evidence.
It is deliberately separate from routine CI and may perform synthetic scoring.
"""

from __future__ import annotations

import json
import logging

from credit_risk.assurance.evidence import ROOT, encode, read_json, verify
from credit_risk.assurance.runtime import COMPOSE, command, request, write_new
from credit_risk.incidents.workflow import build as incidents
from credit_risk.inference.batch import run_batch
from credit_risk.inference.engine import InferenceEngine
from credit_risk.inference.logging import LOGGER
from credit_risk.monitoring.benchmark import acceptance, synthetic_batch
from credit_risk.monitoring.benchmark import rehearse as benchmark
from credit_risk.monitoring.workflow import batch, reference, service
from credit_risk.platform.evidence import publish_evidence
from credit_risk.platform.evidence import verify_evidence as verify_platform
from credit_risk.platform.rehearsal import rehearse as platform
from credit_risk.release.release_b import NEW_EVIDENCE
from credit_risk.robustness.workflow import build as robustness
from credit_risk.robustness.workflow import verify_evidence as verify_robustness


def main() -> None:
    if (ROOT / "reports/platform/phase8_v1").exists():
        verify_collected()
        return
    # Each output is new; any failed measurement stops collection before sign-off.
    platform()
    platform_sha = publish_evidence("experiment/platform/release_b_v1")
    robustness_sha = robustness()
    reference_sha = reference()
    benchmark_sha = benchmark()
    acceptance_sha = acceptance(benchmark_sha)
    incident_log = ROOT / "experiment/release_b/incident-events.jsonl"
    incident_log.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(incident_log, mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    try:
        incident_sha = incidents(benchmark_sha)
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
    engine = InferenceEngine()
    operational = ROOT / "experiment/monitoring/operational"
    operational.mkdir(parents=True)
    input_path = operational / "synthetic.csv"
    synthetic_batch(input_path)
    result = run_batch(
        input_path=input_path,
        as_of_date="2026-09-30",
        snapshot_id="monitoring",
        output_root=operational / "batches",
        config=engine.config,
        engine=engine,
    )
    batch_sha = batch(
        str(input_path),
        str(result.run_root),
        "reports/monitoring/reference_v1",
        reference_sha,
        "reports/monitoring/batch_v1",
    )
    # Docker combines Uvicorn text and safe JSON. Preserve only reviewed event objects.
    raw = command([*COMPOSE, "logs", "--no-log-prefix", "--no-color", "api"])
    events = []
    for line in raw.splitlines() + incident_log.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("event"):
            events.append(event)
    events.extend(
        [
            {
                "event": "service_health_probe",
                "route": "/ready",
                "status": "ready" if request("/ready") == {"status": "ready"} else "unavailable",
            },
            {
                "event": "batch_attempt_completed",
                "operation": "batch",
                "status": result.status,
                "row_count": result.valid_rows,
                "rejection_count": result.rejected_rows,
            },
        ]
    )
    log = operational / "service.jsonl"
    log.write_bytes(b"".join(encode(x).replace(b"\n", b"") + b"\n" for x in events))
    service_sha = service(str(log), "reports/monitoring/service_v1")
    contract = read_json(ROOT / "configs/releases/release_b_v1.json")
    anchors = {
        "platform": platform_sha,
        "robustness": robustness_sha,
        "reference": reference_sha,
        "benchmark": benchmark_sha,
        "acceptance": acceptance_sha,
        "incidents": incident_sha,
        "batch_monitoring": batch_sha,
        "service_monitoring": service_sha,
    }
    for name, sha in anchors.items():
        contract["evidence"][name]["expected_manifest_sha256"] = sha
    contract["status"] = "evidence_collected_ci_and_owner_review_pending"
    write_new(ROOT / "experiment/release_b/proposed-release-contract.json", contract)
    print(encode({"status": "evidence_collected", "anchors": anchors}).decode())


def verify_collected() -> None:
    """A later report/approval commit authenticates existing evidence without scoring."""
    contract = read_json(ROOT / "configs/releases/release_b_v1.json")
    for name, kind in NEW_EVIDENCE.items():
        item = contract["evidence"][name]
        verify(item["root"], item["expected_manifest_sha256"], kind)
    for name, function in (("platform", verify_platform), ("robustness", verify_robustness)):
        item = contract["evidence"][name]
        function(item["root"], item["expected_manifest_sha256"])
    print(encode({"status": "existing_evidence_verified", "scoring_performed": False}).decode())


if __name__ == "__main__":
    main()
