"""Monitoring publications reconcile source inputs with verified batch outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.assurance.evidence import (
    EvidenceError,
    clean_commit,
    hash_file,
    publish,
    safe_path,
    source_map,
    verify,
)
from credit_risk.assurance.validation import check_baseline, validation_cohort
from credit_risk.inference.batch import parse_batch_csv, verify_batch_run
from credit_risk.inference.contracts import load_inference_config
from credit_risk.inference.engine import InferenceEngine
from credit_risk.inference.logging import ALLOWED_LOG_FIELDS
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS, REPAYMENT_STATUS_COLUMNS
from credit_risk.monitoring.drift import compare, profile

REFERENCE = "reports/monitoring/reference_v1"
SOURCE_FILES = [
    "configs/inference/phase6_v1.json",
    "models/selected_v1/manifest.json",
    "configs/governance/phase5_v1.json",
    "docs/adr/0009-release-b-operational-assurance.md",
]


def reference(data_root: str = "data", output: str = REFERENCE) -> str:
    commit = clean_commit()
    destination = safe_path(output, "reports/monitoring")
    if destination.exists():
        raise EvidenceError("Reference already exists.")
    features, target = validation_cohort(data_root)
    engine = InferenceEngine()
    scored = engine.score(features)
    check_baseline(target.to_numpy(), scored.probabilities)
    profiles = {
        column: profile(features[column], column in REPAYMENT_STATUS_COLUMNS)
        for column in PREDICTOR_COLUMNS
    }
    profiles["probability"] = profile(scored.probabilities)
    return publish(
        destination,
        kind="monitor_reference_v1",
        commit=commit,
        sources=source_map(SOURCE_FILES),
        summary={
            "status": "reference_ready",
            "rows": 4800,
            "profiles": profiles,
            "model_sha256": engine.config.bundle.model_sha256,
            "risk_bands": pd.Series(scored.risk_bands).value_counts().sort_index().to_dict(),
            "thresholds": {"warning": 0.1, "investigate": 0.2, "minimum_rows": 200},
        },
    )


def batch(
    input_path: str,
    run_root: str,
    reference_root: str,
    expected_reference_sha256: str,
    output: str,
) -> str:
    commit = clean_commit()
    destination = safe_path(output, "reports/monitoring")
    config = load_inference_config()
    ref = verify(reference_root, expected_reference_sha256, "monitor_reference_v1")
    manifest = verify_batch_run(safe_path(run_root, "experiment"), config=config)
    content = safe_path(input_path).read_bytes()
    if hash_file(input_path) != manifest["input_sha256"]:
        raise EvidenceError("Monitoring input does not match the scored snapshot.")
    if ref["model_sha256"] != config.bundle.model_sha256:
        raise EvidenceError("Monitoring reference model differs from the scored model.")
    parsed = parse_batch_csv(content, config)
    scores = pd.read_csv(
        safe_path(Path(run_root) / "scores.csv"), dtype={"account_id": str}, keep_default_na=False
    )
    if set(scores.account_id) != set(parsed.account_ids):
        raise EvidenceError("Monitoring account coverage differs from batch output.")
    scores = scores.set_index("account_id").loc[list(parsed.account_ids)]
    # Batch output uses full-precision probability; account IDs never enter reports.
    probability_column = "probability_of_default"
    checks = {
        column: compare(ref["profiles"][column], parsed.features[column])
        for column in PREDICTOR_COLUMNS
    }
    checks["probability"] = compare(ref["profiles"]["probability"], scores[probability_column])
    severity = {"clear": 0, "warning": 1, "investigate": 2, "insufficient_data": 3}
    status = max((item["status"] for item in checks.values()), key=severity.__getitem__)
    summary = {
        "status": status,
        "batch_id": manifest["batch_id"],
        "batch_status": manifest["status"],
        "input_sha256": manifest["input_sha256"],
        "reference_sha256": expected_reference_sha256,
        "model_sha256": config.bundle.model_sha256,
        "checks": checks,
        "counts": {
            "input": parsed.input_rows,
            "valid": len(parsed.account_ids),
            "rejected": len(parsed.rejections),
        },
        "rejection_rate": len(parsed.rejections) / parsed.input_rows,
        "risk_bands": scores.risk_band.value_counts().sort_index().to_dict(),
        "action": "human_investigation" if status in {"warning", "investigate"} else status,
        "automatic_model_change": False,
    }
    # Runtime CSVs are deliberately not required by clean-checkout verification.
    sources = source_map(SOURCE_FILES + [str(Path(reference_root) / "evidence-manifest.json")])
    return publish(
        destination, kind="monitor_batch_v1", summary=summary, sources=sources, commit=commit
    )


def summarize_events(lines: list[str]) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except (ValueError, TypeError) as error:
            raise EvidenceError("Service input must contain JSON events only.") from error
        if not isinstance(event, dict) or set(event) - ALLOWED_LOG_FIELDS:
            raise EvidenceError("Service event contains unapproved fields.")
        name = event.get("event")
        if name == "api_request_completed":
            duration = event.get("duration_ms")
            status = event.get("status")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (float, int))
                or not np.isfinite(duration)
                or duration < 0
                or not isinstance(status, str)
                or not status.isdigit()
                or not 100 <= int(status) <= 599
            ):
                raise EvidenceError("Malformed request measurement.")
            requests.append(event)
        elif name == "batch_attempt_completed":
            if event.get("status") not in {"completed", "completed_with_rejections", "failed"}:
                raise EvidenceError("Malformed batch status.")
            batches.append(event)
        elif name == "service_health_probe":
            if event.get("status") not in {"ready", "unavailable"}:
                raise EvidenceError("Malformed health probe.")
            health.append(event)
    durations = [x["duration_ms"] for x in requests]
    server_errors = sum(int(x["status"]) >= 500 for x in requests)
    return {
        "status": "measured" if requests or batches or health else "insufficient_data",
        "request_count": len(requests),
        "server_error_count": server_errors,
        "client_error_count": sum(400 <= int(x["status"]) < 500 for x in requests),
        "server_error_rate": server_errors / len(requests) if requests else None,
        "p95_latency_ms": float(np.percentile(durations, 95)) if durations else None,
        "batch_attempt_count": len(batches),
        "batch_failure_count": sum(x["status"] == "failed" for x in batches),
        "health_probe_count": len(health),
        "health_failure_count": sum(x["status"] == "unavailable" for x in health),
    }


def service(log_path: str, output: str) -> str:
    commit = clean_commit()
    logs = safe_path(log_path, "experiment")
    summary = summarize_events(logs.read_text(encoding="utf-8").splitlines())
    summary["runtime_log_sha256"] = hash_file(logs)
    return publish(
        safe_path(output, "reports/monitoring"),
        kind="monitor_service_v1",
        summary=summary,
        sources=source_map(SOURCE_FILES),
        commit=commit,
    )


def verify_report(root: str, expected: str, kind: str) -> dict[str, Any]:
    if kind not in {"reference", "batch", "service"}:
        raise EvidenceError("Unknown monitoring report kind.")
    return verify(root, expected, f"monitor_{kind}_v1")
