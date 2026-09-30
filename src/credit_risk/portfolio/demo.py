"""Synthetic local walkthrough using existing API, batch CLI and monitoring contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from credit_risk.assurance.evidence import (
    ROOT,
    EvidenceError,
    clean_commit,
    encode,
    safe_path,
    verify,
)
from credit_risk.inference.batch import BatchInferenceError, verify_batch_run
from credit_risk.inference.client import InferenceClientError, predict_v1
from credit_risk.inference.contracts import InferenceContractError, load_inference_config
from credit_risk.inference.logging import ALLOWED_LOG_FIELDS
from credit_risk.monitoring.workflow import batch as monitor_batch

REFERENCE = "reports/monitoring/reference_v1"
REFERENCE_SHA256 = "5f2a43675cbd9f6ed44bf4a421df6647d3e780cf5b2e9eae6ba782e94849c4ac"
AS_OF_DATE = "2026-09-30"  # Synthetic fixture identity, never a claim of current customer data.
ROWS = 400


class DemoError(RuntimeError):
    """A walkthrough invariant failed; keep its isolated receipts for inspection."""


def synthetic_input(content: bytes) -> bytes:
    """Repeat the reviewed operational fixture with unique synthetic identifiers."""
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    rows = list(reader)
    if not reader.fieldnames or not rows or "account_id" not in reader.fieldnames:
        raise DemoError("The synthetic fixture is missing its header or rows.")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    for index in range(ROWS):
        writer.writerow({**rows[index % len(rows)], "account_id": f"synthetic-{index:06d}"})
    return output.getvalue().encode("utf-8")


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)


def _ready(base_url: str) -> None:
    with urllib.request.urlopen(f"{base_url}/ready", timeout=10) as response:
        if json.load(response) != {"status": "ready"}:
            raise DemoError("The local API is not ready.")


def _batch_cli(source: Path, run_id: str, runtime: Path, receipt: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "credit_risk.cli",
                "inference",
                "batch",
                "--input",
                str(source),
                "--as-of-date",
                AS_OF_DATE,
                "--snapshot-id",
                run_id,
                "--output-root",
                str(runtime / "batches"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DemoError("The batch CLI could not complete.") from error
    events = []
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("event"):
            if set(event) - ALLOWED_LOG_FIELDS:
                raise DemoError("Batch output contains unapproved event fields.")
            events.append(event)
    _write_new(
        runtime / receipt,
        b"".join((json.dumps(event, sort_keys=True) + "\n").encode() for event in events),
    )
    terminal = [event for event in events if event["event"] == "batch_attempt_completed"]
    if completed.returncode != 0 or len(terminal) != 1 or terminal[0].get("status") != "completed":
        raise DemoError("Synthetic batch failed; inspect the isolated event receipt.")
    attempt = terminal[0]
    if (
        re.fullmatch(r"[0-9a-f]{32}", str(attempt.get("trace_id", ""))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(attempt.get("batch_id", ""))) is None
    ):
        raise DemoError("Batch CLI did not return an invocation trace.")
    return attempt


def _snapshot(folder: Path) -> dict[str, tuple[str, int]]:
    return {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in folder.iterdir()
    }


def run_demo(run_id: str, api_url: str = "http://127.0.0.1:8080") -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", run_id) is None:
        raise DemoError("run_id must contain 1-48 safe letters, digits, hyphens or underscores.")
    url = urlsplit(api_url)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or url.username is not None
        or url.password is not None
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise DemoError(
            "Use an HTTP loopback API URL without credentials, path, query or fragment."
        )
    api_url = api_url.rstrip("/")
    commit = clean_commit()
    runtime = safe_path(f"experiment/portfolio/week11/{run_id}", "experiment")
    report = safe_path(f"reports/monitoring/release_c_demo/{run_id}", "reports/monitoring")
    if runtime.exists() or report.exists():
        raise DemoError("Run destination already exists; retain it and choose a fresh run_id.")
    reference = verify(REFERENCE, REFERENCE_SHA256, "monitor_reference_v1")
    config = load_inference_config()
    if reference["model_sha256"] != config.bundle.model_sha256:
        raise DemoError("Historical monitoring reference model differs from the reviewed model.")
    _ready(api_url)
    request = json.loads((ROOT / "tests/fixtures/prediction_request.json").read_bytes())
    response = predict_v1(request, base_url=api_url, request_id=f"release-c-{run_id}")
    if (
        response["probability_of_default"] != 0.190382
        or response["risk_band"] != "standard"
        or response["manifest_sha256"] != config.bundle.manifest_sha256
        or response["trace_id"] != f"release-c-{run_id}"
        or len({reason["category"] for reason in response["reasons"]}) != 2
        or not all(
            math.isfinite(reason["contribution_raw_log_odds"]) for reason in response["reasons"]
        )
    ):
        raise DemoError("The live API changed synthetic prediction, explanation or model identity.")
    runtime.mkdir(parents=True)
    _write_new(runtime / "api-response.json", encode(response))
    source = runtime / "synthetic.csv"
    _write_new(
        source, synthetic_input((ROOT / "tests/fixtures/inference_batch_v1.csv").read_bytes())
    )
    first = _batch_cli(source, run_id, runtime, "batch-first-events.jsonl")
    batch_root = runtime / "batches" / AS_OF_DATE / run_id
    manifest = verify_batch_run(batch_root, config=config, expected_batch_id=first.get("batch_id"))
    if (
        manifest["status"] != "completed"
        or manifest["counts"]["valid_rows"] != ROWS
        or manifest["counts"]["rejected_rows"] != 0
        or manifest["policy"]["selected_rows"] != 40
    ):
        raise DemoError("Synthetic batch counts or queue policy changed.")
    before = _snapshot(batch_root)
    second = _batch_cli(source, run_id, runtime, "batch-reuse-events.jsonl")
    if (
        first.get("batch_id") != second.get("batch_id")
        or first["trace_id"] == second["trace_id"]
        or _snapshot(batch_root) != before
    ):
        raise DemoError(
            "Idempotent reuse changed batch identity/files or reused an invocation trace."
        )
    monitor_sha = monitor_batch(
        str(source), str(batch_root), REFERENCE, REFERENCE_SHA256, str(report)
    )
    summary = verify(report, monitor_sha, "monitor_batch_v1")
    if (
        summary["status"] != "investigate"
        or summary["automatic_model_change"] is not False
        or summary["batch_id"] != manifest["batch_id"]
    ):
        raise DemoError("The synthetic monitoring result differs from the planned investigation.")
    result = {
        "status": "synthetic_walkthrough_passed_not_release_approval",
        "implementation_commit": commit,
        "prediction": response["probability_of_default"],
        "rows": ROWS,
        "selected_rows": 40,
        "batch_id": manifest["batch_id"],
        "input_sha256": manifest["input_sha256"],
        "batch_manifest_sha256": before["manifest.json"][0],
        "model_sha256": config.bundle.model_sha256,
        "batch_trace_id": first["trace_id"],
        "reuse_trace_id": second["trace_id"],
        "idempotent_no_rewrite": True,
        "monitoring_status": summary["status"],
        "reference_sha256": REFERENCE_SHA256,
        "monitoring_manifest_sha256": monitor_sha,
        "monitoring_root": report.relative_to(ROOT).as_posix(),
        "runtime_root": runtime.relative_to(ROOT).as_posix(),
        "disposition": "Expected synthetic shift; retain scores and demonstrate human investigation.",
        "automatic_model_change": False,
    }
    _write_new(runtime / "receipt.json", encode(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic local demo; requires clean committed code."
    )
    parser.add_argument(
        "--run-id", required=True, help="Fresh, safe identifier; existing runs are preserved."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    args = parser.parse_args(argv)
    try:
        result = run_demo(args.run_id, args.api_url)
    except (
        DemoError,
        EvidenceError,
        BatchInferenceError,
        InferenceClientError,
        InferenceContractError,
        OSError,
        ValueError,
    ) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
