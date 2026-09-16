"""Atomic, idempotent monthly scoring for operational CSV snapshots."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pydantic import ValidationError

from credit_risk.inference.contracts import (
    ACCOUNT_ID_PATTERN,
    PHASE6_CONFIG_SHA256,
    InferenceConfig,
    OperationalFeatures,
)
from credit_risk.inference.engine import InferenceEngine, InferenceResult
from credit_risk.inference.logging import emit_event
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS

SCORES_COLUMNS = (
    "portfolio_rank",
    "selected_for_review",
    "account_id",
    "as_of_date",
    "snapshot_id",
    "probability_of_default",
    "risk_band",
    "primary_reason_category",
    "primary_reason_direction",
    "primary_reason_contribution_raw_log_odds",
    "secondary_reason_category",
    "secondary_reason_direction",
    "secondary_reason_contribution_raw_log_odds",
    "trace_id",
    "model_id",
    "bundle_id",
    "policy_id",
)
REJECTION_COLUMNS = ("source_row_number", "account_id", "rule_ids")
EXPECTED_RUN_FILES = {"scores.csv", "rejections.csv", "manifest.json"}
INTEGER_PATTERN = re.compile(r"^-?(0|[1-9][0-9]*)$")
SAFE_ID_PATTERN = re.compile(ACCOUNT_ID_PATTERN)


class BatchInferenceError(RuntimeError):
    """Raised when a scoring batch cannot be parsed, published, or verified."""


@dataclass(frozen=True, slots=True)
class Rejection:
    source_row_number: int
    account_id: str
    rule_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedBatch:
    input_rows: int
    account_ids: tuple[str, ...]
    features: pd.DataFrame
    rejections: tuple[Rejection, ...]


@dataclass(frozen=True, slots=True)
class BatchRunResult:
    run_root: Path
    batch_id: str
    status: str
    exit_code: int
    valid_rows: int
    rejected_rows: int
    reused: bool


@dataclass(slots=True)
class _CandidateRow:
    source_row_number: int
    account_id: str
    payload: dict[str, int]
    errors: set[str]


def run_batch(
    *,
    input_path: str | Path,
    as_of_date: str,
    snapshot_id: str,
    output_root: str | Path,
    config: InferenceConfig,
    engine: InferenceEngine,
) -> BatchRunResult:
    """Score one snapshot or verify and reuse an identical completed run."""

    scoring_date = _parse_date(as_of_date)
    if SAFE_ID_PATTERN.fullmatch(snapshot_id) is None:
        raise BatchInferenceError("Snapshot ID must match the reviewed safe identifier pattern.")
    source = Path(input_path)
    try:
        input_bytes = source.read_bytes()
    except OSError as error:
        raise BatchInferenceError(f"Unable to read batch input: {error}") from error
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    parsed = parse_batch_csv(input_bytes, config)
    identity = {
        "input_sha256": input_sha256,
        "as_of_date": scoring_date.isoformat(),
        "snapshot_id": snapshot_id,
        "config_sha256": PHASE6_CONFIG_SHA256,
        "manifest_sha256": config.bundle.manifest_sha256,
        "model_sha256": config.bundle.model_sha256,
    }
    batch_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_root = Path(output_root) / scoring_date.isoformat() / snapshot_id
    if run_root.exists():
        manifest = verify_batch_run(run_root, config=config, expected_batch_id=batch_id)
        return BatchRunResult(
            run_root=run_root,
            batch_id=batch_id,
            status=str(manifest["status"]),
            exit_code=int(config.batch.exit_codes[str(manifest["status"])]),
            valid_rows=int(manifest["counts"]["valid_rows"]),
            rejected_rows=int(manifest["counts"]["rejected_rows"]),
            reused=True,
        )

    inference: InferenceResult | None = None
    if parsed.account_ids:
        inference = engine.score(parsed.features)
    status = (
        "failed"
        if not parsed.account_ids
        else "completed_with_rejections"
        if parsed.rejections
        else "completed"
    )
    scores_bytes, selected_count, band_counts = _scores_bytes(
        parsed.account_ids,
        inference,
        scoring_date.isoformat(),
        snapshot_id,
        batch_id,
        config,
    )
    rejections_bytes = _rejections_bytes(parsed.rejections)
    output_hashes = {
        "scores.csv": hashlib.sha256(scores_bytes).hexdigest(),
        "rejections.csv": hashlib.sha256(rejections_bytes).hexdigest(),
    }
    manifest = {
        "schema_version": "1.0.0",
        "protocol_id": config.protocol_id,
        "batch_id": batch_id,
        "status": status,
        "as_of_date": scoring_date.isoformat(),
        "snapshot_id": snapshot_id,
        "input_sha256": input_sha256,
        "config_sha256": PHASE6_CONFIG_SHA256,
        "model": {
            "model_id": config.bundle.model_id,
            "bundle_id": config.bundle.bundle_id,
            "manifest_sha256": config.bundle.manifest_sha256,
            "model_sha256": config.bundle.model_sha256,
        },
        "policy": {
            "policy_id": config.policy.policy_id,
            "review_capacity_fraction": config.policy.review_capacity_fraction,
            "review_capacity_rounding": config.policy.review_capacity_rounding,
            "selected_rows": selected_count,
        },
        "counts": {
            "input_rows": parsed.input_rows,
            "valid_rows": len(parsed.account_ids),
            "rejected_rows": len(parsed.rejections),
            "risk_bands": band_counts,
        },
        "explanation": {
            "method": config.explanation.method,
            "space": config.explanation.space,
            "top_reason_count": config.explanation.top_reason_count,
            "language_policy": config.explanation.language_policy,
            "maximum_additivity_error": (
                inference.max_additivity_error if inference is not None else None
            ),
            "maximum_probability_error": (
                inference.max_probability_error if inference is not None else None
            ),
        },
        "outputs": output_hashes,
        "privacy": {
            "row_level_values_in_manifest": False,
            "local_paths_in_manifest": False,
            "wall_clock_timestamps_in_manifest": False,
        },
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
    _publish_run(
        run_root,
        {
            "scores.csv": scores_bytes,
            "rejections.csv": rejections_bytes,
            "manifest.json": manifest_bytes,
        },
    )
    emit_event(
        "batch_completed",
        operation="inference_batch",
        status=status,
        batch_id=batch_id,
        model_id=config.bundle.model_id,
        bundle_id=config.bundle.bundle_id,
        policy_id=config.policy.policy_id,
        row_count=len(parsed.account_ids),
        rejection_count=len(parsed.rejections),
    )
    return BatchRunResult(
        run_root=run_root,
        batch_id=batch_id,
        status=status,
        exit_code=config.batch.exit_codes[status],
        valid_rows=len(parsed.account_ids),
        rejected_rows=len(parsed.rejections),
        reused=False,
    )


def parse_batch_csv(content: bytes, config: InferenceConfig) -> ParsedBatch:
    """Parse strict UTF-8 CSV while preserving row-level validation failures."""

    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise BatchInferenceError("Batch input must be valid UTF-8.") from error
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise BatchInferenceError(f"Batch input is malformed CSV: {error}") from error
    if not rows:
        raise BatchInferenceError("Batch input is empty.")
    expected_header = list(config.input.columns)
    if rows[0] != expected_header:
        raise BatchInferenceError("Batch input headers must match the reviewed order exactly.")
    if len(rows) == 1:
        raise BatchInferenceError("Batch input contains no account rows.")

    candidates: list[_CandidateRow] = []
    for source_row_number, values in enumerate(rows[1:], start=2):
        if len(values) != len(expected_header):
            raise BatchInferenceError(
                f"Batch input row {source_row_number} has the wrong number of columns."
            )
        account_id = values[0]
        errors: set[str] = set()
        safe_account_id = account_id if SAFE_ID_PATTERN.fullmatch(account_id) else ""
        if not safe_account_id:
            errors.add("invalid_account_id")
        payload: dict[str, int] = {}
        for column, value in zip(PREDICTOR_COLUMNS, values[1:], strict=True):
            if INTEGER_PATTERN.fullmatch(value) is None:
                errors.add(f"invalid_{column}")
                continue
            payload[column] = int(value)
        if len(payload) == len(PREDICTOR_COLUMNS):
            try:
                OperationalFeatures.model_validate(payload)
            except ValidationError as error:
                for item in error.errors():
                    field = str(item["loc"][0]) if item["loc"] else "row"
                    errors.add(f"invalid_{field}")
        candidates.append(
            _CandidateRow(
                source_row_number=source_row_number,
                account_id=safe_account_id,
                payload=payload,
                errors=errors,
            )
        )

    counts = Counter(row.account_id for row in candidates if row.account_id)
    for row in candidates:
        if row.account_id and counts[row.account_id] > 1:
            row.errors.add("duplicate_account_id")

    valid = [row for row in candidates if not row.errors]
    rejections = tuple(
        Rejection(row.source_row_number, row.account_id, tuple(sorted(row.errors)))
        for row in candidates
        if row.errors
    )
    frame = pd.DataFrame(
        [row.payload for row in valid],
        index=pd.Index([row.account_id for row in valid], name="account_id"),
        columns=PREDICTOR_COLUMNS,
    )
    return ParsedBatch(
        input_rows=len(candidates),
        account_ids=tuple(row.account_id for row in valid),
        features=frame,
        rejections=rejections,
    )


def verify_batch_run(
    run_root: str | Path,
    *,
    config: InferenceConfig,
    expected_batch_id: str | None = None,
) -> dict[str, Any]:
    """Verify an existing batch without loading or scoring a model."""

    root = Path(run_root)
    if not root.is_dir():
        raise BatchInferenceError("Batch run root does not exist or is not a directory.")
    try:
        observed = {path.name for path in root.iterdir() if path.is_file()}
    except OSError as error:
        raise BatchInferenceError(f"Unable to inspect batch run: {error}") from error
    if observed != EXPECTED_RUN_FILES:
        raise BatchInferenceError("Existing batch files differ from the reviewed allowlist.")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BatchInferenceError(f"Unable to parse batch manifest: {error}") from error
    if expected_batch_id is not None and manifest.get("batch_id") != expected_batch_id:
        raise BatchInferenceError("Existing snapshot conflicts with the requested batch identity.")
    if manifest.get("config_sha256") != PHASE6_CONFIG_SHA256:
        raise BatchInferenceError("Existing batch uses a different inference configuration.")
    model = manifest.get("model", {})
    if model != {
        "model_id": config.bundle.model_id,
        "bundle_id": config.bundle.bundle_id,
        "manifest_sha256": config.bundle.manifest_sha256,
        "model_sha256": config.bundle.model_sha256,
    }:
        raise BatchInferenceError("Existing batch model lineage differs from the contract.")
    status = manifest.get("status")
    if status not in config.batch.statuses:
        raise BatchInferenceError("Existing batch has an invalid completion status.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"scores.csv", "rejections.csv"}:
        raise BatchInferenceError("Existing batch output digest allowlist is invalid.")
    for filename, expected_hash in outputs.items():
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise BatchInferenceError(f"Existing batch output digest is invalid for {filename}.")
        try:
            observed_hash = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        except OSError as error:
            raise BatchInferenceError(f"Unable to read batch output {filename}: {error}") from error
        if observed_hash != expected_hash:
            raise BatchInferenceError(f"Existing batch output digest mismatch for {filename}.")
    counts = manifest.get("counts", {})
    if not isinstance(counts, dict) or not all(
        isinstance(counts.get(name), int) for name in ("input_rows", "valid_rows", "rejected_rows")
    ):
        raise BatchInferenceError("Existing batch counts are invalid.")
    if counts["valid_rows"] + counts["rejected_rows"] != counts["input_rows"]:
        raise BatchInferenceError("Existing batch counts do not reconcile.")
    return manifest


def _scores_bytes(
    account_ids: tuple[str, ...],
    inference: InferenceResult | None,
    as_of_date: str,
    snapshot_id: str,
    batch_id: str,
    config: InferenceConfig,
) -> tuple[bytes, int, dict[str, int]]:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(SCORES_COLUMNS)
    if inference is None:
        return (
            stream.getvalue().encode("utf-8"),
            0,
            {name: 0 for name in ("standard", "elevated", "high", "critical")},
        )
    order = sorted(
        range(len(account_ids)),
        key=lambda index: (-float(inference.probabilities[index]), account_ids[index]),
    )
    selected_count = math.floor(len(account_ids) * config.policy.review_capacity_fraction)
    band_counts = {name: 0 for name in ("standard", "elevated", "high", "critical")}
    for rank, index in enumerate(order, start=1):
        reasons = inference.reasons[index]
        band = inference.risk_bands[index]
        band_counts[band] += 1
        trace_id = hashlib.sha256(f"{batch_id}|{account_ids[index]}".encode()).hexdigest()[:32]
        writer.writerow(
            (
                rank,
                str(rank <= selected_count).lower(),
                account_ids[index],
                as_of_date,
                snapshot_id,
                format(float(inference.probabilities[index]), ".17g"),
                band,
                reasons[0].category,
                reasons[0].direction,
                format(reasons[0].contribution_raw_log_odds, ".17g"),
                reasons[1].category,
                reasons[1].direction,
                format(reasons[1].contribution_raw_log_odds, ".17g"),
                trace_id,
                config.bundle.model_id,
                config.bundle.bundle_id,
                config.policy.policy_id,
            )
        )
    return stream.getvalue().encode("utf-8"), selected_count, band_counts


def _rejections_bytes(rejections: tuple[Rejection, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(REJECTION_COLUMNS)
    for rejection in rejections:
        writer.writerow(
            (
                rejection.source_row_number,
                rejection.account_id,
                "|".join(rejection.rule_ids),
            )
        )
    return stream.getvalue().encode("utf-8")


def _publish_run(run_root: Path, files: dict[str, bytes]) -> None:
    run_root.parent.mkdir(parents=True, exist_ok=True)
    # Keep the sibling stage name short enough for deeply nested Windows job paths.
    staging = run_root.with_name(f".stage-{uuid4().hex[:12]}")
    try:
        staging.mkdir()
        for filename, content in files.items():
            (staging / filename).write_bytes(content)
        _replace_directory_with_retry(staging, run_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _replace_directory_with_retry(source: Path, destination: Path) -> None:
    """Tolerate short-lived Windows sync locks without weakening atomic promotion."""

    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise BatchInferenceError("as-of-date must be an ISO date in YYYY-MM-DD form.") from error
    if parsed.isoformat() != value:
        raise BatchInferenceError("as-of-date must use canonical YYYY-MM-DD form.")
    return parsed
