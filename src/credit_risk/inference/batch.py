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
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from credit_risk.inference.contracts import (
    ACCOUNT_ID_PATTERN,
    PHASE6_CONFIG_SHA256,
    RESERVED_SNAPSHOT_IDS,
    InferenceConfig,
    OperationalFeatures,
)
from credit_risk.inference.engine import InferenceEngine, InferenceResult
from credit_risk.inference.logging import emit_event
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.modeling.risk_policy import risk_band

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
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RISK_BANDS = ("standard", "elevated", "high", "critical")
REJECTION_RULE_IDS = {
    "duplicate_account_id",
    "invalid_account_id",
    "invalid_column_count",
    *(f"invalid_{column}" for column in PREDICTOR_COLUMNS),
}


class BatchInferenceError(RuntimeError):
    """Raised when a scoring batch cannot be parsed, published, or verified."""


class _StrictManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())


class BatchModelManifest(_StrictManifestModel):
    model_id: Literal["catboost_fixed"]
    bundle_id: Literal["selected_v1"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BatchPolicyManifest(_StrictManifestModel):
    policy_id: Literal["outreach_top_10_v1"]
    review_capacity_fraction: float
    review_capacity_rounding: Literal["floor"]
    selected_rows: int = Field(ge=0)


class BatchCountsManifest(_StrictManifestModel):
    input_rows: int = Field(ge=1)
    valid_rows: int = Field(ge=0)
    rejected_rows: int = Field(ge=0)
    risk_bands: dict[str, int]

    @model_validator(mode="after")
    def validate_risk_bands(self) -> BatchCountsManifest:
        if set(self.risk_bands) != set(RISK_BANDS) or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in self.risk_bands.values()
        ):
            raise ValueError("risk-band counts must contain four non-negative integers")
        return self


class BatchExplanationManifest(_StrictManifestModel):
    method: Literal["catboost_native_shap_values"]
    space: Literal["raw_log_odds"]
    top_reason_count: Literal[2]
    language_policy: Literal["model_attribution_not_causal_or_adverse_action_reason"]
    maximum_additivity_error: float | None
    maximum_probability_error: float | None


class BatchPrivacyManifest(_StrictManifestModel):
    row_level_values_in_manifest: Literal[False]
    local_paths_in_manifest: Literal[False]
    wall_clock_timestamps_in_manifest: Literal[False]


class BatchManifest(_StrictManifestModel):
    schema_version: Literal["1.0.0"]
    protocol_id: Literal["phase6_v1"]
    batch_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["completed", "completed_with_rejections", "failed"]
    as_of_date: str
    snapshot_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: BatchModelManifest
    policy: BatchPolicyManifest
    counts: BatchCountsManifest
    explanation: BatchExplanationManifest
    outputs: dict[str, str]
    privacy: BatchPrivacyManifest

    @model_validator(mode="after")
    def validate_output_contract(self) -> BatchManifest:
        if set(self.outputs) != {"scores.csv", "rejections.csv"} or any(
            not isinstance(digest, str) or HEX64_PATTERN.fullmatch(digest) is None
            for digest in self.outputs.values()
        ):
            raise ValueError("output digests must cover the two reviewed CSV files")
        return self


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
    if (
        SAFE_ID_PATTERN.fullmatch(snapshot_id) is None
        or snapshot_id in config.batch.reserved_snapshot_ids
    ):
        raise BatchInferenceError(
            "Snapshot ID must match the reviewed safe identifier pattern and cannot be '.' or '..'."
        )
    source = Path(input_path)
    try:
        input_bytes = source.read_bytes()
    except OSError as error:
        raise BatchInferenceError(f"Unable to read batch input: {error}") from error
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    parsed = parse_batch_csv(input_bytes, config)
    identity = _batch_identity(
        input_sha256=input_sha256,
        as_of_date=scoring_date.isoformat(),
        snapshot_id=snapshot_id,
        config_sha256=PHASE6_CONFIG_SHA256,
        manifest_sha256=config.bundle.manifest_sha256,
        model_sha256=config.bundle.model_sha256,
    )
    batch_id = _batch_id(identity)
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
        account_id = values[0] if values else ""
        safe_account_id = account_id if SAFE_ID_PATTERN.fullmatch(account_id) else ""
        if len(values) != len(expected_header):
            width_errors = {"invalid_column_count"}
            if not safe_account_id:
                width_errors.add("invalid_account_id")
            candidates.append(
                _CandidateRow(
                    source_row_number=source_row_number,
                    account_id=safe_account_id,
                    payload={},
                    errors=width_errors,
                )
            )
            continue
        errors: set[str] = set()
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
    """Verify complete batch lineage and semantics without rescoring."""

    root = Path(run_root)
    if root.is_symlink() or not root.is_dir():
        raise BatchInferenceError("Batch run root does not exist or is not a directory.")
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise BatchInferenceError(f"Unable to inspect batch run: {error}") from error
    if {path.name for path in entries} != EXPECTED_RUN_FILES or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise BatchInferenceError("Existing batch files differ from the reviewed allowlist.")
    try:
        manifest = BatchManifest.model_validate_json((root / "manifest.json").read_bytes())
    except OSError as error:
        raise BatchInferenceError(f"Unable to parse batch manifest: {error}") from error
    except ValidationError as error:
        raise BatchInferenceError(f"Invalid batch manifest: {error}") from error

    _validate_manifest_contract(manifest, config, expected_batch_id)
    for filename, expected_hash in manifest.outputs.items():
        try:
            observed_hash = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        except OSError as error:
            raise BatchInferenceError(f"Unable to read batch output {filename}: {error}") from error
        if observed_hash != expected_hash:
            raise BatchInferenceError(f"Existing batch output digest mismatch for {filename}.")

    score_ids = _validate_scores_output(root / "scores.csv", manifest, config)
    rejection_ids = _validate_rejections_output(root / "rejections.csv", manifest)
    if score_ids & rejection_ids:
        raise BatchInferenceError("Scored and rejected account IDs must be disjoint.")
    return manifest.model_dump(mode="json")


def _validate_manifest_contract(
    manifest: BatchManifest,
    config: InferenceConfig,
    expected_batch_id: str | None,
) -> None:
    _parse_date(manifest.as_of_date)
    if (
        SAFE_ID_PATTERN.fullmatch(manifest.snapshot_id) is None
        or manifest.snapshot_id in RESERVED_SNAPSHOT_IDS
    ):
        raise BatchInferenceError("Existing batch snapshot ID is not path-safe.")
    if manifest.config_sha256 != PHASE6_CONFIG_SHA256:
        raise BatchInferenceError("Existing batch uses a different inference configuration.")
    if manifest.model.model_dump(mode="json") != {
        "model_id": config.bundle.model_id,
        "bundle_id": config.bundle.bundle_id,
        "manifest_sha256": config.bundle.manifest_sha256,
        "model_sha256": config.bundle.model_sha256,
    }:
        raise BatchInferenceError("Existing batch model lineage differs from the contract.")
    if (
        manifest.policy.policy_id != config.policy.policy_id
        or manifest.policy.review_capacity_fraction != config.policy.review_capacity_fraction
        or manifest.policy.review_capacity_rounding != config.policy.review_capacity_rounding
    ):
        raise BatchInferenceError("Existing batch policy differs from the contract.")

    counts = manifest.counts
    if counts.valid_rows + counts.rejected_rows != counts.input_rows:
        raise BatchInferenceError("Existing batch counts do not reconcile.")
    if sum(counts.risk_bands.values()) != counts.valid_rows:
        raise BatchInferenceError("Existing batch risk-band counts do not reconcile.")
    expected_selected = math.floor(counts.valid_rows * config.policy.review_capacity_fraction)
    if manifest.policy.selected_rows != expected_selected:
        raise BatchInferenceError("Existing batch selected-row count violates the capacity policy.")
    expected_status = (
        "failed"
        if counts.valid_rows == 0
        else "completed_with_rejections"
        if counts.rejected_rows
        else "completed"
    )
    if manifest.status != expected_status:
        raise BatchInferenceError("Existing batch status does not match its population counts.")

    explanation = manifest.explanation
    if (
        explanation.method != config.explanation.method
        or explanation.space != config.explanation.space
        or explanation.top_reason_count != config.explanation.top_reason_count
        or explanation.language_policy != config.explanation.language_policy
    ):
        raise BatchInferenceError("Existing batch explanation contract differs from the config.")
    additivity_error = explanation.maximum_additivity_error
    probability_error = explanation.maximum_probability_error
    if counts.valid_rows == 0:
        if additivity_error is not None or probability_error is not None:
            raise BatchInferenceError("Failed batches must not report explanation diagnostics.")
    else:
        if (
            additivity_error is None
            or probability_error is None
            or not math.isfinite(additivity_error)
            or not math.isfinite(probability_error)
            or additivity_error < 0.0
            or probability_error < 0.0
        ):
            raise BatchInferenceError("Existing batch explanation diagnostics are invalid.")
        if (
            additivity_error > config.explanation.additivity_absolute_tolerance
            or probability_error > config.explanation.probability_absolute_tolerance
        ):
            raise BatchInferenceError("Existing batch explanation diagnostics exceed tolerances.")

    identity = _batch_identity(
        input_sha256=manifest.input_sha256,
        as_of_date=manifest.as_of_date,
        snapshot_id=manifest.snapshot_id,
        config_sha256=manifest.config_sha256,
        manifest_sha256=manifest.model.manifest_sha256,
        model_sha256=manifest.model.model_sha256,
    )
    recomputed = _batch_id(identity)
    if manifest.batch_id != recomputed:
        raise BatchInferenceError("Existing batch ID does not match its recorded identity.")
    if expected_batch_id is not None and manifest.batch_id != expected_batch_id:
        raise BatchInferenceError("Existing snapshot conflicts with the requested batch identity.")


def _validate_scores_output(
    path: Path,
    manifest: BatchManifest,
    config: InferenceConfig,
) -> set[str]:
    rows = _read_output_rows(path, SCORES_COLUMNS, "scores")
    if len(rows) != manifest.counts.valid_rows:
        raise BatchInferenceError("Score rows do not match the manifest valid-row count.")

    seen: set[str] = set()
    ordering: list[tuple[float, str]] = []
    band_counts = Counter({name: 0 for name in RISK_BANDS})
    selected_rows = 0
    for expected_rank, values in enumerate(rows, start=1):
        row = dict(zip(SCORES_COLUMNS, values, strict=True))
        rank = _parse_output_integer(row["portfolio_rank"], "portfolio rank", minimum=1)
        if rank != expected_rank:
            raise BatchInferenceError("Score portfolio ranks must be contiguous and ordered.")
        if row["selected_for_review"] not in {"true", "false"}:
            raise BatchInferenceError("Score selection flags must be canonical booleans.")
        selected = row["selected_for_review"] == "true"
        expected_selected = expected_rank <= manifest.policy.selected_rows
        if selected != expected_selected:
            raise BatchInferenceError("Score selection flags violate the reviewed capacity policy.")
        selected_rows += int(selected)

        account_id = row["account_id"]
        if SAFE_ID_PATTERN.fullmatch(account_id) is None or account_id in seen:
            raise BatchInferenceError("Score account IDs must be safe and unique.")
        seen.add(account_id)
        if row["as_of_date"] != manifest.as_of_date or row["snapshot_id"] != manifest.snapshot_id:
            raise BatchInferenceError("Score snapshot identity differs from the manifest.")
        probability = _parse_output_float(
            row["probability_of_default"], "probability", minimum=0.0, maximum=1.0
        )
        observed_band = row["risk_band"]
        if observed_band != risk_band(probability, config.prediction.risk_band_thresholds):
            raise BatchInferenceError("Score risk band differs from its probability.")
        band_counts[observed_band] += 1

        categories = (
            row["primary_reason_category"],
            row["secondary_reason_category"],
        )
        if len(set(categories)) != 2 or not set(categories) <= set(
            config.explanation.reason_categories
        ):
            raise BatchInferenceError("Score reason categories violate the reviewed allowlist.")
        for prefix in ("primary", "secondary"):
            contribution = _parse_output_float(
                row[f"{prefix}_reason_contribution_raw_log_odds"],
                f"{prefix} reason contribution",
            )
            expected_direction = (
                config.explanation.direction_labels["positive"]
                if contribution > 0.0
                else config.explanation.direction_labels["negative"]
                if contribution < 0.0
                else "neutral"
            )
            if row[f"{prefix}_reason_direction"] != expected_direction:
                raise BatchInferenceError("Score reason direction differs from its contribution.")
        expected_trace = hashlib.sha256(f"{manifest.batch_id}|{account_id}".encode()).hexdigest()[
            :32
        ]
        if row["trace_id"] != expected_trace:
            raise BatchInferenceError("Score trace ID differs from the deterministic contract.")
        if (
            row["model_id"] != manifest.model.model_id
            or row["bundle_id"] != manifest.model.bundle_id
            or row["policy_id"] != manifest.policy.policy_id
        ):
            raise BatchInferenceError("Score model or policy lineage differs from the manifest.")
        ordering.append((probability, account_id))

    if ordering != sorted(ordering, key=lambda item: (-item[0], item[1])):
        raise BatchInferenceError("Score rows violate deterministic portfolio ordering.")
    if selected_rows != manifest.policy.selected_rows:
        raise BatchInferenceError("Score selection count differs from the manifest.")
    if dict(band_counts) != manifest.counts.risk_bands:
        raise BatchInferenceError("Score risk-band counts differ from the manifest.")
    return seen


def _validate_rejections_output(path: Path, manifest: BatchManifest) -> set[str]:
    rows = _read_output_rows(path, REJECTION_COLUMNS, "rejections")
    if len(rows) != manifest.counts.rejected_rows:
        raise BatchInferenceError("Rejection rows do not match the manifest rejected-row count.")
    seen_source_rows: set[int] = set()
    safe_ids: list[str] = []
    prior_source_row = 1
    parsed_rules: list[tuple[str, tuple[str, ...]]] = []
    for values in rows:
        row = dict(zip(REJECTION_COLUMNS, values, strict=True))
        source_row = _parse_output_integer(
            row["source_row_number"], "rejection source row", minimum=2
        )
        if (
            source_row in seen_source_rows
            or source_row <= prior_source_row
            or source_row > manifest.counts.input_rows + 1
        ):
            raise BatchInferenceError("Rejection source rows must be unique and ordered.")
        seen_source_rows.add(source_row)
        prior_source_row = source_row
        account_id = row["account_id"]
        if account_id and SAFE_ID_PATTERN.fullmatch(account_id) is None:
            raise BatchInferenceError("Rejection evidence contains an unsafe account ID.")
        if account_id:
            safe_ids.append(account_id)
        rules = tuple(row["rule_ids"].split("|"))
        if (
            not rules
            or any(not rule for rule in rules)
            or rules != tuple(sorted(set(rules)))
            or not set(rules) <= REJECTION_RULE_IDS
        ):
            raise BatchInferenceError("Rejection evidence contains invalid rule IDs.")
        if ("invalid_account_id" in rules) != (account_id == ""):
            raise BatchInferenceError("Rejection account ID differs from its validation rules.")
        if "duplicate_account_id" in rules and not account_id:
            raise BatchInferenceError("Duplicate-account rejection must retain a safe account ID.")
        parsed_rules.append((account_id, rules))

    all_safe_counts = Counter(safe_ids)
    duplicate_counts = Counter(
        account_id
        for account_id, rules in parsed_rules
        if account_id and "duplicate_account_id" in rules
    )
    if any(
        count < 2 or count != all_safe_counts[account_id]
        for account_id, count in duplicate_counts.items()
    ):
        raise BatchInferenceError("Duplicate-account rejection evidence is incomplete.")
    return set(safe_ids)


def _read_output_rows(
    path: Path,
    expected_columns: tuple[str, ...],
    description: str,
) -> list[list[str]]:
    try:
        content = path.read_bytes().decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(content, newline=""), strict=True))
    except (OSError, UnicodeError, csv.Error) as error:
        raise BatchInferenceError(f"Unable to parse batch {description} output: {error}") from error
    if not rows or rows[0] != list(expected_columns):
        raise BatchInferenceError(f"Batch {description} headers differ from the contract.")
    if any(len(row) != len(expected_columns) for row in rows[1:]):
        raise BatchInferenceError(f"Batch {description} output contains a malformed row.")
    return rows[1:]


def _parse_output_integer(value: str, description: str, *, minimum: int) -> int:
    if INTEGER_PATTERN.fullmatch(value) is None:
        raise BatchInferenceError(f"Batch {description} must be a canonical integer.")
    parsed = int(value)
    if parsed < minimum:
        raise BatchInferenceError(f"Batch {description} is outside its valid range.")
    return parsed


def _parse_output_float(
    value: str,
    description: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise BatchInferenceError(f"Batch {description} must be numeric.") from error
    if (
        not math.isfinite(parsed)
        or (minimum is not None and parsed < minimum)
        or (maximum is not None and parsed > maximum)
    ):
        raise BatchInferenceError(f"Batch {description} is outside its valid range.")
    return parsed


def _batch_identity(
    *,
    input_sha256: str,
    as_of_date: str,
    snapshot_id: str,
    config_sha256: str,
    manifest_sha256: str,
    model_sha256: str,
) -> dict[str, str]:
    return {
        "input_sha256": input_sha256,
        "as_of_date": as_of_date,
        "snapshot_id": snapshot_id,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "model_sha256": model_sha256,
    }


def _batch_id(identity: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
