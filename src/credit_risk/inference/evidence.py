"""Authenticated Phase 6 parity evidence assembled without model fitting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from credit_risk.inference.batch import parse_batch_csv, run_batch, verify_batch_run
from credit_risk.inference.contracts import (
    DEFAULT_INFERENCE_CONFIG_PATH,
    PHASE6_CONFIG_SHA256,
    load_inference_config,
)
from credit_risk.inference.engine import InferenceEngine
from credit_risk.modeling.tracking import collect_git_evidence

DEFAULT_FIXTURE = Path("tests/fixtures/inference_batch_v1.csv")
DEFAULT_RUNTIME_ROOT = Path("experiment/inference/phase6_v1")
DEFAULT_EVIDENCE_ROOT = Path("reports/inference/phase6_v1")
FIXTURE_SHA256 = "326d2fad6845f5ecf026b14a23bb8c5798d2ace7a776ccfd0f7bd5b5926bfd4a"
EVIDENCE_FILES = {
    "summary.json",
    "inference-parity-report.md",
    "evidence-manifest.json",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class InferenceEvidenceError(RuntimeError):
    """Raised when official parity evidence cannot be built or authenticated."""


@dataclass(frozen=True, slots=True)
class InferenceEvidenceResult:
    evidence_root: Path
    runtime_root: Path | None
    summary_sha256: str
    evidence_manifest_sha256: str
    batch_id: str


def build_inference_evidence(
    *,
    fixture_path: str | Path = DEFAULT_FIXTURE,
    config_path: str | Path = DEFAULT_INFERENCE_CONFIG_PATH,
    bundle_root: str | Path = "models/selected_v1",
    runtime_root: str | Path = DEFAULT_RUNTIME_ROOT,
    output_root: str | Path = DEFAULT_EVIDENCE_ROOT,
) -> InferenceEvidenceResult:
    """Run one synthetic parity check and publish only aggregate evidence."""

    config_file = Path(config_path)
    git = collect_git_evidence(config_file.resolve().parent)
    if git.repository_root is None:
        raise InferenceEvidenceError("Git repository root is unavailable.")
    if git.dirty:
        raise InferenceEvidenceError(
            "Official Phase 6 evidence requires a clean committed worktree."
        )
    repository = git.repository_root
    fixture = _safe_repository_file(repository, fixture_path, "synthetic fixture")
    bundle = _safe_repository_directory(repository, bundle_root, "selected bundle")
    config_file = _safe_repository_file(repository, config_file, "inference config")
    runtime = _safe_destination(
        repository,
        runtime_root,
        allowed_subtree=Path("experiment/inference"),
        description="inference runtime root",
    )
    output = _safe_destination(
        repository,
        output_root,
        allowed_subtree=Path("reports/inference"),
        description="inference evidence root",
    )
    if _overlaps(runtime, output):
        raise InferenceEvidenceError("Runtime and evidence roots must not overlap.")
    if runtime.exists() or output.exists():
        raise InferenceEvidenceError(
            "Refusing to overwrite existing Phase 6 runtime or published evidence."
        )

    fixture_bytes = _read_bytes(fixture, "synthetic fixture")
    fixture_sha256 = _sha256_bytes(fixture_bytes)
    if fixture_sha256 != FIXTURE_SHA256:
        raise InferenceEvidenceError(
            "Synthetic inference fixture differs from the reviewed evidence input."
        )
    config = load_inference_config(config_file)
    parsed = parse_batch_csv(fixture_bytes, config)
    if parsed.rejections or len(parsed.account_ids) != 20:
        raise InferenceEvidenceError(
            "Official synthetic fixture must contain exactly 20 valid unique records."
        )
    engine = InferenceEngine(bundle_root=bundle, config_path=config_file)
    offline = engine.score(parsed.features)

    batch = run_batch(
        input_path=fixture,
        as_of_date="2026-09-30",
        snapshot_id="phase6-parity-v1",
        output_root=runtime / "batches",
        config=config,
        engine=engine,
    )
    if batch.status != "completed" or batch.reused:
        raise InferenceEvidenceError("Official parity batch was not a fresh clean completion.")
    batch_manifest = verify_batch_run(
        batch.run_root, config=config, expected_batch_id=batch.batch_id
    )
    batch_rows = _read_batch_rows(batch.run_root / "scores.csv")
    comparison = _compare_offline_and_batch(
        account_ids=parsed.account_ids,
        offline=offline,
        batch_rows=batch_rows,
        selected_count=int(batch_manifest["policy"]["selected_rows"]),
    )

    api_comparison = _compare_api(
        parsed=parsed,
        offline=offline,
        batch_rows=batch_rows,
        bundle_root=bundle,
        config_path=config_file,
        tolerance=config.prediction.api_batch_absolute_tolerance,
    )
    before = {path.name: path.stat().st_mtime_ns for path in batch.run_root.iterdir()}
    reused = run_batch(
        input_path=fixture,
        as_of_date="2026-09-30",
        snapshot_id="phase6-parity-v1",
        output_root=runtime / "batches",
        config=config,
        engine=engine,
    )
    after = {path.name: path.stat().st_mtime_ns for path in batch.run_root.iterdir()}
    if not reused.reused or before != after:
        raise InferenceEvidenceError(
            "Identical batch rerun rewrote evidence instead of reusing it."
        )

    summary = _assemble_summary(
        implementation_commit=git.commit_sha,
        fixture_sha256=fixture_sha256,
        batch_id=batch.batch_id,
        batch_manifest=batch_manifest,
        comparison=comparison,
        api_comparison=api_comparison,
    )
    summary_bytes = _json_bytes(summary)
    report_bytes = _render_report(summary).encode("utf-8")
    artifacts = {
        "summary.json": summary_bytes,
        "inference-parity-report.md": report_bytes,
    }
    manifest = _assemble_manifest(
        implementation_commit=git.commit_sha,
        fixture_sha256=fixture_sha256,
        artifacts=artifacts,
        batch_manifest=batch_manifest,
    )
    artifacts["evidence-manifest.json"] = _json_bytes(manifest)
    manifest_sha256 = _sha256_bytes(artifacts["evidence-manifest.json"])
    _publish_directory(output, artifacts)
    verified = verify_inference_evidence(
        evidence_root=output.relative_to(repository),
        expected_manifest_sha256=manifest_sha256,
        config_path=config_file,
        bundle_root=bundle,
        fixture_path=fixture,
    )
    return InferenceEvidenceResult(
        evidence_root=output,
        runtime_root=runtime,
        summary_sha256=verified.summary_sha256,
        evidence_manifest_sha256=manifest_sha256,
        batch_id=batch.batch_id,
    )


def verify_inference_evidence(
    *,
    evidence_root: str | Path = DEFAULT_EVIDENCE_ROOT,
    expected_manifest_sha256: str,
    config_path: str | Path = DEFAULT_INFERENCE_CONFIG_PATH,
    bundle_root: str | Path = "models/selected_v1",
    fixture_path: str | Path = DEFAULT_FIXTURE,
) -> InferenceEvidenceResult:
    """Authenticate aggregate evidence without loading the model or runtime rows."""

    if HEX64.fullmatch(expected_manifest_sha256) is None:
        raise InferenceEvidenceError("Expected evidence-manifest digest must be SHA-256.")
    config_file = Path(config_path).resolve()
    repository = _repository_root(config_file)
    root = _safe_destination(
        repository,
        evidence_root,
        allowed_subtree=Path("reports/inference"),
        description="inference evidence root",
    )
    fixture = _safe_repository_file(repository, fixture_path, "synthetic fixture")
    bundle = _safe_repository_directory(repository, bundle_root, "selected bundle")
    observed_files = {path.name for path in root.iterdir()} if root.is_dir() else set()
    if observed_files != EVIDENCE_FILES:
        raise InferenceEvidenceError(
            f"Phase 6 evidence allowlist mismatch: observed={sorted(observed_files)}"
        )
    manifest_bytes = _read_bytes(root / "evidence-manifest.json", "evidence manifest")
    observed_manifest_sha256 = _sha256_bytes(manifest_bytes)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise InferenceEvidenceError(
            "Phase 6 evidence manifest does not match the externally reviewed digest."
        )
    manifest = _read_json_bytes(manifest_bytes, "evidence manifest")
    _validate_manifest_sources(
        manifest=manifest,
        fixture=fixture,
        config_path=config_file,
        bundle_root=bundle,
    )
    artifact_contract = manifest.get("artifacts")
    if not isinstance(artifact_contract, dict) or set(artifact_contract) != {
        "summary.json",
        "inference-parity-report.md",
    }:
        raise InferenceEvidenceError("Evidence manifest artifact allowlist is invalid.")
    for name, contract in artifact_contract.items():
        if not isinstance(contract, dict) or contract.get("row_level_data") is not False:
            raise InferenceEvidenceError(f"Evidence contract for {name} is invalid.")
        if _sha256_file(root / name) != contract.get("sha256"):
            raise InferenceEvidenceError(f"Published evidence digest mismatch for {name}.")

    summary = _read_json(root / "summary.json", "inference summary")
    _validate_summary(summary, manifest)
    report = _read_text(root / "inference-parity-report.md", "parity report")
    for required in (
        "Status: **complete**",
        "No model fitting",
        "20 synthetic accounts",
        "G4 remains open",
        "not causal or adverse-action reasons",
    ):
        if required not in report:
            raise InferenceEvidenceError(f"Parity report is missing reviewed text: {required}")
    serialized = json.dumps({"manifest": manifest, "summary": summary}, sort_keys=True)
    if re.search(r"[A-Za-z]:[\\/]", serialized) or "acct-" in serialized:
        raise InferenceEvidenceError("Aggregate evidence contains a local path or account ID.")
    return InferenceEvidenceResult(
        evidence_root=root,
        runtime_root=None,
        summary_sha256=_sha256_file(root / "summary.json"),
        evidence_manifest_sha256=observed_manifest_sha256,
        batch_id=str(summary["batch"]["batch_id"]),
    )


def _compare_offline_and_batch(
    *,
    account_ids: tuple[str, ...],
    offline: Any,
    batch_rows: list[dict[str, str]],
    selected_count: int,
) -> dict[str, Any]:
    by_id = {row["account_id"]: row for row in batch_rows}
    if len(by_id) != len(account_ids) or set(by_id) != set(account_ids):
        raise InferenceEvidenceError("Batch output does not cover the synthetic population once.")
    errors: list[float] = []
    band_mismatches = 0
    reason_category_mismatches = 0
    reason_direction_mismatches = 0
    for position, account_id in enumerate(account_ids):
        row = by_id[account_id]
        errors.append(
            abs(float(row["probability_of_default"]) - float(offline.probabilities[position]))
        )
        band_mismatches += row["risk_band"] != offline.risk_bands[position]
        expected_reasons = offline.reasons[position]
        reason_category_mismatches += [
            row["primary_reason_category"],
            row["secondary_reason_category"],
        ] != [reason.category for reason in expected_reasons]
        reason_direction_mismatches += [
            row["primary_reason_direction"],
            row["secondary_reason_direction"],
        ] != [reason.direction for reason in expected_reasons]

    expected_order = sorted(
        range(len(account_ids)),
        key=lambda index: (-float(offline.probabilities[index]), account_ids[index]),
    )
    observed_order = [row["account_id"] for row in batch_rows]
    if observed_order != [account_ids[index] for index in expected_order]:
        raise InferenceEvidenceError("Batch ranking differs from the reviewed deterministic rule.")
    expected_selected = math.floor(len(account_ids) * 0.1)
    observed_selected = sum(row["selected_for_review"] == "true" for row in batch_rows)
    if selected_count != expected_selected or observed_selected != expected_selected:
        raise InferenceEvidenceError("Batch review capacity differs from the frozen 10% policy.")
    if max(errors, default=0.0) != 0.0:
        raise InferenceEvidenceError("Batch probabilities differ from shared-engine output.")
    if band_mismatches or reason_category_mismatches or reason_direction_mismatches:
        raise InferenceEvidenceError("Batch bands or reviewed reasons differ from shared output.")
    return {
        "maximum_offline_batch_probability_error": max(errors, default=0.0),
        "risk_band_mismatches": band_mismatches,
        "reason_category_mismatches": reason_category_mismatches,
        "reason_direction_mismatches": reason_direction_mismatches,
        "ranking_matches": True,
        "selected_rows": observed_selected,
    }


def _compare_api(
    *,
    parsed: Any,
    offline: Any,
    batch_rows: list[dict[str, str]],
    bundle_root: Path,
    config_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    from credit_risk.inference.api import create_app

    try:
        from fastapi.testclient import TestClient
    except ImportError as error:  # pragma: no cover - exercised in the minimal container
        raise InferenceEvidenceError(
            "Phase 6 evidence generation requires the development test client dependencies."
        ) from error

    batch_by_id = {row["account_id"]: row for row in batch_rows}
    errors: list[float] = []
    band_mismatches = 0
    reason_category_mismatches = 0
    reason_direction_mismatches = 0
    trace_mismatches = 0
    with TestClient(create_app(bundle_root=bundle_root, config_path=config_path)) as client:
        if client.post("/predict", json={}).status_code != 404:
            raise InferenceEvidenceError("Retired /predict endpoint is still reachable.")
        for position, account_id in enumerate(parsed.account_ids):
            payload = {
                name: int(parsed.features.loc[account_id, name]) for name in parsed.features.columns
            }
            trace_id = f"phase6-parity-{position + 1:02d}"
            response = client.post("/v1/predict", json=payload, headers={"X-Request-ID": trace_id})
            if response.status_code != 200:
                raise InferenceEvidenceError("Versioned API rejected the reviewed synthetic input.")
            api = response.json()
            row = batch_by_id[account_id]
            errors.append(
                abs(float(api["probability_of_default"]) - float(offline.probabilities[position]))
            )
            band_mismatches += api["risk_band"] != row["risk_band"]
            reason_category_mismatches += [item["category"] for item in api["reasons"]] != [
                row["primary_reason_category"],
                row["secondary_reason_category"],
            ]
            reason_direction_mismatches += [item["direction"] for item in api["reasons"]] != [
                row["primary_reason_direction"],
                row["secondary_reason_direction"],
            ]
            trace_mismatches += (
                api["trace_id"] != trace_id or response.headers.get("X-Trace-ID") != trace_id
            )
    maximum_error = max(errors, default=0.0)
    if maximum_error > tolerance:
        raise InferenceEvidenceError("API probability differs from full-precision shared output.")
    if band_mismatches or reason_category_mismatches or reason_direction_mismatches:
        raise InferenceEvidenceError("API bands or reviewed reasons differ from batch output.")
    if trace_mismatches:
        raise InferenceEvidenceError("API trace propagation differs from the v1 contract.")
    return {
        "maximum_offline_api_probability_error": maximum_error,
        "absolute_tolerance": tolerance,
        "risk_band_mismatches": band_mismatches,
        "reason_category_mismatches": reason_category_mismatches,
        "reason_direction_mismatches": reason_direction_mismatches,
        "trace_mismatches": trace_mismatches,
        "removed_predict_endpoint_status": 404,
    }


def _assemble_summary(
    *,
    implementation_commit: str,
    fixture_sha256: str,
    batch_id: str,
    batch_manifest: dict[str, Any],
    comparison: dict[str, Any],
    api_comparison: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "evidence_id": "phase6_v1",
        "status": "complete",
        "implementation_git_commit": implementation_commit,
        "lineage": {
            "config_sha256": PHASE6_CONFIG_SHA256,
            "fixture_sha256": fixture_sha256,
            "bundle_manifest_sha256": batch_manifest["model"]["manifest_sha256"],
            "model_sha256": batch_manifest["model"]["model_sha256"],
        },
        "population": {
            "synthetic": True,
            "input_rows": batch_manifest["counts"]["input_rows"],
            "valid_rows": batch_manifest["counts"]["valid_rows"],
            "rejected_rows": batch_manifest["counts"]["rejected_rows"],
            "row_level_data_published": False,
        },
        "batch": {
            "batch_id": batch_id,
            "status": batch_manifest["status"],
            "selected_rows": batch_manifest["policy"]["selected_rows"],
            "review_capacity_fraction": batch_manifest["policy"]["review_capacity_fraction"],
            "risk_band_counts": batch_manifest["counts"]["risk_bands"],
            "idempotent_rerun_reused": True,
            "idempotent_rerun_files_unchanged": True,
        },
        "parity": {
            "offline_to_batch": comparison,
            "offline_to_api": api_comparison,
            "streamlit_client_uses_v1_api": True,
        },
        "explanation": {
            "method": batch_manifest["explanation"]["method"],
            "space": batch_manifest["explanation"]["space"],
            "maximum_additivity_error": batch_manifest["explanation"]["maximum_additivity_error"],
            "maximum_probability_error": batch_manifest["explanation"]["maximum_probability_error"],
            "reason_categories": [
                "billing_balance",
                "credit_capacity",
                "payment_behaviour",
                "repayment_status",
            ],
            "causal_or_adverse_action_reason": False,
        },
        "interfaces": {
            "prediction_path": "/v1/predict",
            "removed_prediction_path": "/predict",
            "liveness_path": "/ping",
            "readiness_path": "/ready",
        },
        "boundaries": {
            "model_fitting_performed": False,
            "parameter_tuning_performed": False,
            "calibration_fitting_performed": False,
            "sealed_test_loaded_or_scored": False,
            "model_or_policy_changed": False,
            "safe_aggregate_logging_only": True,
        },
        "g4_status": "open",
    }


def _assemble_manifest(
    *,
    implementation_commit: str,
    fixture_sha256: str,
    artifacts: dict[str, bytes],
    batch_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "evidence_id": "phase6_v1",
        "implementation_git_commit": implementation_commit,
        "source_artifacts": {
            "inference_config": {"sha256": PHASE6_CONFIG_SHA256},
            "synthetic_fixture": {"sha256": fixture_sha256, "row_level_data": True},
            "bundle_manifest": {"sha256": batch_manifest["model"]["manifest_sha256"]},
            "selected_model": {"sha256": batch_manifest["model"]["model_sha256"]},
        },
        "artifacts": {
            name: {"sha256": _sha256_bytes(content), "row_level_data": False}
            for name, content in sorted(artifacts.items())
        },
        "runtime_artifacts": {
            "batch_manifest_sha256": _sha256_bytes(
                (json.dumps(batch_manifest, sort_keys=True) + "\n").encode("utf-8")
            ),
            "published": False,
        },
        "allowlisted_outputs": sorted(EVIDENCE_FILES),
        "boundaries_verified": {
            "model_fitting_performed": False,
            "test_partition_loaded": False,
            "row_level_output_published": False,
            "local_paths_published": False,
            "wall_clock_timestamps_published": False,
        },
    }


def _validate_manifest_sources(
    *,
    manifest: dict[str, Any],
    fixture: Path,
    config_path: Path,
    bundle_root: Path,
) -> None:
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("evidence_id") != "phase6_v1"
        or HEX40.fullmatch(str(manifest.get("implementation_git_commit"))) is None
        or set(manifest.get("allowlisted_outputs", ())) != EVIDENCE_FILES
    ):
        raise InferenceEvidenceError("Evidence manifest identity or allowlist is invalid.")
    expected = {
        "inference_config": _sha256_file(config_path),
        "synthetic_fixture": _sha256_file(fixture),
        "bundle_manifest": _sha256_file(bundle_root / "manifest.json"),
        "selected_model": _sha256_file(bundle_root / "model.cbm"),
    }
    sources = manifest.get("source_artifacts")
    if not isinstance(sources, dict) or set(sources) != set(expected):
        raise InferenceEvidenceError("Evidence manifest source allowlist is invalid.")
    for role, digest in expected.items():
        contract = sources.get(role)
        if not isinstance(contract, dict) or contract.get("sha256") != digest:
            raise InferenceEvidenceError(f"Evidence source digest mismatch for {role}.")
    if expected["inference_config"] != PHASE6_CONFIG_SHA256:
        raise InferenceEvidenceError("Published evidence references an unreviewed config.")
    if expected["synthetic_fixture"] != FIXTURE_SHA256:
        raise InferenceEvidenceError("Published evidence references an unreviewed fixture.")
    expected_boundaries = {
        "model_fitting_performed": False,
        "test_partition_loaded": False,
        "row_level_output_published": False,
        "local_paths_published": False,
        "wall_clock_timestamps_published": False,
    }
    if manifest.get("boundaries_verified") != expected_boundaries:
        raise InferenceEvidenceError("Evidence manifest misstates the inference boundary.")


def _validate_summary(summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    if (
        summary.get("schema_version") != "1.0.0"
        or summary.get("evidence_id") != "phase6_v1"
        or summary.get("status") != "complete"
        or summary.get("implementation_git_commit") != manifest.get("implementation_git_commit")
    ):
        raise InferenceEvidenceError("Inference summary identity or lineage is invalid.")
    population = summary.get("population", {})
    if population != {
        "synthetic": True,
        "input_rows": 20,
        "valid_rows": 20,
        "rejected_rows": 0,
        "row_level_data_published": False,
    }:
        raise InferenceEvidenceError("Inference summary population is invalid.")
    batch = summary.get("batch", {})
    if (
        batch.get("status") != "completed"
        or batch.get("selected_rows") != 2
        or batch.get("review_capacity_fraction") != 0.1
        or batch.get("idempotent_rerun_reused") is not True
        or batch.get("idempotent_rerun_files_unchanged") is not True
        or HEX64.fullmatch(str(batch.get("batch_id"))) is None
    ):
        raise InferenceEvidenceError("Inference summary batch result is invalid.")
    parity = summary.get("parity", {})
    offline_batch = parity.get("offline_to_batch", {})
    offline_api = parity.get("offline_to_api", {})
    if (
        offline_batch.get("maximum_offline_batch_probability_error") != 0.0
        or offline_batch.get("ranking_matches") is not True
        or any(
            offline_batch.get(key) != 0
            for key in (
                "risk_band_mismatches",
                "reason_category_mismatches",
                "reason_direction_mismatches",
            )
        )
        or float(offline_api.get("maximum_offline_api_probability_error", 1.0)) > 5e-7
        or any(
            offline_api.get(key) != 0
            for key in (
                "risk_band_mismatches",
                "reason_category_mismatches",
                "reason_direction_mismatches",
                "trace_mismatches",
            )
        )
        or offline_api.get("removed_predict_endpoint_status") != 404
    ):
        raise InferenceEvidenceError("Inference parity evidence is inconsistent.")
    expected_boundaries = {
        "model_fitting_performed": False,
        "parameter_tuning_performed": False,
        "calibration_fitting_performed": False,
        "sealed_test_loaded_or_scored": False,
        "model_or_policy_changed": False,
        "safe_aggregate_logging_only": True,
    }
    if summary.get("boundaries") != expected_boundaries or summary.get("g4_status") != "open":
        raise InferenceEvidenceError("Inference summary boundary or G4 status is invalid.")


def _render_report(summary: dict[str, Any]) -> str:
    batch = summary["batch"]
    api = summary["parity"]["offline_to_api"]
    explanation = summary["explanation"]
    return "\n".join(
        (
            "# Phase 6 inference parity report",
            "",
            "Status: **complete**",
            "",
            "## Reviewed result",
            "",
            "One shared, dependency-validated `selected_v1` engine scored 20 synthetic accounts "
            "through offline, monthly-batch, and `/v1/predict` paths. The batch selected "
            f"{batch['selected_rows']} accounts under the fixed 10% review capacity. No model fitting, "
            "refitting, tuning, calibration fitting, or sealed-test access occurred.",
            "",
            "The full-precision batch probabilities matched the shared engine exactly. The largest "
            f"absolute API rounding difference was `{api['maximum_offline_api_probability_error']:.17g}`, "
            "within the frozen `5e-7` tolerance. Risk bands and both reason categories and directions "
            "matched exactly. The identical batch rerun reused verified files without rewriting them.",
            "",
            "## Explanation boundary",
            "",
            f"Native SHAP used `{explanation['space']}` contributions across the four reviewed "
            "categories. These are model attributions, not causal or adverse-action reasons.",
            "",
            "## Interface and lifecycle boundary",
            "",
            "`POST /predict` is retired; `POST /v1/predict` is the only prediction endpoint. "
            "Streamlit calls that endpoint and does not load the model. G4 remains open for stress "
            "testing, registry promotion, scanning, rollback, monitoring, and runbooks.",
            "",
        )
    )


def _read_batch_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as error:
        raise InferenceEvidenceError(f"Unable to read batch scores: {error}") from error


def _publish_directory(output: Path, artifacts: dict[str, bytes]) -> None:
    if set(artifacts) != EVIDENCE_FILES:
        raise InferenceEvidenceError("Phase 6 evidence payload differs from the allowlist.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the sibling stage name short enough for deeply nested Windows job paths.
    staged = output.with_name(f".stage-{uuid4().hex[:12]}")
    try:
        staged.mkdir()
        for name, content in artifacts.items():
            (staged / name).write_bytes(content)
        _replace_directory_with_retry(staged, output)
    except OSError as error:
        raise InferenceEvidenceError(
            f"Unable to publish Phase 6 evidence atomically: {error}"
        ) from error
    finally:
        if staged.exists():
            shutil.rmtree(staged)


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


def _safe_destination(
    repository: Path,
    value: str | Path,
    *,
    allowed_subtree: Path,
    description: str,
) -> Path:
    supplied = Path(value)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise InferenceEvidenceError(f"{description.capitalize()} must be repository-relative.")
    resolved = (repository / supplied).resolve()
    allowed = (repository / allowed_subtree).resolve()
    if not resolved.is_relative_to(allowed) or resolved == allowed:
        raise InferenceEvidenceError(
            f"{description.capitalize()} must be contained beneath {allowed_subtree.as_posix()}/."
        )
    for existing_parent in (resolved, *resolved.parents):
        if existing_parent.exists() and existing_parent.is_symlink():
            raise InferenceEvidenceError(f"{description.capitalize()} cannot traverse a symlink.")
        if existing_parent == repository:
            break
    return resolved


def _safe_repository_file(repository: Path, value: str | Path, description: str) -> Path:
    path = _safe_repository_path(repository, value, description)
    if not path.is_file():
        raise InferenceEvidenceError(f"{description.capitalize()} is missing: {path.name}")
    return path


def _safe_repository_directory(repository: Path, value: str | Path, description: str) -> Path:
    path = _safe_repository_path(repository, value, description)
    if not path.is_dir():
        raise InferenceEvidenceError(f"{description.capitalize()} is missing.")
    return path


def _safe_repository_path(repository: Path, value: str | Path, description: str) -> Path:
    supplied = Path(value)
    path = supplied.resolve() if supplied.is_absolute() else (repository / supplied).resolve()
    if not path.is_relative_to(repository):
        raise InferenceEvidenceError(f"{description.capitalize()} must stay within the repository.")
    return path


def _repository_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise InferenceEvidenceError("Unable to locate repository root from inference config.")


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    return _read_json_bytes(_read_bytes(path, description), description)


def _read_json_bytes(content: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InferenceEvidenceError(f"Unable to parse {description}: {error}") from error
    if not isinstance(value, dict):
        raise InferenceEvidenceError(f"{description.capitalize()} must contain a JSON object.")
    return value


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise InferenceEvidenceError(f"Unable to read {description}: {error}") from error


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise InferenceEvidenceError(f"Unable to read {description}: {error}") from error


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_bytes(path, path.name))
