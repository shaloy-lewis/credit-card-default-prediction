"""One-time, prediction-only evaluation of the sealed test partition."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from credit_risk.data.manifest import DEFAULT_DATASET_MANIFEST_PATH, DEFAULT_SPLIT_CONFIG_PATH
from credit_risk.data.workflow import DataWorkflowError, verify_dataset
from credit_risk.modeling.dataset import ModelingDataError, load_governed_test_data
from credit_risk.modeling.metrics import MetricValidationError, evaluate_predictions
from credit_risk.modeling.risk_policy import RiskPolicyError, risk_band
from credit_risk.modeling.selected_bundle import (
    SelectedBundleError,
    load_selected_bundle,
)
from credit_risk.modeling.selection_analysis import metrics_payload
from credit_risk.modeling.selection_models import SelectionModelError
from credit_risk.modeling.tracking import TrackingError, collect_git_evidence

DEFAULT_AUTHORIZATION_PATH = Path("configs/modeling/final_test_v1.json")
DEFAULT_APPROVAL_PATH = Path("configs/modeling/final_test_v1.approval.json")
DEFAULT_BUNDLE_ROOT = Path("models/selected_v1")
DEFAULT_RUNTIME_ROOT = Path("experiment/final-test-v1")
DEFAULT_OUTPUT_ROOT = Path("reports/modeling/final_test_v1")
SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "final-test-report.md"
STARTED_FILENAME = "evaluation-started.json"
COMPLETED_FILENAME = "evaluation-completed.json"
PREDICTIONS_FILENAME = "test_predictions.csv"


class FinalTestWorkflowError(RuntimeError):
    """Raised when the one-time test cannot proceed safely."""


class FinalTestApproval(BaseModel):
    """Reviewed, deterministic approval for exactly one prediction-only evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())

    schema_version: Literal["1.0.0"]
    approval_id: Literal["final_test_v1_approval"]
    status: Literal["approved_once"]
    frozen_authorization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workflow_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_evaluations: Literal[1]
    training: Literal["prohibited"]
    refitting: Literal["prohibited"]
    retuning: Literal["prohibited"]
    force_override: Literal["prohibited"]
    dirty_execution: Literal["prohibited"]


class FrozenFinalTestAuthorization(BaseModel):
    """Strict subset of the already-reviewed final-test gate contract."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    schema_version: Literal["1.0.0"]
    authorization_id: Literal["final_test_v1"]
    status: Literal["frozen_not_executed"]
    selection_evidence: dict[str, Any]
    test_contract: dict[str, Any]
    frozen_gates: dict[str, float]
    execution: dict[str, Any]

    @model_validator(mode="after")
    def enforce_one_time_contract(self) -> FrozenFinalTestAuthorization:
        contract = self.test_contract
        execution = self.execution
        expected_prohibitions = {
            "training": "prohibited",
            "refitting": "prohibited",
            "retuning": "prohibited",
            "force_override": "prohibited",
        }
        if any(contract.get(name) != value for name, value in expected_prohibitions.items()):
            raise ValueError("frozen authorization no longer prohibits model changes")
        if (
            contract.get("maximum_evaluations") != 1
            or contract.get("required_unique_accounts") != 6000
        ):
            raise ValueError("frozen authorization must permit exactly one 6,000-row evaluation")
        if contract.get("calibration") != "identity":
            raise ValueError("frozen authorization must retain identity calibration")
        if execution != {
            "authorized": False,
            "holdout_loaded_during_freeze": False,
            "requires_separate_explicit_request": True,
        }:
            raise ValueError("frozen authorization execution state changed unexpectedly")
        if set(self.frozen_gates) != {
            "minimum_average_precision",
            "maximum_brier_score",
            "minimum_lift_at_0_1",
        }:
            raise ValueError("frozen authorization gates changed unexpectedly")
        return self


@dataclass(frozen=True, slots=True)
class FinalTestResult:
    summary_path: Path
    report_path: Path
    started_receipt_path: Path
    completed_receipt_path: Path
    predictions_path: Path
    summary_sha256: str
    report_sha256: str
    predictions_sha256: str
    g2_closed: bool


def run_final_test(
    *,
    data_root: str | Path = "data",
    authorization_path: str | Path = DEFAULT_AUTHORIZATION_PATH,
    approval_path: str | Path = DEFAULT_APPROVAL_PATH,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    runtime_root: str | Path = DEFAULT_RUNTIME_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> FinalTestResult:
    """Score the sealed test once; this function contains no fitting operation."""

    authorization_file = Path(authorization_path)
    approval_file = Path(approval_path)
    try:
        git = collect_git_evidence(authorization_file.resolve().parent)
        if git.dirty:
            raise FinalTestWorkflowError(
                "Final-test execution requires a clean committed worktree."
            )
        if git.repository_root is None:
            raise FinalTestWorkflowError("Git repository root is unavailable.")
        repository_root = git.repository_root
        output = _safe_destination(repository_root, output_root, "final-test output")
        runtime = _safe_destination(repository_root, runtime_root, "final-test runtime")
        if output.exists():
            raise FinalTestWorkflowError(
                f"Final-test output already exists and permanently prevents reevaluation: {output}"
            )

        authorization_bytes = authorization_file.read_bytes()
        approval_bytes = approval_file.read_bytes()
        authorization = FrozenFinalTestAuthorization.model_validate_json(authorization_bytes)
        approval = FinalTestApproval.model_validate_json(approval_bytes)
        authorization_sha = _sha256_bytes(authorization_bytes)
        approval_sha = _sha256_bytes(approval_bytes)
        if approval.frozen_authorization_sha256 != authorization_sha:
            raise FinalTestWorkflowError("Approval does not match the frozen authorization digest.")
        workflow_sha = _sha256_file(Path(__file__))
        if approval.workflow_sha256 != workflow_sha:
            raise FinalTestWorkflowError(
                "Final-test workflow source differs from the reviewed approval."
            )

        selection_root = repository_root / "reports" / "modeling" / "selection_v1"
        bundle = _safe_destination(repository_root, bundle_root, "selected bundle")
        manifest_path = bundle / "manifest.json"
        model_path = bundle / "model.cbm"
        _verify_selection_evidence(
            authorization,
            selection_root=selection_root,
            manifest_path=manifest_path,
            model_path=model_path,
        )
        if _sha256_file(manifest_path) != approval.manifest_sha256:
            raise FinalTestWorkflowError(
                "Approved manifest digest differs from the selected bundle."
            )
        if _sha256_file(model_path) != approval.model_sha256:
            raise FinalTestWorkflowError("Approved model digest differs from the selected bundle.")
        manifest, selected_model = load_selected_bundle(bundle, trusted=True)
        if manifest.holdout_evaluated is not False:
            raise FinalTestWorkflowError("Selected bundle already claims holdout evaluation.")

        verification = verify_dataset(
            data_root,
            DEFAULT_DATASET_MANIFEST_PATH,
            DEFAULT_SPLIT_CONFIG_PATH,
        )
        selection_summary = json.loads((selection_root / "summary.json").read_bytes())
        _verify_lineage(selection_summary, verification)

        evaluation_id = _evaluation_id(
            authorization_sha=authorization_sha,
            approval_sha=approval_sha,
            model_sha=manifest.model_sha256,
            assignment_sha=verification.assignment_sha256,
            git_commit=git.commit_sha,
        )
        evaluation_runtime = runtime / evaluation_id
        if evaluation_runtime.exists():
            raise FinalTestWorkflowError(
                "A runtime record already exists for this authorized evaluation; rerun is prohibited."
            )
        output.mkdir(parents=True, exist_ok=False)
        evaluation_runtime.mkdir(parents=True, exist_ok=False)
        started = {
            "schema_version": "1.0.0",
            "evaluation_id": evaluation_id,
            "status": "started",
            "authorization_sha256": authorization_sha,
            "approval_sha256": approval_sha,
            "manifest_sha256": approval.manifest_sha256,
            "model_sha256": approval.model_sha256,
            "assignment_sha256": verification.assignment_sha256,
            "git_commit": git.commit_sha,
            "evaluation_count": 1,
            "training_performed": False,
            "refitting_performed": False,
            "retuning_performed": False,
        }
        started_path = output / STARTED_FILENAME
        _write_json_atomic(started_path, started)

        governed = load_governed_test_data(data_root=data_root)
        _verify_test_lineage(governed.lineage, selection_summary)
        if len(governed.account_ids) != authorization.test_contract["required_unique_accounts"]:
            raise FinalTestWorkflowError("Sealed test coverage differs from the authorization.")
        probabilities = selected_model.predict_proba(governed.X)
        if probabilities.shape != (6000,):
            raise FinalTestWorkflowError(
                "Selected model did not return one score per test account."
            )
        metrics = evaluate_predictions(
            governed.y.to_numpy(), probabilities, probabilities=probabilities
        )
        metric_payload = metrics_payload(metrics)
        lift_at_ten = next(
            item["lift"] for item in metric_payload["capacities"] if item["capacity"] == 0.1
        )
        gates: dict[str, dict[str, Any]] = {
            "average_precision": {
                "observed": metric_payload["discrimination"]["average_precision"],
                "threshold": authorization.frozen_gates["minimum_average_precision"],
                "operator": ">=",
            },
            "brier_score": {
                "observed": metric_payload["probability"]["brier_score"],
                "threshold": authorization.frozen_gates["maximum_brier_score"],
                "operator": "<=",
            },
            "lift_at_0_1": {
                "observed": lift_at_ten,
                "threshold": authorization.frozen_gates["minimum_lift_at_0_1"],
                "operator": ">=",
            },
        }
        gates["average_precision"]["passed"] = (
            gates["average_precision"]["observed"] >= gates["average_precision"]["threshold"]
        )
        gates["brier_score"]["passed"] = (
            gates["brier_score"]["observed"] <= gates["brier_score"]["threshold"]
        )
        gates["lift_at_0_1"]["passed"] = (
            gates["lift_at_0_1"]["observed"] >= gates["lift_at_0_1"]["threshold"]
        )
        g2_closed = all(bool(gate["passed"]) for gate in gates.values())
        bands = np.asarray(
            [risk_band(float(value), manifest.risk_band_thresholds) for value in probabilities]
        )
        predictions_path = evaluation_runtime / PREDICTIONS_FILENAME
        _write_predictions(
            predictions_path,
            governed.account_ids.to_numpy(),
            governed.y.to_numpy(),
            probabilities,
            bands,
        )
        predictions_sha = _sha256_file(predictions_path)
        band_counts = {
            name: int(np.sum(bands == name))
            for name in ("standard", "elevated", "high", "critical")
        }
        summary = {
            "schema_version": "1.0.0",
            "evaluation_id": evaluation_id,
            "status": "complete",
            "g2_status": "closed" if g2_closed else "open_failed_test",
            "model": {
                "model_id": manifest.selected_model_id,
                "bundle_id": manifest.bundle_id,
                "manifest_sha256": approval.manifest_sha256,
                "model_sha256": approval.model_sha256,
                "calibration": manifest.calibration,
            },
            "population": {
                "partition": "test",
                "rows": len(governed.account_ids),
                "unique_accounts": int(governed.account_ids.nunique()),
                "target_counts": {
                    str(label): int(count)
                    for label, count in governed.y.value_counts().sort_index().items()
                },
                "assignment_sha256": verification.assignment_sha256,
            },
            "execution": {
                "evaluation_count": 1,
                "maximum_evaluations": 1,
                "training_performed": False,
                "refitting_performed": False,
                "retuning_performed": False,
                "cross_validation_performed": False,
            },
            "metrics": metric_payload,
            "gates": gates,
            "risk_bands": {
                "thresholds": manifest.risk_band_thresholds,
                "counts": band_counts,
            },
            "lineage": {
                **asdict(governed.lineage),
                "authorization_sha256": authorization_sha,
                "approval_sha256": approval_sha,
                "git_commit": git.commit_sha,
            },
            "runtime_artifacts": {
                "test_predictions_sha256": predictions_sha,
                "row_level_data_committed": False,
            },
        }
        summary_path = output / SUMMARY_FILENAME
        _write_json_atomic(summary_path, summary)
        summary_sha = _sha256_file(summary_path)
        report_path = output / REPORT_FILENAME
        _write_text_atomic(report_path, _render_report(summary, summary_sha))
        report_sha = _sha256_file(report_path)
        completed = {
            "schema_version": "1.0.0",
            "evaluation_id": evaluation_id,
            "status": "complete",
            "g2_closed": g2_closed,
            "summary_sha256": summary_sha,
            "report_sha256": report_sha,
            "test_predictions_sha256": predictions_sha,
            "evaluation_count": 1,
        }
        completed_path = output / COMPLETED_FILENAME
        _write_json_atomic(completed_path, completed)
        return FinalTestResult(
            summary_path=summary_path,
            report_path=report_path,
            started_receipt_path=started_path,
            completed_receipt_path=completed_path,
            predictions_path=predictions_path,
            summary_sha256=summary_sha,
            report_sha256=report_sha,
            predictions_sha256=predictions_sha,
            g2_closed=g2_closed,
        )
    except FinalTestWorkflowError:
        raise
    except (
        DataWorkflowError,
        MetricValidationError,
        ModelingDataError,
        OSError,
        RiskPolicyError,
        SelectedBundleError,
        SelectionModelError,
        TrackingError,
        ValidationError,
        ValueError,
    ) as error:
        raise FinalTestWorkflowError(str(error)) from error


def _verify_selection_evidence(
    authorization: FrozenFinalTestAuthorization,
    *,
    selection_root: Path,
    manifest_path: Path,
    model_path: Path,
) -> None:
    expected = authorization.selection_evidence
    paths = {
        "summary_sha256": selection_root / "summary.json",
        "report_sha256": selection_root / "selection-report.md",
        "manifest_sha256": manifest_path,
        "model_sha256": model_path,
    }
    mismatches = [name for name, path in paths.items() if _sha256_file(path) != expected.get(name)]
    if mismatches:
        raise FinalTestWorkflowError(
            f"Selection evidence differs from the frozen authorization: {sorted(mismatches)}"
        )


def _verify_lineage(selection_summary: dict[str, Any], verification: Any) -> None:
    expected = selection_summary["reproducibility"]["data_lineage"]
    observed = {
        "source_sha256": verification.source_sha256,
        "dataset_manifest_sha256": verification.dataset_manifest_sha256,
        "canonical_sha256": verification.canonical_sha256,
        "quality_report_sha256": verification.quality_report_sha256,
        "split_config_sha256": verification.split_config_sha256,
        "assignment_sha256": verification.assignment_sha256,
        "split_manifest_sha256": verification.split_manifest_sha256,
    }
    mismatches = [name for name, value in observed.items() if expected.get(name) != value]
    if mismatches:
        raise FinalTestWorkflowError(f"Verified data lineage changed: {sorted(mismatches)}")


def _verify_test_lineage(lineage: Any, selection_summary: dict[str, Any]) -> None:
    expected = selection_summary["reproducibility"]["data_lineage"]
    observed = asdict(lineage)
    mismatches = [name for name, value in observed.items() if expected.get(name) != value]
    if mismatches:
        raise FinalTestWorkflowError(f"Sealed test lineage changed: {sorted(mismatches)}")


def _evaluation_id(**inputs: str) -> str:
    content = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def _write_predictions(
    path: Path,
    account_ids: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    bands: np.ndarray,
) -> None:
    if len(set(int(value) for value in account_ids)) != 6000:
        raise FinalTestWorkflowError("Test predictions must cover 6,000 unique accounts.")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("account_id", "target", "probability", "risk_band"))
    rows = sorted(
        zip(account_ids, target, probabilities, bands, strict=True), key=lambda row: int(row[0])
    )
    for account_id, label, probability, band in rows:
        writer.writerow(
            (int(account_id), int(label), format(float(probability), ".17g"), str(band))
        )
    _write_bytes_atomic(path, output.getvalue().encode("utf-8"))


def _render_report(summary: dict[str, Any], summary_sha: str) -> str:
    metrics = summary["metrics"]
    lift = next(item["lift"] for item in metrics["capacities"] if item["capacity"] == 0.1)
    gate_rows = [
        f"| {name} | {gate['observed']:.6f} | {gate['operator']} {gate['threshold']:.6f} | "
        f"{'pass' if gate['passed'] else 'fail'} |"
        for name, gate in summary["gates"].items()
    ]
    return "\n".join(
        [
            "# One-time final-test report",
            "",
            f"- **G2 status:** {summary['g2_status']}",
            "- **Model:** `catboost_fixed` from bundle `selected_v1`",
            "- **Evaluation boundary:** exactly 6,000 sealed test accounts, scored once",
            "- **Training, refitting, retuning, and cross-validation:** not performed",
            f"- **Average precision:** {metrics['discrimination']['average_precision']:.6f}",
            f"- **Brier score:** {metrics['probability']['brier_score']:.6f}",
            f"- **Lift at 10%:** {lift:.6f}",
            "",
            "| Gate | Observed | Frozen requirement | Result |",
            "| --- | ---: | ---: | --- |",
            *gate_rows,
            "",
            "Row-level test predictions remain ignored; their checksum is retained in the summary.",
            f"Summary SHA-256: `{summary_sha}`",
            "",
        ]
    )


def _safe_destination(repository_root: Path, path: str | Path, description: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise FinalTestWorkflowError(f"{description} must remain inside the repository.") from error
    return resolved


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes_atomic(path, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))


def _write_text_atomic(path: Path, content: str) -> None:
    _write_bytes_atomic(path, content.encode("utf-8"))


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
