"""Freeze validation-derived gates without loading the sealed test partition."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import ValidationError

from credit_risk.modeling.selected_bundle import BundleManifest
from credit_risk.modeling.tracking import TrackingError, collect_git_evidence

DEFAULT_AUTHORIZATION_PATH = Path("configs/modeling/final_test_v1.json")


class FinalTestFreezeError(RuntimeError):
    """Raised when test authorization cannot be frozen without touching test data."""


def freeze_final_test_authorization(
    *,
    selection_root: str | Path = "reports/modeling/selection_v1",
    bundle_root: str | Path = "models/selected_v1",
    output: str | Path = DEFAULT_AUTHORIZATION_PATH,
) -> Path:
    """Freeze absolute gates from reviewed validation evidence; never load data or a model."""

    selection = Path(selection_root)
    bundle = Path(bundle_root)
    output_path = Path(output)
    if output_path.exists():
        raise FinalTestFreezeError(f"Refusing to overwrite frozen authorization {output_path}.")
    try:
        git = collect_git_evidence(selection)
        if git.dirty:
            raise FinalTestFreezeError(
                "Freeze-test requires committed selection evidence and a clean worktree."
            )
        summary_path = selection / "summary.json"
        report_path = selection / "selection-report.md"
        manifest_path = bundle / "manifest.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = BundleManifest.model_validate_json(manifest_path.read_bytes())
        model_path = bundle / manifest.model_filename
        model_sha = _sha256(model_path)
    except FinalTestFreezeError:
        raise
    except (OSError, json.JSONDecodeError, ValidationError, TrackingError) as error:
        raise FinalTestFreezeError(f"Unable to verify selection evidence: {error}") from error
    if model_sha != manifest.model_sha256:
        raise FinalTestFreezeError("Selected model digest differs from the reviewed manifest.")
    if summary.get("holdout", {}).get("evaluated") is not False:
        raise FinalTestFreezeError("Selection evidence must state that the holdout is unevaluated.")
    selected_id = summary.get("selection", {}).get("selected_model_id")
    if selected_id != manifest.selected_model_id:
        raise FinalTestFreezeError("Selection summary and bundle identify different winners.")
    selected = next(
        (model for model in summary.get("models", ()) if model.get("model_id") == selected_id),
        None,
    )
    if selected is None:
        raise FinalTestFreezeError("Selection summary is missing the selected validation result.")
    metrics = selected["validation_metrics"]
    lift = next(item["lift"] for item in metrics["capacities"] if item["capacity"] == 0.1)
    authorization: dict[str, Any] = {
        "schema_version": "1.0.0",
        "authorization_id": "final_test_v1",
        "status": "frozen_not_executed",
        "selection_evidence": {
            "evidence_commit": git.commit_sha,
            "implementation_commit": summary["reproducibility"]["git_commit"],
            "summary_sha256": _sha256(summary_path),
            "report_sha256": _sha256(report_path),
            "manifest_sha256": _sha256(manifest_path),
            "model_sha256": model_sha,
            "selected_model_id": selected_id,
        },
        "test_contract": {
            "required_unique_accounts": 6000,
            "maximum_evaluations": 1,
            "training": "prohibited",
            "refitting": "prohibited",
            "retuning": "prohibited",
            "force_override": "prohibited",
            "calibration": "identity",
            "risk_band_thresholds": manifest.risk_band_thresholds,
        },
        "frozen_gates": {
            "minimum_average_precision": metrics["discrimination"]["average_precision"] - 0.03,
            "maximum_brier_score": metrics["probability"]["brier_score"] + 0.02,
            "minimum_lift_at_0_1": lift - 0.30,
        },
        "execution": {
            "authorized": False,
            "requires_separate_explicit_request": True,
            "holdout_loaded_during_freeze": False,
        },
    }
    content = (json.dumps(authorization, sort_keys=True) + "\n").encode("utf-8")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except OSError as error:
        raise FinalTestFreezeError(
            f"Unable to publish final-test authorization: {error}"
        ) from error
    return output_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
