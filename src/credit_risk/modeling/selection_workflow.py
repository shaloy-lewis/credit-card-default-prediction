"""Governed four-fit selection, tracking, and atomic release publication."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import numpy as np

from credit_risk.modeling.dataset import ModelingDataError, load_governed_development_data
from credit_risk.modeling.metrics import MetricValidationError, evaluate_predictions
from credit_risk.modeling.selected_bundle import (
    BundleManifest,
    SelectedBundleError,
    load_selected_bundle,
    population_sha256,
    write_manifest,
    write_model_artifact,
)
from credit_risk.modeling.selection_analysis import (
    SelectionAnalysisError,
    ValidationResult,
    bootstrap_validation_metrics,
    calibration_diagnostics,
    flat_tracking_metrics,
    metrics_payload,
    risk_band_thresholds,
    select_validation_winner,
)
from credit_risk.modeling.selection_contracts import (
    DEFAULT_SELECTION_CONFIG_PATH,
    SelectionContractError,
    load_selection_config,
    selection_config_sha256,
)
from credit_risk.modeling.selection_models import (
    SelectionModelError,
    fit_one_pass_models,
)
from credit_risk.modeling.tracking import (
    ModelRunPayload,
    TrackingDependencyError,
    TrackingError,
    TrackingRunResult,
    collect_git_evidence,
    collect_package_versions,
    mark_tracking_run_failed,
    track_selection_runs,
)

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_TRACKING_ROOT = Path("experiment/mlflow")
DEFAULT_OUTPUT_ROOT = Path("reports/modeling/selection_v1")
DEFAULT_BUNDLE_ROOT = Path("models/selected_v1")
SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "selection-report.md"
PREDICTIONS_FILENAME = "validation_predictions.csv"
BOOTSTRAP_FILENAME = "bootstrap_intervals.json"


class SelectionWorkflowError(RuntimeError):
    """Raised when selection cannot complete without violating governance."""


@dataclass(frozen=True, slots=True)
class SelectionWorkflowResult:
    selected_model_id: str
    summary_path: Path
    report_path: Path
    manifest_path: Path
    model_path: Path
    validation_predictions_path: Path
    bootstrap_path: Path
    summary_sha256: str
    report_sha256: str
    manifest_sha256: str
    model_sha256: str
    tracking: TrackingRunResult


def run_model_selection(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config_path: str | Path = DEFAULT_SELECTION_CONFIG_PATH,
    tracking_root: str | Path = DEFAULT_TRACKING_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
) -> SelectionWorkflowResult:
    """Fit four fixed models once, select on validation, and publish the exact winner."""

    try:
        config = load_selection_config(config_path)
        git = collect_git_evidence(Path(config_path).resolve().parent)
        if git.dirty:
            raise SelectionWorkflowError(
                "Official selection requires a clean committed worktree; commit reviewed code first."
            )
        versions = collect_package_versions(tuple(config.dependencies))
        if versions != config.dependencies:
            raise SelectionWorkflowError(
                f"Installed dependency versions differ from the frozen protocol: {versions}"
            )
        repository_root = git.repository_root
        if repository_root is None:
            raise SelectionWorkflowError("Git repository root is unavailable.")
        output = _safe_destination(repository_root, output_root, "selection output")
        bundle = _safe_destination(repository_root, bundle_root, "selected bundle")
        tracking = _safe_destination(repository_root, tracking_root, "tracking root")
        runtime = tracking / "selection-runtime" / git.commit_sha
        for path in (output, bundle, runtime):
            if path.exists():
                raise SelectionWorkflowError(
                    f"Refusing to overwrite existing governed output: {path}"
                )

        governed = load_governed_development_data(
            data_root=data_root,
            feature_contract_path=config.data.feature_contract_path,
            manifest_path=config.data.dataset_manifest_path,
            split_config_path=config.data.split_config_path,
        )
        fold_zero = governed.assignments["cv_fold_r0"].eq(0)
        X_train = governed.X.loc[~fold_zero].copy()
        y_train = governed.y.loc[~fold_zero].copy()
        X_validation = governed.X.loc[fold_zero].copy()
        y_validation = governed.y.loc[fold_zero].copy()
        _validate_selection_split(
            X_train.index.to_numpy(),
            y_train.to_numpy(),
            X_validation.index.to_numpy(),
            y_validation.to_numpy(),
            config,
        )

        fitted_models = fit_one_pass_models(X_train, y_train, config)
        result_items: list[ValidationResult] = []
        for model in fitted_models:
            probabilities = model.predict_proba(X_validation)
            result_items.append(
                ValidationResult(
                    model_id=model.model_id,
                    probabilities=probabilities,
                    metrics=evaluate_predictions(
                        y_validation.to_numpy(), probabilities, probabilities=probabilities
                    ),
                )
            )
        validation_results = tuple(result_items)
        # Scoring is deliberately repeatable, but fitting above occurs exactly once per model.
        decision = select_validation_winner(validation_results, config)
        selected_model = next(
            model for model in fitted_models if model.model_id == decision.selected_model_id
        )
        selected_result = next(
            result for result in decision.results if result.model_id == decision.selected_model_id
        )
        calibration = calibration_diagnostics(
            y_validation.to_numpy(), selected_result.probabilities
        )
        thresholds = risk_band_thresholds(
            selected_result.probabilities, config.selection.risk_band_quantiles
        )
        bootstrap = bootstrap_validation_metrics(
            y_validation.to_numpy(),
            selected_result.probabilities,
            resamples=config.selection.bootstrap_resamples,
            random_state=config.selection.bootstrap_random_state,
        )

        tracking.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="selection-v1-", dir=tracking) as temporary:
            stage = Path(temporary)
            staged_output = stage / "output"
            staged_bundle = stage / "bundle"
            staged_runtime = stage / "runtime"
            for path in (staged_output, staged_bundle, staged_runtime):
                path.mkdir()
            predictions_path = staged_runtime / PREDICTIONS_FILENAME
            _write_predictions(
                predictions_path,
                X_validation.index.to_numpy(),
                y_validation.to_numpy(),
                decision.results,
            )
            predictions_sha = _sha256_file(predictions_path)
            bootstrap_path = staged_runtime / BOOTSTRAP_FILENAME
            _write_json(bootstrap_path, bootstrap)
            bootstrap_sha = _sha256_file(bootstrap_path)
            model_path, model_sha = write_model_artifact(selected_model, staged_bundle)
            selected_contract = next(
                model for model in config.models if model.model_id == decision.selected_model_id
            )
            manifest = BundleManifest(
                schema_version="1.0.0",
                bundle_id="selected_v1",
                selected_model_id=cast(
                    Literal[
                        "logistic_l2",
                        "random_forest",
                        "hist_gradient_boosting",
                        "catboost_fixed",
                    ],
                    decision.selected_model_id,
                ),
                model_filename=cast(Literal["model.joblib", "model.cbm"], model_path.name),
                model_sha256=model_sha,
                selection_config_sha256=selection_config_sha256(config_path),
                training_population_sha256=population_sha256(
                    X_train.index.to_numpy(), y_train.to_numpy()
                ),
                validation_population_sha256=population_sha256(
                    X_validation.index.to_numpy(), y_validation.to_numpy()
                ),
                validation_predictions_sha256=predictions_sha,
                feature_order=config.predictor_columns,
                feature_handling=selected_contract.feature_handling,
                class_order=(0, 1),
                fixed_parameters=selected_contract.parameters,
                dependencies=versions,
                git_commit=git.commit_sha,
                git_dirty=False,
                validation_metrics=metrics_payload(selected_result.metrics),
                selection_outcome=_decision_payload(decision),
                calibration="identity",
                risk_band_thresholds=thresholds,
                fit_count=4,
                winner_refitted=False,
                holdout_evaluated=False,
                trusted_local_serialization=True,
            )
            manifest_path = write_manifest(manifest, staged_bundle)
            manifest_sha = _sha256_file(manifest_path)
            _, reloaded = load_selected_bundle(staged_bundle, trusted=True)
            reloaded_probabilities = reloaded.predict_proba(X_validation)
            if not np.array_equal(reloaded_probabilities, selected_result.probabilities):
                raise SelectionWorkflowError(
                    "Reloaded winner predictions differ from the exact selected estimator."
                )

            summary = _summary_payload(
                config_sha=selection_config_sha256(config_path),
                config=config,
                git=git,
                versions=versions,
                lineage=asdict(governed.lineage),
                decision=decision,
                calibration=calibration,
                thresholds=thresholds,
                bootstrap_sha=bootstrap_sha,
                predictions_sha=predictions_sha,
                manifest_sha=manifest_sha,
                model_sha=model_sha,
            )
            summary_path = staged_output / SUMMARY_FILENAME
            _write_json(summary_path, summary)
            summary_sha = _sha256_file(summary_path)
            report_path = staged_output / REPORT_FILENAME
            report_path.write_text(
                _render_report(summary, summary_sha), encoding="utf-8", newline="\n"
            )
            report_sha = _sha256_file(report_path)
            tracking_result = track_selection_runs(
                tracking_root=tracking,
                parent_parameters={
                    "protocol_id": config.protocol_id,
                    "selection_config_sha256": selection_config_sha256(config_path),
                    "fit_count": 4,
                    "winner_refitted": False,
                    "holdout_evaluated": False,
                },
                parent_tags={
                    "git_commit": git.commit_sha,
                    "selected_model_id": decision.selected_model_id,
                },
                model_runs=tuple(
                    ModelRunPayload(
                        model_name=result.model_id,
                        parameters=_tracking_parameters(
                            next(
                                model.parameters
                                for model in config.models
                                if model.model_id == result.model_id
                            )
                        ),
                        metrics=flat_tracking_metrics(result.metrics),
                    )
                    for result in decision.results
                ),
                artifacts=(
                    summary_path,
                    report_path,
                    predictions_path,
                    bootstrap_path,
                    manifest_path,
                    model_path,
                ),
            )
            try:
                _promote_directories(
                    ((staged_output, output), (staged_bundle, bundle), (staged_runtime, runtime))
                )
            except Exception as error:
                try:
                    mark_tracking_run_failed(tracking_result, failure_stage="atomic_publication")
                except TrackingError:
                    pass
                raise SelectionWorkflowError(
                    f"Atomic selection publication failed: {error}"
                ) from error

        return SelectionWorkflowResult(
            selected_model_id=decision.selected_model_id,
            summary_path=output / SUMMARY_FILENAME,
            report_path=output / REPORT_FILENAME,
            manifest_path=bundle / "manifest.json",
            model_path=bundle / model_path.name,
            validation_predictions_path=runtime / PREDICTIONS_FILENAME,
            bootstrap_path=runtime / BOOTSTRAP_FILENAME,
            summary_sha256=summary_sha,
            report_sha256=report_sha,
            manifest_sha256=manifest_sha,
            model_sha256=model_sha,
            tracking=tracking_result,
        )
    except SelectionWorkflowError:
        raise
    except (
        ModelingDataError,
        MetricValidationError,
        SelectedBundleError,
        SelectionAnalysisError,
        SelectionContractError,
        SelectionModelError,
        TrackingDependencyError,
        TrackingError,
        OSError,
        ValueError,
    ) as error:
        raise SelectionWorkflowError(str(error)) from error


def _validate_selection_split(
    train_ids: np.ndarray,
    train_target: np.ndarray,
    validation_ids: np.ndarray,
    validation_target: np.ndarray,
    config: Any,
) -> None:
    if (
        len(train_ids) != config.data.training_rows
        or len(validation_ids) != config.data.validation_rows
    ):
        raise SelectionWorkflowError("Train/validation row counts differ from the frozen protocol.")
    if set(train_ids) & set(validation_ids) or len(set(train_ids) | set(validation_ids)) != 24000:
        raise SelectionWorkflowError(
            "Train and validation IDs must be disjoint and cover development."
        )
    observed_train = {str(label): int(np.sum(train_target == label)) for label in (0, 1)}
    observed_validation = {str(label): int(np.sum(validation_target == label)) for label in (0, 1)}
    if (
        observed_train != config.data.training_target_counts
        or observed_validation != config.data.validation_target_counts
    ):
        raise SelectionWorkflowError(
            "Train/validation class counts differ from the frozen protocol."
        )


def _write_predictions(
    path: Path,
    account_ids: np.ndarray,
    target: np.ndarray,
    results: tuple[ValidationResult, ...],
) -> None:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("model_id", "account_id", "target", "probability"))
    for result in results:
        if len(result.probabilities) != len(account_ids):
            raise SelectionWorkflowError("Validation predictions do not cover every account.")
        for account_id, label, probability in zip(
            account_ids, target, result.probabilities, strict=True
        ):
            writer.writerow(
                (result.model_id, int(account_id), int(label), format(float(probability), ".17g"))
            )
    path.write_bytes(output.getvalue().encode("utf-8"))


def _decision_payload(decision: Any) -> dict[str, Any]:
    return {
        "selected_model_id": decision.selected_model_id,
        "best_eligible_average_precision": decision.best_eligible_average_precision,
        "models": [asdict(item) for item in decision.decisions],
        "selection_rule": "guardrails_then_ap_equivalence_then_simplicity",
    }


def _tracking_parameters(parameters: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Represent JSON null deterministically within MLflow's scalar parameter contract."""

    return {name: "none" if value is None else value for name, value in parameters.items()}


def _summary_payload(**values: Any) -> dict[str, Any]:
    config = values["config"]
    decision = values["decision"]
    return {
        "schema_version": "1.0.0",
        "experiment_id": "selection_v1",
        "status": "complete",
        "protocol": {
            "selection_config_sha256": values["config_sha"],
            "parameter_tuning": False,
            "cross_validation_iteration": False,
            "fit_count": 4,
            "winner_refitted": False,
            "calibration": "identity",
        },
        "reproducibility": {
            "git_commit": values["git"].commit_sha,
            "git_dirty": False,
            "dependencies": values["versions"],
            "data_lineage": values["lineage"],
        },
        "population": {
            "partition": "development_only",
            "training_rows": config.data.training_rows,
            "validation_rows": config.data.validation_rows,
            "sealed_test_rows": config.data.sealed_test_rows,
            "holdout_accessed": False,
        },
        "models": [
            {
                "model_id": result.model_id,
                "fixed_parameters": next(
                    model.parameters for model in config.models if model.model_id == result.model_id
                ),
                "validation_metrics": metrics_payload(result.metrics),
                "decision": next(
                    asdict(item) for item in decision.decisions if item.model_id == result.model_id
                ),
            }
            for result in decision.results
        ],
        "selection": _decision_payload(decision),
        "selected_model": {
            "model_id": decision.selected_model_id,
            "calibration_diagnostics": values["calibration"],
            "risk_band_thresholds": values["thresholds"],
        },
        "runtime_artifacts": {
            "validation_predictions_sha256": values["predictions_sha"],
            "bootstrap_intervals_sha256": values["bootstrap_sha"],
        },
        "bundle": {
            "manifest_sha256": values["manifest_sha"],
            "model_sha256": values["model_sha"],
            "trusted_local_serialization": True,
        },
        "holdout": {
            "evaluated": False,
            "authorization_frozen": False,
            "g2_status": "open",
        },
    }


def _render_report(summary: dict[str, Any], summary_sha: str) -> str:
    rows = []
    for model in summary["models"]:
        metrics = model["validation_metrics"]
        capacity = next(item for item in metrics["capacities"] if item["capacity"] == 0.1)
        rows.append(
            "| {model} | {ap:.6f} | {brier:.6f} | {lift:.6f} | {eligible} |".format(
                model=model["model_id"],
                ap=metrics["discrimination"]["average_precision"],
                brier=metrics["probability"]["brier_score"],
                lift=capacity["lift"],
                eligible=str(model["decision"]["eligible"]).lower(),
            )
        )
    return "\n".join(
        (
            "# One-pass governed model-selection report",
            "",
            f"**Deterministic summary SHA-256:** `{summary_sha}`",
            "",
            "Each fixed classifier was fitted exactly once on 19,200 development rows and scored "
            "on the same 4,800 validation accounts. No cross-validation loop, tuning, calibration "
            "fit, winner refit, or sealed-test access occurred.",
            "",
            "| Model | Average precision | Brier | Lift@10% | Eligible |",
            "| --- | ---: | ---: | ---: | --- |",
            *rows,
            "",
            f"Selected model: **{summary['selection']['selected_model_id']}**.",
            "Identity calibration and validation-derived risk bands are frozen in the bundle.",
            "The joblib format uses pickle semantics and must be loaded only from a trusted, "
            "digest-verified local bundle.",
            "",
            "## Governance boundary",
            "",
            "The 6,000-row test partition remains sealed. G2 stays open until a separately "
            "authorized, one-time test evaluation passes gates frozen from this validation result.",
            "Historical Phase 2/3 evidence is retained for audit but is not an executable workflow.",
            "",
        )
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_destination(repository_root: Path, path: str | Path, description: str) -> Path:
    candidate = Path(path)
    resolved = (
        (repository_root / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise SelectionWorkflowError(f"{description} must remain inside the repository.") from error
    return resolved


def _promote_directories(pairs: tuple[tuple[Path, Path], ...]) -> None:
    promoted: list[tuple[Path, Path]] = []
    try:
        for staged, destination in pairs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, destination)
            promoted.append((staged, destination))
    except Exception:
        for staged, destination in reversed(promoted):
            if destination.exists():
                os.replace(destination, staged)
        raise
