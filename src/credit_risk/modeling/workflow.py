"""Governed orchestration for the Phase 2 baseline experiment.

Only development-fold data can enter this workflow.  It produces deterministic
portfolio evidence while MLflow run identifiers and the row-level OOF file stay
in the ignored tracking area as operational evidence.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import Any, Final

import numpy as np

from credit_risk.modeling.baselines import (
    BaselineValidationError,
    fit_fold_baselines,
)
from credit_risk.modeling.contracts import (
    DEFAULT_BASELINE_CONFIG_PATH,
    BaselineExperimentConfig,
    parse_baseline_config,
    parse_feature_contract,
)
from credit_risk.modeling.dataset import (
    GovernedDevelopmentData,
    ModelingDataError,
    load_governed_development_data,
)
from credit_risk.modeling.metrics import (
    MetricValidationError,
    PredictionMetrics,
    RepeatSummary,
    evaluate_predictions,
    summarize_repeat_values,
)
from credit_risk.modeling.tracking import (
    BASELINE_MODEL_NAMES,
    DEFAULT_EXPERIMENT_NAME,
    GitEvidence,
    ModelRunPayload,
    TrackingDependencyError,
    TrackingError,
    TrackingRunResult,
    collect_git_evidence,
    collect_package_versions,
    ensure_mlflow_available,
    mark_tracking_run_failed,
    track_baseline_runs,
)

DEFAULT_DATA_ROOT: Final[Path] = Path("data")
DEFAULT_TRACKING_ROOT: Final[Path] = Path("experiment/mlflow")
DEFAULT_OUTPUT_ROOT: Final[Path] = Path("reports/modeling/baseline_v1")
PROVISIONAL_OUTPUT_ROOT: Final[Path] = Path("experiment/provisional/baseline_v1")
SUMMARY_FILENAME: Final[str] = "summary.json"
REPORT_FILENAME: Final[str] = "baseline-report.md"
OOF_FILENAME: Final[str] = "oof_predictions.csv"
LOGISTIC_DIAGNOSTICS_FILENAME: Final[str] = "logistic_fold_diagnostics.json"


class BaselineWorkflowError(RuntimeError):
    """Raised when baseline evidence cannot be produced without violating a gate."""


@dataclass(frozen=True, slots=True)
class BaselineExperimentResult:
    """Published evidence and operational tracking identifiers for one execution."""

    summary_path: Path
    report_path: Path
    oof_predictions_path: Path
    logistic_diagnostics_path: Path
    summary_sha256: str
    report_sha256: str
    oof_predictions_sha256: str
    logistic_diagnostics_sha256: str
    tracking: TrackingRunResult


@dataclass(frozen=True, slots=True)
class _PredictionBlock:
    repeat_index: int
    fold_index: int
    account_ids: np.ndarray
    target: np.ndarray
    scores: Mapping[str, np.ndarray]
    logistic_diagnostic: _LogisticFoldDiagnostic


@dataclass(frozen=True, slots=True)
class _LogisticFoldDiagnostic:
    repeat_index: int
    fold_index: int
    transformed_feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    iterations: int


@dataclass(frozen=True, slots=True)
class _ModelEvaluation:
    fold_metrics: tuple[tuple[int, int, PredictionMetrics], ...]
    repeat_metrics: tuple[tuple[int, PredictionMetrics], ...]
    combined_oof_metrics: PredictionMetrics
    repeat_summaries: Mapping[str, RepeatSummary]


def run_baseline_experiment(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config_path: str | Path = DEFAULT_BASELINE_CONFIG_PATH,
    tracking_root: str | Path = DEFAULT_TRACKING_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    allow_dirty: bool = False,
    repo_root: str | Path = ".",
) -> BaselineExperimentResult:
    """Run the three approved baselines and publish governed experiment evidence."""

    try:
        return _run_baseline_experiment(
            data_root=Path(data_root),
            config_path=Path(config_path),
            tracking_root=Path(tracking_root),
            output_root=Path(output_root),
            allow_dirty=allow_dirty,
            repo_root=Path(repo_root),
        )
    except BaselineWorkflowError:
        raise
    except (TrackingDependencyError, TrackingError) as error:
        raise BaselineWorkflowError(str(error)) from error
    except (BaselineValidationError, MetricValidationError, ModelingDataError) as error:
        raise BaselineWorkflowError(f"Baseline experiment validation failed: {error}") from error
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise BaselineWorkflowError(f"Baseline experiment failed: {error}") from error


def _run_baseline_experiment(
    *,
    data_root: Path,
    config_path: Path,
    tracking_root: Path,
    output_root: Path,
    allow_dirty: bool,
    repo_root: Path,
) -> BaselineExperimentResult:
    ensure_mlflow_available()
    package_versions = collect_package_versions()
    git = collect_git_evidence(repo_root)
    repo_root = git.repository_root or repo_root.resolve()
    data_root = _root_relative_path(data_root, repo_root)
    tracking_root = _root_relative_path(tracking_root, repo_root)
    output_root = _root_relative_path(output_root, repo_root)
    _enforce_git_output_policy(
        git,
        allow_dirty=allow_dirty,
        output_root=output_root,
        repo_root=repo_root,
    )
    config_path = _resolve_entrypoint_file(config_path, repo_root, "baseline configuration")
    python_version = platform.python_version()
    uv_lock_sha256 = _sha256(_read_regular_file(repo_root / "uv.lock", "dependency lock"))
    config_bytes = _read_regular_file(config_path, "baseline configuration")
    config_sha256 = _sha256(config_bytes)
    config = parse_baseline_config(config_bytes, source=config_path)
    feature_contract_path = _resolve_configured_path(
        config_path=config_path,
        configured_path=config.feature_contract_path,
        repo_root=repo_root,
        label="feature contract",
    )
    dataset_manifest_path = _resolve_configured_path(
        config_path=config_path,
        configured_path=config.dataset_manifest_path,
        repo_root=repo_root,
        label="dataset manifest",
    )
    split_config_path = _resolve_configured_path(
        config_path=config_path,
        configured_path=config.split_config_path,
        repo_root=repo_root,
        label="split configuration",
    )
    feature_contract_bytes = _read_regular_file(feature_contract_path, "feature contract")
    feature_contract_sha256 = _sha256(feature_contract_bytes)
    if feature_contract_sha256 != config.feature_contract_sha256:
        raise BaselineWorkflowError(
            "Baseline configuration feature-contract hash does not match the checked-in bytes."
        )
    parse_feature_contract(feature_contract_bytes, source=feature_contract_path)
    governed = load_governed_development_data(
        data_root=data_root,
        feature_contract_path=feature_contract_path,
        manifest_path=dataset_manifest_path,
        split_config_path=split_config_path,
    )
    _validate_runtime_contract(config, governed)

    blocks = _score_all_folds(governed, config=config)
    capacities = tuple(float(capacity) for capacity in config.evaluation.capacities)
    evaluations = _evaluate_models(blocks, capacities=capacities)
    prediction_kinds = {
        config.baselines.prevalence.model_id: config.baselines.prevalence.prediction_kind,
        config.baselines.repayment_rule.model_id: (config.baselines.repayment_rule.prediction_kind),
        config.baselines.logistic.model_id: config.baselines.logistic.prediction_kind,
    }
    oof_bytes = _oof_csv_bytes(blocks, prediction_kinds=prediction_kinds)
    oof_sha256 = _sha256(oof_bytes)
    oof_path = tracking_root / "execution-artifacts" / oof_sha256 / OOF_FILENAME
    _publish_immutable_runtime_artifact(oof_path, oof_bytes)
    diagnostics_bytes = _logistic_diagnostics_bytes(blocks)
    diagnostics_sha256 = _sha256(diagnostics_bytes)
    diagnostics_path = (
        tracking_root / "execution-artifacts" / diagnostics_sha256 / LOGISTIC_DIAGNOSTICS_FILENAME
    )
    _publish_immutable_runtime_artifact(diagnostics_path, diagnostics_bytes)

    summary = _summary_payload(
        config=config,
        config_sha256=config_sha256,
        feature_contract_sha256=feature_contract_sha256,
        governed=governed,
        git=git,
        python_version=python_version,
        uv_lock_sha256=uv_lock_sha256,
        package_versions=package_versions,
        evaluations=evaluations,
        oof_sha256=oof_sha256,
        diagnostics_sha256=diagnostics_sha256,
    )
    summary_bytes = _json_bytes(summary)
    summary_sha256 = _sha256(summary_bytes)
    report_bytes = _report_bytes(summary, summary_sha256=summary_sha256)
    report_sha256 = _sha256(report_bytes)

    runtime_evidence_root = tracking_root / "execution-artifacts" / summary_sha256
    staged_summary = runtime_evidence_root / SUMMARY_FILENAME
    staged_report = runtime_evidence_root / REPORT_FILENAME
    _publish_immutable_runtime_artifact(staged_summary, summary_bytes)
    _publish_immutable_runtime_artifact(staged_report, report_bytes)

    tracking = track_baseline_runs(
        tracking_root=tracking_root,
        experiment_name=config.experiment_name or DEFAULT_EXPERIMENT_NAME,
        parent_run_name=config.experiment_id,
        parent_parameters=_parent_parameters(
            config=config,
            config_sha256=config_sha256,
            feature_contract_sha256=feature_contract_sha256,
            governed=governed,
            git=git,
            python_version=python_version,
            uv_lock_sha256=uv_lock_sha256,
            package_versions=package_versions,
        ),
        parent_tags={
            "experiment_id": config.experiment_id,
            "partition": config.partition,
            "holdout_evaluated": "false",
            "git_dirty": str(git.dirty).lower(),
        },
        model_runs=_tracking_payloads(evaluations, config=config),
        artifacts=(staged_summary, staged_report, oof_path, diagnostics_path),
    )
    summary_path = output_root / SUMMARY_FILENAME
    report_path = output_root / REPORT_FILENAME
    try:
        _promote_outputs(
            {
                summary_path: summary_bytes,
                report_path: report_bytes,
            }
        )
    except BaselineWorkflowError as publication_error:
        try:
            mark_tracking_run_failed(tracking, failure_stage="evidence_publication")
        except TrackingError as tracking_error:
            raise BaselineWorkflowError(
                f"{publication_error} The MLflow parent run could not be marked failed: "
                f"{tracking_error}"
            ) from publication_error
        raise

    return BaselineExperimentResult(
        summary_path=summary_path,
        report_path=report_path,
        oof_predictions_path=oof_path,
        logistic_diagnostics_path=diagnostics_path,
        summary_sha256=summary_sha256,
        report_sha256=report_sha256,
        oof_predictions_sha256=oof_sha256,
        logistic_diagnostics_sha256=diagnostics_sha256,
        tracking=tracking,
    )


def _validate_runtime_contract(
    config: BaselineExperimentConfig,
    governed: GovernedDevelopmentData,
) -> None:
    if config.partition != "development":
        raise BaselineWorkflowError(
            "Baseline experiments may access only the sealed development partition."
        )
    if config.positive_label != 1:
        raise BaselineWorkflowError("Baseline experiments require positive_label=1.")
    if governed.n_repeats != 3:
        raise BaselineWorkflowError(
            f"Governed baseline evaluation requires 3 repeats, found {governed.n_repeats}."
        )
    if governed.n_splits != 5:
        raise BaselineWorkflowError(
            f"Governed baseline evaluation requires 5 folds, found {governed.n_splits}."
        )
    configured_model_ids = (
        config.baselines.prevalence.model_id,
        config.baselines.repayment_rule.model_id,
        config.baselines.logistic.model_id,
    )
    if configured_model_ids != BASELINE_MODEL_NAMES:
        raise BaselineWorkflowError(
            "Baseline configuration does not contain the three approved model IDs in order."
        )


def _enforce_git_output_policy(
    git: GitEvidence,
    *,
    allow_dirty: bool,
    output_root: Path,
    repo_root: Path,
) -> None:
    if not git.dirty:
        return
    if not allow_dirty:
        raise BaselineWorkflowError(
            "Git worktree is dirty. Commit or stash reviewed changes, or pass --allow-dirty "
            "to record the content-sensitive diff hash in experiment lineage."
        )
    requested_output = (
        output_root.resolve()
        if output_root.is_absolute()
        else (repo_root.resolve() / output_root).resolve()
    )
    provisional_output = (repo_root.resolve() / PROVISIONAL_OUTPUT_ROOT).resolve()
    if requested_output != provisional_output:
        raise BaselineWorkflowError(
            "Dirty experiments must publish only to the ignored provisional output root. Use "
            "--output-root experiment/provisional/baseline_v1 with --allow-dirty."
        )


def _score_all_folds(
    governed: GovernedDevelopmentData,
    *,
    config: BaselineExperimentConfig,
) -> tuple[_PredictionBlock, ...]:
    blocks: list[_PredictionBlock] = []
    for repeat_index in range(governed.n_repeats):
        for fold_index in range(governed.n_splits):
            fold = governed.fold(repeat_index, fold_index)
            fitted = fit_fold_baselines(
                fold.X_train,
                fold.y_train,
                config=config.baselines,
            )
            predictions = fitted.predict(fold.X_validation)
            target = np.asarray(fold.y_validation, dtype=np.int8)
            account_ids = np.asarray(fold.validation_account_ids, dtype=np.int64)
            scores = {
                "fold_prevalence": np.asarray(predictions.prevalence, dtype=np.float64),
                "repayment_burden_rule": np.asarray(predictions.repayment_rule, dtype=np.float64),
                "logistic_l2": np.asarray(predictions.logistic_l2, dtype=np.float64),
            }
            _validate_prediction_block(account_ids, target, scores)
            classifier = fitted.logistic.pipeline.named_steps["classifier"]
            coefficients = tuple(float(value) for value in classifier.coef_[0])
            diagnostic = _LogisticFoldDiagnostic(
                repeat_index=repeat_index,
                fold_index=fold_index,
                transformed_feature_names=fitted.logistic.transformed_feature_names,
                coefficients=coefficients,
                intercept=float(classifier.intercept_[0]),
                iterations=int(classifier.n_iter_[0]),
            )
            _validate_logistic_diagnostic(diagnostic)
            blocks.append(
                _PredictionBlock(
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    account_ids=account_ids,
                    target=target,
                    scores=scores,
                    logistic_diagnostic=diagnostic,
                )
            )
    _validate_oof_coverage(governed, blocks)
    return tuple(blocks)


def _validate_logistic_diagnostic(diagnostic: _LogisticFoldDiagnostic) -> None:
    if (
        not diagnostic.transformed_feature_names
        or len(diagnostic.transformed_feature_names) != len(diagnostic.coefficients)
        or len(set(diagnostic.transformed_feature_names))
        != len(diagnostic.transformed_feature_names)
        or diagnostic.iterations < 1
        or not np.isfinite((*diagnostic.coefficients, diagnostic.intercept)).all()
    ):
        raise BaselineWorkflowError("Logistic fold diagnostics are incomplete or non-finite.")


def _validate_prediction_block(
    account_ids: np.ndarray,
    target: np.ndarray,
    scores: Mapping[str, np.ndarray],
) -> None:
    rows = len(account_ids)
    if rows < 1 or target.shape != (rows,):
        raise BaselineWorkflowError(
            "Fold targets and account IDs must be aligned non-empty vectors."
        )
    if len(np.unique(account_ids)) != rows:
        raise BaselineWorkflowError("A validation fold contains duplicate account IDs.")
    if not np.isin(target, (0, 1)).all():
        raise BaselineWorkflowError("A validation fold contains a non-binary target.")
    if tuple(scores) != BASELINE_MODEL_NAMES:
        raise BaselineWorkflowError("Fold predictions do not contain the three approved baselines.")
    for model_name, values in scores.items():
        if values.shape != (rows,) or not np.isfinite(values).all():
            raise BaselineWorkflowError(
                f"Baseline {model_name} returned misaligned or non-finite validation scores."
            )
    for model_name in ("fold_prevalence", "logistic_l2"):
        values = scores[model_name]
        if np.any((values < 0.0) | (values > 1.0)):
            raise BaselineWorkflowError(f"Baseline {model_name} returned invalid probabilities.")


def _validate_oof_coverage(
    governed: GovernedDevelopmentData,
    blocks: Sequence[_PredictionBlock],
) -> None:
    expected_ids = np.sort(np.asarray(governed.account_ids, dtype=np.int64))
    for repeat_index in range(governed.n_repeats):
        repeat_blocks = [block for block in blocks if block.repeat_index == repeat_index]
        actual_ids = np.sort(np.concatenate([block.account_ids for block in repeat_blocks]))
        if not np.array_equal(actual_ids, expected_ids):
            raise BaselineWorkflowError(
                f"OOF coverage for repeat {repeat_index} is not exactly one prediction per "
                "development account."
            )


def _evaluate_models(
    blocks: Sequence[_PredictionBlock],
    *,
    capacities: tuple[float, ...],
) -> dict[str, _ModelEvaluation]:
    evaluations: dict[str, _ModelEvaluation] = {}
    repeats = sorted({block.repeat_index for block in blocks})
    for model_name in BASELINE_MODEL_NAMES:
        fold_metrics: list[tuple[int, int, PredictionMetrics]] = []
        repeat_metrics: list[tuple[int, PredictionMetrics]] = []
        for block in blocks:
            fold_metrics.append(
                (
                    block.repeat_index,
                    block.fold_index,
                    _evaluate_block(block, model_name=model_name, capacities=capacities),
                )
            )
        for repeat_index in repeats:
            repeat_blocks = [block for block in blocks if block.repeat_index == repeat_index]
            repeat_metrics.append(
                (
                    repeat_index,
                    _evaluate_concatenated(
                        repeat_blocks,
                        model_name=model_name,
                        capacities=capacities,
                    ),
                )
            )
        combined = _evaluate_concatenated(
            blocks,
            model_name=model_name,
            capacities=capacities,
        )
        flattened_repeats = [_flatten_metrics(metrics) for _, metrics in repeat_metrics]
        metric_names = tuple(sorted(flattened_repeats[0]))
        repeat_summaries = {
            metric_name: summarize_repeat_values(
                [values[metric_name] for values in flattened_repeats]
            )
            for metric_name in metric_names
        }
        evaluations[model_name] = _ModelEvaluation(
            fold_metrics=tuple(fold_metrics),
            repeat_metrics=tuple(repeat_metrics),
            combined_oof_metrics=combined,
            repeat_summaries=repeat_summaries,
        )
    return evaluations


def _evaluate_block(
    block: _PredictionBlock,
    *,
    model_name: str,
    capacities: tuple[float, ...],
) -> PredictionMetrics:
    values = block.scores[model_name]
    return evaluate_predictions(
        block.target,
        values,
        probabilities=values if model_name != "repayment_burden_rule" else None,
        capacities=capacities,
    )


def _evaluate_concatenated(
    blocks: Sequence[_PredictionBlock],
    *,
    model_name: str,
    capacities: tuple[float, ...],
) -> PredictionMetrics:
    target = np.concatenate([block.target for block in blocks])
    values = np.concatenate([block.scores[model_name] for block in blocks])
    return evaluate_predictions(
        target,
        values,
        probabilities=values if model_name != "repayment_burden_rule" else None,
        capacities=capacities,
    )


def _summary_payload(
    *,
    config: BaselineExperimentConfig,
    config_sha256: str,
    feature_contract_sha256: str,
    governed: GovernedDevelopmentData,
    git: GitEvidence,
    python_version: str,
    uv_lock_sha256: str,
    package_versions: Mapping[str, str],
    evaluations: Mapping[str, _ModelEvaluation],
    oof_sha256: str,
    diagnostics_sha256: str,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_name in BASELINE_MODEL_NAMES:
        evaluation = evaluations[model_name]
        models[model_name] = {
            "descriptive_pooled_oof_metrics": _structured_metrics(evaluation.combined_oof_metrics),
            "repeat_metrics": [
                {
                    "repeat_index": repeat_index,
                    **_structured_metrics(metrics),
                }
                for repeat_index, metrics in evaluation.repeat_metrics
            ],
            "fold_metrics": [
                {
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                    **_structured_metrics(metrics),
                }
                for repeat_index, fold_index, metrics in evaluation.fold_metrics
            ],
            "repeat_summaries": {
                metric_name: asdict(summary)
                for metric_name, summary in sorted(evaluation.repeat_summaries.items())
            },
        }
    return {
        "schema_version": "1.0.0",
        "experiment": {
            "experiment_id": config.experiment_id,
            "experiment_name": config.experiment_name,
            "partition": config.partition,
            "positive_label": config.positive_label,
            "random_state": config.random_state,
            "primary_metric": config.evaluation.primary_metric,
            "probability_guardrail": config.evaluation.probability_guardrail,
            "primary_capacity_metric": {
                "metric": config.evaluation.primary_capacity_metric.metric,
                "capacity": config.evaluation.primary_capacity_metric.capacity,
            },
            "capacities": list(config.evaluation.capacities),
            "baseline_names": list(BASELINE_MODEL_NAMES),
        },
        "data": {
            "development_rows": int(len(governed.account_ids)),
            "n_splits": governed.n_splits,
            "n_repeats": governed.n_repeats,
            "holdout_evaluated": False,
        },
        "lineage": asdict(governed.lineage),
        "reproducibility": {
            "baseline_config_sha256": config_sha256,
            "feature_contract_sha256": feature_contract_sha256,
            "git_commit_sha": git.commit_sha,
            "git_dirty": git.dirty,
            "git_diff_sha256": git.diff_sha256,
            "python_version": python_version,
            "uv_lock_sha256": uv_lock_sha256,
            "package_versions": dict(sorted(package_versions.items())),
        },
        "runtime_artifacts": {
            "oof_predictions_filename": OOF_FILENAME,
            "oof_predictions_sha256": oof_sha256,
            "logistic_diagnostics_filename": LOGISTIC_DIAGNOSTICS_FILENAME,
            "logistic_diagnostics_sha256": diagnostics_sha256,
            "contains_holdout_rows": False,
            "contains_fitted_models": False,
        },
        "models": models,
    }


def _tracking_payloads(
    evaluations: Mapping[str, _ModelEvaluation],
    *,
    config: BaselineExperimentConfig,
) -> tuple[ModelRunPayload, ...]:
    parameters: dict[str, Mapping[str, str | int | float | bool]] = {
        "fold_prevalence": {
            "strategy": config.baselines.prevalence.kind,
            "prediction_kind": config.baselines.prevalence.prediction_kind,
        },
        "repayment_burden_rule": {
            "strategy": config.baselines.repayment_rule.kind,
            "prediction_kind": config.baselines.repayment_rule.prediction_kind,
            "lag_weights": ",".join(
                str(weight) for weight in config.baselines.repayment_rule.recency_weights
            ),
            "status_columns": ",".join(config.baselines.repayment_rule.status_columns),
            "negative_value_floor": config.baselines.repayment_rule.negative_value_floor,
            "aggregation": config.baselines.repayment_rule.aggregation,
        },
        "logistic_l2": {
            "strategy": config.baselines.logistic.kind,
            "prediction_kind": config.baselines.logistic.prediction_kind,
            "status_columns": ",".join(config.baselines.logistic.status_columns),
            "status_encoding": config.baselines.logistic.status_encoding,
            "status_categories": ",".join(
                str(category) for category in config.baselines.logistic.status_categories
            ),
            "status_drop": config.baselines.logistic.status_drop,
            "handle_unknown": config.baselines.logistic.handle_unknown,
            "monetary_columns": ",".join(config.baselines.logistic.monetary_columns),
            "scaler": config.baselines.logistic.scaler,
            "penalty": config.baselines.logistic.penalty,
            "C": config.baselines.logistic.c,
            "solver": config.baselines.logistic.solver,
            "class_weight": "none",
            "fit_intercept": config.baselines.logistic.fit_intercept,
            "max_iter": config.baselines.logistic.max_iter,
            "tol": config.baselines.logistic.tolerance,
            "random_state": config.baselines.logistic.random_state,
        },
    }
    payloads: list[ModelRunPayload] = []
    for model_name in BASELINE_MODEL_NAMES:
        evaluation = evaluations[model_name]
        metrics: dict[str, float] = {
            f"descriptive_pooled_oof.{name}": value
            for name, value in _flatten_metrics(evaluation.combined_oof_metrics).items()
        }
        for repeat_index, repeat_metrics in evaluation.repeat_metrics:
            metrics.update(
                {
                    f"repeat.r{repeat_index}.{name}": value
                    for name, value in _flatten_metrics(repeat_metrics).items()
                }
            )
        for repeat_index, fold_index, fold_metrics in evaluation.fold_metrics:
            metrics.update(
                {
                    f"fold.r{repeat_index}.f{fold_index}.{name}": value
                    for name, value in _flatten_metrics(fold_metrics).items()
                }
            )
        for metric_name, summary in evaluation.repeat_summaries.items():
            for statistic in ("mean", "standard_deviation", "minimum", "maximum"):
                metrics[f"repeat_summary.{metric_name}.{statistic}"] = float(
                    getattr(summary, statistic)
                )
        payloads.append(
            ModelRunPayload(
                model_name=model_name,
                parameters=parameters[model_name],
                metrics=metrics,
            )
        )
    return tuple(payloads)


def _parent_parameters(
    *,
    config: BaselineExperimentConfig,
    config_sha256: str,
    feature_contract_sha256: str,
    governed: GovernedDevelopmentData,
    git: GitEvidence,
    python_version: str,
    uv_lock_sha256: str,
    package_versions: Mapping[str, str],
) -> dict[str, str | int | float | bool]:
    parameters: dict[str, str | int | float | bool] = {
        "baseline_config_sha256": config_sha256,
        "feature_contract_sha256": feature_contract_sha256,
        "git_commit_sha": git.commit_sha,
        "git_dirty": git.dirty,
        "git_diff_sha256": git.diff_sha256,
        "python_version": python_version,
        "uv_lock_sha256": uv_lock_sha256,
        "development_rows": int(len(governed.account_ids)),
        "n_splits": governed.n_splits,
        "n_repeats": governed.n_repeats,
        "random_state": config.random_state,
        "primary_metric": config.evaluation.primary_metric,
        "probability_guardrail": config.evaluation.probability_guardrail,
        "primary_capacity_metric": (
            f"{config.evaluation.primary_capacity_metric.metric}"
            f"@{config.evaluation.primary_capacity_metric.capacity:g}"
        ),
    }
    parameters.update(
        {f"lineage.{key}": str(value) for key, value in asdict(governed.lineage).items()}
    )
    parameters.update(
        {f"package.{name}": version for name, version in sorted(package_versions.items())}
    )
    return parameters


def _structured_metrics(metrics: PredictionMetrics) -> dict[str, Any]:
    return {
        "discrimination": asdict(metrics.discrimination),
        "probability": asdict(metrics.probability) if metrics.probability is not None else None,
        "capacities": [asdict(capacity) for capacity in metrics.capacities],
    }


def _flatten_metrics(metrics: PredictionMetrics) -> dict[str, float]:
    values = {
        "average_precision": metrics.discrimination.average_precision,
        "roc_auc": metrics.discrimination.roc_auc,
        "ks": metrics.discrimination.ks,
        "gini": metrics.discrimination.gini,
    }
    if metrics.probability is not None:
        values["brier_score"] = metrics.probability.brier_score
        values["log_loss"] = metrics.probability.log_loss
    for capacity in metrics.capacities:
        prefix = f"capacity_{_capacity_label(capacity.capacity)}"
        values[f"{prefix}.precision"] = capacity.precision
        values[f"{prefix}.recall"] = capacity.recall
        values[f"{prefix}.lift"] = capacity.lift
        values[f"{prefix}.expected_true_positives"] = capacity.expected_true_positives
    return {key: float(value) for key, value in sorted(values.items())}


def _capacity_label(capacity: float) -> str:
    return f"{capacity:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _oof_csv_bytes(
    blocks: Sequence[_PredictionBlock],
    *,
    prediction_kinds: Mapping[str, str],
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "account_id",
            "model_id",
            "repeat_index",
            "fold_index",
            "prediction_kind",
            "score",
        )
    )
    rows: list[tuple[int, str, int, int, str, float]] = []
    for block in blocks:
        for row_index, account_id in enumerate(block.account_ids):
            for model_name in BASELINE_MODEL_NAMES:
                score = float(block.scores[model_name][row_index])
                rows.append(
                    (
                        int(account_id),
                        model_name,
                        block.repeat_index,
                        block.fold_index,
                        prediction_kinds[model_name],
                        score,
                    )
                )
    model_order = {name: index for index, name in enumerate(BASELINE_MODEL_NAMES)}
    rows.sort(key=lambda row: (model_order[row[1]], row[2], row[3], row[0]))
    for account_id, model_name, repeat_index, fold_index, prediction_kind, score in rows:
        writer.writerow(
            (
                account_id,
                model_name,
                repeat_index,
                fold_index,
                prediction_kind,
                _format_float(score),
            )
        )
    return output.getvalue().encode("utf-8")


def _logistic_diagnostics_bytes(blocks: Sequence[_PredictionBlock]) -> bytes:
    diagnostics = tuple(block.logistic_diagnostic for block in blocks)
    if not diagnostics:
        raise BaselineWorkflowError("Logistic diagnostics require at least one fold.")
    feature_names = diagnostics[0].transformed_feature_names
    if any(diagnostic.transformed_feature_names != feature_names for diagnostic in diagnostics):
        raise BaselineWorkflowError("Logistic transformed features differ across reviewed folds.")
    payload = {
        "schema_version": "1.0.0",
        "model_id": "logistic_l2",
        "transformed_feature_names": list(feature_names),
        "folds": [
            {
                "repeat_index": diagnostic.repeat_index,
                "fold_index": diagnostic.fold_index,
                "iterations": diagnostic.iterations,
                "intercept": diagnostic.intercept,
                "coefficients": list(diagnostic.coefficients),
            }
            for diagnostic in diagnostics
        ],
    }
    return _json_bytes(payload)


def _report_bytes(summary: Mapping[str, Any], *, summary_sha256: str) -> bytes:
    models = summary["models"]
    experiment = summary["experiment"]
    reproducibility = summary["reproducibility"]
    lineage = summary["lineage"]
    lines = [
        "# Governed baseline experiment report",
        "",
        f"- **Experiment:** `{experiment['experiment_id']}`",
        "- **Evaluation boundary:** sealed development folds only; holdout rows were not "
        "exposed to model fitting, scoring, or evaluation",
        f"- **Primary metric:** `{experiment['primary_metric']}`",
        f"- **Probability guardrail:** `{experiment['probability_guardrail']}`",
        "- **Primary capacity metric:** "
        f"`{experiment['primary_capacity_metric']['metric']}@"
        f"{experiment['primary_capacity_metric']['capacity']}`",
        f"- **Deterministic summary SHA-256:** `{summary_sha256}`",
        "",
        "## Protocol result across complete repeats",
        "",
        "The headline values are means of the three complete repeated-CV evaluations. "
        "`average_precision` is non-interpolated average precision, not a trapezoidal "
        "PR-curve area.",
        "",
        "| Baseline | Average precision | ROC-AUC | KS | Gini | Brier score | Log loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in BASELINE_MODEL_NAMES:
        repeat = models[model_name]["repeat_summaries"]
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{model_name}`",
                    _display_float(repeat["average_precision"]["mean"]),
                    _display_float(repeat["roc_auc"]["mean"]),
                    _display_float(repeat["ks"]["mean"]),
                    _display_float(repeat["gini"]["mean"]),
                    (
                        _display_float(repeat["brier_score"]["mean"])
                        if "brier_score" in repeat
                        else "n/a"
                    ),
                    (_display_float(repeat["log_loss"]["mean"]) if "log_loss" in repeat else "n/a"),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Repeat-level variation",
            "",
            "These are descriptive results across the three fixed CV repeats; they are not an "
            "independence-based confidence interval.",
            "",
            "| Baseline | Average precision mean | Std | Min | Max |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for model_name in BASELINE_MODEL_NAMES:
        repeat = models[model_name]["repeat_summaries"]["average_precision"]
        lines.append(
            f"| `{model_name}` | {_display_float(repeat['mean'])} | "
            f"{_display_float(repeat['standard_deviation'])} | "
            f"{_display_float(repeat['minimum'])} | "
            f"{_display_float(repeat['maximum'])} |"
        )

    lines.extend(("", "## Capacity evidence", ""))
    lines.extend(
        (
            "| Baseline | Capacity | Precision | Recall | Lift |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for model_name in BASELINE_MODEL_NAMES:
        repeat = models[model_name]["repeat_summaries"]
        for capacity in experiment["capacities"]:
            prefix = f"capacity_{_capacity_label(float(capacity))}"
            lines.append(
                f"| `{model_name}` | {100.0 * float(capacity):.0f}% | "
                f"{_display_float(repeat[f'{prefix}.precision']['mean'])} | "
                f"{_display_float(repeat[f'{prefix}.recall']['mean'])} | "
                f"{_display_float(repeat[f'{prefix}.lift']['mean'])} |"
            )

    lines.extend(
        (
            "",
            "## Reproducibility and governance",
            "",
            f"- Canonical data SHA-256: `{lineage['canonical_sha256']}`",
            f"- Split assignment SHA-256: `{lineage['assignment_sha256']}`",
            f"- Baseline config SHA-256: `{reproducibility['baseline_config_sha256']}`",
            f"- Feature contract SHA-256: `{reproducibility['feature_contract_sha256']}`",
            f"- Git commit: `{reproducibility['git_commit_sha']}`",
            f"- Dirty worktree recorded: `{str(reproducibility['git_dirty']).lower()}`",
            f"- Git diff SHA-256: `{reproducibility['git_diff_sha256']}`",
            f"- Python version: `{reproducibility['python_version']}`",
            f"- Dependency lock SHA-256: `{reproducibility['uv_lock_sha256']}`",
            "- No fitted estimator, pickle, raw source, or holdout row is stored in this report "
            "or the MLflow evidence artifacts.",
            "- Fold-level logistic convergence and coefficient diagnostics are stored as "
            "non-executable JSON runtime evidence.",
            "- The repayment rule is a ranking score, so Brier score and log loss are "
            "intentionally not reported for it.",
            "- The machine-readable summary retains pooled OOF metrics as descriptive "
            "evidence only; each development account appears once in every repeat, so pooled "
            "values are not the protocol-level headline.",
            "- Results describe this published 2005 Taiwan dataset and do not establish "
            "causal impact, India-specific performance, or production suitability.",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _promote_outputs(payloads: Mapping[Path, bytes]) -> None:
    previous: dict[Path, bytes | None] = {}
    changed: list[Path] = []
    try:
        for destination in payloads:
            if destination.exists() and not destination.is_file():
                raise BaselineWorkflowError(
                    f"Baseline report destination is not a regular file: {destination}"
                )
            previous[destination] = destination.read_bytes() if destination.is_file() else None
    except BaselineWorkflowError:
        raise
    except OSError as error:
        raise BaselineWorkflowError(
            f"Unable to preflight baseline report destinations: {error}"
        ) from error

    for destination, content in payloads.items():
        if previous[destination] == content:
            continue
        try:
            _write_bytes_atomically(destination, content)
            changed.append(destination)
        except OSError as error:
            _rollback_outputs(changed, previous)
            raise BaselineWorkflowError(
                f"Unable to publish baseline report evidence atomically: {error}"
            ) from error


def _rollback_outputs(changed: Sequence[Path], previous: Mapping[Path, bytes | None]) -> None:
    rollback_failures: list[str] = []
    for destination in reversed(changed):
        original = previous[destination]
        try:
            if original is None:
                destination.unlink(missing_ok=True)
            else:
                _write_bytes_atomically(destination, original)
        except OSError as error:  # pragma: no cover - catastrophic filesystem failure
            rollback_failures.append(f"{destination}: {error}")
    if rollback_failures:  # pragma: no cover - catastrophic filesystem failure
        raise BaselineWorkflowError(
            "Baseline report rollback failed: " + "; ".join(rollback_failures)
        )


def _publish_immutable_runtime_artifact(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file():
            raise BaselineWorkflowError(f"Runtime artifact destination is not a file: {path}")
        if path.read_bytes() != content:
            raise BaselineWorkflowError(
                f"Runtime artifact hash-addressed destination contains different bytes: {path}"
            )
        return
    try:
        _write_bytes_atomically(path, content)
    except OSError as error:
        raise BaselineWorkflowError(f"Unable to publish runtime evidence: {error}") from error


def _write_bytes_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".partial",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _read_regular_file(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise BaselineWorkflowError(f"The {label} is missing or not a file: {path}")
    return path.read_bytes()


def _resolve_configured_path(
    *,
    config_path: Path,
    configured_path: str,
    repo_root: Path,
    label: str,
) -> Path:
    relative_path = Path(configured_path)
    posix_path = PurePosixPath(configured_path)
    windows_path = PureWindowsPath(configured_path)
    if (
        relative_path.is_absolute()
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or relative_path.suffix.lower() != ".json"
        or "\\" in configured_path
    ):
        raise BaselineWorkflowError(
            f"Configured {label} path must be safe and repository-relative: {configured_path}"
        )
    repository_candidate = (repo_root.resolve() / relative_path).resolve()
    if repository_candidate.is_file():
        return repository_candidate
    sibling_candidate = (config_path.resolve().parent / relative_path.name).resolve()
    if len(posix_path.parts) == 1 and sibling_candidate.is_file():
        return sibling_candidate
    raise BaselineWorkflowError(
        f"Configured {label} is missing or not a file. Checked "
        f"{repository_candidate} and {sibling_candidate}."
    )


def _root_relative_path(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_entrypoint_file(path: Path, repo_root: Path, label: str) -> Path:
    candidate = _root_relative_path(path, repo_root)
    if not candidate.is_file():
        raise BaselineWorkflowError(f"The {label} is missing or not a file: {candidate}")
    return candidate


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _display_float(value: float) -> str:
    return format(float(value), ".6f")
