"""Governed orchestration for the Phase 3 CatBoost candidate experiment."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final

import numpy as np

from credit_risk.data.manifest import DEFAULT_DATASET_MANIFEST_PATH, DEFAULT_SPLIT_CONFIG_PATH
from credit_risk.modeling.candidate_contracts import (
    DEFAULT_CANDIDATE_CONFIG_PATH,
    CandidateExperimentConfig,
    FeatureView,
    SampledConfiguration,
    parse_candidate_config,
)
from credit_risk.modeling.candidate_selection import (
    CandidateScore,
    CandidateSelection,
    CandidateSelectionError,
    select_candidate,
)
from credit_risk.modeling.candidates import (
    CandidateFoldDiagnostics,
    CandidateModelError,
    fit_candidate_fold,
)
from credit_risk.modeling.contracts import ModelingContractError, parse_feature_contract
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
    DEFAULT_VERSION_PACKAGES,
    GitEvidence,
    ModelRunPayload,
    TrackingDependencyError,
    TrackingError,
    TrackingRunResult,
    collect_git_evidence,
    collect_package_versions,
    ensure_mlflow_available,
    mark_tracking_run_failed,
    track_candidate_runs,
)

DEFAULT_DATA_ROOT: Final[Path] = Path("data")
DEFAULT_TRACKING_ROOT: Final[Path] = Path("experiment/mlflow")
DEFAULT_OUTPUT_ROOT: Final[Path] = Path("reports/modeling/candidate_v1")
PROVISIONAL_OUTPUT_ROOT: Final[Path] = Path("experiment/provisional/candidate_v1")
SUMMARY_FILENAME: Final[str] = "summary.json"
REPORT_FILENAME: Final[str] = "candidate-report.md"
OOF_FILENAME: Final[str] = "oof_predictions.csv"
DIAGNOSTICS_FILENAME: Final[str] = "fold_diagnostics.json"
CANDIDATE_VERSION_PACKAGES: Final[tuple[str, ...]] = (*DEFAULT_VERSION_PACKAGES, "catboost")


class CandidateWorkflowError(RuntimeError):
    """Raised when Phase 3 evidence cannot be produced without violating a gate."""


@dataclass(frozen=True, slots=True)
class CandidateExperimentResult:
    """Published aggregate evidence and ignored runtime artifacts."""

    summary_path: Path
    report_path: Path
    oof_predictions_path: Path
    fold_diagnostics_path: Path
    summary_sha256: str
    report_sha256: str
    oof_predictions_sha256: str
    fold_diagnostics_sha256: str
    selected_model_id: str
    selected_configuration_id: str
    catboost_advances: bool
    tracking: TrackingRunResult


@dataclass(frozen=True, slots=True)
class _CandidateBlock:
    variant_id: str
    role: str
    feature_view: str
    configuration: SampledConfiguration
    repeat_index: int
    fold_index: int
    account_ids: np.ndarray
    target: np.ndarray
    probabilities: np.ndarray
    diagnostics: CandidateFoldDiagnostics


@dataclass(frozen=True, slots=True)
class _VariantEvaluation:
    variant_id: str
    role: str
    feature_view: str
    configuration: SampledConfiguration
    fold_metrics: tuple[tuple[int, int, PredictionMetrics], ...]
    repeat_metrics: tuple[tuple[int, PredictionMetrics], ...]
    combined_oof_metrics: PredictionMetrics
    repeat_summaries: Mapping[str, RepeatSummary]


def run_candidate_experiment(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config_path: str | Path = DEFAULT_CANDIDATE_CONFIG_PATH,
    tracking_root: str | Path = DEFAULT_TRACKING_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    allow_dirty: bool = False,
    repo_root: str | Path = ".",
) -> CandidateExperimentResult:
    """Run the frozen development-only CatBoost search and diagnostics."""

    try:
        return _run_candidate_experiment(
            data_root=Path(data_root),
            config_path=Path(config_path),
            tracking_root=Path(tracking_root),
            output_root=Path(output_root),
            allow_dirty=allow_dirty,
            repo_root=Path(repo_root),
        )
    except CandidateWorkflowError:
        raise
    except (TrackingDependencyError, TrackingError) as error:
        raise CandidateWorkflowError(str(error)) from error
    except (
        CandidateModelError,
        CandidateSelectionError,
        MetricValidationError,
        ModelingContractError,
        ModelingDataError,
    ) as error:
        raise CandidateWorkflowError(f"Candidate experiment validation failed: {error}") from error
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise CandidateWorkflowError(f"Candidate experiment failed: {error}") from error


def _run_candidate_experiment(
    *,
    data_root: Path,
    config_path: Path,
    tracking_root: Path,
    output_root: Path,
    allow_dirty: bool,
    repo_root: Path,
) -> CandidateExperimentResult:
    ensure_mlflow_available()
    package_versions = collect_package_versions(CANDIDATE_VERSION_PACKAGES)
    git = collect_git_evidence(repo_root)
    repo_root = git.repository_root or repo_root.resolve()
    data_root = _root_relative(data_root, repo_root)
    tracking_root = _root_relative(tracking_root, repo_root)
    output_root = _root_relative(output_root, repo_root)
    _enforce_output_policy(
        git,
        allow_dirty=allow_dirty,
        output_root=output_root,
        repo_root=repo_root,
    )
    config_path = _resolve_file(config_path, repo_root, "candidate configuration")
    config_bytes = _read_file(config_path, "candidate configuration")
    config_sha256 = _sha256(config_bytes)
    config = parse_candidate_config(config_bytes, source=config_path)
    feature_contract_path, feature_contract_sha256 = _validate_references(config, repo_root)
    _validate_package_versions(config, package_versions)

    governed = load_governed_development_data(
        data_root=data_root,
        feature_contract_path=feature_contract_path,
        manifest_path=repo_root / DEFAULT_DATASET_MANIFEST_PATH,
        split_config_path=repo_root / DEFAULT_SPLIT_CONFIG_PATH,
    )
    _validate_runtime_contract(config, governed, feature_contract_sha256)

    full_view = _view(config, "operational_full")
    blocks: list[_CandidateBlock] = []
    for sampled in config.candidate.search.sampled_configurations:
        blocks.extend(_score_variant(governed, config, full_view, sampled, role="search"))
    search_evaluations = _evaluate_variants(blocks, config=config)
    selection = select_candidate(
        tuple(
            _candidate_score(search_evaluations[_variant_id(full_view.view_id, sampled)], sampled)
            for sampled in config.candidate.search.sampled_configurations
        ),
        config.advancement_gate,
    )
    for view_id in ("repayment_status_only", "monetary_only"):
        blocks.extend(
            _score_variant(
                governed,
                config,
                _view(config, view_id),
                selection.selected_configuration,
                role="diagnostic_ablation",
            )
        )
    if len(blocks) != config.candidate.search.maximum_fold_fits:
        raise CandidateWorkflowError(
            "Candidate fit budget mismatch: "
            f"expected={config.candidate.search.maximum_fold_fits}, observed={len(blocks)}."
        )
    _validate_oof_coverage(governed, blocks, expected_variants=14)
    evaluations = _evaluate_variants(blocks, config=config)

    oof_bytes, oof_rows = _oof_bytes(blocks, evaluations)
    expected_oof_rows = 14 * governed.n_repeats * len(governed.account_ids)
    if oof_rows != expected_oof_rows:
        raise CandidateWorkflowError(
            f"Candidate OOF row count mismatch: expected={expected_oof_rows}, observed={oof_rows}."
        )
    oof_sha256 = _sha256(oof_bytes)
    oof_path = tracking_root / "execution-artifacts" / oof_sha256 / OOF_FILENAME
    _publish_immutable(oof_path, oof_bytes)

    diagnostics_bytes = _diagnostics_bytes(blocks)
    diagnostics_sha256 = _sha256(diagnostics_bytes)
    diagnostics_path = (
        tracking_root / "execution-artifacts" / diagnostics_sha256 / DIAGNOSTICS_FILENAME
    )
    _publish_immutable(diagnostics_path, diagnostics_bytes)

    summary = _summary_payload(
        config=config,
        config_sha256=config_sha256,
        feature_contract_sha256=feature_contract_sha256,
        governed=governed,
        git=git,
        python_version=platform.python_version(),
        uv_lock_sha256=_sha256(_read_file(repo_root / "uv.lock", "dependency lock")),
        package_versions=package_versions,
        evaluations=evaluations,
        selection=selection,
        oof_sha256=oof_sha256,
        oof_rows=oof_rows,
        diagnostics_sha256=diagnostics_sha256,
    )
    summary_bytes = _json_bytes(summary)
    summary_sha256 = _sha256(summary_bytes)
    report_bytes = _report_bytes(summary, summary_sha256=summary_sha256)
    report_sha256 = _sha256(report_bytes)
    runtime_root = tracking_root / "execution-artifacts" / summary_sha256
    staged_summary = runtime_root / SUMMARY_FILENAME
    staged_report = runtime_root / REPORT_FILENAME
    _publish_immutable(staged_summary, summary_bytes)
    _publish_immutable(staged_report, report_bytes)

    tracking = track_candidate_runs(
        tracking_root=tracking_root,
        parent_parameters=_parent_parameters(summary),
        parent_tags={
            "experiment_id": config.protocol_id,
            "partition": "development",
            "holdout_evaluated": "false",
            "git_dirty": str(git.dirty).lower(),
            "selected_model_id": selection.selected_model_id,
        },
        variant_runs=_tracking_payloads(evaluations),
        artifacts=(staged_summary, staged_report, oof_path, diagnostics_path),
    )
    summary_path = output_root / SUMMARY_FILENAME
    report_path = output_root / REPORT_FILENAME
    try:
        _promote_outputs({summary_path: summary_bytes, report_path: report_bytes})
    except CandidateWorkflowError as publication_error:
        try:
            mark_tracking_run_failed(tracking, failure_stage="evidence_publication")
        except TrackingError as tracking_error:
            raise CandidateWorkflowError(
                f"{publication_error} The MLflow parent run could not be marked failed: "
                f"{tracking_error}"
            ) from publication_error
        raise
    return CandidateExperimentResult(
        summary_path=summary_path,
        report_path=report_path,
        oof_predictions_path=oof_path,
        fold_diagnostics_path=diagnostics_path,
        summary_sha256=summary_sha256,
        report_sha256=report_sha256,
        oof_predictions_sha256=oof_sha256,
        fold_diagnostics_sha256=diagnostics_sha256,
        selected_model_id=selection.selected_model_id,
        selected_configuration_id=selection.selected_configuration.configuration_id,
        catboost_advances=selection.catboost_advances,
        tracking=tracking,
    )


def _validate_references(config: CandidateExperimentConfig, repo_root: Path) -> tuple[Path, str]:
    evidence = config.baseline_evidence
    summary_bytes = _verified_reference(
        repo_root,
        evidence.summary_path,
        evidence.summary_sha256,
        "baseline summary",
    )
    _verified_reference(
        repo_root,
        evidence.report_path,
        evidence.report_sha256,
        "baseline report",
    )
    feature_bytes = _verified_reference(
        repo_root,
        config.data_contract.feature_contract_path,
        config.data_contract.feature_contract_sha256,
        "feature contract",
    )
    parse_feature_contract(feature_bytes, source=config.data_contract.feature_contract_path)
    baseline = json.loads(summary_bytes)
    try:
        logistic = baseline["models"][evidence.reference_model_id]["repeat_summaries"]
        observed_reference = {
            "average_precision_mean": logistic["average_precision"]["mean"],
            "average_precision_standard_deviation": logistic["average_precision"][
                "standard_deviation"
            ],
            "brier_score_mean": logistic["brier_score"]["mean"],
            "lift_at_0_1_mean": logistic["capacity_0_1.lift"]["mean"],
            "roc_auc_mean": logistic["roc_auc"]["mean"],
        }
        if baseline["data"]["holdout_evaluated"] is not False:
            raise CandidateWorkflowError("Baseline evidence unexpectedly contains holdout results.")
    except (KeyError, TypeError) as error:
        raise CandidateWorkflowError(f"Baseline evidence has an invalid schema: {error}") from error
    if observed_reference != evidence.reference_metrics.model_dump():
        raise CandidateWorkflowError(
            "Candidate baseline reference metrics do not match the report."
        )
    return repo_root / config.data_contract.feature_contract_path, _sha256(feature_bytes)


def _verified_reference(
    repo_root: Path,
    relative_path: str,
    expected_sha256: str,
    label: str,
) -> bytes:
    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise CandidateWorkflowError(f"Candidate {label} escapes the repository: {path}") from error
    content = _read_file(path, label)
    observed = _sha256(content)
    if observed != expected_sha256:
        raise CandidateWorkflowError(
            f"Candidate {label} hash mismatch: expected={expected_sha256}, observed={observed}."
        )
    return content


def _validate_package_versions(
    config: CandidateExperimentConfig,
    versions: Mapping[str, str],
) -> None:
    expected = {
        "catboost": config.candidate.library_version,
        "scikit-learn": config.candidate.search.sampler_version,
    }
    mismatches = {
        package: (required, versions.get(package))
        for package, required in expected.items()
        if versions.get(package) != required
    }
    if mismatches:
        raise CandidateWorkflowError(f"Candidate dependency version mismatch: {mismatches}")


def _validate_runtime_contract(
    config: CandidateExperimentConfig,
    governed: GovernedDevelopmentData,
    feature_contract_sha256: str,
) -> None:
    if len(governed.account_ids) != config.data_contract.expected_rows:
        raise CandidateWorkflowError(
            "Governed development row count differs from candidate config."
        )
    if governed.n_splits != 5 or governed.n_repeats != 3:
        raise CandidateWorkflowError("Candidate evaluation requires exactly 5 folds and 3 repeats.")
    if governed.lineage.assignment_sha256 != config.data_contract.split_assignment_sha256:
        raise CandidateWorkflowError(
            "Candidate split-assignment lineage does not match the config."
        )
    if governed.lineage.feature_contract_sha256 != feature_contract_sha256:
        raise CandidateWorkflowError(
            "Candidate feature-contract lineage does not match runtime data."
        )
    if tuple(governed.predictors.columns) != _view(config, "operational_full").predictor_columns:
        raise CandidateWorkflowError("Governed predictors differ from the full candidate view.")
    if set(governed.assignments["partition"].astype(str)) != {"development"}:
        raise CandidateWorkflowError(
            "Candidate data interface exposed a non-development partition."
        )


def _score_variant(
    governed: GovernedDevelopmentData,
    config: CandidateExperimentConfig,
    view: FeatureView,
    sampled: SampledConfiguration,
    *,
    role: str,
) -> tuple[_CandidateBlock, ...]:
    blocks: list[_CandidateBlock] = []
    categorical = tuple(
        column
        for column in config.candidate.feature_handling.native_categorical_columns
        if column in view.predictor_columns
    )
    variant_id = _variant_id(view.view_id, sampled)
    for repeat_index in range(governed.n_repeats):
        for fold_index in range(governed.n_splits):
            fold = governed.fold(repeat_index, fold_index)
            result = fit_candidate_fold(
                fold.X_train.loc[:, list(view.predictor_columns)],
                fold.y_train,
                fold.X_validation.loc[:, list(view.predictor_columns)],
                fold.y_validation,
                predictor_columns=view.predictor_columns,
                categorical_columns=categorical,
                fixed_parameters=config.candidate.fixed_parameters,
                sampled_parameters=sampled.parameters,
            )
            account_ids = np.asarray(fold.validation_account_ids, dtype=np.int64)
            target = np.asarray(fold.y_validation, dtype=np.int8)
            probabilities = np.asarray(result.probabilities, dtype=np.float64)
            _validate_block(account_ids, target, probabilities)
            blocks.append(
                _CandidateBlock(
                    variant_id=variant_id,
                    role=role,
                    feature_view=view.view_id,
                    configuration=sampled,
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    account_ids=account_ids,
                    target=target,
                    probabilities=probabilities,
                    diagnostics=result.diagnostics,
                )
            )
    return tuple(blocks)


def _validate_block(
    account_ids: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    rows = len(account_ids)
    if rows < 1 or target.shape != (rows,) or probabilities.shape != (rows,):
        raise CandidateWorkflowError("Candidate fold outputs are empty or misaligned.")
    if len(np.unique(account_ids)) != rows:
        raise CandidateWorkflowError("Candidate validation fold contains duplicate account IDs.")
    if not np.array_equal(np.unique(target), np.asarray([0, 1], dtype=np.int8)):
        raise CandidateWorkflowError("Candidate validation fold must contain both binary classes.")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise CandidateWorkflowError("Candidate fold returned invalid probabilities.")


def _validate_oof_coverage(
    governed: GovernedDevelopmentData,
    blocks: Sequence[_CandidateBlock],
    *,
    expected_variants: int,
) -> None:
    expected_ids = np.sort(np.asarray(governed.account_ids, dtype=np.int64))
    variants = tuple(dict.fromkeys(block.variant_id for block in blocks))
    if len(variants) != expected_variants:
        raise CandidateWorkflowError(
            f"Candidate evidence requires {expected_variants} unique variants, found {len(variants)}."
        )
    for variant_id in variants:
        variant_blocks = [block for block in blocks if block.variant_id == variant_id]
        if len(variant_blocks) != governed.n_splits * governed.n_repeats:
            raise CandidateWorkflowError(
                f"Candidate variant {variant_id} has incomplete fold coverage."
            )
        for repeat_index in range(governed.n_repeats):
            repeat_blocks = [
                block for block in variant_blocks if block.repeat_index == repeat_index
            ]
            if {block.fold_index for block in repeat_blocks} != set(range(governed.n_splits)):
                raise CandidateWorkflowError(
                    f"Candidate variant {variant_id} repeat {repeat_index} has invalid folds."
                )
            actual_ids = np.sort(np.concatenate([block.account_ids for block in repeat_blocks]))
            if not np.array_equal(actual_ids, expected_ids):
                raise CandidateWorkflowError(
                    f"Candidate variant {variant_id} repeat {repeat_index} lacks exact OOF coverage."
                )


def _evaluate_variants(
    blocks: Sequence[_CandidateBlock],
    *,
    config: CandidateExperimentConfig,
) -> dict[str, _VariantEvaluation]:
    capacities = tuple(float(value) for value in config.evaluation.reported_capacities)
    variants = tuple(dict.fromkeys(block.variant_id for block in blocks))
    evaluations: dict[str, _VariantEvaluation] = {}
    for variant_id in variants:
        variant_blocks = [block for block in blocks if block.variant_id == variant_id]
        first = variant_blocks[0]
        fold_metrics = tuple(
            (
                block.repeat_index,
                block.fold_index,
                evaluate_predictions(
                    block.target,
                    block.probabilities,
                    probabilities=block.probabilities,
                    capacities=capacities,
                ),
            )
            for block in variant_blocks
        )
        repeat_metrics: list[tuple[int, PredictionMetrics]] = []
        for repeat_index in range(3):
            repeat_blocks = [
                block for block in variant_blocks if block.repeat_index == repeat_index
            ]
            repeat_metrics.append((repeat_index, _evaluate_concatenated(repeat_blocks, capacities)))
        combined = _evaluate_concatenated(variant_blocks, capacities)
        flattened = [_flatten_metrics(metrics) for _, metrics in repeat_metrics]
        summaries = {
            metric_name: summarize_repeat_values([values[metric_name] for values in flattened])
            for metric_name in sorted(flattened[0])
        }
        evaluations[variant_id] = _VariantEvaluation(
            variant_id=variant_id,
            role=first.role,
            feature_view=first.feature_view,
            configuration=first.configuration,
            fold_metrics=fold_metrics,
            repeat_metrics=tuple(repeat_metrics),
            combined_oof_metrics=combined,
            repeat_summaries=summaries,
        )
    return evaluations


def _evaluate_concatenated(
    blocks: Sequence[_CandidateBlock],
    capacities: tuple[float, ...],
) -> PredictionMetrics:
    target = np.concatenate([block.target for block in blocks])
    probabilities = np.concatenate([block.probabilities for block in blocks])
    return evaluate_predictions(
        target,
        probabilities,
        probabilities=probabilities,
        capacities=capacities,
    )


def _candidate_score(
    evaluation: _VariantEvaluation,
    configuration: SampledConfiguration,
) -> CandidateScore:
    summaries = evaluation.repeat_summaries
    return CandidateScore(
        configuration=configuration,
        average_precision_mean=summaries["average_precision"].mean,
        average_precision_standard_deviation=summaries["average_precision"].standard_deviation,
        brier_score_mean=summaries["brier_score"].mean,
        lift_at_0_1_mean=summaries["capacity_0_1.lift"].mean,
    )


def _summary_payload(
    *,
    config: CandidateExperimentConfig,
    config_sha256: str,
    feature_contract_sha256: str,
    governed: GovernedDevelopmentData,
    git: GitEvidence,
    python_version: str,
    uv_lock_sha256: str,
    package_versions: Mapping[str, str],
    evaluations: Mapping[str, _VariantEvaluation],
    selection: CandidateSelection,
    oof_sha256: str,
    oof_rows: int,
    diagnostics_sha256: str,
) -> dict[str, Any]:
    gate_outcomes = dict(selection.gate_outcomes)
    variants: dict[str, Any] = {}
    baseline = config.baseline_evidence.reference_metrics
    for variant_id, evaluation in evaluations.items():
        summaries = evaluation.repeat_summaries
        gate = gate_outcomes.get(evaluation.configuration.configuration_id)
        variants[variant_id] = {
            "role": evaluation.role,
            "feature_view": evaluation.feature_view,
            "configuration_id": evaluation.configuration.configuration_id,
            "parameters": evaluation.configuration.parameters.model_dump(),
            "eligible_for_advancement": evaluation.role == "search",
            "gate_outcome": asdict(gate) if evaluation.role == "search" and gate else None,
            "deltas_from_logistic": {
                "average_precision": summaries["average_precision"].mean
                - baseline.average_precision_mean,
                "brier_score": summaries["brier_score"].mean - baseline.brier_score_mean,
                "lift_at_0_1": summaries["capacity_0_1.lift"].mean - baseline.lift_at_0_1_mean,
            },
            "descriptive_pooled_oof_metrics": _structured_metrics(evaluation.combined_oof_metrics),
            "repeat_metrics": [
                {"repeat_index": repeat_index, **_structured_metrics(metrics)}
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
                name: asdict(summary) for name, summary in sorted(summaries.items())
            },
        }
    return {
        "schema_version": "1.0.0",
        "experiment": {
            "protocol_id": config.protocol_id,
            "experiment_name": "credit-risk-candidate-v1",
            "candidate_model_id": config.candidate.model_id,
            "primary_metric": config.evaluation.primary_metric,
            "probability_guardrail": config.evaluation.probability_guardrail,
            "capacities": list(config.evaluation.reported_capacities),
            "sampled_configuration_ids": [
                item.configuration_id for item in config.candidate.search.sampled_configurations
            ],
        },
        "data": {
            "development_rows": len(governed.account_ids),
            "holdout_evaluated": False,
            "n_repeats": governed.n_repeats,
            "n_splits": governed.n_splits,
            "partition": "development",
        },
        "lineage": asdict(governed.lineage),
        "baseline_reference": config.baseline_evidence.model_dump(),
        "reproducibility": {
            "candidate_config_sha256": config_sha256,
            "feature_contract_sha256": feature_contract_sha256,
            "git_commit_sha": git.commit_sha,
            "git_dirty": git.dirty,
            "git_diff_sha256": git.diff_sha256,
            "python_version": python_version,
            "uv_lock_sha256": uv_lock_sha256,
            "package_versions": dict(sorted(package_versions.items())),
        },
        "fit_budget": {
            "maximum_fold_fits": config.candidate.search.maximum_fold_fits,
            "completed_fold_fits": config.candidate.search.maximum_fold_fits,
            "search_fold_fits": 180,
            "diagnostic_fold_fits": 30,
            "evaluated_variants": 14,
        },
        "selection": {
            "selected_configuration_id": selection.selected_configuration.configuration_id,
            "selected_model_id": selection.selected_model_id,
            "catboost_advances": selection.catboost_advances,
            "equivalent_configuration_ids": list(selection.equivalent_configuration_ids),
            "diagnostic_views_eligible_for_advancement": False,
            "gate_id": config.advancement_gate.gate_id,
        },
        "runtime_artifacts": {
            "oof_predictions_filename": OOF_FILENAME,
            "oof_predictions_sha256": oof_sha256,
            "oof_prediction_rows": oof_rows,
            "fold_diagnostics_filename": DIAGNOSTICS_FILENAME,
            "fold_diagnostics_sha256": diagnostics_sha256,
            "contains_holdout_rows": False,
            "contains_fitted_models": False,
        },
        "variants": variants,
    }


def _oof_bytes(
    blocks: Sequence[_CandidateBlock],
    evaluations: Mapping[str, _VariantEvaluation],
) -> tuple[bytes, int]:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "account_id",
            "variant_id",
            "feature_view",
            "configuration_id",
            "repeat_index",
            "fold_index",
            "probability",
        )
    )
    rows = 0
    for variant_id in evaluations:
        variant_blocks = [block for block in blocks if block.variant_id == variant_id]
        first = variant_blocks[0]
        for repeat_index in range(3):
            repeat_blocks = [
                block for block in variant_blocks if block.repeat_index == repeat_index
            ]
            account_ids = np.concatenate([block.account_ids for block in repeat_blocks])
            probabilities = np.concatenate([block.probabilities for block in repeat_blocks])
            fold_indexes = np.concatenate(
                [
                    np.full(len(block.account_ids), block.fold_index, dtype=np.int8)
                    for block in repeat_blocks
                ]
            )
            order = np.argsort(account_ids, kind="stable")
            for position in order:
                writer.writerow(
                    (
                        int(account_ids[position]),
                        variant_id,
                        first.feature_view,
                        first.configuration.configuration_id,
                        repeat_index,
                        int(fold_indexes[position]),
                        _format_float(probabilities[position]),
                    )
                )
                rows += 1
    return output.getvalue().encode("utf-8"), rows


def _diagnostics_bytes(blocks: Sequence[_CandidateBlock]) -> bytes:
    payload = {
        "schema_version": "1.0.0",
        "model_id": "catboost_v1",
        "fit_count": len(blocks),
        "fits": [
            {
                "variant_id": block.variant_id,
                "role": block.role,
                "feature_view": block.feature_view,
                "configuration_id": block.configuration.configuration_id,
                "repeat_index": block.repeat_index,
                "fold_index": block.fold_index,
                **asdict(block.diagnostics),
            }
            for block in blocks
        ],
    }
    return _json_bytes(payload)


def _report_bytes(summary: Mapping[str, Any], *, summary_sha256: str) -> bytes:
    selection = summary["selection"]
    baseline = summary["baseline_reference"]["reference_metrics"]
    lines = [
        "# Governed CatBoost candidate report",
        "",
        f"- **Protocol:** `{summary['experiment']['protocol_id']}`",
        "- **Evaluation boundary:** 24,000 development rows using the reviewed 5-fold × "
        "3-repeat assignments; the holdout was not fitted, scored, or evaluated",
        f"- **Deterministic summary SHA-256:** `{summary_sha256}`",
        f"- **Completed fold fits:** `{summary['fit_budget']['completed_fold_fits']}`",
        "",
        "## Advancement decision",
        "",
        f"- **Selected configuration:** `{selection['selected_configuration_id']}`",
        f"- **Phase 4 candidate:** `{selection['selected_model_id']}`",
        f"- **CatBoost advances:** `{str(selection['catboost_advances']).lower()}`",
        "- Reduced feature views are diagnostic only and cannot advance.",
        "",
        "## Full-view bounded search",
        "",
        "| Configuration | AP | AP std | Brier | Lift@10% | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant in summary["variants"].values():
        if variant["role"] != "search":
            continue
        repeat = variant["repeat_summaries"]
        gate = variant["gate_outcome"]
        lines.append(
            f"| `{variant['configuration_id']}` | "
            f"{_display_float(repeat['average_precision']['mean'])} | "
            f"{_display_float(repeat['average_precision']['standard_deviation'])} | "
            f"{_display_float(repeat['brier_score']['mean'])} | "
            f"{_display_float(repeat['capacity_0_1.lift']['mean'])} | "
            f"{str(gate['eligible']).lower()} |"
        )
    lines.extend(
        (
            "",
            "## Diagnostic feature-family ablations",
            "",
            "| Feature view | Predictors | AP | AP delta vs logistic | Brier | Lift@10% |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for variant in summary["variants"].values():
        if variant["role"] != "diagnostic_ablation":
            continue
        repeat = variant["repeat_summaries"]
        lines.append(
            f"| `{variant['feature_view']}` | "
            f"{6 if variant['feature_view'] == 'repayment_status_only' else 13} | "
            f"{_display_float(repeat['average_precision']['mean'])} | "
            f"{_display_float(variant['deltas_from_logistic']['average_precision'])} | "
            f"{_display_float(repeat['brier_score']['mean'])} | "
            f"{_display_float(repeat['capacity_0_1.lift']['mean'])} |"
        )
    lines.extend(
        (
            "",
            "## Reference and governance",
            "",
            f"- Logistic AP reference: `{_display_float(baseline['average_precision_mean'])}`",
            f"- Logistic Brier reference: `{_display_float(baseline['brier_score_mean'])}`",
            f"- Logistic lift@10% reference: `{_display_float(baseline['lift_at_0_1_mean'])}`",
            f"- Candidate config SHA-256: `{summary['reproducibility']['candidate_config_sha256']}`",
            f"- Git commit: `{summary['reproducibility']['git_commit_sha']}`",
            f"- Dirty worktree recorded: `{str(summary['reproducibility']['git_dirty']).lower()}`",
            "- No fitted estimator, executable model, raw row, MLflow identifier, or holdout "
            "result is committed.",
            "- These development-CV results are model-selection evidence, not an unbiased final "
            "performance estimate.",
            "- Results describe the published 2005 Taiwan sample and do not establish causal "
            "impact, India-specific performance, or production suitability.",
            "- Calibration, uncertainty, policy selection, and one-time holdout evaluation "
            "remain Phase 4 work.",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _tracking_payloads(
    evaluations: Mapping[str, _VariantEvaluation],
) -> tuple[ModelRunPayload, ...]:
    payloads: list[ModelRunPayload] = []
    for variant_id, evaluation in evaluations.items():
        parameters: dict[str, str | int | float | bool] = {
            "role": evaluation.role,
            "feature_view": evaluation.feature_view,
            "configuration_id": evaluation.configuration.configuration_id,
            **evaluation.configuration.parameters.model_dump(),
        }
        metrics: dict[str, float] = {}
        for name, summary in evaluation.repeat_summaries.items():
            for statistic in ("mean", "standard_deviation", "minimum", "maximum"):
                metrics[f"repeat_summary.{name}.{statistic}"] = float(getattr(summary, statistic))
        for repeat_index, fold_index, fold_metrics in evaluation.fold_metrics:
            for name, value in _flatten_metrics(fold_metrics).items():
                metrics[f"fold.r{repeat_index}.f{fold_index}.{name}"] = value
        payloads.append(
            ModelRunPayload(model_name=variant_id, parameters=parameters, metrics=metrics)
        )
    return tuple(payloads)


def _parent_parameters(summary: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    reproducibility = summary["reproducibility"]
    selection = summary["selection"]
    lineage = summary["lineage"]
    parameters: dict[str, str | int | float | bool] = {
        "candidate_config_sha256": reproducibility["candidate_config_sha256"],
        "feature_contract_sha256": reproducibility["feature_contract_sha256"],
        "git_commit_sha": reproducibility["git_commit_sha"],
        "git_dirty": reproducibility["git_dirty"],
        "git_diff_sha256": reproducibility["git_diff_sha256"],
        "python_version": reproducibility["python_version"],
        "uv_lock_sha256": reproducibility["uv_lock_sha256"],
        "selected_configuration_id": selection["selected_configuration_id"],
        "selected_model_id": selection["selected_model_id"],
        "catboost_advances": selection["catboost_advances"],
        "completed_fold_fits": summary["fit_budget"]["completed_fold_fits"],
    }
    parameters.update({f"lineage.{key}": str(value) for key, value in lineage.items()})
    parameters.update(
        {
            f"package.{name}": version
            for name, version in reproducibility["package_versions"].items()
        }
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


def _view(config: CandidateExperimentConfig, view_id: str) -> FeatureView:
    for view in config.feature_views:
        if view.view_id == view_id:
            return view
    raise CandidateWorkflowError(f"Candidate feature view is missing: {view_id}")


def _variant_id(view_id: str, sampled: SampledConfiguration) -> str:
    return f"{view_id}__{sampled.configuration_id}"


def _capacity_label(capacity: float) -> str:
    return f"{capacity:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _enforce_output_policy(
    git: GitEvidence,
    *,
    allow_dirty: bool,
    output_root: Path,
    repo_root: Path,
) -> None:
    if not git.dirty:
        return
    if not allow_dirty:
        raise CandidateWorkflowError(
            "Git worktree is dirty. Commit reviewed changes or use --allow-dirty with the "
            "candidate provisional output root."
        )
    provisional = (repo_root / PROVISIONAL_OUTPUT_ROOT).resolve()
    if output_root.resolve() != provisional:
        raise CandidateWorkflowError(
            "Dirty candidate experiments must use --output-root "
            "experiment/provisional/candidate_v1."
        )


def _promote_outputs(payloads: Mapping[Path, bytes]) -> None:
    previous: dict[Path, bytes | None] = {}
    changed: list[Path] = []
    try:
        for destination in payloads:
            if destination.exists() and not destination.is_file():
                raise CandidateWorkflowError(
                    f"Candidate report destination is not a regular file: {destination}"
                )
            previous[destination] = destination.read_bytes() if destination.is_file() else None
        for destination, content in payloads.items():
            if previous[destination] == content:
                continue
            _write_atomic(destination, content)
            changed.append(destination)
    except CandidateWorkflowError:
        _rollback(changed, previous)
        raise
    except OSError as error:
        _rollback(changed, previous)
        raise CandidateWorkflowError(
            f"Unable to publish candidate report evidence: {error}"
        ) from error


def _rollback(changed: Sequence[Path], previous: Mapping[Path, bytes | None]) -> None:
    failures: list[str] = []
    for destination in reversed(changed):
        try:
            original = previous[destination]
            if original is None:
                destination.unlink(missing_ok=True)
            else:
                _write_atomic(destination, original)
        except OSError as error:  # pragma: no cover - catastrophic filesystem failure
            failures.append(f"{destination}: {error}")
    if failures:  # pragma: no cover - catastrophic filesystem failure
        raise CandidateWorkflowError("Candidate evidence rollback failed: " + "; ".join(failures))


def _publish_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file():
            raise CandidateWorkflowError(f"Runtime artifact destination is not a file: {path}")
        if path.read_bytes() != content:
            raise CandidateWorkflowError(
                f"Hash-addressed runtime artifact contains different bytes: {path}"
            )
        return
    try:
        _write_atomic(path, content)
    except OSError as error:
        raise CandidateWorkflowError(
            f"Unable to publish candidate runtime evidence: {error}"
        ) from error


def _write_atomic(path: Path, content: bytes) -> None:
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


def _resolve_file(path: Path, repo_root: Path, label: str) -> Path:
    candidate = _root_relative(path, repo_root)
    if not candidate.is_file():
        raise CandidateWorkflowError(f"The {label} is missing or not a file: {candidate}")
    return candidate


def _root_relative(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root.resolve() / path).resolve()


def _read_file(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise CandidateWorkflowError(f"The {label} is missing or not a file: {path}")
    return path.read_bytes()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _display_float(value: float) -> str:
    return format(float(value), ".6f")
