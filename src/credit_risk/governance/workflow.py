"""Atomic, prediction-only Phase 5 governance evidence workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from credit_risk.governance.contracts import (
    DEFAULT_GOVERNANCE_CONFIG_PATH,
    GovernanceConfig,
    governance_config_sha256,
    load_governance_config,
)
from credit_risk.governance.explanations import ExplanationResult, explain_validation_sample
from credit_risk.governance.fairness import FairnessResult, analyse_subgroups
from credit_risk.modeling.contracts import AUDIT_COLUMNS, PREDICTOR_COLUMNS
from credit_risk.modeling.dataset import GovernedDevelopmentData, load_governed_development_data
from credit_risk.modeling.metrics import PredictionMetrics, evaluate_predictions
from credit_risk.modeling.risk_policy import risk_band
from credit_risk.modeling.selected_bundle import load_selected_bundle
from credit_risk.modeling.tracking import collect_git_evidence

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_BUNDLE_ROOT = Path("models/selected_v1")
DEFAULT_RUNTIME_ROOT = Path("experiment/governance/phase5_v1")
DEFAULT_OUTPUT_ROOT = Path("reports/governance/phase5_v1")


class GovernanceWorkflowError(RuntimeError):
    """Raised when Phase 5 evidence cannot be produced without violating governance."""


@dataclass(frozen=True, slots=True)
class GovernanceWorkflowResult:
    evidence_root: Path
    runtime_root: Path
    evidence_manifest_sha256: str
    summary_sha256: str
    g3_result: str
    review_trigger_count: int


def run_governance_build(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config_path: str | Path = DEFAULT_GOVERNANCE_CONFIG_PATH,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    runtime_root: str | Path = DEFAULT_RUNTIME_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> GovernanceWorkflowResult:
    """Build validation-only governance and native-SHAP evidence without fitting."""

    try:
        config = load_governance_config(config_path)
        git = collect_git_evidence(Path(config_path).resolve().parent)
        if git.dirty:
            raise GovernanceWorkflowError(
                "Official Phase 5 evidence requires a clean committed worktree."
            )
        if git.repository_root is None:
            raise GovernanceWorkflowError("Git repository root is unavailable.")
        repository = git.repository_root
        output = _safe_publication_destination(
            repository,
            output_root,
            allowed_subtree=Path("reports/governance"),
            description="governance evidence root",
        )
        runtime = _safe_publication_destination(
            repository,
            runtime_root,
            allowed_subtree=Path("experiment/governance"),
            description="governance runtime root",
        )
        _validate_publication_separation(output, runtime)
        bundle = _safe_repository_input(repository, bundle_root, "selected bundle root")
        if output.exists() or runtime.exists():
            raise GovernanceWorkflowError(
                "Refusing to overwrite existing Phase 5 evidence or runtime artifacts."
            )

        governed = load_governed_development_data(data_root=data_root)
        _validate_development_boundary(governed, config)
        _validate_data_lineage(governed, config)
        _validate_repository_lineage(repository, config)
        predictors, target, audit = _validation_slice(governed, config)
        _prove_audit_exclusion(predictors, audit, config)

        manifest, model = load_selected_bundle(
            bundle,
            trusted=True,
            expected_manifest_sha256=config.lineage["bundle_manifest_sha256"],
            required_dependencies=tuple(config.dependencies),
        )
        if manifest.model_sha256 != config.lineage["model_sha256"]:
            raise GovernanceWorkflowError(
                "Selected model digest differs from the Phase 5 contract."
            )
        probabilities = model.predict_proba(predictors)
        metrics = evaluate_predictions(
            target.to_numpy(), probabilities, probabilities=probabilities
        )
        _validate_metric_parity(metrics, config)
        bands = np.asarray(
            [risk_band(float(score), manifest.risk_band_thresholds) for score in probabilities]
        )

        fairness = analyse_subgroups(
            audit.loc[:, list(AUDIT_COLUMNS)],
            target,
            probabilities,
            threshold=config.prediction.q90,
            contract=config.fairness,
        )
        _validate_expected_triggers(fairness, config)
        explanations = explain_validation_sample(
            model,
            predictors,
            target,
            probabilities,
            bands,
            config.explanation,
        )

        runtime.parent.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        stage_id = uuid4().hex
        staged_runtime = runtime.with_name(f".{runtime.name}.stage-{stage_id}")
        staged_output = output.with_name(f".{output.name}.stage-{stage_id}")
        staged_runtime.mkdir()
        staged_output.mkdir()
        try:
            _write_predictions(
                staged_runtime / "validation_predictions.csv",
                predictors.index.to_numpy(dtype=np.int64),
                target.to_numpy(dtype=np.int8),
                probabilities,
                bands,
            )
            _write_shap(staged_runtime / "sampled_shap_values.csv", explanations)
            _write_json(staged_runtime / "subgroup_bootstrap.json", fairness.bootstrap)
            runtime_hashes = {
                name: _sha256_file(staged_runtime / name) for name in config.outputs.runtime
            }

            final_summary = _read_json(repository / "reports/modeling/final_test_v1/summary.json")
            summary = _summary_payload(
                config=config,
                config_sha=governance_config_sha256(config_path),
                git_commit=git.commit_sha,
                governed=governed,
                manifest=manifest,
                metrics=metrics,
                probabilities=probabilities,
                bands=bands,
                fairness=fairness,
                explanations=explanations,
                runtime_hashes=runtime_hashes,
                final_summary=final_summary,
            )
            _write_json(staged_output / "summary.json", summary)
            summary_sha = _sha256_file(staged_output / "summary.json")
            documents = _render_documents(summary, summary_sha)
            for filename, content in documents.items():
                _write_text(staged_output / filename, content)
            artifact_hashes = {
                name: _sha256_file(staged_output / name)
                for name in config.outputs.committed
                if name != "evidence-manifest.json"
            }
            evidence_manifest = {
                "schema_version": "1.0.0",
                "governance_id": config.governance_id,
                "configuration_sha256": governance_config_sha256(config_path),
                "artifacts": {
                    name: {"sha256": digest, "row_level_data": False}
                    for name, digest in artifact_hashes.items()
                },
                "runtime_artifacts": {
                    name: {"sha256": digest, "committed": False}
                    for name, digest in runtime_hashes.items()
                },
                "prohibitions_verified": {
                    "fitting_performed": False,
                    "full_dataset_integrity_verification_performed": True,
                    "test_explanations_generated": False,
                    "final_test_predictions_loaded": False,
                    "test_partition_returned": False,
                    "test_partition_selected": False,
                    "test_predictions_generated": False,
                    "test_subgroup_analysis_performed": False,
                },
            }
            _write_json(staged_output / "evidence-manifest.json", evidence_manifest)
            _validate_staged_evidence(
                staged_output, config, configuration_sha256=governance_config_sha256(config_path)
            )
            try:
                _promote_directories(((staged_runtime, runtime), (staged_output, output)))
            except OSError as error:
                raise GovernanceWorkflowError(
                    f"Atomic Phase 5 publication failed: {error}"
                ) from error
        finally:
            for staged in (staged_runtime, staged_output):
                if staged.exists():
                    shutil.rmtree(staged)
        return GovernanceWorkflowResult(
            evidence_root=output,
            runtime_root=runtime,
            evidence_manifest_sha256=_sha256_file(output / "evidence-manifest.json"),
            summary_sha256=_sha256_file(output / "summary.json"),
            g3_result=config.review.g3_result,
            review_trigger_count=len(fairness.triggers),
        )
    except (GovernanceWorkflowError, ModuleNotFoundError):
        raise
    except Exception as error:
        raise GovernanceWorkflowError(f"Phase 5 governance build failed: {error}") from error


def verify_governance_evidence(
    *,
    expected_manifest_sha256: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config_path: str | Path = DEFAULT_GOVERNANCE_CONFIG_PATH,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    runtime_root: str | Path = DEFAULT_RUNTIME_ROOT,
    evidence_root: str | Path = DEFAULT_OUTPUT_ROOT,
    aggregate_only: bool = False,
) -> GovernanceWorkflowResult:
    """Verify reviewed evidence, data lineage, and bundle without rescoring any row."""

    try:
        config = load_governance_config(config_path)
        git = collect_git_evidence(Path(config_path).resolve().parent)
        if git.repository_root is None:
            raise GovernanceWorkflowError("Git repository root is unavailable.")
        repository = git.repository_root
        evidence = _safe_repository_input(repository, evidence_root, "governance evidence root")
        runtime = _safe_publication_destination(
            repository,
            runtime_root,
            allowed_subtree=Path("experiment/governance"),
            description="governance runtime root",
        )
        bundle = _safe_repository_input(repository, bundle_root, "selected bundle root")
        governed = load_governed_development_data(data_root=data_root)
        _validate_development_boundary(governed, config)
        _validate_data_lineage(governed, config)
        _validate_repository_lineage(repository, config)
        manifest, _ = load_selected_bundle(
            bundle,
            trusted=True,
            expected_manifest_sha256=config.lineage["bundle_manifest_sha256"],
            required_dependencies=tuple(config.dependencies),
        )
        if manifest.model_sha256 != config.lineage["model_sha256"]:
            raise GovernanceWorkflowError(
                "Selected model digest differs from the Phase 5 contract."
            )
        evidence_manifest = _validate_staged_evidence(
            evidence,
            config,
            configuration_sha256=governance_config_sha256(config_path),
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if not aggregate_only:
            _validate_runtime_evidence(runtime, config, evidence_manifest)
        summary = _read_json(evidence / "summary.json")
        if summary.get("g3", {}).get("result") != config.review.g3_result:
            raise GovernanceWorkflowError("Published G3 result differs from the frozen protocol.")
        trigger_count = len(summary.get("subgroup_review", {}).get("triggers", []))
        return GovernanceWorkflowResult(
            evidence_root=evidence,
            runtime_root=runtime,
            evidence_manifest_sha256=_sha256_file(evidence / "evidence-manifest.json"),
            summary_sha256=_sha256_file(evidence / "summary.json"),
            g3_result=config.review.g3_result,
            review_trigger_count=trigger_count,
        )
    except (GovernanceWorkflowError, ModuleNotFoundError):
        raise
    except Exception as error:
        raise GovernanceWorkflowError(f"Phase 5 governance verification failed: {error}") from error


def _validation_slice(
    governed: GovernedDevelopmentData, config: GovernanceConfig
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    assignments = governed.assignments
    if set(assignments["partition"].unique()) != {"development"}:
        raise GovernanceWorkflowError("Governance input contains a non-development partition.")
    mask = assignments[config.population.assignment_column].eq(config.population.validation_fold)
    account_ids = assignments.index[mask]
    predictors = governed.X.loc[account_ids].copy()
    target = governed.y.loc[account_ids].copy()
    audit = governed.audit.loc[account_ids].copy()
    counts = {str(key): int(value) for key, value in target.value_counts().sort_index().items()}
    if len(account_ids) != config.population.rows or counts != config.population.target_counts:
        raise GovernanceWorkflowError(
            f"Validation population differs from the frozen contract: rows={len(account_ids)}, "
            f"target_counts={counts}"
        )
    if tuple(predictors.columns) != config.features.predictor_columns:
        raise GovernanceWorkflowError("Validation predictors differ from the frozen feature order.")
    if not account_ids.is_unique or not account_ids.is_monotonic_increasing:
        raise GovernanceWorkflowError("Validation account IDs must be unique and sorted.")
    return predictors, target, audit


def _validate_development_boundary(
    governed: GovernedDevelopmentData, config: GovernanceConfig
) -> None:
    """Prove that the modelling boundary contains only the complete development partition."""

    expected_index = governed.assignments.index
    aligned_indexes = (
        governed.account_ids,
        governed.X.index,
        governed.y.index,
        governed.audit.index,
    )
    if (
        len(expected_index) != config.population.development_rows
        or any(not index.equals(expected_index) for index in aligned_indexes)
        or not expected_index.is_unique
        or not expected_index.is_monotonic_increasing
    ):
        raise GovernanceWorkflowError(
            "Governance input must contain the complete, aligned development population."
        )
    partitions = governed.assignments["partition"]
    if partitions.isna().any() or not partitions.eq("development").all():
        raise GovernanceWorkflowError(
            "The governance modelling boundary must not return any test-partition account."
        )


def _prove_audit_exclusion(
    predictors: pd.DataFrame, audit: pd.DataFrame, config: GovernanceConfig
) -> None:
    forbidden = set(config.features.forbidden_predictor_columns)
    if forbidden & set(predictors.columns):
        raise GovernanceWorkflowError(
            "A prohibited identifier, target, or demographic entered prediction."
        )
    if not set(config.features.audit_columns).issubset(audit.columns):
        raise GovernanceWorkflowError("Required audit fields are missing.")
    joined = predictors.join(audit.loc[:, list(config.features.audit_columns)])
    projected = joined.loc[:, list(config.features.predictor_columns)].copy()
    changed = joined.copy()
    for column in config.features.audit_columns:
        changed[column] = pd.to_numeric(changed[column], errors="raise") + 10_000
    mutated_projection = changed.loc[:, list(config.features.predictor_columns)]
    if not projected.equals(mutated_projection) or not projected.equals(predictors):
        raise GovernanceWorkflowError(
            "Changing an audit field altered the projected predictor frame."
        )


def _validate_metric_parity(metrics: PredictionMetrics, config: GovernanceConfig) -> None:
    if metrics.probability is None:
        raise GovernanceWorkflowError("Selected-bundle validation lacks probability metrics.")
    observed = {
        "average_precision": metrics.discrimination.average_precision,
        "brier_score": metrics.probability.brier_score,
        "lift_at_0_1": next(item.lift for item in metrics.capacities if item.capacity == 0.1),
    }
    mismatches = {
        name: (expected, observed[name])
        for name, expected in config.prediction.expected_validation_metrics.items()
        if abs(observed[name] - expected) > config.prediction.metric_absolute_tolerance
    }
    if mismatches:
        raise GovernanceWorkflowError(
            f"Validation metrics differ from selection evidence: {mismatches}"
        )


def _validate_data_lineage(governed: GovernedDevelopmentData, config: GovernanceConfig) -> None:
    checks = {
        "source_sha256": governed.lineage.source_sha256,
        "canonical_sha256": governed.lineage.canonical_sha256,
        "assignment_sha256": governed.lineage.assignment_sha256,
        "reviewed_split_lock_sha256": governed.lineage.reviewed_split_lock_sha256,
        "feature_contract_sha256": governed.lineage.feature_contract_sha256,
    }
    mismatches = {
        name: (config.lineage[name], observed)
        for name, observed in checks.items()
        if config.lineage.get(name) != observed
    }
    if mismatches:
        raise GovernanceWorkflowError(f"Governed data lineage mismatch: {mismatches}")


def _validate_repository_lineage(repository: Path, config: GovernanceConfig) -> None:
    files = {
        "feature_contract_sha256": repository / "configs/modeling/feature_contract_v1.json",
        "reviewed_split_lock_sha256": repository / "configs/data/split_v1.lock.json",
        "selection_summary_sha256": repository / "reports/modeling/selection_v1/summary.json",
        "selection_report_sha256": repository / "reports/modeling/selection_v1/selection-report.md",
        "final_test_summary_sha256": repository / "reports/modeling/final_test_v1/summary.json",
        "final_test_report_sha256": repository
        / "reports/modeling/final_test_v1/final-test-report.md",
        "bundle_manifest_sha256": repository / "models/selected_v1/manifest.json",
        "model_sha256": repository / "models/selected_v1/model.cbm",
    }
    mismatches = {
        name: (config.lineage.get(name), _sha256_file(path))
        for name, path in files.items()
        if config.lineage.get(name) != _sha256_file(path)
    }
    if mismatches:
        raise GovernanceWorkflowError(f"Reviewed repository lineage mismatch: {mismatches}")
    selection = _read_json(files["selection_summary_sha256"])
    if (
        selection.get("runtime_artifacts", {}).get("validation_predictions_sha256")
        != config.lineage["selection_validation_predictions_sha256"]
        or selection.get("population", {}).get("holdout_accessed") is not False
    ):
        raise GovernanceWorkflowError("Selection evidence violates the Phase 5 lineage boundary.")
    final = _read_json(files["final_test_summary_sha256"])
    if (
        final.get("g2_status") != "closed"
        or final.get("execution", {}).get("evaluation_count") != 1
    ):
        raise GovernanceWorkflowError(
            "Final-test aggregate evidence is not the consumed G2 decision."
        )


def _validate_expected_triggers(result: FairnessResult, config: GovernanceConfig) -> None:
    observed = {
        (item["axis"], item["group"], item["metric"], item["direction"]) for item in result.triggers
    }
    expected = {
        (item.axis, item.group, item.metric, item.direction)
        for item in config.review.expected_triggers
    }
    if observed != expected:
        raise GovernanceWorkflowError(
            f"Subgroup review triggers differ from the predeclared disposition: {sorted(observed)}"
        )


def _summary_payload(
    *,
    config: GovernanceConfig,
    config_sha: str,
    git_commit: str,
    governed: GovernedDevelopmentData,
    manifest: Any,
    metrics: PredictionMetrics,
    probabilities: np.ndarray,
    bands: np.ndarray,
    fairness: FairnessResult,
    explanations: ExplanationResult,
    runtime_hashes: dict[str, str],
    final_summary: dict[str, Any],
) -> dict[str, Any]:
    assert metrics.probability is not None
    return {
        "schema_version": "1.0.0",
        "governance_id": config.governance_id,
        "status": "complete",
        "execution": {
            "training_performed": False,
            "refitting_performed": False,
            "parameter_tuning_performed": False,
            "cross_validation_performed": False,
            "calibration_fitting_performed": False,
            "validation_prediction_passes": 1,
        },
        "data_boundary": {
            "full_dataset_integrity_verification_performed": True,
            "test_explanations_generated": False,
            "final_test_predictions_loaded": False,
            "test_partition_returned": False,
            "test_partition_selected": False,
            "test_predictions_generated": False,
            "test_subgroup_analysis_performed": False,
        },
        "lineage": {
            "git_commit": git_commit,
            "git_dirty": False,
            "configuration_sha256": config_sha,
            "source_sha256": governed.lineage.source_sha256,
            "canonical_sha256": governed.lineage.canonical_sha256,
            "assignment_sha256": governed.lineage.assignment_sha256,
            "feature_contract_sha256": governed.lineage.feature_contract_sha256,
            "reviewed_split_lock_sha256": governed.lineage.reviewed_split_lock_sha256,
            "selection_summary_sha256": config.lineage["selection_summary_sha256"],
            "final_test_summary_sha256": config.lineage["final_test_summary_sha256"],
            "bundle_manifest_sha256": config.lineage["bundle_manifest_sha256"],
            "model_sha256": manifest.model_sha256,
        },
        "population": {
            "partition": config.population.partition,
            "rows": config.population.rows,
            "target_counts": config.population.target_counts,
            "validation_fold": config.population.validation_fold,
            "unique_accounts": config.population.rows,
        },
        "model": {
            "model_id": manifest.selected_model_id,
            "bundle_id": manifest.bundle_id,
            "calibration": manifest.calibration,
            "predictor_count": len(PREDICTOR_COLUMNS),
            "audit_fields_passed_to_estimator": False,
            "audit_input_invariance_verified": True,
        },
        "validation_metrics": {
            "average_precision": metrics.discrimination.average_precision,
            "roc_auc": metrics.discrimination.roc_auc,
            "brier_score": metrics.probability.brier_score,
            "log_loss": metrics.probability.log_loss,
            "lift_at_0_1": next(item.lift for item in metrics.capacities if item.capacity == 0.1),
            "mean_probability": float(np.mean(probabilities)),
        },
        "policy": {
            "threshold_name": "q90",
            "threshold": config.prediction.q90,
            "review_capacity": 0.1,
            "risk_band_counts": {
                name: int(np.sum(bands == name))
                for name in ("standard", "elevated", "high", "critical")
            },
        },
        "subgroup_review": {
            "overall": fairness.overall,
            "groups": fairness.groups,
            "triggers": fairness.triggers,
            "trigger_policy": config.review.trigger_policy,
            "support_policy": config.fairness.support.model_dump(mode="json"),
            "uncertainty": {
                "performance_intervals": config.fairness.bootstrap.model_dump(mode="json"),
                "prevalence_interval": config.fairness.prevalence_interval.model_dump(mode="json"),
            },
        },
        "explanations": {
            "method": config.explanation.method,
            "space": config.explanation.space,
            "sample_rows": config.explanation.sample_rows,
            "sampling": "seed_42_stratified_by_target_and_risk_band",
            "stratum_counts": _explanation_stratum_counts(explanations),
            "max_raw_additivity_error": explanations.max_additivity_error,
            "max_sigmoid_probability_error": explanations.max_probability_error,
            "feature_summary": explanations.feature_summary,
            "reason_category_summary": explanations.category_summary,
            "language_policy": config.explanation.language_policy,
            "row_level_values_committed": False,
        },
        "final_test_aggregate_reference": {
            "g2_status": final_summary["g2_status"],
            "evaluation_count": final_summary["execution"]["evaluation_count"],
            "average_precision": final_summary["metrics"]["discrimination"]["average_precision"],
            "brier_score": final_summary["metrics"]["probability"]["brier_score"],
            "lift_at_0_1": next(
                item["lift"]
                for item in final_summary["metrics"]["capacities"]
                if item["capacity"] == 0.1
            ),
            "row_level_predictions_loaded": False,
        },
        "runtime_artifacts": {
            "hashes": runtime_hashes,
            "committed": False,
            "row_level_data_committed": False,
        },
        "g3": {
            "result": config.review.g3_result,
            "conditions": config.review.disposition,
            "fairness_certification_claimed": False,
            "regulatory_compliance_claimed": False,
        },
    }


def _render_documents(summary: dict[str, Any], summary_sha: str) -> dict[str, str]:
    triggers = summary["subgroup_review"]["triggers"]
    trigger_lines = "\n".join(
        f"- `{item['axis']}={item['group']}`: {item['metric']}={item['observed']:.6f} "
        f"triggered {item['direction']}."
        for item in triggers
    )
    metrics = summary["validation_metrics"]
    conditions = "\n".join(f"- {_condition_sentence(item)}" for item in summary["g3"]["conditions"])
    fairness_rows = "\n".join(_fairness_row(item) for item in summary["subgroup_review"]["groups"])
    explanation_rows = "\n".join(
        f"| {item['name'].replace('_', ' ')} | {item['mean_absolute_contribution']:.6f} | "
        f"{item['mean_signed_contribution']:.6f} | {item['mean_direction']} |"
        for item in summary["explanations"]["reason_category_summary"]
    )
    return {
        "governance-report.md": f"""# Phase 5 Governance Review

Status: **closed with conditions**

Summary SHA-256: `{summary_sha}`

The reviewed `selected_v1` bundle was scored once on the 4,800-row development-validation
slice. Full canonical-file verification was performed for integrity, but no test account was
selected, returned, scored, explained, or included in subgroup analysis. No model fitting,
calibration fitting, or cross-validation occurred.

Validation AP was {metrics["average_precision"]:.6f}, Brier score was
{metrics["brier_score"]:.6f}, and lift at 10% was {metrics["lift_at_0_1"]:.6f}.

Native CatBoost SHAP values were checked in raw-log-odds space for 1,000 deterministic
validation rows. Contributions are model attributions, not causal or adverse-action reasons.

| Attribution category | Mean absolute contribution | Mean signed contribution | Mean direction |
| --- | ---: | ---: | --- |
{explanation_rows}

## Conditions

{conditions}
""",
        "fairness-report.md": f"""# Validation Subgroup Review

This is a validation-only diagnostic, not a fairness certification, India compliance review,
or proof of production suitability. Demographics were excluded from the estimator and retained
only for audit.

The q90 threshold represents 10% review capacity. Supported groups required at least 100 rows,
25 defaults, and 25 non-defaults; smaller groups report counts only. Prevalence uses a two-sided
95% Wilson score interval. The remaining measures use 500 seed-42 within-group stratified
bootstrap resamples and percentile 95% intervals.

## Predeclared review triggers

{trigger_lines}

The thresholds were frozen before the planning preview exposed these results and were not changed
after the two triggers became visible.

## Group evidence

| Axis | Group | Support | Rows | Prevalence [95% CI] | Mean probability [95% CI] | Calibration gap [95% CI] | Brier [95% CI] | Selection [95% CI] | TPR [95% CI] | FPR [95% CI] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{fairness_rows}

Both triggers require human review. Education remains audit-only, and the model is restricted to
human-owned outreach prioritisation. No automatic rejection follows from a trigger.
""",
        "model-card.md": f"""# Model Card — selected_v1

## Intended use

Prioritise a human-owned retention or support outreach queue in a local portfolio demonstration.
The model must not make adverse-action, lending, pricing, or eligibility decisions.

## Model and inputs

`catboost_fixed` uses 19 operational credit-limit, repayment-status, billing, and payment fields.
Sex, education, marital status, age, account ID, and target are excluded from prediction.
Identity calibration and validation-derived risk thresholds are unchanged.

## Evidence

- Validation: AP {metrics["average_precision"]:.6f}; Brier {metrics["brier_score"]:.6f}; lift@10%
  {metrics["lift_at_0_1"]:.6f}.
- Final test (aggregate committed evidence only): AP
  {summary["final_test_aggregate_reference"]["average_precision"]:.6f}; Brier
  {summary["final_test_aggregate_reference"]["brier_score"]:.6f}; lift@10%
  {summary["final_test_aggregate_reference"]["lift_at_0_1"]:.6f}.
- Explanations: native SHAP in raw-log-odds space, labelled risk-increasing or
  risk-mitigating and never represented as causal.
- G3: closed with conditions following two predeclared education selection-rate triggers.

## Limitations and controls

The 2005 Taiwan dataset is not representative production data for India. Fairness, regulatory
compliance, monitoring, registry promotion, and rollback readiness are not claimed. Retraining is
a separately governed process. Representative production data and monitoring are required before
real-world use.
""",
        "risk-register.md": """# Phase 5 Risk Register

| Risk | Current control | Status / owner handoff |
| --- | --- | --- |
| Demographic disparity | Audit-only fields and subgroup triggers | Open: governance owner review |
| Misleading explanations | Native-SHAP additivity checks and non-causal language | Controlled with conditions |
| Geographic and temporal transportability | 2005 Taiwan limitation documented | Open: representative data required |
| Calibration or feature drift | Identity calibration documented | Open: Phase 10 monitoring |
| Unsafe automation | Human-owned outreach only; adverse action prohibited | Controlled with conditions |
| Privacy and logging | Row-level evidence remains ignored and uncommitted | Open: production privacy design |
| Artifact integrity | Manifest and model digests verified before loading | Controlled |
| Consumed-test protection | Final test permanently retired after one evaluation | Controlled |
| Missing registry and rollback | No registry promotion claim | Open: Phase 8 controls |
| Missing production monitoring | No production-readiness claim | Open: Phase 10 controls |
""",
        "g3-review.md": f"""# G3 Review Decision

Decision: **closed_with_conditions**.

The gate confirms demographic exclusion, predictor invariance to audit-field changes, validation
subgroup evidence, native-SHAP numerical checks, and explicit prohibited uses. Full-file integrity
verification parsed the canonical snapshot, but no test account was selected, returned, scored,
explained, or audited by subgroup. This does not certify fairness, regulatory compliance, or
production suitability.

Predeclared triggers:

{trigger_lines}

Required conditions are human review of these triggers, audit-only demographics, human-owned
outreach prioritisation, no adverse action, no India/compliance claim, and representative data plus
monitoring before real use.
""",
    }


def _condition_sentence(value: str) -> str:
    labels = {
        "retain_all_demographics_as_audit_only": "Retain all demographics as audit-only fields.",
        "restrict_use_to_human_owned_outreach_prioritisation": (
            "Restrict use to human-owned outreach prioritisation."
        ),
        "prohibit_adverse_action": "Prohibit adverse action.",
        "prohibit_india_or_compliance_claims": "Prohibit India-specific or compliance claims.",
        "require_representative_data_and_monitoring_before_real_use": (
            "Require representative data and monitoring before real use."
        ),
    }
    return labels.get(value, value.replace("_", " ").capitalize() + ".")


def _explanation_stratum_counts(result: ExplanationResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target, band in zip(result.sampled_target, result.sampled_risk_bands, strict=True):
        key = f"target_{int(target)}__{band}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _fairness_row(group: dict[str, Any]) -> str:
    if group["status"] == "insufficient_support":
        return (
            f"| {group['axis']} | {group['group']} | insufficient | {group['rows']} | — | — | — | "
            "— | — | — | — |"
        )
    metrics = group["metrics"]
    intervals = group["confidence_intervals"]

    def estimate(name: str) -> str:
        interval = intervals[name]
        return f"{metrics[name]:.4f} [{interval['lower']:.4f}, {interval['upper']:.4f}]"

    return (
        f"| {group['axis']} | {group['group']} | reviewed | {group['rows']} | "
        f"{estimate('target_prevalence')} | {estimate('mean_probability')} | "
        f"{estimate('calibration_in_the_large')} | {estimate('brier_score')} | "
        f"{estimate('selection_rate_at_q90')} | {estimate('true_positive_rate_at_q90')} | "
        f"{estimate('false_positive_rate_at_q90')} |"
    )


def _write_predictions(
    path: Path,
    account_ids: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    bands: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("account_id", "target", "probability", "risk_band"))
        for row in zip(account_ids, target, probabilities, bands, strict=True):
            writer.writerow((int(row[0]), int(row[1]), format(float(row[2]), ".17g"), row[3]))


def _write_shap(path: Path, result: ExplanationResult) -> None:
    headers = (
        "account_id",
        "target",
        "probability",
        "risk_band",
        "base_value",
        "raw_score",
        *(f"shap__{name}" for name in PREDICTOR_COLUMNS),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        for index in range(len(result.sampled_account_ids)):
            writer.writerow(
                (
                    int(result.sampled_account_ids[index]),
                    int(result.sampled_target[index]),
                    format(float(result.sampled_probabilities[index]), ".17g"),
                    result.sampled_risk_bands[index],
                    format(float(result.base_values[index]), ".17g"),
                    format(float(result.raw_scores[index]), ".17g"),
                    *(format(float(value), ".17g") for value in result.shap_values[index]),
                )
            )


def _validate_staged_evidence(
    root: Path,
    config: GovernanceConfig,
    *,
    configuration_sha256: str,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        observed = {path.name for path in root.iterdir() if path.is_file()}
    except OSError as error:
        raise GovernanceWorkflowError(f"Unable to inspect governance evidence: {error}") from error
    if observed != set(config.outputs.committed):
        raise GovernanceWorkflowError(
            f"Governance evidence differs from the allowlist: {sorted(observed)}"
        )
    manifest_path = root / "evidence-manifest.json"
    if expected_manifest_sha256 is not None:
        _validate_sha256(expected_manifest_sha256, "Expected evidence manifest digest")
        observed_manifest_sha256 = _sha256_file(manifest_path)
        if observed_manifest_sha256 != expected_manifest_sha256:
            raise GovernanceWorkflowError(
                "Evidence manifest digest differs from the reviewed digest: "
                f"expected={expected_manifest_sha256}, observed={observed_manifest_sha256}"
            )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("governance_id") != config.governance_id
        or manifest.get("configuration_sha256") != configuration_sha256
    ):
        raise GovernanceWorkflowError("Evidence manifest identity differs from Phase 5.")
    expected_names = set(config.outputs.committed) - {"evidence-manifest.json"}
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != expected_names:
        raise GovernanceWorkflowError("Evidence manifest does not cover every committed artifact.")
    for filename in expected_names:
        if artifacts[filename].get("sha256") != _sha256_file(root / filename):
            raise GovernanceWorkflowError(f"Governance artifact digest mismatch: {filename}")
        if artifacts[filename].get("row_level_data") is not False:
            raise GovernanceWorkflowError(
                f"Governance artifact has an unsafe data claim: {filename}"
            )
    summary = _read_json(root / "summary.json")
    execution = summary.get("execution", {})
    if any(
        execution.get(key) is not False
        for key in (
            "training_performed",
            "refitting_performed",
            "parameter_tuning_performed",
            "cross_validation_performed",
            "calibration_fitting_performed",
        )
    ):
        raise GovernanceWorkflowError("Published evidence violates a no-training/no-test claim.")
    expected_boundary = {
        "full_dataset_integrity_verification_performed": True,
        "test_explanations_generated": False,
        "final_test_predictions_loaded": False,
        "test_partition_returned": False,
        "test_partition_selected": False,
        "test_predictions_generated": False,
        "test_subgroup_analysis_performed": False,
    }
    if summary.get("data_boundary") != expected_boundary:
        raise GovernanceWorkflowError("Published evidence misstates the reviewed test boundary.")
    if manifest.get("prohibitions_verified") != {
        "fitting_performed": False,
        **expected_boundary,
    }:
        raise GovernanceWorkflowError("Evidence manifest misstates the reviewed test boundary.")
    if summary.get("population", {}).get("rows") != config.population.rows:
        raise GovernanceWorkflowError("Published evidence has the wrong validation population.")
    return manifest


def _validate_runtime_evidence(
    root: Path, config: GovernanceConfig, evidence_manifest: dict[str, Any]
) -> None:
    try:
        observed = {path.name for path in root.iterdir()}
    except OSError as error:
        raise GovernanceWorkflowError(
            f"Unable to inspect governance runtime evidence: {error}"
        ) from error
    expected = set(config.outputs.runtime)
    if observed != expected:
        raise GovernanceWorkflowError(
            f"Governance runtime evidence differs from the allowlist: {sorted(observed)}"
        )
    runtime_artifacts = evidence_manifest.get("runtime_artifacts", {})
    if set(runtime_artifacts) != expected:
        raise GovernanceWorkflowError(
            "Evidence manifest does not cover every governance runtime artifact."
        )
    for filename in expected:
        item = runtime_artifacts[filename]
        if item.get("committed") is not False:
            raise GovernanceWorkflowError(
                f"Governance runtime artifact has an unsafe publication claim: {filename}"
            )
        if item.get("sha256") != _sha256_file(root / filename):
            raise GovernanceWorkflowError(
                f"Governance runtime artifact digest mismatch: {filename}"
            )


def _validate_sha256(value: str, description: str) -> None:
    if len(value) != 64:
        raise GovernanceWorkflowError(f"{description} must contain 64 hexadecimal characters.")
    try:
        int(value, 16)
    except ValueError as error:
        raise GovernanceWorkflowError(
            f"{description} must contain 64 hexadecimal characters."
        ) from error


def _safe_repository_input(repository: Path, path: str | Path, description: str) -> Path:
    candidate = Path(path)
    resolved = (
        (repository / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    )
    try:
        resolved.relative_to(repository.resolve())
    except ValueError as error:
        raise GovernanceWorkflowError(
            f"{description} must remain inside the repository."
        ) from error
    return resolved


def _safe_publication_destination(
    repository: Path,
    path: str | Path,
    *,
    allowed_subtree: Path,
    description: str,
) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.drive:
        raise GovernanceWorkflowError(f"{description} must be repository-relative.")
    repository_root = repository.resolve()
    allowed_root = (repository_root / allowed_subtree).resolve()
    resolved = (repository_root / candidate).resolve()
    try:
        relative = resolved.relative_to(allowed_root)
    except ValueError as error:
        raise GovernanceWorkflowError(
            f"{description} must remain beneath {allowed_subtree.as_posix()}/."
        ) from error
    if not relative.parts:
        raise GovernanceWorkflowError(
            f"{description} must name a child beneath {allowed_subtree.as_posix()}/."
        )
    return resolved


def _validate_publication_separation(*paths: Path) -> None:
    resolved = tuple(path.resolve() for path in paths)
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise GovernanceWorkflowError(
                    "Governance evidence and runtime publication roots must not overlap."
                )


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))


def _write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GovernanceWorkflowError(
            f"Unable to read governed JSON {path.name}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise GovernanceWorkflowError(f"Governed JSON {path.name} must contain an object.")
    return value


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise GovernanceWorkflowError(f"Unable to hash governed file {path}: {error}") from error
