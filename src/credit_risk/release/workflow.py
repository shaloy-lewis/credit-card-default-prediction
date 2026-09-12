"""Non-computational Release A evidence assembly and verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from credit_risk.modeling.tracking import collect_git_evidence
from credit_risk.release.contracts import (
    DEFAULT_RELEASE_CONFIG_PATH,
    DEFAULT_UNCERTAINTY_SOURCE,
    RELEASE_OUTPUTS,
    ArtifactReference,
    ReleaseAConfig,
    ReleaseContractError,
    load_release_config,
    release_config_sha256,
)

DEFAULT_OUTPUT_ROOT = Path("reports/releases/release_a_v1")
MODEL_ORDER = ("logistic_l2", "random_forest", "hist_gradient_boosting", "catboost_fixed")
BASELINE_ORDER = ("fold_prevalence", "repayment_burden_rule", "logistic_l2")
CAPACITIES = (0.05, 0.1, 0.2)
UNCERTAINTY_METRICS = ("average_precision", "brier_score", "lift_at_0_1")


class ReleaseWorkflowError(RuntimeError):
    """Raised when Release A evidence cannot be assembled or authenticated."""


@dataclass(frozen=True, slots=True)
class ReleaseWorkflowResult:
    """Digest-rich outcome from release publication or verification."""

    evidence_root: Path
    summary_sha256: str
    evidence_manifest_sha256: str
    status: str


def run_release_build(
    *,
    data_root: str | Path = "data",
    config_path: str | Path = DEFAULT_RELEASE_CONFIG_PATH,
    uncertainty_source: str | Path = DEFAULT_UNCERTAINTY_SOURCE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> ReleaseWorkflowResult:
    """Publish a Release A dossier using existing aggregate evidence only."""

    try:
        config = load_release_config(config_path)
        git = collect_git_evidence(".")
        if git.repository_root is None:
            raise ReleaseWorkflowError("Git repository root is unavailable.")
        if git.dirty:
            raise ReleaseWorkflowError(
                "Official Release A publication requires a clean committed worktree."
            )
        repository = git.repository_root
        output = _safe_release_root(repository, output_root, must_exist=False)
        if output.exists():
            raise ReleaseWorkflowError(f"Release A evidence already exists: {output}")

        sources = _load_and_authenticate_sources(repository, config)
        uncertainty_path = _resolve_uncertainty_source(repository, config, uncertainty_source)
        uncertainty_bytes = _read_bytes(uncertainty_path, "validation uncertainty")
        if _sha256_bytes(uncertainty_bytes) != config.uncertainty_source.sha256:
            raise ReleaseWorkflowError(
                "Validation uncertainty digest differs from the reviewed selection evidence."
            )
        uncertainty = _read_json_bytes(uncertainty_bytes, "validation uncertainty")
        _validate_source_semantics(config, sources, uncertainty)

        data_result = _verify_offline_data(
            data_root=data_root,
            manifest_path=config.source_artifacts["data_manifest"].path,
            split_config_path=config.source_artifacts["split_config"].path,
        )
        _validate_data_verification(data_result, sources)

        summary = _assemble_summary(
            config=config,
            config_sha256=release_config_sha256(config_path),
            implementation_commit=git.commit_sha,
            sources=sources,
            uncertainty=uncertainty,
        )
        summary_bytes = _json_bytes(summary)
        report_bytes = _render_report(summary, _sha256_bytes(summary_bytes)).encode("utf-8")
        artifact_bytes = {
            "summary.json": summary_bytes,
            "release-a-report.md": report_bytes,
            "validation-uncertainty.json": uncertainty_bytes,
        }
        manifest = _assemble_manifest(
            config=config,
            config_sha256=release_config_sha256(config_path),
            implementation_commit=git.commit_sha,
            artifact_bytes=artifact_bytes,
        )
        artifact_bytes["evidence-manifest.json"] = _json_bytes(manifest)
        expected_manifest_sha256 = _sha256_bytes(artifact_bytes["evidence-manifest.json"])

        def validate_staged_evidence(staged: Path) -> None:
            _validate_release_directory(
                config=config,
                config_sha256=release_config_sha256(config_path),
                evidence_root=staged,
                expected_manifest_sha256=expected_manifest_sha256,
                sources=sources,
            )

        _publish_directory_atomically(
            output,
            artifact_bytes,
            validate=validate_staged_evidence,
        )
        return _validate_release_directory(
            config=config,
            config_sha256=release_config_sha256(config_path),
            evidence_root=output,
            expected_manifest_sha256=expected_manifest_sha256,
            sources=sources,
        )
    except ReleaseWorkflowError:
        raise
    except (
        ReleaseContractError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
    ) as error:
        raise ReleaseWorkflowError(f"Release A build failed: {error}") from error


def verify_release_evidence(
    *,
    expected_manifest_sha256: str,
    config_path: str | Path = DEFAULT_RELEASE_CONFIG_PATH,
    evidence_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> ReleaseWorkflowResult:
    """Authenticate a committed Release A dossier without runtime data or scoring."""

    try:
        _validate_sha256(expected_manifest_sha256, "Expected Release A manifest digest")
        config = load_release_config(config_path)
        repository = _repository_root(Path(config_path))
        evidence = _safe_release_root(repository, evidence_root, must_exist=True)
        sources = _load_and_authenticate_sources(repository, config)
        return _validate_release_directory(
            config=config,
            config_sha256=release_config_sha256(config_path),
            evidence_root=evidence,
            expected_manifest_sha256=expected_manifest_sha256,
            sources=sources,
        )
    except ReleaseWorkflowError:
        raise
    except (
        ReleaseContractError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
    ) as error:
        raise ReleaseWorkflowError(f"Release A verification failed: {error}") from error


def _verify_offline_data(
    *, data_root: str | Path, manifest_path: str | Path, split_config_path: str | Path
) -> Any:
    from credit_risk.data.workflow import verify_dataset

    return verify_dataset(
        data_root=data_root,
        manifest_path=manifest_path,
        split_config_path=split_config_path,
    )


def _load_and_authenticate_sources(repository: Path, config: ReleaseAConfig) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for role, reference in config.source_artifacts.items():
        path = _safe_repository_input(repository, reference)
        observed = _sha256_file(path)
        if observed != reference.sha256:
            raise ReleaseWorkflowError(
                f"Reviewed source digest mismatch for {role}: "
                f"expected={reference.sha256}, observed={observed}"
            )
        if path.suffix == ".json":
            sources[role] = _read_json(path, role)
        else:
            sources[role] = path
    return sources


def _validate_source_semantics(
    config: ReleaseAConfig, sources: dict[str, Any], uncertainty: dict[str, Any]
) -> None:
    baseline = sources["baseline_summary"]
    selection = sources["selection_summary"]
    bundle = sources["bundle_manifest"]
    authorization = sources["final_test_authorization"]
    approval = sources["final_test_approval"]
    started = sources["final_test_started_receipt"]
    completed = sources["final_test_completed_receipt"]
    final_test = sources["final_test_summary"]

    if (
        baseline.get("schema_version") != "1.0.0"
        or baseline.get("experiment", {}).get("experiment_id") != "baseline_v1"
    ):
        raise ReleaseWorkflowError("Historical baseline summary is incompatible with Release A.")
    if tuple(baseline.get("experiment", {}).get("baseline_names", ())) != BASELINE_ORDER:
        raise ReleaseWorkflowError("Historical baseline models differ from the reviewed protocol.")
    if baseline.get("data", {}).get("holdout_evaluated") is not False:
        raise ReleaseWorkflowError("Historical baseline evidence misstates holdout use.")
    if baseline.get("reproducibility", {}).get("git_dirty") is not False:
        raise ReleaseWorkflowError("Historical baseline evidence was not produced cleanly.")

    protocol = selection.get("protocol", {})
    if protocol != {
        "calibration": "identity",
        "cross_validation_iteration": False,
        "fit_count": 4,
        "parameter_tuning": False,
        "selection_config_sha256": config.source_artifacts["selection_config"].sha256,
        "winner_refitted": False,
    }:
        raise ReleaseWorkflowError("Selection protocol differs from the reviewed four-fit release.")
    if [item.get("model_id") for item in selection.get("models", [])] != list(MODEL_ORDER):
        raise ReleaseWorkflowError("Selection model order differs from the reviewed comparison.")
    if selection.get("selection", {}).get("selected_model_id") != "catboost_fixed":
        raise ReleaseWorkflowError("Selection evidence does not identify the reviewed winner.")
    if selection.get("population") != {
        "holdout_accessed": False,
        "partition": "development_only",
        "sealed_test_rows": 6000,
        "training_rows": 19200,
        "validation_rows": 4800,
    }:
        raise ReleaseWorkflowError("Selection population differs from the reviewed split.")
    if selection.get("holdout", {}).get("evaluated") is not False:
        raise ReleaseWorkflowError("Selection evidence misstates holdout evaluation.")
    if selection.get("bundle") != {
        "manifest_sha256": config.source_artifacts["bundle_manifest"].sha256,
        "model_sha256": config.source_artifacts["bundle_model"].sha256,
        "trusted_local_serialization": True,
    }:
        raise ReleaseWorkflowError("Selection evidence identifies a different model bundle.")
    if selection.get("runtime_artifacts", {}).get("bootstrap_intervals_sha256") != (
        config.uncertainty_source.sha256
    ):
        raise ReleaseWorkflowError("Selection evidence identifies different uncertainty evidence.")

    selected_metrics = selection["models"][-1]["validation_metrics"]
    _validate_uncertainty(config, uncertainty, selected_metrics)
    calibration = selection.get("selected_model", {}).get("calibration_diagnostics", {})
    if (
        calibration.get("method") != "identity"
        or calibration.get("calibrator_fitted") is not False
        or len(calibration.get("reliability_bins", [])) != 10
    ):
        raise ReleaseWorkflowError("Identity-calibration evidence is incomplete.")

    if (
        bundle.get("selected_model_id") != "catboost_fixed"
        or bundle.get("model_filename") != "model.cbm"
        or bundle.get("fit_count") != 4
        or bundle.get("winner_refitted") is not False
        or bundle.get("holdout_evaluated") is not False
        or bundle.get("calibration") != "identity"
        or bundle.get("model_sha256") != config.source_artifacts["bundle_model"].sha256
    ):
        raise ReleaseWorkflowError("Selected bundle manifest differs from Release A evidence.")

    if (
        authorization.get("status") != "frozen_not_executed"
        or authorization.get("test_contract", {}).get("maximum_evaluations") != 1
        or authorization.get("test_contract", {}).get("training") != "prohibited"
        or authorization.get("test_contract", {}).get("refitting") != "prohibited"
        or authorization.get("test_contract", {}).get("retuning") != "prohibited"
    ):
        raise ReleaseWorkflowError("Final-test authorization differs from the frozen contract.")
    if (
        approval.get("maximum_evaluations") != 1
        or approval.get("status") != "approved_once"
        or approval.get("workflow_sha256")
        != config.source_artifacts["executed_evaluator_source"].sha256
        or approval.get("frozen_authorization_sha256")
        != config.source_artifacts["final_test_authorization"].sha256
        or approval.get("manifest_sha256") != config.source_artifacts["bundle_manifest"].sha256
        or approval.get("model_sha256") != config.source_artifacts["bundle_model"].sha256
    ):
        raise ReleaseWorkflowError("Final-test approval chain is inconsistent.")
    if (
        started.get("status") != "started"
        or started.get("evaluation_count") != 1
        or started.get("approval_sha256") != config.source_artifacts["final_test_approval"].sha256
        or started.get("manifest_sha256") != config.source_artifacts["bundle_manifest"].sha256
        or started.get("model_sha256") != config.source_artifacts["bundle_model"].sha256
        or completed.get("status") != "complete"
        or completed.get("evaluation_count") != 1
        or completed.get("evaluation_id") != started.get("evaluation_id")
        or completed.get("summary_sha256") != config.source_artifacts["final_test_summary"].sha256
        or completed.get("report_sha256") != config.source_artifacts["final_test_report"].sha256
    ):
        raise ReleaseWorkflowError("Final-test durable receipts are inconsistent.")
    gates = final_test.get("gates", {})
    if (
        final_test.get("status") != "complete"
        or final_test.get("g2_status") != "closed"
        or final_test.get("population", {}).get("rows") != 6000
        or final_test.get("population", {}).get("unique_accounts") != 6000
        or final_test.get("execution")
        != {
            "cross_validation_performed": False,
            "evaluation_count": 1,
            "maximum_evaluations": 1,
            "refitting_performed": False,
            "retuning_performed": False,
            "training_performed": False,
        }
        or set(gates) != {"average_precision", "brier_score", "lift_at_0_1"}
        or not all(item.get("passed") is True for item in gates.values())
        or final_test.get("evaluation_id") != started.get("evaluation_id")
        or final_test.get("model", {}).get("manifest_sha256")
        != config.source_artifacts["bundle_manifest"].sha256
        or final_test.get("model", {}).get("model_sha256")
        != config.source_artifacts["bundle_model"].sha256
    ):
        raise ReleaseWorkflowError("Final-test evidence does not satisfy the frozen release gates.")
    _validate_capacities(selected_metrics.get("capacities"), 4800, "validation")
    _validate_capacities(final_test.get("metrics", {}).get("capacities"), 6000, "final test")

    baseline_lineage = baseline.get("lineage", {})
    selection_lineage = selection.get("reproducibility", {}).get("data_lineage", {})
    final_lineage = final_test.get("lineage", {})
    for key in ("source_sha256", "canonical_sha256", "assignment_sha256"):
        values = {baseline_lineage.get(key), selection_lineage.get(key), final_lineage.get(key)}
        if len(values) != 1 or None in values:
            raise ReleaseWorkflowError(f"Release A data lineage differs for {key}.")


def _validate_uncertainty(
    config: ReleaseAConfig,
    uncertainty: dict[str, Any],
    selected_metrics: dict[str, Any],
) -> None:
    expected_header = {
        "confidence_level": config.uncertainty_source.confidence_level,
        "method": config.uncertainty_source.method,
        "random_state": config.uncertainty_source.random_state,
        "resamples": config.uncertainty_source.resamples,
    }
    if {key: uncertainty.get(key) for key in expected_header} != expected_header:
        raise ReleaseWorkflowError("Validation uncertainty metadata differs from the reviewed run.")
    metrics = uncertainty.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(UNCERTAINTY_METRICS):
        raise ReleaseWorkflowError("Validation uncertainty metrics differ from the allowlist.")
    expected_points = {
        "average_precision": selected_metrics["discrimination"]["average_precision"],
        "brier_score": selected_metrics["probability"]["brier_score"],
        "lift_at_0_1": selected_metrics["capacities"][1]["lift"],
    }
    for name, expected_point in expected_points.items():
        interval = metrics[name]
        if set(interval) != {"lower", "point", "upper"}:
            raise ReleaseWorkflowError(f"Validation uncertainty interval is malformed: {name}")
        if interval["point"] != expected_point:
            raise ReleaseWorkflowError(f"Validation uncertainty point differs for {name}.")
        if not interval["lower"] <= interval["point"] <= interval["upper"]:
            raise ReleaseWorkflowError(f"Validation uncertainty interval is unordered: {name}")


def _validate_capacities(items: Any, population: int, description: str) -> None:
    if not isinstance(items, list) or tuple(item.get("capacity") for item in items) != CAPACITIES:
        raise ReleaseWorkflowError(f"{description.capitalize()} capacities differ from the policy.")
    for item in items:
        expected_count = int(population * item["capacity"])
        if item.get("selected_count") != expected_count:
            raise ReleaseWorkflowError(
                f"{description.capitalize()} capacity has the wrong selected count."
            )
        for metric in ("precision", "recall", "lift", "expected_true_positives"):
            if not isinstance(item.get(metric), int | float):
                raise ReleaseWorkflowError(
                    f"{description.capitalize()} capacity is missing {metric}."
                )


def _validate_data_verification(result: Any, sources: dict[str, Any]) -> None:
    lineage = sources["selection_summary"]["reproducibility"]["data_lineage"]
    expected = {
        "source_sha256": lineage["source_sha256"],
        "dataset_manifest_sha256": lineage["dataset_manifest_sha256"],
        "canonical_sha256": lineage["canonical_sha256"],
        "quality_report_sha256": lineage["quality_report_sha256"],
        "split_config_sha256": lineage["split_config_sha256"],
        "assignment_sha256": lineage["assignment_sha256"],
        "split_manifest_sha256": lineage["split_manifest_sha256"],
    }
    observed = {name: getattr(result, name, None) for name in expected}
    if observed != expected or getattr(result, "reviewed_lock_verified", False) is not True:
        raise ReleaseWorkflowError("Offline data verification differs from reviewed lineage.")


def _assemble_summary(
    *,
    config: ReleaseAConfig,
    config_sha256: str,
    implementation_commit: str,
    sources: dict[str, Any],
    uncertainty: dict[str, Any],
) -> dict[str, Any]:
    baseline = sources["baseline_summary"]
    selection = sources["selection_summary"]
    final_test = sources["final_test_summary"]
    selected = selection["models"][-1]
    baseline_models = []
    for model_id in BASELINE_ORDER:
        summaries = baseline["models"][model_id]["repeat_summaries"]
        baseline_models.append(
            {
                "model_id": model_id,
                "average_precision": summaries["average_precision"]["mean"],
                "brier_score": summaries.get("brier_score", {}).get("mean"),
                "lift_at_0_1": summaries["capacity_0_1.lift"]["mean"],
            }
        )
    return {
        "schema_version": "1.0.0",
        "release_id": config.release_id,
        "milestone": config.milestone,
        "status": "complete",
        "lineage": {
            "release_config_sha256": config_sha256,
            "implementation_git_commit": implementation_commit,
            "git_dirty": False,
            "source_artifacts": {
                role: reference.model_dump(mode="json")
                for role, reference in sorted(config.source_artifacts.items())
            },
        },
        "data": {
            "verification": "offline_complete_snapshot_integrity",
            "modeling_use": "none",
            "dataset_id": "uci_credit_default",
            "rows": 30000,
            "development_rows": 24000,
            "test_rows": 6000,
            "source_sha256": selection["reproducibility"]["data_lineage"]["source_sha256"],
            "canonical_sha256": selection["reproducibility"]["data_lineage"]["canonical_sha256"],
            "assignment_sha256": selection["reproducibility"]["data_lineage"]["assignment_sha256"],
        },
        "baselines": {
            "status": "historical_reviewed_evidence",
            "protocol": "five_fold_three_repeat_development_only",
            "models": baseline_models,
        },
        "selection": {
            "fit_count": selection["protocol"]["fit_count"],
            "parameter_tuning": selection["protocol"]["parameter_tuning"],
            "cross_validation_iteration": selection["protocol"]["cross_validation_iteration"],
            "winner_refitted": selection["protocol"]["winner_refitted"],
            "models": [
                {
                    "model_id": item["model_id"],
                    "validation_metrics": item["validation_metrics"],
                    "decision": item["decision"],
                }
                for item in selection["models"]
            ],
            "selected_model_id": selection["selection"]["selected_model_id"],
            "selection_rule": selection["selection"]["selection_rule"],
            "model_sha256": selection["bundle"]["model_sha256"],
            "manifest_sha256": selection["bundle"]["manifest_sha256"],
        },
        "calibration": selection["selected_model"]["calibration_diagnostics"],
        "uncertainty": {
            **uncertainty,
            "population": config.uncertainty_source.population,
            "final_test_intervals_computed": False,
        },
        "capacity": {
            "validation": selected["validation_metrics"]["capacities"],
            "final_test": final_test["metrics"]["capacities"],
        },
        "final_test": {
            "status": final_test["status"],
            "g2_status": final_test["g2_status"],
            "evaluation_count": final_test["execution"]["evaluation_count"],
            "maximum_evaluations": final_test["execution"]["maximum_evaluations"],
            "permanently_consumed": True,
            "metrics": final_test["metrics"],
            "gates": final_test["gates"],
        },
        "release_criteria": {criterion: "passed" for criterion in config.release_criteria},
        "evidence_boundary": {
            "full_dataset_integrity_verification_performed": True,
            "model_deserialized": False,
            "prediction_generated": False,
            "training_performed": False,
            "bootstrap_generated": False,
            "test_partition_selected": False,
            "test_predictions_loaded": False,
            "final_test_reevaluated": False,
            "stress_evidence": config.governance.stress_evidence,
        },
        "claims": {
            "causal_impact": False,
            "india_specific_validity": False,
            "regulatory_compliance": False,
            "fairness_certification": False,
            "production_suitability": False,
        },
    }


def _assemble_manifest(
    *,
    config: ReleaseAConfig,
    config_sha256: str,
    implementation_commit: str,
    artifact_bytes: dict[str, bytes],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "release_id": config.release_id,
        "configuration_sha256": config_sha256,
        "implementation_git_commit": implementation_commit,
        "artifacts": {
            name: {"row_level_data": False, "sha256": _sha256_bytes(content)}
            for name, content in sorted(artifact_bytes.items())
        },
        "source_artifacts": {
            role: reference.model_dump(mode="json")
            for role, reference in sorted(config.source_artifacts.items())
        },
        "boundaries_verified": {
            "model_deserialized": False,
            "prediction_generated": False,
            "training_performed": False,
            "bootstrap_generated": False,
            "test_partition_selected": False,
            "test_predictions_loaded": False,
            "final_test_reevaluated": False,
        },
    }


def _render_report(summary: dict[str, Any], summary_sha256: str) -> str:
    baseline_rows = "\n".join(
        "| {model_id} | {average_precision:.6f} | {brier} | {lift_at_0_1:.6f} |".format(
            model_id=item["model_id"],
            average_precision=item["average_precision"],
            brier=("n/a" if item["brier_score"] is None else f"{item['brier_score']:.6f}"),
            lift_at_0_1=item["lift_at_0_1"],
        )
        for item in summary["baselines"]["models"]
    )
    selection_rows = "\n".join(
        f"| {item['model_id']} | "
        f"{item['validation_metrics']['discrimination']['average_precision']:.6f} | "
        f"{item['validation_metrics']['probability']['brier_score']:.6f} | "
        f"{item['validation_metrics']['capacities'][1]['lift']:.6f} | "
        f"{str(item['decision']['eligible']).lower()} |"
        for item in summary["selection"]["models"]
    )
    uncertainty_rows = "\n".join(
        f"| {name} | {item['point']:.6f} | [{item['lower']:.6f}, {item['upper']:.6f}] |"
        for name, item in summary["uncertainty"]["metrics"].items()
    )
    reliability_rows = "\n".join(
        f"| {item['bin']} | {item['rows']} | {item['mean_probability']:.6f} | "
        f"{item['observed_event_rate']:.6f} |"
        for item in summary["calibration"]["reliability_bins"]
    )
    final_metrics = summary["final_test"]["metrics"]
    return f"""# Release A — defensible model

Status: **complete**

Summary SHA-256: `{summary_sha256}`

Release A consolidates the reviewed model evidence without fitting, scoring, or
loading row-level predictions. The selected artifact is the exact four-fit
validation winner and the one authorized final-test evaluation is permanently
consumed.

## Reproducible data and baselines

The checksum-pinned UCI snapshot contains 30,000 rows with a sealed 24,000-row
development partition and 6,000-row test partition. The source, canonical data,
and assignment hashes are bound in the machine-readable summary.

| Baseline | Average precision | Brier | Lift@10% |
| --- | ---: | ---: | ---: |
{baseline_rows}

## Fixed model comparison

| Model | Validation AP | Validation Brier | Validation lift@10% | Eligible |
| --- | ---: | ---: | ---: | --- |
{selection_rows}

Selected model: **{summary["selection"]["selected_model_id"]}**. Exactly four
fits were performed, with no tuning, cross-validation iteration, or winner refit.

## Identity calibration

No calibrator was fitted. Identity calibration retained mean probability
`{summary["calibration"]["mean_probability"]:.6f}` against observed prevalence
`{summary["calibration"]["observed_prevalence"]:.6f}`. Ten-bin ECE is
`{summary["calibration"]["expected_calibration_error_10_equal_count_bins"]:.6f}`.

| Bin | Rows | Mean probability | Observed event rate |
| ---: | ---: | ---: | ---: |
{reliability_rows}

## Validation-only uncertainty

Intervals use 500 seed-42 stratified prediction-only percentile-bootstrap
resamples. They are validation intervals, not final-test intervals.

| Metric | Point | 95% interval |
| --- | ---: | ---: |
{uncertainty_rows}

## Capacity-aware evaluation

### Validation

{_capacity_table(summary["capacity"]["validation"])}

### One-time final test

{_capacity_table(summary["capacity"]["final_test"])}

Final-test average precision is
`{final_metrics["discrimination"]["average_precision"]:.6f}`, Brier score is
`{final_metrics["probability"]["brier_score"]:.6f}`, and lift at 10% is
`{summary["capacity"]["final_test"][1]["lift"]:.6f}`. All three frozen gates
passed, closing G2. The evaluation cannot be rerun.

## Evidence boundary and limitations

This dossier performed complete-snapshot integrity verification only. It did not
deserialize a model, generate predictions, regenerate bootstrap evidence, select
test accounts, or load final-test predictions. Robustness and population-shift
stress evidence is explicitly deferred to G4/Release B.

Results describe a single 2005 Taiwan dataset. They do not establish causal
impact, India-specific validity, regulatory compliance, fairness certification,
production suitability, or realised business value.
"""


def _capacity_table(items: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        f"| {item['capacity']:.0%} | {item['selected_count']} | "
        f"{item['precision']:.6f} | {item['recall']:.6f} | {item['lift']:.6f} | "
        f"{item['expected_true_positives']:.6f} |"
        for item in items
    )
    return (
        "| Capacity | Selected | Precision | Recall | Lift | Expected true positives |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{rows}"
    )


def _validate_release_directory(
    *,
    config: ReleaseAConfig,
    config_sha256: str,
    evidence_root: Path,
    expected_manifest_sha256: str,
    sources: dict[str, Any],
) -> ReleaseWorkflowResult:
    observed_files = {path.name for path in evidence_root.iterdir()}
    if observed_files != set(RELEASE_OUTPUTS):
        raise ReleaseWorkflowError(
            f"Release A evidence differs from the allowlist: {sorted(observed_files)}"
        )
    manifest_path = evidence_root / "evidence-manifest.json"
    observed_manifest_sha256 = _sha256_file(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise ReleaseWorkflowError(
            "Release A manifest digest differs from the reviewed digest: "
            f"expected={expected_manifest_sha256}, observed={observed_manifest_sha256}"
        )
    manifest = _read_json(manifest_path, "Release A evidence manifest")
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("release_id") != config.release_id
        or manifest.get("configuration_sha256") != config_sha256
    ):
        raise ReleaseWorkflowError("Release A evidence manifest is incompatible.")
    expected_sources = {
        role: reference.model_dump(mode="json")
        for role, reference in sorted(config.source_artifacts.items())
    }
    if manifest.get("source_artifacts") != expected_sources:
        raise ReleaseWorkflowError("Release A manifest source bindings differ from the protocol.")
    expected_artifacts = set(RELEASE_OUTPUTS) - {"evidence-manifest.json"}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ReleaseWorkflowError("Release A manifest artifact allowlist is incomplete.")
    for filename in expected_artifacts:
        item = artifacts[filename]
        if item.get("row_level_data") is not False:
            raise ReleaseWorkflowError(f"Release artifact has an unsafe row-data claim: {filename}")
        if item.get("sha256") != _sha256_file(evidence_root / filename):
            raise ReleaseWorkflowError(f"Release artifact digest mismatch: {filename}")
    expected_boundaries = {
        "model_deserialized": False,
        "prediction_generated": False,
        "training_performed": False,
        "bootstrap_generated": False,
        "test_partition_selected": False,
        "test_predictions_loaded": False,
        "final_test_reevaluated": False,
    }
    if manifest.get("boundaries_verified") != expected_boundaries:
        raise ReleaseWorkflowError("Release A manifest misstates the evidence boundary.")

    uncertainty_bytes = _read_bytes(
        evidence_root / "validation-uncertainty.json", "published uncertainty"
    )
    if _sha256_bytes(uncertainty_bytes) != config.uncertainty_source.sha256:
        raise ReleaseWorkflowError("Published uncertainty is not byte-identical to its source.")
    uncertainty = _read_json_bytes(uncertainty_bytes, "published uncertainty")
    _validate_source_semantics(config, sources, uncertainty)
    summary = _read_json(evidence_root / "summary.json", "Release A summary")
    _validate_published_summary(config, summary, manifest, uncertainty, sources)
    report = _read_text(evidence_root / "release-a-report.md", "Release A report")
    for required in (
        "Status: **complete**",
        "Validation-only uncertainty",
        "Identity calibration",
        "Capacity-aware evaluation",
        "The evaluation cannot be rerun.",
        "deferred to G4/Release B",
    ):
        if required not in report:
            raise ReleaseWorkflowError(f"Release A report is missing reviewed text: {required}")
    return ReleaseWorkflowResult(
        evidence_root=evidence_root,
        summary_sha256=_sha256_file(evidence_root / "summary.json"),
        evidence_manifest_sha256=observed_manifest_sha256,
        status="complete",
    )


def _validate_published_summary(
    config: ReleaseAConfig,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    uncertainty: dict[str, Any],
    sources: dict[str, Any],
) -> None:
    if (
        summary.get("schema_version") != "1.0.0"
        or summary.get("release_id") != config.release_id
        or summary.get("milestone") != config.milestone
        or summary.get("status") != "complete"
    ):
        raise ReleaseWorkflowError("Release A summary identity or status is invalid.")
    if summary.get("lineage", {}).get("release_config_sha256") != manifest.get(
        "configuration_sha256"
    ):
        raise ReleaseWorkflowError("Release A summary configuration lineage is inconsistent.")
    if summary.get("lineage", {}).get("implementation_git_commit") != manifest.get(
        "implementation_git_commit"
    ):
        raise ReleaseWorkflowError("Release A implementation lineage is inconsistent.")
    if summary.get("lineage", {}).get("git_dirty") is not False:
        raise ReleaseWorkflowError("Release A summary does not record a clean implementation.")
    if summary.get("uncertainty", {}).get("metrics") != uncertainty.get("metrics"):
        raise ReleaseWorkflowError("Release A summary uncertainty differs from the published file.")
    if summary.get("release_criteria") != {
        criterion: "passed" for criterion in config.release_criteria
    }:
        raise ReleaseWorkflowError("Release A summary does not pass every frozen criterion.")
    if summary.get("selection", {}).get("selected_model_id") != "catboost_fixed":
        raise ReleaseWorkflowError("Release A summary identifies the wrong selected model.")
    if summary.get("final_test", {}).get("g2_status") != "closed":
        raise ReleaseWorkflowError("Release A summary does not retain closed G2 status.")
    if summary.get("evidence_boundary", {}).get("stress_evidence") != ("deferred_to_g4_release_b"):
        raise ReleaseWorkflowError("Release A summary misstates the stress-evidence boundary.")
    if any(summary.get("claims", {}).values()):
        raise ReleaseWorkflowError("Release A summary contains an unsupported positive claim.")
    if (
        summary.get("data", {}).get("assignment_sha256")
        != sources["selection_summary"]["reproducibility"]["data_lineage"]["assignment_sha256"]
    ):
        raise ReleaseWorkflowError("Release A summary data lineage is inconsistent.")


def _resolve_uncertainty_source(
    repository: Path, config: ReleaseAConfig, supplied: str | Path
) -> Path:
    expected = _safe_repository_path(repository, config.uncertainty_source.path)
    observed = _safe_repository_path(repository, supplied)
    if observed != expected:
        raise ReleaseWorkflowError(
            "Uncertainty source must be the exact reviewed selection-runtime path."
        )
    return observed


def _safe_repository_input(repository: Path, reference: ArtifactReference) -> Path:
    path = _safe_repository_path(repository, reference.path)
    if not path.is_file():
        raise ReleaseWorkflowError(f"Reviewed source artifact is missing: {reference.path}")
    return path


def _safe_repository_path(repository: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.drive:
        raise ReleaseWorkflowError("Release paths must be repository-relative.")
    resolved = (repository.resolve() / candidate).resolve()
    try:
        resolved.relative_to(repository.resolve())
    except ValueError as error:
        raise ReleaseWorkflowError("Release path escapes the repository.") from error
    return resolved


def _safe_release_root(repository: Path, path: str | Path, *, must_exist: bool) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.drive:
        raise ReleaseWorkflowError("Release output root must be repository-relative.")
    repository_root = repository.resolve()
    allowed_root = (repository_root / "reports" / "releases").resolve()
    resolved = (repository_root / candidate).resolve()
    try:
        relative = resolved.relative_to(allowed_root)
    except ValueError as error:
        raise ReleaseWorkflowError(
            "Release evidence root must remain beneath reports/releases/."
        ) from error
    if not relative.parts:
        raise ReleaseWorkflowError("Release evidence root must name a release child directory.")
    if must_exist and not resolved.is_dir():
        raise ReleaseWorkflowError(f"Release A evidence directory is missing: {resolved}")
    return resolved


def _repository_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ReleaseWorkflowError(f"Unable to locate repository root from {config_path}")


def _publish_directory_atomically(
    output: Path,
    artifacts: dict[str, bytes],
    *,
    validate: Callable[[Path], None] | None = None,
) -> None:
    if set(artifacts) != set(RELEASE_OUTPUTS):
        raise ReleaseWorkflowError("Release publication payload differs from the allowlist.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(f".{output.name}.stage-{uuid4().hex}")
    try:
        staged.mkdir()
        for name, content in artifacts.items():
            (staged / name).write_bytes(content)
        if validate is not None:
            validate(staged)
        os.replace(staged, output)
    except OSError as error:
        raise ReleaseWorkflowError(
            f"Unable to publish Release A evidence atomically: {error}"
        ) from error
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    return _read_json_bytes(_read_bytes(path, description), description)


def _read_json_bytes(content: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseWorkflowError(f"Unable to parse {description}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseWorkflowError(f"{description.capitalize()} must contain a JSON object.")
    return value


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseWorkflowError(f"Unable to read {description}: {error}") from error


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ReleaseWorkflowError(f"Unable to read {description}: {error}") from error


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReleaseWorkflowError(f"Unable to hash governed file {path}: {error}") from error
    return digest.hexdigest()


def _validate_sha256(value: str, description: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ReleaseWorkflowError(
            f"{description} must contain 64 lowercase hexadecimal characters."
        )
