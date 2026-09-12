from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import credit_risk.governance.workflow as workflow
from credit_risk.governance.explanations import ExplanationResult
from credit_risk.governance.fairness import FairnessResult
from credit_risk.governance.workflow import (
    GovernanceWorkflowError,
    run_governance_build,
    verify_governance_evidence,
)
from credit_risk.modeling.contracts import AUDIT_COLUMNS, PREDICTOR_COLUMNS
from credit_risk.modeling.metrics import evaluate_predictions
from credit_risk.modeling.tracking import GitEvidence


class _FakeModel:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities
        self.calls = 0

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        assert tuple(frame.columns) == PREDICTOR_COLUMNS
        return self.probabilities.copy()


def _governed(config: Any) -> SimpleNamespace:
    ids = pd.Index(range(1, 24001), name="account_id")
    predictors = pd.DataFrame(
        {column: np.zeros(len(ids), dtype=np.int64) for column in PREDICTOR_COLUMNS},
        index=ids,
    )
    target_values = np.concatenate(
        (
            np.zeros(3738, dtype=np.int8),
            np.ones(1062, dtype=np.int8),
            np.zeros(len(ids) - 4800, dtype=np.int8),
        )
    )
    target = pd.Series(target_values, index=ids, name="default_next_month")
    audit = pd.DataFrame(
        {
            "account_id": ids,
            "default_next_month": target_values,
            "sex_code": np.where(np.arange(len(ids)) % 2, 1, 2),
            "education_code": 2,
            "marital_status_code": 1,
            "age_years": 35,
        },
        index=ids,
    )
    folds = np.ones(len(ids), dtype=np.int8)
    folds[:4800] = 0
    assignments = pd.DataFrame({"partition": "development", "cv_fold_r0": folds}, index=ids)
    lineage = SimpleNamespace(
        source_sha256=config.lineage["source_sha256"],
        canonical_sha256=config.lineage["canonical_sha256"],
        assignment_sha256=config.lineage["assignment_sha256"],
        reviewed_split_lock_sha256=config.lineage["reviewed_split_lock_sha256"],
        feature_contract_sha256=config.lineage["feature_contract_sha256"],
    )
    return SimpleNamespace(
        account_ids=ids,
        X=predictors,
        y=target,
        audit=audit,
        assignments=assignments,
        lineage=lineage,
    )


def _fairness(config: Any) -> FairnessResult:
    triggers = tuple(
        {
            "axis": item.axis,
            "group": item.group,
            "metric": item.metric,
            "direction": item.direction,
            "observed": 0.7 if item.direction == "below_lower_bound" else 1.3,
            "threshold": 0.8 if item.direction == "below_lower_bound" else 1.25,
            "disposition": "documented_human_review_required",
        }
        for item in config.review.expected_triggers
    )
    return FairnessResult(
        overall={"rows": 4800, "selection_rate_at_q90": 0.1},
        groups=(),
        triggers=triggers,
        bootstrap={"resamples": 500, "groups": {}},
    )


def _explanations(config: Any) -> ExplanationResult:
    rows = config.explanation.sample_rows
    return ExplanationResult(
        sampled_account_ids=np.arange(1, rows + 1),
        sampled_target=np.zeros(rows, dtype=np.int8),
        sampled_probabilities=np.full(rows, 0.1),
        sampled_risk_bands=np.full(rows, "standard"),
        base_values=np.zeros(rows),
        raw_scores=np.zeros(rows),
        shap_values=np.zeros((rows, 19)),
        feature_summary=(),
        category_summary=(),
        max_additivity_error=0.0,
        max_probability_error=0.0,
    )


def _patch_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = workflow.load_governance_config()
    governed = _governed(config)
    validation_mask = governed.assignments["cv_fold_r0"].eq(0)
    target = governed.y.loc[validation_mask].to_numpy()
    probabilities = np.where(target == 1, 0.7, 0.1).astype(float)
    fake_model = _FakeModel(probabilities)
    manifest = SimpleNamespace(
        model_sha256=config.lineage["model_sha256"],
        selected_model_id="catboost_fixed",
        bundle_id="selected_v1",
        calibration="identity",
        risk_band_thresholds={"q80": 0.3, "q90": config.prediction.q90, "q95": 0.8},
    )
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _path: GitEvidence("a" * 40, False, "b" * 64, tmp_path),
    )
    monkeypatch.setattr(workflow, "load_governed_development_data", lambda **_kwargs: governed)
    monkeypatch.setattr(workflow, "_validate_repository_lineage", lambda *_args: None)
    monkeypatch.setattr(
        workflow, "load_selected_bundle", lambda *_args, **_kwargs: (manifest, fake_model)
    )
    monkeypatch.setattr(workflow, "_validate_metric_parity", lambda *_args: None)
    monkeypatch.setattr(workflow, "analyse_subgroups", lambda *_args, **_kwargs: _fairness(config))
    monkeypatch.setattr(
        workflow,
        "explain_validation_sample",
        lambda *_args, **_kwargs: _explanations(config),
    )
    monkeypatch.setattr(
        workflow,
        "_read_json",
        lambda path: (
            {
                "g2_status": "closed",
                "execution": {"evaluation_count": 1},
                "metrics": {
                    "discrimination": {"average_precision": 0.54},
                    "probability": {"brier_score": 0.13},
                    "capacities": [{"capacity": 0.1, "lift": 3.0}],
                },
            }
            if path.name == "summary.json" and "final_test_v1" in path.as_posix()
            else json.loads(path.read_bytes())
        ),
    )
    return config, fake_model


def test_build_scores_once_publishes_allowlisted_evidence_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, model = _patch_success(tmp_path, monkeypatch)
    output = tmp_path / "reports/governance/phase5_v1"
    runtime = tmp_path / "experiment/governance/phase5_v1"

    result = run_governance_build(
        data_root=tmp_path / "data",
        bundle_root=tmp_path / "models/selected_v1",
        runtime_root=Path("experiment/governance/phase5_v1"),
        output_root=Path("reports/governance/phase5_v1"),
    )

    assert model.calls == 1
    assert result.g3_result == "closed_with_conditions"
    assert result.review_trigger_count == 2
    assert {path.name for path in output.iterdir()} == set(config.outputs.committed)
    assert {path.name for path in runtime.iterdir()} == set(config.outputs.runtime)
    assert not list(output.parent.glob(".phase5_v1.stage-*"))
    assert not list(runtime.parent.glob(".phase5_v1.stage-*"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["execution"]["training_performed"] is False
    assert summary["data_boundary"] == {
        "full_dataset_integrity_verification_performed": True,
        "test_explanations_generated": False,
        "final_test_predictions_loaded": False,
        "test_partition_returned": False,
        "test_partition_selected": False,
        "test_predictions_generated": False,
        "test_subgroup_analysis_performed": False,
    }
    assert summary["explanations"]["row_level_values_committed"] is False
    assert sum(summary["explanations"]["stratum_counts"].values()) == 1000
    assert summary["g3"]["result"] == "closed_with_conditions"
    fairness_report = (output / "fairness-report.md").read_text(encoding="utf-8")
    assert "## Group evidence" in fairness_report
    assert "\n| Axis | Group | Support | Rows | Prevalence [95% CI] |" in fairness_report
    assert "Prohibit India-specific" in (output / "governance-report.md").read_text(
        encoding="utf-8"
    )

    verified = verify_governance_evidence(
        expected_manifest_sha256=result.evidence_manifest_sha256,
        data_root=tmp_path / "data",
        bundle_root=tmp_path / "models/selected_v1",
        runtime_root=Path("experiment/governance/phase5_v1"),
        evidence_root=output,
    )
    assert verified.summary_sha256 == result.summary_sha256

    runtime_file = runtime / "validation_predictions.csv"
    original_runtime = runtime_file.read_bytes()
    runtime_file.write_bytes(b"corrupt\n")
    with pytest.raises(GovernanceWorkflowError, match="runtime artifact digest mismatch"):
        verify_governance_evidence(
            expected_manifest_sha256=result.evidence_manifest_sha256,
            data_root=tmp_path / "data",
            bundle_root=tmp_path / "models/selected_v1",
            runtime_root=Path("experiment/governance/phase5_v1"),
            evidence_root=output,
        )
    aggregate = verify_governance_evidence(
        expected_manifest_sha256=result.evidence_manifest_sha256,
        data_root=tmp_path / "data",
        bundle_root=tmp_path / "models/selected_v1",
        runtime_root=Path("experiment/governance/phase5_v1"),
        evidence_root=output,
        aggregate_only=True,
    )
    assert aggregate.summary_sha256 == result.summary_sha256
    runtime_file.write_bytes(original_runtime)

    runtime_file.unlink()
    with pytest.raises(
        GovernanceWorkflowError, match="runtime evidence differs from the allowlist"
    ):
        verify_governance_evidence(
            expected_manifest_sha256=result.evidence_manifest_sha256,
            data_root=tmp_path / "data",
            bundle_root=tmp_path / "models/selected_v1",
            runtime_root=Path("experiment/governance/phase5_v1"),
            evidence_root=output,
        )
    runtime_file.write_bytes(original_runtime)

    (output / "unexpected.txt").write_text("not allowlisted", encoding="utf-8")
    with pytest.raises(GovernanceWorkflowError, match="differs from the allowlist"):
        verify_governance_evidence(
            expected_manifest_sha256=result.evidence_manifest_sha256,
            data_root=tmp_path / "data",
            bundle_root=tmp_path / "models/selected_v1",
            runtime_root=Path("experiment/governance/phase5_v1"),
            evidence_root=output,
        )
    (output / "unexpected.txt").unlink()
    (output / "model-card.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(GovernanceWorkflowError, match="digest mismatch"):
        verify_governance_evidence(
            expected_manifest_sha256=result.evidence_manifest_sha256,
            data_root=tmp_path / "data",
            bundle_root=tmp_path / "models/selected_v1",
            runtime_root=Path("experiment/governance/phase5_v1"),
            evidence_root=output,
        )
    altered_manifest = json.loads((output / "evidence-manifest.json").read_bytes())
    altered_manifest["artifacts"]["model-card.md"]["sha256"] = workflow._sha256_file(
        output / "model-card.md"
    )
    workflow._write_json(output / "evidence-manifest.json", altered_manifest)
    with pytest.raises(GovernanceWorkflowError, match="manifest digest differs"):
        verify_governance_evidence(
            expected_manifest_sha256=result.evidence_manifest_sha256,
            data_root=tmp_path / "data",
            bundle_root=tmp_path / "models/selected_v1",
            runtime_root=Path("experiment/governance/phase5_v1"),
            evidence_root=output,
        )


def test_dirty_and_unsafe_paths_fail_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _path: GitEvidence("a" * 40, True, "b" * 64, tmp_path),
    )
    monkeypatch.setattr(
        workflow,
        "load_governed_development_data",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("data was accessed")),
    )
    with pytest.raises(GovernanceWorkflowError, match="clean committed"):
        run_governance_build()

    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _path: GitEvidence("a" * 40, False, "b" * 64, tmp_path),
    )
    with pytest.raises(GovernanceWorkflowError, match="repository-relative"):
        run_governance_build(output_root=tmp_path.parent / "outside")
    with pytest.raises(GovernanceWorkflowError, match="repository-relative"):
        run_governance_build(runtime_root=tmp_path / "experiment/governance/absolute")


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("output_root", Path("src/governance-evidence"), "beneath reports/governance"),
        ("output_root", Path(".git/governance-evidence"), "beneath reports/governance"),
        ("output_root", Path("models/selected_v1/evidence"), "beneath reports/governance"),
        ("runtime_root", Path("data/governance-runtime"), "beneath experiment/governance"),
        (
            "output_root",
            Path("reports/governance/../../configs/evidence"),
            "beneath reports/governance",
        ),
    ),
)
def test_publication_paths_reject_protected_or_traversing_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keyword: str,
    value: Path,
    message: str,
) -> None:
    _patch_success(tmp_path, monkeypatch)
    arguments = {
        "data_root": tmp_path / "data",
        "bundle_root": tmp_path / "models/selected_v1",
        "runtime_root": Path("experiment/governance/test-run"),
        "output_root": Path("reports/governance/test-run"),
    }
    arguments[keyword] = value
    with pytest.raises(GovernanceWorkflowError, match=message):
        run_governance_build(**arguments)


def test_publication_paths_reject_symlink_escape_and_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_success(tmp_path, monkeypatch)
    allowed = tmp_path / "reports/governance"
    allowed.mkdir(parents=True)
    escape = allowed / "escape"
    try:
        escape.symlink_to(tmp_path.parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")
    with pytest.raises(GovernanceWorkflowError, match="beneath reports/governance"):
        run_governance_build(
            data_root=tmp_path / "data",
            bundle_root=tmp_path / "models/selected_v1",
            runtime_root=Path("experiment/governance/test-run"),
            output_root=Path("reports/governance/escape/evidence"),
        )

    with pytest.raises(GovernanceWorkflowError, match="must not overlap"):
        workflow._validate_publication_separation(
            tmp_path / "reports/governance/a",
            tmp_path / "reports/governance/a/child",
        )


def test_atomic_failure_leaves_no_partial_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_success(tmp_path, monkeypatch)
    monkeypatch.setattr(
        workflow,
        "_promote_directories",
        lambda _pairs: (_ for _ in ()).throw(OSError("promotion stopped")),
    )
    output = tmp_path / "reports/governance/phase5_v1"
    runtime = tmp_path / "experiment/governance/phase5_v1"

    with pytest.raises(GovernanceWorkflowError, match="Atomic Phase 5 publication failed"):
        run_governance_build(
            data_root=tmp_path / "data",
            bundle_root=tmp_path / "models/selected_v1",
            runtime_root=Path("experiment/governance/phase5_v1"),
            output_root=Path("reports/governance/phase5_v1"),
        )

    assert not output.exists()
    assert not runtime.exists()


def test_existing_destination_and_unexpected_failure_are_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_success(tmp_path, monkeypatch)
    output = tmp_path / "reports/governance/phase5_v1"
    output.mkdir(parents=True)
    with pytest.raises(GovernanceWorkflowError, match="Refusing to overwrite"):
        run_governance_build(
            data_root=tmp_path / "data",
            bundle_root=tmp_path / "models/selected_v1",
            runtime_root=Path("experiment/governance/phase5_v1"),
            output_root=Path("reports/governance/phase5_v1"),
        )

    output.rmdir()
    monkeypatch.setattr(
        workflow,
        "load_governed_development_data",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("unexpected failure")),
    )
    with pytest.raises(GovernanceWorkflowError, match="build failed: unexpected failure"):
        run_governance_build(
            data_root=tmp_path / "data",
            bundle_root=tmp_path / "models/selected_v1",
            runtime_root=Path("experiment/governance/phase5_v1"),
            output_root=Path("reports/governance/phase5_v1"),
        )


def test_no_training_or_test_loader_is_reachable_from_workflow() -> None:
    source = inspect.getsource(workflow)

    assert ".fit(" not in source
    assert "load_governed_test_data" not in source
    assert "load_final_test" not in source


def test_metric_parity_and_trigger_mismatch_fail() -> None:
    config = workflow.load_governance_config()
    labels = np.asarray([0, 0, 1, 1])
    metrics = evaluate_predictions(labels, [0.1, 0.2, 0.8, 0.9], probabilities=[0.1, 0.2, 0.8, 0.9])
    with pytest.raises(GovernanceWorkflowError, match="differ from selection"):
        workflow._validate_metric_parity(metrics, config)
    with pytest.raises(GovernanceWorkflowError, match="lacks probability"):
        workflow._validate_metric_parity(replace(metrics, probability=None), config)

    fairness = FairnessResult(overall={}, groups=(), triggers=(), bootstrap={})
    with pytest.raises(GovernanceWorkflowError, match="triggers differ"):
        workflow._validate_expected_triggers(fairness, config)


def test_audit_projection_rejects_demographic_leakage() -> None:
    config = workflow.load_governance_config()
    predictors = pd.DataFrame({column: [0] for column in PREDICTOR_COLUMNS})
    audit = pd.DataFrame({column: [1] for column in AUDIT_COLUMNS})
    workflow._prove_audit_exclusion(predictors, audit, config)

    leaked = predictors.copy()
    leaked["sex_code"] = 1
    with pytest.raises(GovernanceWorkflowError, match="prohibited"):
        workflow._prove_audit_exclusion(leaked, audit, config)
    with pytest.raises(GovernanceWorkflowError, match="audit fields are missing"):
        workflow._prove_audit_exclusion(predictors, audit.drop(columns=["age_years"]), config)


def test_validation_slice_rejects_wrong_partition_population_columns_and_ids() -> None:
    config = workflow.load_governance_config()
    governed = _governed(config)
    wrong_partition = governed.assignments.copy()
    wrong_partition.iloc[0, wrong_partition.columns.get_loc("partition")] = "test"
    governed.assignments = wrong_partition
    with pytest.raises(GovernanceWorkflowError, match="non-development"):
        workflow._validation_slice(governed, config)

    governed = _governed(config)
    governed.assignments = governed.assignments.drop(index=1)
    with pytest.raises(GovernanceWorkflowError, match="population differs"):
        workflow._validation_slice(governed, config)

    governed = _governed(config)
    governed.X = governed.X.loc[:, list(reversed(PREDICTOR_COLUMNS))]
    with pytest.raises(GovernanceWorkflowError, match="feature order"):
        workflow._validation_slice(governed, config)

    governed = _governed(config)
    governed.assignments = governed.assignments.sort_index(ascending=False)
    with pytest.raises(GovernanceWorkflowError, match="unique and sorted"):
        workflow._validation_slice(governed, config)


def test_development_boundary_rejects_test_rows_and_misaligned_population() -> None:
    config = workflow.load_governance_config()
    governed = _governed(config)
    workflow._validate_development_boundary(governed, config)

    governed.assignments.iloc[0, governed.assignments.columns.get_loc("partition")] = "test"
    with pytest.raises(GovernanceWorkflowError, match="must not return any test-partition"):
        workflow._validate_development_boundary(governed, config)

    governed = _governed(config)
    governed.X = governed.X.iloc[:-1]
    with pytest.raises(GovernanceWorkflowError, match="complete, aligned development"):
        workflow._validate_development_boundary(governed, config)


def test_lineage_and_repository_evidence_are_strict() -> None:
    config = workflow.load_governance_config()
    governed = _governed(config)
    workflow._validate_data_lineage(governed, config)
    governed.lineage.source_sha256 = "0" * 64
    with pytest.raises(GovernanceWorkflowError, match="data lineage mismatch"):
        workflow._validate_data_lineage(governed, config)

    workflow._validate_repository_lineage(Path.cwd(), config)
    altered = config.model_copy(update={"lineage": {**config.lineage, "model_sha256": "0" * 64}})
    with pytest.raises(GovernanceWorkflowError, match="repository lineage mismatch"):
        workflow._validate_repository_lineage(Path.cwd(), altered)


def test_json_helpers_reject_invalid_or_missing_files(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    with pytest.raises(GovernanceWorkflowError, match="Unable to read governed JSON"):
        workflow._read_json(invalid)
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(GovernanceWorkflowError, match="must contain an object"):
        workflow._read_json(array)
    with pytest.raises(GovernanceWorkflowError, match="Unable to hash"):
        workflow._sha256_file(tmp_path / "missing")
    for digest in ("short", "g" * 64):
        with pytest.raises(GovernanceWorkflowError, match="64 hexadecimal"):
            workflow._validate_sha256(digest, "Test digest")


def test_document_helpers_render_supported_and_suppressed_groups() -> None:
    supported = {
        "axis": "sex_code",
        "group": "1",
        "status": "reviewed",
        "rows": 100,
        "metrics": {
            "target_prevalence": 0.2,
            "mean_probability": 0.21,
            "calibration_in_the_large": 0.01,
            "brier_score": 0.13,
            "selection_rate_at_q90": 0.1,
            "true_positive_rate_at_q90": 0.3,
            "false_positive_rate_at_q90": 0.05,
        },
        "confidence_intervals": {
            "target_prevalence": {"lower": 0.12, "upper": 0.29},
            "mean_probability": {"lower": 0.18, "upper": 0.24},
            "calibration_in_the_large": {"lower": -0.01, "upper": 0.03},
            "brier_score": {"lower": 0.11, "upper": 0.15},
            "selection_rate_at_q90": {"lower": 0.07, "upper": 0.13},
            "true_positive_rate_at_q90": {"lower": 0.22, "upper": 0.38},
            "false_positive_rate_at_q90": {"lower": 0.02, "upper": 0.08},
        },
    }
    unsupported = {
        "axis": "age_band",
        "group": "60_100",
        "status": "insufficient_support",
        "rows": 56,
    }

    assert "| reviewed | 100 |" in workflow._fairness_row(supported)
    assert "0.2000 [0.1200, 0.2900]" in workflow._fairness_row(supported)
    assert "| insufficient | 56 |" in workflow._fairness_row(unsupported)
    assert workflow._condition_sentence("prohibit_india_or_compliance_claims") == (
        "Prohibit India-specific or compliance claims."
    )
    assert workflow._condition_sentence("new_condition") == "New condition."
