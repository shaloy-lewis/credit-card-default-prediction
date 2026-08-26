"""Tests for governed baseline orchestration and deterministic evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from credit_risk.modeling import workflow
from credit_risk.modeling.baselines import PREDICTOR_COLUMNS
from credit_risk.modeling.dataset import GovernedDevelopmentData, ModelingLineage
from credit_risk.modeling.tracking import (
    BASELINE_MODEL_NAMES,
    GitEvidence,
    TrackingDependencyError,
    TrackingError,
    TrackingRunResult,
)
from credit_risk.modeling.workflow import BaselineWorkflowError, run_baseline_experiment


def test_baseline_workflow_is_deterministic_and_tracks_only_approved_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    config_path, config, data_calls = _configure_runtime(tmp_path, monkeypatch, captured)
    kwargs = {
        "data_root": tmp_path / "data",
        "config_path": config_path,
        "tracking_root": tmp_path / "experiment" / "mlflow",
        "output_root": tmp_path / "reports",
        "repo_root": tmp_path,
    }

    first = run_baseline_experiment(**kwargs)
    first_summary = first.summary_path.read_bytes()
    first_report = first.report_path.read_bytes()
    first_oof = first.oof_predictions_path.read_bytes()
    second = run_baseline_experiment(**kwargs)

    assert first.summary_sha256 == second.summary_sha256
    assert first.report_sha256 == second.report_sha256
    assert first.oof_predictions_sha256 == second.oof_predictions_sha256
    assert second.summary_path.read_bytes() == first_summary
    assert second.report_path.read_bytes() == first_report
    assert second.oof_predictions_path.read_bytes() == first_oof
    assert len(captured) == 2
    assert len(data_calls) == 2
    for data_call in data_calls:
        assert data_call["feature_contract_path"] == (tmp_path / "feature_contract.json")
        assert data_call["manifest_path"] == (tmp_path / "dataset_manifest.json")
        assert data_call["split_config_path"] == (tmp_path / "split_config.json")

    summary = json.loads(first_summary)
    assert list(summary["models"]) == sorted(BASELINE_MODEL_NAMES)
    assert summary["experiment"]["experiment_name"] == config.experiment_name
    assert summary["data"] == {
        "development_rows": 10,
        "holdout_evaluated": False,
        "n_repeats": 3,
        "n_splits": 5,
    }
    assert summary["runtime_artifacts"]["contains_holdout_rows"] is False
    assert summary["runtime_artifacts"]["contains_fitted_models"] is False
    serialized = first_summary.decode("utf-8")
    assert "parent-run" not in serialized
    assert str(tmp_path) not in serialized
    assert "timestamp" not in serialized.lower()
    for model_id in BASELINE_MODEL_NAMES:
        evidence = summary["models"][model_id]
        assert "descriptive_pooled_oof_metrics" in evidence
        assert "combined_oof_metrics" not in evidence
        assert len(evidence["fold_metrics"]) == 15
        assert len(evidence["repeat_metrics"]) == 3
        assert "average_precision" in evidence["repeat_summaries"]

    oof_lines = first_oof.decode("utf-8").splitlines()
    assert oof_lines[0] == ("account_id,model_id,repeat_index,fold_index,prediction_kind,score")
    assert len(oof_lines) == 1 + 10 * 3 * 3
    assert "target" not in oof_lines[0]
    assert all("pickle" not in line for line in oof_lines)
    assert oof_lines[1].split(",")[1] == "fold_prevalence"
    assert oof_lines[31].split(",")[1] == "repayment_burden_rule"
    assert oof_lines[61].split(",")[1] == "logistic_l2"

    tracked = captured[0]
    assert tracked["experiment_name"] == "credit-risk-baseline-v1"
    assert tuple(payload.model_name for payload in tracked["model_runs"]) == (BASELINE_MODEL_NAMES)
    assert [path.name for path in tracked["artifacts"]] == [
        "summary.json",
        "baseline-report.md",
        "oof_predictions.csv",
        "logistic_fold_diagnostics.json",
    ]
    assert tracked["artifact_bytes"]["oof_predictions.csv"] == first_oof
    diagnostics = json.loads(tracked["artifact_bytes"]["logistic_fold_diagnostics.json"])
    assert diagnostics["model_id"] == "logistic_l2"
    assert len(diagnostics["folds"]) == 15
    assert tracked["parent_tags"]["holdout_evaluated"] == "false"
    report = first_report.decode("utf-8")
    assert "Average precision" in report
    assert "not a trapezoidal PR-curve area" in report
    assert "pooled OOF metrics as descriptive evidence only" in report
    assert all(line == line.rstrip() for line in report.splitlines())


def test_dirty_gate_runs_before_loading_governed_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow, "ensure_mlflow_available", lambda: None)
    monkeypatch.setattr(workflow, "collect_package_versions", lambda: {"mlflow": "3.15.0"})
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _root: GitEvidence("a" * 40, True, "b" * 64),
    )
    monkeypatch.setattr(
        workflow,
        "parse_baseline_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("config should not load")),
    )

    with pytest.raises(BaselineWorkflowError, match="Git worktree is dirty.*--allow-dirty"):
        run_baseline_experiment(config_path=tmp_path / "missing.json", repo_root=tmp_path)


def test_modeling_dependency_preflight_wins_before_git_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow,
        "ensure_mlflow_available",
        lambda: (_ for _ in ()).throw(
            TrackingDependencyError(
                "MLflow is unavailable; install the project with the 'modeling' extra."
            )
        ),
    )
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _root: (_ for _ in ()).throw(AssertionError("git should not run")),
    )

    with pytest.raises(BaselineWorkflowError, match="install.*modeling.*extra"):
        run_baseline_experiment(config_path=tmp_path / "missing.json", repo_root=tmp_path)


def test_dirty_override_cannot_publish_to_reviewed_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow, "ensure_mlflow_available", lambda: None)
    monkeypatch.setattr(workflow, "collect_package_versions", lambda: {"mlflow": "3.15.0"})
    dirty = GitEvidence("a" * 40, True, "b" * 64)
    monkeypatch.setattr(workflow, "collect_git_evidence", lambda _root: dirty)

    with pytest.raises(BaselineWorkflowError, match="experiment/provisional/baseline_v1"):
        run_baseline_experiment(
            config_path=tmp_path / "missing.json",
            output_root=Path("reports/modeling/baseline_v1"),
            repo_root=tmp_path,
            allow_dirty=True,
        )

    workflow._enforce_git_output_policy(
        dirty,
        allow_dirty=True,
        output_root=Path("experiment/provisional/baseline_v1"),
        repo_root=tmp_path,
    )


def test_dirty_override_rejects_arbitrary_nonreviewed_output_root(tmp_path: Path) -> None:
    dirty = GitEvidence("a" * 40, True, "b" * 64)

    with pytest.raises(BaselineWorkflowError, match="only to the ignored provisional"):
        workflow._enforce_git_output_policy(
            dirty,
            allow_dirty=True,
            output_root=Path("scratch/another-report"),
            repo_root=tmp_path,
        )


def test_tracking_failure_does_not_publish_partial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    config_path, _config, _data_calls = _configure_runtime(tmp_path, monkeypatch, captured)
    output_root = tmp_path / "reports"
    output_root.mkdir()
    summary_path = output_root / "summary.json"
    report_path = output_root / "baseline-report.md"
    summary_path.write_bytes(b"reviewed-summary\n")
    report_path.write_bytes(b"reviewed-report\n")
    monkeypatch.setattr(
        workflow,
        "track_baseline_runs",
        lambda **_kwargs: (_ for _ in ()).throw(TrackingError("database unavailable")),
    )

    with pytest.raises(BaselineWorkflowError, match="database unavailable"):
        run_baseline_experiment(
            data_root=tmp_path / "data",
            config_path=config_path,
            tracking_root=tmp_path / "tracking",
            output_root=output_root,
            repo_root=tmp_path,
        )

    assert summary_path.read_bytes() == b"reviewed-summary\n"
    assert report_path.read_bytes() == b"reviewed-report\n"


def test_publication_failure_marks_completed_tracking_parent_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    config_path, _config, _data_calls = _configure_runtime(tmp_path, monkeypatch, captured)
    failures: list[tuple[TrackingRunResult, str]] = []
    monkeypatch.setattr(
        workflow,
        "_promote_outputs",
        lambda _payloads: (_ for _ in ()).throw(
            BaselineWorkflowError("simulated publication failure")
        ),
    )
    monkeypatch.setattr(
        workflow,
        "mark_tracking_run_failed",
        lambda result, *, failure_stage: failures.append((result, failure_stage)),
    )

    with pytest.raises(BaselineWorkflowError, match="simulated publication failure"):
        run_baseline_experiment(
            data_root=tmp_path / "data",
            config_path=config_path,
            tracking_root=tmp_path / "tracking",
            output_root=tmp_path / "reports",
            repo_root=tmp_path,
        )

    assert len(captured) == 1
    assert failures == [
        (
            TrackingRunResult(
                tracking_uri="sqlite:///tracking/mlflow.db",
                experiment_name="credit-risk-baseline-v1",
                parent_run_id="parent-run-1",
                child_run_ids=tuple(
                    (model_id, f"child-{model_id}") for model_id in BASELINE_MODEL_NAMES
                ),
            ),
            "evidence_publication",
        )
    ]


@pytest.mark.parametrize(
    ("field_name", "label", "filename"),
    (
        ("dataset_manifest_path", "dataset manifest", "dataset_manifest.json"),
        ("split_config_path", "split configuration", "split_config.json"),
    ),
)
def test_missing_governed_config_paths_fail_before_data_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    label: str,
    filename: str,
) -> None:
    root = tmp_path / field_name
    captured: list[dict[str, Any]] = []
    config_path, _config, data_calls = _configure_runtime(root, monkeypatch, captured)
    (root / filename).unlink()

    with pytest.raises(BaselineWorkflowError, match=f"Configured {label} is missing"):
        run_baseline_experiment(
            data_root=root / "data",
            config_path=config_path,
            tracking_root=root / "tracking",
            output_root=root / "reports",
            repo_root=root,
        )

    assert data_calls == []


@pytest.mark.parametrize("invalid_path", ("../manifest.json", "manifest.txt", "bad\\path.json"))
def test_invalid_governed_config_paths_fail_actionably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_path: str,
) -> None:
    captured: list[dict[str, Any]] = []
    config_path, config, data_calls = _configure_runtime(tmp_path, monkeypatch, captured)
    config.dataset_manifest_path = invalid_path

    with pytest.raises(BaselineWorkflowError, match="safe and repository-relative"):
        run_baseline_experiment(
            data_root=tmp_path / "data",
            config_path=config_path,
            tracking_root=tmp_path / "tracking",
            output_root=tmp_path / "reports",
            repo_root=tmp_path,
        )

    assert data_calls == []


@pytest.mark.parametrize(
    ("partition", "positive_label", "expected"),
    (("test", 1, "development partition"), ("development", 0, "positive_label=1")),
)
def test_runtime_contract_rejects_unapproved_evaluation_boundaries(
    partition: str,
    positive_label: int,
    expected: str,
) -> None:
    config = _config(SimpleNamespace(), partition=partition, positive_label=positive_label)
    with pytest.raises(BaselineWorkflowError, match=expected):
        workflow._validate_runtime_contract(config, _governed_data())


def test_atomic_promotion_rolls_back_first_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "summary.json"
    second = tmp_path / "baseline-report.md"
    first.write_bytes(b"old-summary")
    second.write_bytes(b"old-report")
    real_writer = workflow._write_bytes_atomically
    calls = 0

    def fail_second(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publication failure")
        real_writer(path, content)

    monkeypatch.setattr(workflow, "_write_bytes_atomically", fail_second)
    with pytest.raises(BaselineWorkflowError, match="atomically"):
        workflow._promote_outputs({first: b"new-summary", second: b"new-report"})

    assert first.read_bytes() == b"old-summary"
    assert second.read_bytes() == b"old-report"


def test_atomic_promotion_preflights_every_destination_before_writing(tmp_path: Path) -> None:
    first = tmp_path / "summary.json"
    second = tmp_path / "baseline-report.md"
    first.write_bytes(b"old-summary")
    second.mkdir()

    with pytest.raises(BaselineWorkflowError, match="not a regular file"):
        workflow._promote_outputs({first: b"new-summary", second: b"new-report"})

    assert first.read_bytes() == b"old-summary"


def _configure_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    captured: list[dict[str, Any]],
) -> tuple[Path, SimpleNamespace, list[dict[str, Any]]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "uv.lock").write_bytes(b"fixture dependency lock\n")
    feature_contract = tmp_path / "feature_contract.json"
    feature_contract.write_bytes(b'{"schema_version":"1.0.0"}\n')
    (tmp_path / "dataset_manifest.json").write_bytes(b'{"dataset_id":"fixture"}\n')
    (tmp_path / "split_config.json").write_bytes(b'{"dataset_id":"fixture"}\n')
    config_path = tmp_path / "baseline.json"
    config_path.write_bytes(b'{"experiment_id":"baseline_v1"}\n')
    config = _config(feature_contract)
    governed = _governed_data()
    data_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(workflow, "ensure_mlflow_available", lambda: None)
    monkeypatch.setattr(
        workflow,
        "collect_package_versions",
        lambda: {
            "credit-risk-early-warning": "0.1.0",
            "mlflow": "3.15.0",
            "numpy": "1.26.4",
            "pandas": "2.2.2",
            "scikit-learn": "1.4.2",
        },
    )
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _root: GitEvidence("a" * 40, False, "b" * 64),
    )
    monkeypatch.setattr(
        workflow,
        "parse_baseline_config",
        lambda _content, *, source: config,
    )
    monkeypatch.setattr(
        workflow,
        "parse_feature_contract",
        lambda _content, *, source: SimpleNamespace(),
    )

    def fake_data_loader(**kwargs: Any) -> GovernedDevelopmentData:
        data_calls.append(kwargs)
        return governed

    monkeypatch.setattr(workflow, "load_governed_development_data", fake_data_loader)

    def fake_tracking(**kwargs: Any) -> TrackingRunResult:
        captured.append(
            {
                **kwargs,
                "artifacts": tuple(Path(path) for path in kwargs["artifacts"]),
                "artifact_bytes": {
                    Path(path).name: Path(path).read_bytes() for path in kwargs["artifacts"]
                },
            }
        )
        return TrackingRunResult(
            tracking_uri="sqlite:///tracking/mlflow.db",
            experiment_name=str(kwargs["experiment_name"]),
            parent_run_id=f"parent-run-{len(captured)}",
            child_run_ids=tuple(
                (model_id, f"child-{model_id}") for model_id in BASELINE_MODEL_NAMES
            ),
        )

    monkeypatch.setattr(workflow, "track_baseline_runs", fake_tracking)
    return config_path, config, data_calls


def _config(
    feature_contract: Path | SimpleNamespace,
    *,
    partition: str = "development",
    positive_label: int = 1,
) -> SimpleNamespace:
    if isinstance(feature_contract, Path):
        contract_path = feature_contract.name
        contract_sha256 = hashlib.sha256(feature_contract.read_bytes()).hexdigest()
    else:
        contract_path = "feature_contract.json"
        contract_sha256 = "c" * 64
    return SimpleNamespace(
        experiment_id="baseline_v1",
        experiment_name="credit-risk-baseline-v1",
        feature_contract_path=contract_path,
        dataset_manifest_path="dataset_manifest.json",
        split_config_path="split_config.json",
        feature_contract_sha256=contract_sha256,
        partition=partition,
        positive_label=positive_label,
        random_state=42,
        baselines=SimpleNamespace(
            prevalence=SimpleNamespace(
                model_id="fold_prevalence",
                kind="train_fold_prevalence",
                prediction_kind="probability",
            ),
            repayment_rule=SimpleNamespace(
                model_id="repayment_burden_rule",
                kind="weighted_positive_repayment_status",
                prediction_kind="risk_score",
                status_columns=tuple(f"repayment_status_lag_{lag}" for lag in range(6)),
                recency_weights=(6, 5, 4, 3, 2, 1),
                negative_value_floor=0,
                aggregation="sum",
            ),
            logistic=SimpleNamespace(
                model_id="logistic_l2",
                kind="logistic_regression",
                prediction_kind="probability",
                status_columns=tuple(f"repayment_status_lag_{lag}" for lag in range(6)),
                status_encoding="one_hot",
                status_categories=tuple(range(-2, 10)),
                status_drop="first",
                handle_unknown="error",
                monetary_columns=(
                    "credit_limit_ntd",
                    *(f"bill_amount_ntd_lag_{lag}" for lag in range(6)),
                    *(f"payment_amount_ntd_lag_{lag}" for lag in range(6)),
                ),
                scaler="standard",
                penalty="l2",
                c=1.0,
                solver="lbfgs",
                class_weight=None,
                fit_intercept=True,
                max_iter=2_000,
                tolerance=1e-8,
                random_state=42,
            ),
        ),
        evaluation=SimpleNamespace(
            primary_metric="average_precision",
            probability_guardrail="brier_score",
            primary_capacity_metric=SimpleNamespace(metric="lift", capacity=0.1),
            capacities=(0.05, 0.10, 0.20),
        ),
    )


def _governed_data() -> GovernedDevelopmentData:
    account_ids = pd.Index(range(1, 11), name="account_id", dtype="int64")
    rows = np.arange(10, dtype=np.int64)
    values: dict[str, np.ndarray] = {"credit_limit_ntd": 50_000 + 1_000 * rows}
    for lag in range(6):
        values[f"repayment_status_lag_{lag}"] = ((rows + lag) % 4) - 1
        values[f"bill_amount_ntd_lag_{lag}"] = 1_000 + 100 * rows + lag
        values[f"payment_amount_ntd_lag_{lag}"] = 100 + 10 * rows + lag
    predictors = pd.DataFrame(values, index=account_ids).loc[:, PREDICTOR_COLUMNS]
    target = pd.Series([0, 1] * 5, index=account_ids, name="default_next_month")
    assignments = pd.DataFrame(
        {f"cv_fold_r{repeat}": np.repeat(np.arange(5), 2) for repeat in range(3)},
        index=account_ids,
    )
    lineage = ModelingLineage(
        dataset_id="fixture",
        dataset_version="v1",
        source_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        canonical_sha256="3" * 64,
        quality_report_sha256="4" * 64,
        split_config_sha256="5" * 64,
        assignment_sha256="6" * 64,
        split_manifest_sha256="7" * 64,
        reviewed_split_lock_sha256="8" * 64,
        feature_contract_sha256="9" * 64,
    )
    return GovernedDevelopmentData(
        account_ids=account_ids,
        predictors=predictors,
        target=target,
        audit=pd.DataFrame(index=account_ids),
        assignments=assignments,
        lineage=lineage,
        n_splits=5,
        n_repeats=3,
    )
