"""End-to-end evidence for the governed development-only baseline workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

import credit_risk.modeling.workflow as baseline_workflow
from credit_risk.data.workflow import build_dataset
from credit_risk.modeling.tracking import BASELINE_MODEL_NAMES, GitEvidence
from credit_risk.modeling.workflow import run_baseline_experiment
from tests.unit.data.helpers import source_frame, write_json, write_workflow_contract

pytestmark = [pytest.mark.integration, pytest.mark.training]


def test_synthetic_offline_baseline_experiment_is_complete_and_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise governed data, all baselines, metrics, MLflow, and publication."""

    data_root, config_path = _write_synthetic_experiment(tmp_path)
    git = GitEvidence(commit_sha="a" * 40, dirty=False, diff_sha256="b" * 64)
    monkeypatch.setattr(baseline_workflow, "collect_git_evidence", lambda _root: git)
    tracking_root = tmp_path / "experiment" / "mlflow"
    output_root = tmp_path / "reports" / "baseline_v1"
    arguments = {
        "data_root": data_root,
        "config_path": config_path,
        "tracking_root": tracking_root,
        "output_root": output_root,
        "repo_root": tmp_path,
    }

    first = run_baseline_experiment(**arguments)
    first_summary = first.summary_path.read_bytes()
    first_report = first.report_path.read_bytes()
    first_oof = first.oof_predictions_path.read_bytes()
    second = run_baseline_experiment(**arguments)

    assert second.summary_sha256 == first.summary_sha256
    assert second.report_sha256 == first.report_sha256
    assert second.oof_predictions_sha256 == first.oof_predictions_sha256
    assert second.logistic_diagnostics_sha256 == first.logistic_diagnostics_sha256
    assert second.summary_path.read_bytes() == first_summary
    assert second.report_path.read_bytes() == first_report
    assert second.oof_predictions_path.read_bytes() == first_oof
    assert (
        second.logistic_diagnostics_path.read_bytes()
        == first.logistic_diagnostics_path.read_bytes()
    )

    summary = json.loads(first_summary)
    assert summary["data"] == {
        "development_rows": 80,
        "holdout_evaluated": False,
        "n_repeats": 3,
        "n_splits": 5,
    }
    assert tuple(summary["models"]) == tuple(sorted(BASELINE_MODEL_NAMES))
    assert summary["reproducibility"]["git_dirty"] is False
    assert summary["reproducibility"]["git_commit_sha"] == "a" * 40
    assert summary["runtime_artifacts"]["contains_holdout_rows"] is False
    assert summary["runtime_artifacts"]["contains_fitted_models"] is False

    oof = pd.read_csv(first.oof_predictions_path)
    assert tuple(oof.columns) == (
        "account_id",
        "model_id",
        "repeat_index",
        "fold_index",
        "prediction_kind",
        "score",
    )
    assert len(oof) == 80 * 3 * len(BASELINE_MODEL_NAMES)
    assert set(oof["model_id"]) == set(BASELINE_MODEL_NAMES)
    assert oof.groupby(["model_id", "repeat_index"])["account_id"].nunique().eq(80).all()
    assert not oof.duplicated(["account_id", "model_id", "repeat_index"]).any()

    assignments = pd.read_csv(
        data_root / "splits" / "fixture_credit_default" / "v1" / "split_assignments.csv"
    )
    test_ids = set(assignments.loc[assignments["partition"].eq("test"), "account_id"])
    assert test_ids
    assert test_ids.isdisjoint(set(oof["account_id"]))

    _assert_tracking_contract(first.tracking.tracking_uri, first.tracking.parent_run_id)
    _assert_tracking_contract(second.tracking.tracking_uri, second.tracking.parent_run_id)
    assert not list(tracking_root.rglob("*.pkl"))
    diagnostics = json.loads(first.logistic_diagnostics_path.read_bytes())
    assert len(diagnostics["folds"]) == 15
    assert len(diagnostics["transformed_feature_names"]) == 79


def _write_synthetic_experiment(root: Path) -> tuple[Path, Path]:
    (root / "uv.lock").write_bytes(Path("uv.lock").read_bytes())
    data_root, manifest_path, split_path = write_workflow_contract(root, source_frame())
    built = build_dataset(data_root, manifest_path, split_path, offline=True)
    reviewed_lock = split_path.with_suffix(".lock.json")
    reviewed_lock.write_bytes(built.paths.split_manifest.read_bytes())

    feature_payload = json.loads(
        Path("configs/modeling/feature_contract_v1.json").read_text(encoding="utf-8")
    )
    feature_payload["contract_id"] = "fixture_credit_default_features_v1"
    feature_payload["dataset"] = {
        "dataset_id": "fixture_credit_default",
        "dataset_version": "v1",
    }
    feature_payload["expected_development"] = {
        "rows": 80,
        "target_counts": {"0": 40, "1": 40},
    }
    feature_payload["lineage"] = {
        "source_sha256": built.source_sha256,
        "dataset_manifest_sha256": built.dataset_manifest_sha256,
        "canonical_sha256": built.canonical_sha256,
        "split_config_sha256": built.split_config_sha256,
        "assignment_sha256": built.assignment_sha256,
        "reviewed_split_lock_sha256": hashlib.sha256(reviewed_lock.read_bytes()).hexdigest(),
    }
    feature_path = root / "configs" / "feature_contract.json"
    write_json(feature_path, feature_payload)

    baseline_payload = json.loads(
        Path("configs/modeling/baseline_v1.json").read_text(encoding="utf-8")
    )
    baseline_payload.update(
        {
            "feature_contract_path": feature_path.name,
            "feature_contract_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            "dataset_manifest_path": manifest_path.name,
            "split_config_path": split_path.name,
        }
    )
    baseline_path = root / "configs" / "baseline.json"
    write_json(baseline_path, baseline_payload)
    return data_root, baseline_path


def _assert_tracking_contract(tracking_uri: str, parent_run_id: str) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    parent = client.get_run(parent_run_id)
    assert parent.data.tags["holdout_evaluated"] == "false"
    assert parent.data.tags["run_role"] == "baseline_parent"
    assert parent.data.params["development_rows"] == "80"
    assert {artifact.path for artifact in client.list_artifacts(parent_run_id)} == {
        "evidence",
        "runtime",
    }

    child_runs = client.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )
    assert {run.data.tags["model_id"] for run in child_runs} == set(BASELINE_MODEL_NAMES)
    assert all(run.data.tags["run_role"] == "baseline_model" for run in child_runs)
    assert all("repeat_summary.average_precision.mean" in run.data.metrics for run in child_runs)
