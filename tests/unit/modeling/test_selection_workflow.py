from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import credit_risk.modeling.selection_workflow as workflow
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS, REPAYMENT_STATUS_COLUMNS
from credit_risk.modeling.dataset import ModelingLineage
from credit_risk.modeling.selection_contracts import load_selection_config
from credit_risk.modeling.selection_workflow import SelectionWorkflowError, run_model_selection
from credit_risk.modeling.tracking import GitEvidence, TrackingRunResult


class _FakeModel:
    def __init__(self, model_id: str, probabilities: np.ndarray) -> None:
        self.model_id = model_id
        self.probabilities = probabilities

    def predict_proba(self, _features: pd.DataFrame) -> np.ndarray:
        return self.probabilities.copy()


def test_selection_workflow_uses_four_fitted_doubles_and_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_selection_config()
    governed, validation_probabilities = _governed_data()
    fitted_calls: list[int] = []

    monkeypatch.setattr(workflow, "load_selection_config", lambda _path: config)
    monkeypatch.setattr(workflow, "selection_config_sha256", lambda _path: "c" * 64)
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _path: GitEvidence("a" * 40, False, "b" * 64, tmp_path),
    )
    monkeypatch.setattr(workflow, "collect_package_versions", lambda _names: config.dependencies)
    monkeypatch.setattr(workflow, "load_governed_development_data", lambda **_kwargs: governed)

    def fake_fit(X_train: pd.DataFrame, y_train: pd.Series, _config: Any):
        fitted_calls.append(len(X_train))
        assert len(y_train) == 19200
        return tuple(
            _FakeModel(model.model_id, validation_probabilities) for model in config.models
        )

    monkeypatch.setattr(workflow, "fit_one_pass_models", fake_fit)
    monkeypatch.setattr(
        workflow,
        "bootstrap_validation_metrics",
        lambda *_args, **_kwargs: {"resamples": 500, "metrics": {}},
    )

    def fake_write(model: Any, destination: Path) -> tuple[Path, str]:
        assert model.model_id == "logistic_l2"
        path = destination / "model.joblib"
        path.write_bytes(b"trusted-model-double")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(workflow, "write_model_artifact", fake_write)
    monkeypatch.setattr(
        workflow,
        "load_selected_bundle",
        lambda _root, trusted: (
            None,
            _FakeModel("logistic_l2", validation_probabilities),
        ),
    )
    tracked: dict[str, Any] = {}

    def fake_track(**kwargs: Any) -> TrackingRunResult:
        tracked.update(kwargs)
        return TrackingRunResult(
            "sqlite:///tracking.db",
            "credit-risk-selection-v1",
            "parent",
            tuple(
                (model.model_name, f"child-{index}")
                for index, model in enumerate(kwargs["model_runs"])
            ),
        )

    monkeypatch.setattr(workflow, "track_selection_runs", fake_track)

    result = run_model_selection(
        data_root=tmp_path / "data",
        config_path=Path("config.json"),
        tracking_root=tmp_path / "experiment",
        output_root=tmp_path / "reports",
        bundle_root=tmp_path / "models",
    )

    assert fitted_calls == [19200]
    assert result.selected_model_id == "logistic_l2"
    assert result.summary_path.is_file()
    assert result.manifest_path.is_file()
    assert result.validation_predictions_path.is_file()
    assert len(tracked["model_runs"]) == 4
    assert all(
        value is not None for run in tracked["model_runs"] for value in run.parameters.values()
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["protocol"]["fit_count"] == 4
    assert summary["protocol"]["cross_validation_iteration"] is False
    assert summary["protocol"]["winner_refitted"] is False
    assert summary["population"]["holdout_accessed"] is False


def test_dirty_gate_precedes_data_or_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_selection_config()
    monkeypatch.setattr(workflow, "load_selection_config", lambda _path: config)
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _path: GitEvidence("a" * 40, True, "b" * 64, tmp_path),
    )
    monkeypatch.setattr(
        workflow,
        "load_governed_development_data",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("data must remain closed")),
    )

    with pytest.raises(SelectionWorkflowError, match="clean committed worktree"):
        run_model_selection(config_path=Path("config.json"))


def test_version_and_destination_gates_precede_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_selection_config()
    monkeypatch.setattr(workflow, "load_selection_config", lambda _path: config)
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _path: GitEvidence("a" * 40, False, "b" * 64, tmp_path),
    )
    monkeypatch.setattr(workflow, "collect_package_versions", lambda _names: {"wrong": "1"})
    with pytest.raises(SelectionWorkflowError, match="dependency versions differ"):
        run_model_selection(config_path=Path("config.json"))

    monkeypatch.setattr(workflow, "collect_package_versions", lambda _names: config.dependencies)
    output = tmp_path / "reports"
    output.mkdir()
    with pytest.raises(SelectionWorkflowError, match="Refusing to overwrite"):
        run_model_selection(
            config_path=Path("config.json"),
            tracking_root=tmp_path / "experiment",
            output_root=output,
            bundle_root=tmp_path / "models",
        )

    with pytest.raises(SelectionWorkflowError, match="inside the repository"):
        run_model_selection(
            config_path=Path("config.json"),
            tracking_root=tmp_path / "experiment",
            output_root=tmp_path.parent / "outside",
            bundle_root=tmp_path / "models",
        )


def test_publication_failure_marks_parent_failed_and_preserves_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_selection_config()
    governed, validation_probabilities = _governed_data()
    monkeypatch.setattr(workflow, "load_selection_config", lambda _path: config)
    monkeypatch.setattr(workflow, "selection_config_sha256", lambda _path: "c" * 64)
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _path: GitEvidence("a" * 40, False, "b" * 64, tmp_path),
    )
    monkeypatch.setattr(workflow, "collect_package_versions", lambda _names: config.dependencies)
    monkeypatch.setattr(workflow, "load_governed_development_data", lambda **_kwargs: governed)
    monkeypatch.setattr(
        workflow,
        "fit_one_pass_models",
        lambda *_args: tuple(
            _FakeModel(model.model_id, validation_probabilities) for model in config.models
        ),
    )
    monkeypatch.setattr(
        workflow,
        "bootstrap_validation_metrics",
        lambda *_args, **_kwargs: {"resamples": 500},
    )

    def fake_write(_model: Any, destination: Path) -> tuple[Path, str]:
        path = destination / "model.joblib"
        path.write_bytes(b"model")
        return path, hashlib.sha256(b"model").hexdigest()

    monkeypatch.setattr(workflow, "write_model_artifact", fake_write)
    monkeypatch.setattr(
        workflow,
        "load_selected_bundle",
        lambda *_args, **_kwargs: (None, _FakeModel("logistic_l2", validation_probabilities)),
    )
    tracking = TrackingRunResult("uri", "experiment", "parent", ())
    monkeypatch.setattr(workflow, "track_selection_runs", lambda **_kwargs: tracking)
    failures: list[str] = []
    monkeypatch.setattr(
        workflow,
        "mark_tracking_run_failed",
        lambda _result, failure_stage: failures.append(failure_stage),
    )
    monkeypatch.setattr(
        workflow,
        "_promote_directories",
        lambda _pairs: (_ for _ in ()).throw(OSError("promotion failed")),
    )

    with pytest.raises(SelectionWorkflowError, match="Atomic selection publication failed"):
        run_model_selection(
            config_path=Path("config.json"),
            tracking_root=tmp_path / "experiment",
            output_root=tmp_path / "reports",
            bundle_root=tmp_path / "models",
        )

    assert failures == ["atomic_publication"]
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "models").exists()


def _governed_data() -> tuple[SimpleNamespace, np.ndarray]:
    ids = pd.Index(range(1, 24001), name="account_id")
    payload = {column: np.zeros(len(ids), dtype=np.int64) for column in PREDICTOR_COLUMNS}
    for column in REPAYMENT_STATUS_COLUMNS:
        payload[column] = np.zeros(len(ids), dtype=np.int8)
    predictors = pd.DataFrame(payload, index=ids)
    train = np.concatenate((np.zeros(14953, dtype=np.int8), np.ones(4247, dtype=np.int8)))
    validation = np.concatenate((np.zeros(3738, dtype=np.int8), np.ones(1062, dtype=np.int8)))
    target = pd.Series(np.concatenate((train, validation)), index=ids)
    assignments = pd.DataFrame(
        {
            "cv_fold_r0": np.concatenate(
                (np.ones(19200, dtype=np.int8), np.zeros(4800, dtype=np.int8))
            )
        },
        index=ids,
    )
    lineage = ModelingLineage(
        dataset_id="uci_credit_default",
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
    validation_probabilities = np.where(validation == 1, 0.8, 0.2).astype(np.float64)
    return SimpleNamespace(
        X=predictors,
        y=target,
        assignments=assignments,
        lineage=lineage,
    ), validation_probabilities
