"""Prediction-only and replay-safe final-test workflow tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import credit_risk.modeling.final_test_workflow as workflow
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.modeling.dataset import GovernedTestData, ModelingLineage
from credit_risk.modeling.final_test_workflow import FinalTestWorkflowError, run_final_test
from credit_risk.modeling.selection_models import SelectionModelError
from credit_risk.modeling.tracking import GitEvidence


class _PredictionOnlyModel:
    def __init__(self, probabilities: np.ndarray, *, fail: bool = False) -> None:
        self.probabilities = probabilities
        self.fail = fail
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, *_args: Any, **_kwargs: Any) -> None:
        self.fit_calls += 1
        raise AssertionError("final-test must never fit")

    def predict_proba(self, _features: pd.DataFrame) -> np.ndarray:
        self.predict_calls += 1
        if self.fail:
            raise SelectionModelError("controlled scoring failure")
        return self.probabilities.copy()


def test_final_test_scores_once_without_fit_and_publishes_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, paths = _arrange(tmp_path, monkeypatch, passing=True)

    result = run_final_test(**paths)

    assert result.g2_closed is True
    assert model.fit_calls == 0
    assert model.predict_calls == 1
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["population"] == {
        "assignment_sha256": "g" * 64,
        "partition": "test",
        "rows": 6000,
        "target_counts": {"0": 4673, "1": 1327},
        "unique_accounts": 6000,
    }
    assert summary["execution"] == {
        "cross_validation_performed": False,
        "evaluation_count": 1,
        "maximum_evaluations": 1,
        "refitting_performed": False,
        "retuning_performed": False,
        "training_performed": False,
    }
    assert all(gate["passed"] for gate in summary["gates"].values())
    assert result.started_receipt_path.is_file()
    assert result.completed_receipt_path.is_file()
    assert result.predictions_path.is_file()

    with pytest.raises(FinalTestWorkflowError, match="permanently prevents reevaluation"):
        run_final_test(**paths)
    assert model.predict_calls == 1


def test_failed_frozen_gates_publish_evidence_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, paths = _arrange(tmp_path, monkeypatch, passing=False)

    result = run_final_test(**paths)

    assert result.g2_closed is False
    assert model.fit_calls == 0
    assert model.predict_calls == 1
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["g2_status"] == "open_failed_test"
    assert not all(gate["passed"] for gate in summary["gates"].values())
    assert (
        json.loads(result.completed_receipt_path.read_text(encoding="utf-8"))["g2_closed"] is False
    )


def test_scoring_failure_leaves_started_receipt_and_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, paths = _arrange(tmp_path, monkeypatch, passing=True, scoring_failure=True)

    with pytest.raises(FinalTestWorkflowError, match="controlled scoring failure"):
        run_final_test(**paths)

    output = Path(paths["output_root"])
    assert model.fit_calls == 0
    assert model.predict_calls == 1
    assert (output / workflow.STARTED_FILENAME).is_file()
    assert not (output / workflow.COMPLETED_FILENAME).exists()
    with pytest.raises(FinalTestWorkflowError, match="permanently prevents reevaluation"):
        run_final_test(**paths)
    assert model.predict_calls == 1


def test_preflight_rejects_approval_drift_before_data_or_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, paths = _arrange(tmp_path, monkeypatch, passing=True)
    approval_path = Path(paths["approval_path"])
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["frozen_authorization_sha256"] = "0" * 64
    _write_json(approval_path, approval)

    with pytest.raises(FinalTestWorkflowError, match="frozen authorization digest"):
        run_final_test(**paths)

    assert model.fit_calls == 0
    assert model.predict_calls == 0
    assert not Path(paths["output_root"]).exists()


@pytest.mark.parametrize(
    ("probability", "expected"),
    ((0.19, "standard"), (0.2, "elevated"), (0.5, "high"), (0.8, "critical")),
)
def test_final_test_risk_band_boundaries(
    probability: float,
    expected: str,
) -> None:
    assert workflow.risk_band(probability, {"q80": 0.2, "q90": 0.5, "q95": 0.8}) == expected


def _arrange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    passing: bool,
    scoring_failure: bool = False,
) -> tuple[_PredictionOnlyModel, dict[str, Path]]:
    lineage = ModelingLineage(
        dataset_id="uci_credit_default",
        dataset_version="v1",
        source_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        canonical_sha256="c" * 64,
        quality_report_sha256="d" * 64,
        split_config_sha256="e" * 64,
        assignment_sha256="g" * 64,
        split_manifest_sha256="h" * 64,
        reviewed_split_lock_sha256="i" * 64,
        feature_contract_sha256="j" * 64,
    )
    account_ids = pd.Index(np.arange(1, 6001), name="account_id")
    target_values = np.asarray([0] * 4673 + [1] * 1327, dtype="int8")
    target = pd.Series(target_values, index=account_ids, dtype="int8")
    predictors = pd.DataFrame(
        np.zeros((6000, len(PREDICTOR_COLUMNS)), dtype="int64"),
        index=account_ids,
        columns=PREDICTOR_COLUMNS,
    )
    governed = GovernedTestData(
        account_ids=account_ids,
        predictors=predictors,
        target=target,
        audit=pd.DataFrame(index=account_ids),
        lineage=lineage,
    )
    probabilities = (
        np.concatenate((np.full(4673, 0.05), np.full(1327, 0.85)))
        if passing
        else np.full(6000, 0.2)
    )
    model = _PredictionOnlyModel(probabilities, fail=scoring_failure)

    selection_root = tmp_path / "reports" / "modeling" / "selection_v1"
    selection_root.mkdir(parents=True)
    summary_path = selection_root / "summary.json"
    report_path = selection_root / "selection-report.md"
    _write_json(summary_path, {"reproducibility": {"data_lineage": asdict(lineage)}})
    report_path.write_text("reviewed selection\n", encoding="utf-8")
    bundle_root = tmp_path / "models" / "selected_v1"
    bundle_root.mkdir(parents=True)
    manifest_path = bundle_root / "manifest.json"
    model_path = bundle_root / "model.cbm"
    manifest_path.write_text("reviewed manifest\n", encoding="utf-8")
    model_path.write_bytes(b"reviewed model")

    authorization_path = tmp_path / "configs" / "modeling" / "final_test_v1.json"
    authorization = {
        "schema_version": "1.0.0",
        "authorization_id": "final_test_v1",
        "status": "frozen_not_executed",
        "selection_evidence": {
            "summary_sha256": _sha(summary_path),
            "report_sha256": _sha(report_path),
            "manifest_sha256": _sha(manifest_path),
            "model_sha256": _sha(model_path),
        },
        "test_contract": {
            "required_unique_accounts": 6000,
            "maximum_evaluations": 1,
            "training": "prohibited",
            "refitting": "prohibited",
            "retuning": "prohibited",
            "force_override": "prohibited",
            "calibration": "identity",
        },
        "frozen_gates": {
            "minimum_average_precision": 0.5265104548302114,
            "maximum_brier_score": 0.15353854208377515,
            "minimum_lift_at_0_1": 2.910922787193974,
        },
        "execution": {
            "authorized": False,
            "holdout_loaded_during_freeze": False,
            "requires_separate_explicit_request": True,
        },
    }
    _write_json(authorization_path, authorization)
    approval_path = tmp_path / "configs" / "modeling" / "final_test_v1.approval.json"
    _write_json(
        approval_path,
        {
            "schema_version": "1.0.0",
            "approval_id": "final_test_v1_approval",
            "status": "approved_once",
            "frozen_authorization_sha256": _sha(authorization_path),
            "workflow_sha256": _sha(Path(workflow.__file__)),
            "manifest_sha256": _sha(manifest_path),
            "model_sha256": _sha(model_path),
            "maximum_evaluations": 1,
            "training": "prohibited",
            "refitting": "prohibited",
            "retuning": "prohibited",
            "force_override": "prohibited",
            "dirty_execution": "prohibited",
        },
    )
    verification = SimpleNamespace(
        source_sha256=lineage.source_sha256,
        dataset_manifest_sha256=lineage.dataset_manifest_sha256,
        canonical_sha256=lineage.canonical_sha256,
        quality_report_sha256=lineage.quality_report_sha256,
        split_config_sha256=lineage.split_config_sha256,
        assignment_sha256=lineage.assignment_sha256,
        split_manifest_sha256=lineage.split_manifest_sha256,
    )
    manifest = SimpleNamespace(
        selected_model_id="catboost_fixed",
        bundle_id="selected_v1",
        model_sha256=_sha(model_path),
        calibration="identity",
        risk_band_thresholds={"q80": 0.2, "q90": 0.5, "q95": 0.8},
        holdout_evaluated=False,
    )
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _path: GitEvidence("f" * 40, False, "0" * 64, tmp_path),
    )
    monkeypatch.setattr(workflow, "verify_dataset", lambda *_args, **_kwargs: verification)
    monkeypatch.setattr(workflow, "load_governed_test_data", lambda **_kwargs: governed)
    monkeypatch.setattr(
        workflow, "load_selected_bundle", lambda *_args, **_kwargs: (manifest, model)
    )
    return model, {
        "data_root": tmp_path / "data",
        "authorization_path": authorization_path,
        "approval_path": approval_path,
        "bundle_root": bundle_root,
        "runtime_root": tmp_path / "experiment" / "final-test-v1",
        "output_root": tmp_path / "reports" / "modeling" / "final_test_v1",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
