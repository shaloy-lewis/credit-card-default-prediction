from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import credit_risk.modeling.final_test as final_test
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.modeling.final_test import FinalTestFreezeError, freeze_final_test_authorization
from credit_risk.modeling.selected_bundle import BundleManifest, write_manifest
from credit_risk.modeling.tracking import GitEvidence


def test_freeze_authorization_uses_only_reviewed_validation_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, bundle = _evidence(tmp_path)
    monkeypatch.setattr(
        final_test,
        "collect_git_evidence",
        lambda _path: GitEvidence("f" * 40, False, "0" * 64, tmp_path),
    )

    output = freeze_final_test_authorization(
        selection_root=selection,
        bundle_root=bundle,
        output=tmp_path / "final_test_v1.json",
    )

    authorization = json.loads(output.read_text(encoding="utf-8"))
    assert authorization["frozen_gates"] == pytest.approx(
        {
            "minimum_average_precision": 0.52,
            "maximum_brier_score": 0.16,
            "minimum_lift_at_0_1": 2.8,
        }
    )
    assert authorization["test_contract"]["required_unique_accounts"] == 6000
    assert authorization["execution"]["authorized"] is False
    assert authorization["execution"]["holdout_loaded_during_freeze"] is False


def test_freeze_rejects_dirty_evidence_or_model_digest_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, bundle = _evidence(tmp_path)
    monkeypatch.setattr(
        final_test,
        "collect_git_evidence",
        lambda _path: GitEvidence("f" * 40, True, "0" * 64, tmp_path),
    )
    with pytest.raises(FinalTestFreezeError, match="clean worktree"):
        freeze_final_test_authorization(
            selection_root=selection,
            bundle_root=bundle,
            output=tmp_path / "authorization.json",
        )

    monkeypatch.setattr(
        final_test,
        "collect_git_evidence",
        lambda _path: GitEvidence("f" * 40, False, "0" * 64, tmp_path),
    )
    (bundle / "model.joblib").write_bytes(b"changed")
    with pytest.raises(FinalTestFreezeError, match="digest differs"):
        freeze_final_test_authorization(
            selection_root=selection,
            bundle_root=bundle,
            output=tmp_path / "authorization.json",
        )


def test_freeze_refuses_overwrite_and_holdout_or_winner_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, bundle = _evidence(tmp_path)
    monkeypatch.setattr(
        final_test,
        "collect_git_evidence",
        lambda _path: GitEvidence("f" * 40, False, "0" * 64, tmp_path),
    )
    output = tmp_path / "authorization.json"
    output.write_text("frozen", encoding="utf-8")
    with pytest.raises(FinalTestFreezeError, match="Refusing to overwrite"):
        freeze_final_test_authorization(selection_root=selection, bundle_root=bundle, output=output)

    output.unlink()
    summary_path = selection / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["holdout"]["evaluated"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(FinalTestFreezeError, match="holdout is unevaluated"):
        freeze_final_test_authorization(selection_root=selection, bundle_root=bundle, output=output)

    summary["holdout"]["evaluated"] = False
    summary["selection"]["selected_model_id"] = "random_forest"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(FinalTestFreezeError, match="different winners"):
        freeze_final_test_authorization(selection_root=selection, bundle_root=bundle, output=output)


def _evidence(tmp_path: Path) -> tuple[Path, Path]:
    selection = tmp_path / "reports"
    bundle = tmp_path / "bundle"
    selection.mkdir()
    bundle.mkdir()
    summary = {
        "reproducibility": {"git_commit": "a" * 40},
        "holdout": {"evaluated": False},
        "selection": {"selected_model_id": "logistic_l2"},
        "models": [
            {
                "model_id": "logistic_l2",
                "validation_metrics": {
                    "discrimination": {"average_precision": 0.55},
                    "probability": {"brier_score": 0.14},
                    "capacities": [{"capacity": 0.1, "lift": 3.1}],
                },
            }
        ],
    }
    (selection / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    (selection / "selection-report.md").write_text("# reviewed\n", encoding="utf-8")
    model_path = bundle / "model.joblib"
    model_path.write_bytes(b"model")
    manifest = BundleManifest(
        schema_version="1.0.0",
        bundle_id="selected_v1",
        selected_model_id="logistic_l2",
        model_filename="model.joblib",
        model_sha256=hashlib.sha256(b"model").hexdigest(),
        selection_config_sha256="1" * 64,
        training_population_sha256="2" * 64,
        validation_population_sha256="3" * 64,
        validation_predictions_sha256="4" * 64,
        feature_order=PREDICTOR_COLUMNS,
        feature_handling="handling",
        class_order=(0, 1),
        fixed_parameters={},
        dependencies={},
        git_commit="a" * 40,
        git_dirty=False,
        validation_metrics={},
        selection_outcome={"selected_model_id": "logistic_l2"},
        calibration="identity",
        risk_band_thresholds={"q80": 0.2, "q90": 0.3, "q95": 0.4},
        fit_count=4,
        winner_refitted=False,
        holdout_evaluated=False,
        trusted_local_serialization=True,
    )
    write_manifest(manifest, bundle)
    return selection, bundle
