from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

import credit_risk.modeling.selected_bundle as bundles
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.modeling.selected_bundle import (
    BundleManifest,
    SelectedBundleError,
    load_selected_bundle,
    write_manifest,
    write_model_artifact,
)
from credit_risk.modeling.selection_models import FittedSelectionModel


def test_joblib_bundle_requires_trust_and_digest_validation(tmp_path: Path) -> None:
    fitted = FittedSelectionModel("logistic_l2", Pipeline([]), "handling")
    model_path, model_sha = write_model_artifact(fitted, tmp_path)
    write_manifest(_manifest(model_sha), tmp_path)

    with pytest.raises(SelectedBundleError, match="pickle semantics"):
        load_selected_bundle(tmp_path)

    manifest, loaded = load_selected_bundle(tmp_path, trusted=True)
    assert manifest.model_sha256 == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert loaded.model_id == "logistic_l2"

    model_path.write_bytes(model_path.read_bytes() + b"corrupt")
    with pytest.raises(SelectedBundleError, match="digest mismatch"):
        load_selected_bundle(tmp_path, trusted=True)


def test_bundle_rejects_extra_files_and_wrong_estimator_type(tmp_path: Path) -> None:
    fitted = FittedSelectionModel("logistic_l2", object(), "handling")
    with pytest.raises(SelectedBundleError, match="unexpected estimator type"):
        write_model_artifact(fitted, tmp_path)

    fitted = FittedSelectionModel("logistic_l2", Pipeline([]), "handling")
    _, model_sha = write_model_artifact(fitted, tmp_path)
    write_manifest(_manifest(model_sha), tmp_path)
    (tmp_path / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(SelectedBundleError, match="allowlist"):
        load_selected_bundle(tmp_path, trusted=True)


def test_native_catboost_bundle_uses_canonical_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeCatBoost:
        def save_model(self, path: str, *, format: str) -> None:
            assert format == "cbm"
            Path(path).write_bytes(b"native-catboost")

        def load_model(self, path: str, *, format: str) -> None:
            assert format == "cbm"
            assert Path(path).read_bytes() == b"native-catboost"

    monkeypatch.setattr(bundles, "CatBoostClassifier", FakeCatBoost)
    fitted = FittedSelectionModel("catboost_fixed", FakeCatBoost(), "native")
    model_path, model_sha = write_model_artifact(fitted, tmp_path)
    write_manifest(
        _manifest(
            model_sha,
            model_id="catboost_fixed",
            filename="model.cbm",
            handling="native",
        ),
        tmp_path,
    )

    manifest, loaded = load_selected_bundle(tmp_path)

    assert model_path.name == "model.cbm"
    assert manifest.selected_model_id == "catboost_fixed"
    assert loaded.model_id == "catboost_fixed"


def test_bundle_rejects_invalid_manifest_features_type_and_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SelectedBundleError, match="Unable to read"):
        load_selected_bundle(tmp_path, trusted=True)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SelectedBundleError, match="Invalid selected bundle manifest"):
        load_selected_bundle(tmp_path, trusted=True)

    fitted = FittedSelectionModel("logistic_l2", Pipeline([]), "handling")
    _, model_sha = write_model_artifact(fitted, tmp_path)
    write_manifest(_manifest(model_sha).model_copy(update={"feature_order": ("wrong",)}), tmp_path)
    with pytest.raises(SelectedBundleError, match="feature and class order"):
        load_selected_bundle(tmp_path, trusted=True)

    write_manifest(_manifest(model_sha), tmp_path)
    monkeypatch.setattr(bundles.joblib, "load", lambda _path: object())
    with pytest.raises(SelectedBundleError, match="model type differs"):
        load_selected_bundle(tmp_path, trusted=True)

    with pytest.raises(SelectedBundleError, match="aligned vectors"):
        bundles.population_sha256(np.asarray([[1, 2]]), np.asarray([0, 1]))


def _manifest(
    model_sha: str,
    *,
    model_id: str = "logistic_l2",
    filename: str = "model.joblib",
    handling: str = "one_hot_status_standard_scale_monetary",
) -> BundleManifest:
    return BundleManifest(
        schema_version="1.0.0",
        bundle_id="selected_v1",
        selected_model_id=model_id,
        model_filename=filename,
        model_sha256=model_sha,
        selection_config_sha256="a" * 64,
        training_population_sha256="b" * 64,
        validation_population_sha256="c" * 64,
        validation_predictions_sha256="d" * 64,
        feature_order=PREDICTOR_COLUMNS,
        feature_handling=handling,
        class_order=(0, 1),
        fixed_parameters={"C": 1.0},
        dependencies={"joblib": "1.5.3"},
        git_commit="e" * 40,
        git_dirty=False,
        validation_metrics={"average_precision": 0.5},
        selection_outcome={"selected_model_id": model_id},
        calibration="identity",
        risk_band_thresholds={"q80": 0.2, "q90": 0.3, "q95": 0.4},
        fit_count=4,
        winner_refitted=False,
        holdout_evaluated=False,
        trusted_local_serialization=True,
    )
