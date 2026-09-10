"""Checksum-protected local bundle for the exact validation winner."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from catboost import CatBoostClassifier
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.modeling.selection_models import FittedSelectionModel


class SelectedBundleError(RuntimeError):
    """Raised when a selected-model bundle is unsafe or incompatible."""


class BundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())

    schema_version: Literal["1.0.0"]
    bundle_id: Literal["selected_v1"]
    selected_model_id: Literal[
        "logistic_l2", "random_forest", "hist_gradient_boosting", "catboost_fixed"
    ]
    model_filename: Literal["model.joblib", "model.cbm"]
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_population_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_population_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_order: tuple[str, ...]
    feature_handling: str
    class_order: tuple[Literal[0, 1], Literal[0, 1]]
    fixed_parameters: dict[str, Any]
    dependencies: dict[str, str]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    validation_metrics: dict[str, Any]
    selection_outcome: dict[str, Any]
    calibration: Literal["identity"]
    risk_band_thresholds: dict[str, float]
    fit_count: Literal[4]
    winner_refitted: Literal[False]
    holdout_evaluated: Literal[False]
    trusted_local_serialization: Literal[True]

    @model_validator(mode="after")
    def compatible_bundle(self) -> BundleManifest:
        if self.feature_order != PREDICTOR_COLUMNS or self.class_order != (0, 1):
            raise ValueError("feature and class order differ from the release contract")
        expected_filename = (
            "model.cbm" if self.selected_model_id == "catboost_fixed" else "model.joblib"
        )
        if self.model_filename != expected_filename:
            raise ValueError("model filename differs from the selected model family")
        if self.selection_outcome.get("selected_model_id") != self.selected_model_id:
            raise ValueError("selection outcome differs from the bundled model")
        if set(self.risk_band_thresholds) != {"q80", "q90", "q95"}:
            raise ValueError("risk-band thresholds must contain q80, q90, and q95")
        thresholds = tuple(self.risk_band_thresholds[name] for name in ("q80", "q90", "q95"))
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in thresholds):
            raise ValueError("risk-band thresholds must be finite probabilities")
        if tuple(sorted(thresholds)) != thresholds:
            raise ValueError("risk-band thresholds must be ordered")
        return self


def write_model_artifact(model: FittedSelectionModel, destination: Path) -> tuple[Path, str]:
    """Serialize one already-fitted winner without calling fit."""

    destination.mkdir(parents=True, exist_ok=True)
    if model.model_id == "catboost_fixed":
        model_path = destination / "model.cbm"
        if not isinstance(model.estimator, CatBoostClassifier):
            raise SelectedBundleError("CatBoost winner has an unexpected estimator type.")
        model.estimator.save_model(str(model_path), format="cbm")
    else:
        model_path = destination / "model.joblib"
        expected_types = {
            "logistic_l2": Pipeline,
            "random_forest": RandomForestClassifier,
            "hist_gradient_boosting": HistGradientBoostingClassifier,
        }
        expected = expected_types.get(model.model_id)
        if expected is None or not isinstance(model.estimator, expected):
            raise SelectedBundleError(f"{model.model_id} winner has an unexpected estimator type.")
        joblib.dump(model.estimator, model_path, compress=3)
    return model_path, _sha256_file(model_path)


def write_manifest(manifest: BundleManifest, destination: Path) -> Path:
    path = destination / "manifest.json"
    path.write_bytes((json.dumps(manifest.model_dump(mode="json"), sort_keys=True) + "\n").encode())
    return path


def load_selected_bundle(
    bundle_root: str | Path,
    *,
    trusted: bool = False,
) -> tuple[BundleManifest, FittedSelectionModel]:
    """Load a digest-verified bundle; joblib requires explicit trust acknowledgement."""

    root = Path(bundle_root)
    try:
        manifest = BundleManifest.model_validate_json((root / "manifest.json").read_bytes())
    except OSError as error:
        raise SelectedBundleError(f"Unable to read selected bundle manifest: {error}") from error
    except ValidationError as error:
        raise SelectedBundleError(f"Invalid selected bundle manifest: {error}") from error
    if manifest.feature_order != PREDICTOR_COLUMNS or manifest.class_order != (0, 1):
        raise SelectedBundleError("Selected bundle feature or class order is incompatible.")
    expected_files = {"manifest.json", manifest.model_filename}
    try:
        observed_files = {path.name for path in root.iterdir() if path.is_file()}
    except OSError as error:
        raise SelectedBundleError(f"Unable to inspect selected bundle: {error}") from error
    if observed_files != expected_files:
        raise SelectedBundleError(
            f"Selected bundle files differ from the allowlist: expected={sorted(expected_files)}, "
            f"observed={sorted(observed_files)}"
        )
    model_path = root / manifest.model_filename
    observed_hash = _sha256_file(model_path)
    if observed_hash != manifest.model_sha256:
        raise SelectedBundleError(
            "Selected model digest mismatch: "
            f"expected={manifest.model_sha256}, observed={observed_hash}"
        )
    if manifest.model_filename == "model.cbm":
        estimator: Any = CatBoostClassifier()
        estimator.load_model(str(model_path), format="cbm")
    else:
        if not trusted:
            raise SelectedBundleError(
                "Joblib uses pickle semantics; load only trusted local bundles and pass trusted=True."
            )
        try:
            estimator = joblib.load(model_path)
        except Exception as error:
            raise SelectedBundleError(
                f"Unable to deserialize trusted joblib model: {error}"
            ) from error
        expected_types = {
            "logistic_l2": Pipeline,
            "random_forest": RandomForestClassifier,
            "hist_gradient_boosting": HistGradientBoostingClassifier,
        }
        expected = expected_types.get(manifest.selected_model_id)
        if expected is None or not isinstance(estimator, expected):
            raise SelectedBundleError("Trusted joblib model type differs from its manifest.")
    return manifest, FittedSelectionModel(
        model_id=manifest.selected_model_id,
        estimator=estimator,
        feature_handling=manifest.feature_handling,
    )


def population_sha256(account_ids: np.ndarray, target: np.ndarray) -> str:
    ids = np.asarray(account_ids, dtype="<i8")
    labels = np.asarray(target, dtype="i1")
    if ids.ndim != 1 or labels.ndim != 1 or len(ids) != len(labels):
        raise SelectedBundleError("Population IDs and targets must be aligned vectors.")
    digest = hashlib.sha256()
    digest.update(ids.tobytes())
    digest.update(labels.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise SelectedBundleError(f"Unable to hash bundle file {path}: {error}") from error
