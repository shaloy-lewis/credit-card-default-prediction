"""Digest-verified serving adapter for the reviewed selected-model bundle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk.modeling.risk_policy import RiskBand, risk_band
from credit_risk.modeling.selected_bundle import (
    BundleManifest,
    SelectedBundleError,
    load_selected_bundle,
)

SELECTED_V1_MANIFEST_SHA256 = "df5ce6ce07b268f57fa3bf72c97cd32f8ebb66695d7157139942c91e46d7cd88"
SERVING_DEPENDENCIES = (
    "catboost",
    "joblib",
    "numpy",
    "pandas",
    "pydantic",
    "scikit-learn",
)


class SelectedPredictPipeline:
    """Load one reviewed bundle and expose prediction plus frozen risk policy."""

    def __init__(self, bundle_root: str | Path = "models/selected_v1") -> None:
        self.bundle_root = Path(bundle_root)
        self.manifest, self.model = load_selected_bundle(
            self.bundle_root,
            trusted=True,
            expected_manifest_sha256=SELECTED_V1_MANIFEST_SHA256,
            required_dependencies=SERVING_DEPENDENCIES,
        )
        if self.manifest.selected_model_id != "catboost_fixed":
            raise SelectedBundleError(
                "The selected_v1 serving release requires the reviewed catboost_fixed winner."
            )

    def predict(self, features: pd.DataFrame) -> tuple[float, RiskBand]:
        probabilities = self.model.predict_proba(features)
        if len(probabilities) != 1:
            raise ValueError("Online prediction requires exactly one account per request.")
        probability = float(probabilities[0])
        return probability, risk_band(probability, self.manifest.risk_band_thresholds)

    @property
    def bundle_manifest(self) -> BundleManifest:
        return self.manifest
