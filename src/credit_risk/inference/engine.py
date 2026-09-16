"""Shared vectorised prediction and native-SHAP engine."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from credit_risk.inference.contracts import (
    DEFAULT_INFERENCE_CONFIG_PATH,
    REQUIRED_SERVING_DEPENDENCIES,
    InferenceConfig,
    load_inference_config,
)
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS, REPAYMENT_STATUS_COLUMNS
from credit_risk.modeling.risk_policy import RiskBand, risk_band
from credit_risk.modeling.selected_bundle import BundleManifest, load_selected_bundle


class InferenceError(RuntimeError):
    """Raised when governed prediction or explanation output is invalid."""


@dataclass(frozen=True, slots=True)
class ReasonAttribution:
    category: str
    direction: str
    contribution_raw_log_odds: float


@dataclass(frozen=True, slots=True)
class InferenceResult:
    probabilities: np.ndarray
    risk_bands: tuple[RiskBand, ...]
    reasons: tuple[tuple[ReasonAttribution, ...], ...]
    category_contributions: np.ndarray
    max_additivity_error: float
    max_probability_error: float


class InferenceEngine:
    """Load one reviewed bundle and expose one shared scoring implementation."""

    def __init__(
        self,
        bundle_root: str | Path = "models/selected_v1",
        config_path: str | Path = DEFAULT_INFERENCE_CONFIG_PATH,
    ) -> None:
        self.config = load_inference_config(config_path)
        root = Path(bundle_root)
        self._validate_bundle_files(root)
        self.manifest, self.model = load_selected_bundle(
            root,
            trusted=True,
            expected_manifest_sha256=self.config.bundle.manifest_sha256,
            required_dependencies=REQUIRED_SERVING_DEPENDENCIES,
        )
        if self.manifest.selected_model_id != self.config.bundle.model_id:
            raise InferenceError("Selected bundle model differs from the inference contract.")
        if self.manifest.risk_band_thresholds != self.config.prediction.risk_band_thresholds:
            raise InferenceError("Selected bundle risk thresholds differ from the contract.")
        if not isinstance(self.model.estimator, CatBoostClassifier):
            raise InferenceError("Phase 6 requires the reviewed CatBoost estimator.")

    def _validate_bundle_files(self, root: Path) -> None:
        expected = {
            root / "manifest.json": self.config.bundle.manifest_sha256,
            root / "model.cbm": self.config.bundle.model_sha256,
        }
        for path, digest in expected.items():
            try:
                observed = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                raise InferenceError(f"Unable to read reviewed bundle file: {error}") from error
            if observed != digest:
                raise InferenceError(
                    f"Reviewed bundle digest mismatch for {path.name}: "
                    f"expected={digest}, observed={observed}"
                )

    def score(self, features: pd.DataFrame) -> InferenceResult:
        """Score and explain one or more unique operational rows without fitting."""

        if not isinstance(features, pd.DataFrame) or features.empty:
            raise InferenceError("Inference features must be a non-empty DataFrame.")
        if tuple(features.columns) != PREDICTOR_COLUMNS or not features.index.is_unique:
            raise InferenceError("Inference features violate the ordered operational contract.")
        try:
            probabilities = np.asarray(self.model.predict_proba(features), dtype=np.float64)
        except Exception as error:
            raise InferenceError(f"Selected-model prediction failed: {error}") from error
        if probabilities.shape != (len(features),) or not np.isfinite(probabilities).all():
            raise InferenceError(
                "Selected model returned invalid probability dimensions or values."
            )
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise InferenceError("Selected model returned probability outside [0, 1].")

        prepared = features.copy()
        for column in REPAYMENT_STATUS_COLUMNS:
            prepared[column] = prepared[column].astype("int64").astype(str)
        pool = Pool(prepared, cat_features=list(REPAYMENT_STATUS_COLUMNS))
        estimator = self.model.estimator
        try:
            native = np.asarray(
                estimator.get_feature_importance(pool, type="ShapValues"), dtype=np.float64
            )
            raw_scores = np.asarray(
                estimator.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64
            ).reshape(-1)
        except Exception as error:
            raise InferenceError(f"Native SHAP calculation failed: {error}") from error
        expected_shape = (len(features), self.config.explanation.shap_output_columns)
        if native.shape != expected_shape or raw_scores.shape != (len(features),):
            raise InferenceError(
                f"Native SHAP output shape must be {expected_shape}, observed {native.shape}."
            )
        if not np.isfinite(native).all() or not np.isfinite(raw_scores).all():
            raise InferenceError("Native SHAP values and raw scores must be finite.")

        contributions = native[:, :-1]
        reconstructed = contributions.sum(axis=1) + native[:, -1]
        additivity_error = float(np.max(np.abs(reconstructed - raw_scores)))
        if additivity_error > self.config.explanation.additivity_absolute_tolerance:
            raise InferenceError(
                f"Native SHAP additivity failed: maximum_error={additivity_error:.17g}"
            )
        reconstructed_probability = _sigmoid(raw_scores)
        probability_error = float(np.max(np.abs(reconstructed_probability - probabilities)))
        if probability_error > self.config.explanation.probability_absolute_tolerance:
            raise InferenceError(
                f"Native SHAP probability parity failed: maximum_error={probability_error:.17g}"
            )

        positions = {name: index for index, name in enumerate(PREDICTOR_COLUMNS)}
        categories = tuple(sorted(self.config.explanation.reason_categories))
        category_values = np.column_stack(
            [
                contributions[
                    :, [positions[name] for name in self.config.explanation.reason_categories[key]]
                ].sum(axis=1)
                for key in categories
            ]
        )
        reasons = tuple(
            _top_reasons(categories, category_values[row], self.config)
            for row in range(len(features))
        )
        bands = tuple(
            risk_band(float(probability), self.config.prediction.risk_band_thresholds)
            for probability in probabilities
        )
        return InferenceResult(
            probabilities=probabilities,
            risk_bands=bands,
            reasons=reasons,
            category_contributions=category_values,
            max_additivity_error=additivity_error,
            max_probability_error=probability_error,
        )

    @property
    def bundle_manifest(self) -> BundleManifest:
        return self.manifest


def _top_reasons(
    categories: tuple[str, ...], values: np.ndarray, config: InferenceConfig
) -> tuple[ReasonAttribution, ...]:
    ordered = sorted(
        zip(categories, values, strict=True),
        key=lambda item: (-abs(float(item[1])), item[0]),
    )[: config.explanation.top_reason_count]
    labels = config.explanation.direction_labels
    return tuple(
        ReasonAttribution(
            category=category,
            direction=(
                labels["positive"]
                if float(value) > 0.0
                else labels["negative"]
                if float(value) < 0.0
                else "neutral"
            ),
            contribution_raw_log_odds=float(value),
        )
        for category, value in ordered
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    if not all(math.isfinite(float(value)) for value in result):
        raise InferenceError("Sigmoid reconstruction returned a non-finite value.")
    return result
