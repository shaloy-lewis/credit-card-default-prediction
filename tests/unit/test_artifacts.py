"""Unit tests for the trusted inference-artifact contract."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from credit_risk.artifacts import ArtifactValidationError, load_artifact_bundle
from credit_risk.utils.constants import OUTLIER_COLUMNS


class FakePreprocessor:
    def __init__(self, feature_names: tuple[str, ...] = ("feature",)) -> None:
        self.feature_names = feature_names

    def transform(self, features):
        return features

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(self.feature_names)


class FakeModel:
    def __init__(self, feature_names: tuple[str, ...] = ("feature",)) -> None:
        self.feature_names_ = list(feature_names)
        self.classes_ = np.asarray([0, 1])
        self.feature_importances_ = np.ones(len(feature_names))

    def predict_proba(self, features) -> np.ndarray:
        return np.asarray([[0.5, 0.5]])

    def get_feature_importance(self, *args, **kwargs) -> np.ndarray:
        return self.feature_importances_


def valid_thresholds() -> dict[str, dict[str, float]]:
    return {
        "low_perc": {column: 0.0 for column in OUTLIER_COLUMNS},
        "high_perc": {column: 1.0 for column in OUTLIER_COLUMNS},
    }


def write_artifacts(
    artifact_dir: Path,
    *,
    model: object | None = None,
    preprocessor: object | None = None,
    thresholds: object | None = None,
) -> None:
    (artifact_dir / "model.pkl").write_bytes(pickle.dumps(model or FakeModel()))
    (artifact_dir / "preprocessor.pkl").write_bytes(
        pickle.dumps(preprocessor or FakePreprocessor())
    )
    (artifact_dir / "outlier_threshold.json").write_text(
        json.dumps(thresholds if thresholds is not None else valid_thresholds()),
        encoding="utf-8",
    )


def test_load_artifact_bundle_accepts_matching_artifacts(tmp_path: Path) -> None:
    write_artifacts(tmp_path)

    bundle = load_artifact_bundle(tmp_path)

    assert bundle.transformed_feature_names == ("feature",)
    assert bundle.model.classes_.tolist() == [0, 1]


def test_load_artifact_bundle_lists_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError, match="model.pkl.*preprocessor.pkl"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_rejects_corrupt_pickle(tmp_path: Path) -> None:
    write_artifacts(tmp_path)
    (tmp_path / "model.pkl").write_text("not a pickle", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="Could not deserialize model"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_rejects_malformed_threshold_json(tmp_path: Path) -> None:
    write_artifacts(tmp_path)
    (tmp_path / "outlier_threshold.json").write_text("{", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="valid JSON"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_rejects_missing_threshold_section(tmp_path: Path) -> None:
    thresholds = valid_thresholds()
    del thresholds["high_perc"]
    write_artifacts(tmp_path, thresholds=thresholds)

    with pytest.raises(ArtifactValidationError, match="missing keys: high_perc"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_rejects_unexpected_threshold_section(tmp_path: Path) -> None:
    thresholds = valid_thresholds()
    thresholds["unexpected"] = {}
    write_artifacts(tmp_path, thresholds=thresholds)

    with pytest.raises(ArtifactValidationError, match="unexpected keys: unexpected"):
        load_artifact_bundle(tmp_path)


@pytest.mark.parametrize("invalid_value", [True, "1", float("nan"), float("inf")])
def test_load_artifact_bundle_rejects_non_numeric_or_non_finite_thresholds(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    thresholds = valid_thresholds()
    thresholds["low_perc"][OUTLIER_COLUMNS[0]] = invalid_value  # type: ignore[assignment]
    write_artifacts(tmp_path, thresholds=thresholds)

    with pytest.raises(ArtifactValidationError, match="must be (numeric|finite)"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_rejects_inverted_threshold_bounds(tmp_path: Path) -> None:
    thresholds = valid_thresholds()
    thresholds["low_perc"][OUTLIER_COLUMNS[0]] = 2.0
    write_artifacts(tmp_path, thresholds=thresholds)

    with pytest.raises(ArtifactValidationError, match="lower bounds greater than upper bounds"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_rejects_feature_name_mismatch(tmp_path: Path) -> None:
    write_artifacts(
        tmp_path,
        model=FakeModel(("model_feature",)),
        preprocessor=FakePreprocessor(("preprocessor_feature",)),
    )

    with pytest.raises(ArtifactValidationError, match="do not exactly match"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_requires_model_methods(tmp_path: Path) -> None:
    write_artifacts(tmp_path, model=object())

    with pytest.raises(ArtifactValidationError, match="missing required callable method"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_rejects_non_binary_or_reordered_classes(tmp_path: Path) -> None:
    model = FakeModel()
    model.classes_ = np.asarray([1, 0])
    write_artifacts(tmp_path, model=model)

    with pytest.raises(ArtifactValidationError, match=r"binary classes \[0, 1\]"):
        load_artifact_bundle(tmp_path)


def test_load_artifact_bundle_rejects_feature_importance_width_mismatch(
    tmp_path: Path,
) -> None:
    model = FakeModel()
    model.feature_importances_ = np.ones(2)
    write_artifacts(tmp_path, model=model)

    with pytest.raises(ArtifactValidationError, match="one value per transformed feature"):
        load_artifact_bundle(tmp_path)
