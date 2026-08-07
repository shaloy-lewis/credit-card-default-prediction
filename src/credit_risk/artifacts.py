"""Loading and validation for the trusted inference artifacts."""

import json
import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from credit_risk.utils.constants import OUTLIER_COLUMNS

MODEL_FILENAME = "model.pkl"
PREPROCESSOR_FILENAME = "preprocessor.pkl"
THRESHOLD_FILENAME = "outlier_threshold.json"
REQUIRED_ARTIFACT_FILENAMES = (
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
    THRESHOLD_FILENAME,
)


class ArtifactValidationError(RuntimeError):
    """Raised when inference artifacts are missing, corrupt, or incompatible."""


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """A validated, immutable container for the inference artifacts."""

    model: Any
    preprocessor: Any
    outlier_threshold: Mapping[str, Mapping[str, float]]
    transformed_feature_names: tuple[str, ...]


def load_artifact_bundle(artifact_dir: str | Path) -> ArtifactBundle:
    """Load and validate trusted, local inference artifacts from ``artifact_dir``.

    Pickle can execute arbitrary code while loading. Callers must only pass an
    artifact directory whose contents they trust.
    """

    directory = Path(artifact_dir)
    paths = {filename: directory / filename for filename in REQUIRED_ARTIFACT_FILENAMES}
    missing = [filename for filename, path in paths.items() if not path.is_file()]
    if missing:
        missing_list = ", ".join(missing)
        raise ArtifactValidationError(
            f"Artifact directory '{directory}' is missing required file(s): {missing_list}."
        )

    model = _load_trusted_pickle(paths[MODEL_FILENAME], "model")
    preprocessor = _load_trusted_pickle(paths[PREPROCESSOR_FILENAME], "preprocessor")
    thresholds = _load_thresholds(paths[THRESHOLD_FILENAME])
    transformed_feature_names = _validate_model_and_preprocessor(model, preprocessor)

    return ArtifactBundle(
        model=model,
        preprocessor=preprocessor,
        outlier_threshold=thresholds,
        transformed_feature_names=transformed_feature_names,
    )


def _load_trusted_pickle(path: Path, artifact_name: str) -> Any:
    try:
        with path.open("rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as exc:
        raise ArtifactValidationError(
            f"Could not deserialize {artifact_name} artifact '{path}'. "
            "Confirm that it is a valid, trusted pickle created with compatible dependencies. "
            f"Original error: {exc}"
        ) from exc


def _load_thresholds(path: Path) -> Mapping[str, Mapping[str, float]]:
    try:
        with path.open(encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"Could not read outlier thresholds from '{path}' as valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ArtifactValidationError(
            f"Outlier threshold artifact '{path}' must contain a JSON object."
        )

    expected_sections = {"low_perc", "high_perc"}
    actual_sections = set(payload)
    if actual_sections != expected_sections:
        raise ArtifactValidationError(
            _schema_difference_message(
                f"Outlier threshold artifact '{path}'",
                actual_sections,
                expected_sections,
            )
        )

    validated: dict[str, Mapping[str, float]] = {}
    expected_columns = set(OUTLIER_COLUMNS)
    for section in ("low_perc", "high_perc"):
        values = payload[section]
        if not isinstance(values, dict):
            raise ArtifactValidationError(
                f"Outlier threshold section '{section}' in '{path}' must be a JSON object."
            )

        actual_columns = set(values)
        if actual_columns != expected_columns:
            raise ArtifactValidationError(
                _schema_difference_message(
                    f"Outlier threshold section '{section}' in '{path}'",
                    actual_columns,
                    expected_columns,
                )
            )

        section_values: dict[str, float] = {}
        for column in OUTLIER_COLUMNS:
            value = values[column]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ArtifactValidationError(
                    f"Outlier threshold '{section}.{column}' in '{path}' must be numeric."
                )
            try:
                numeric_value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ArtifactValidationError(
                    f"Outlier threshold '{section}.{column}' in '{path}' must be finite."
                ) from exc
            if not math.isfinite(numeric_value):
                raise ArtifactValidationError(
                    f"Outlier threshold '{section}.{column}' in '{path}' must be finite."
                )
            section_values[column] = numeric_value

        validated[section] = MappingProxyType(section_values)

    low_values = validated["low_perc"]
    high_values = validated["high_perc"]
    invalid_bounds = [
        column for column in OUTLIER_COLUMNS if low_values[column] > high_values[column]
    ]
    if invalid_bounds:
        raise ArtifactValidationError(
            f"Outlier threshold artifact '{path}' has lower bounds greater than upper bounds "
            f"for: {', '.join(invalid_bounds)}."
        )

    return MappingProxyType(validated)


def _schema_difference_message(subject: str, actual: set[str], expected: set[str]) -> str:
    details: list[str] = []
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        details.append(f"missing keys: {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected keys: {', '.join(unexpected)}")
    return f"{subject} has an invalid schema ({'; '.join(details)})."


def _validate_model_and_preprocessor(model: Any, preprocessor: Any) -> tuple[str, ...]:
    _require_methods(preprocessor, "Preprocessor", ("transform", "get_feature_names_out"))
    _require_methods(model, "Model", ("predict_proba", "get_feature_importance"))

    try:
        preprocessor_names = tuple(preprocessor.get_feature_names_out())
    except Exception as exc:
        raise ArtifactValidationError(
            "Preprocessor could not provide transformed feature names; confirm that it is fitted. "
            f"Original error: {exc}"
        ) from exc

    if not preprocessor_names or not all(isinstance(name, str) for name in preprocessor_names):
        raise ArtifactValidationError(
            "Preprocessor transformed feature names must be a non-empty sequence of strings."
        )

    try:
        model_names = tuple(model.feature_names_)
    except Exception as exc:
        raise ArtifactValidationError(
            "Model must expose transformed feature names through 'feature_names_'."
        ) from exc

    if model_names != preprocessor_names:
        raise ArtifactValidationError(
            "Model feature_names_ do not exactly match the preprocessor's transformed feature names."
        )

    try:
        classes = np.asarray(model.classes_)
    except Exception as exc:
        raise ArtifactValidationError(
            "Model must expose fitted classes through 'classes_'."
        ) from exc
    if classes.ndim != 1 or classes.shape != (2,) or not np.array_equal(classes, [0, 1]):
        raise ArtifactValidationError(
            "Model classes_ must be the binary classes [0, 1] in that order."
        )

    try:
        feature_importances = np.asarray(model.feature_importances_)
    except Exception as exc:
        raise ArtifactValidationError(
            "Model must expose global feature importance through 'feature_importances_'."
        ) from exc
    if feature_importances.ndim != 1 or feature_importances.shape[0] != len(preprocessor_names):
        raise ArtifactValidationError(
            "Model feature_importances_ must be one-dimensional and have one value per "
            f"transformed feature (expected {len(preprocessor_names)})."
        )

    return preprocessor_names


def _require_methods(obj: Any, artifact_name: str, method_names: tuple[str, ...]) -> None:
    missing = [name for name in method_names if not callable(getattr(obj, name, None))]
    if missing:
        raise ArtifactValidationError(
            f"{artifact_name} artifact is missing required callable method(s): {', '.join(missing)}."
        )
