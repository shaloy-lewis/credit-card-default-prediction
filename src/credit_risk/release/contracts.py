"""Frozen contract for the Release A audit dossier."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_RELEASE_CONFIG_PATH = Path("configs/releases/release_a_v1.json")
DEFAULT_UNCERTAINTY_SOURCE = Path(
    "experiment/mlflow/selection-runtime/"
    "f7c99f257fe756f6db6bac449a7ef4f48a899ea4/bootstrap_intervals.json"
)

SOURCE_ARTIFACT_ROLES = frozenset(
    {
        "data_manifest",
        "split_lock",
        "split_config",
        "feature_contract",
        "baseline_config",
        "baseline_summary",
        "baseline_report",
        "selection_config",
        "selection_summary",
        "selection_report",
        "bundle_manifest",
        "bundle_model",
        "final_test_authorization",
        "final_test_approval",
        "final_test_started_receipt",
        "final_test_completed_receipt",
        "final_test_summary",
        "final_test_report",
        "executed_evaluator_source",
    }
)
RELEASE_CRITERIA = (
    "reproducible_data_and_split_protocol",
    "fixed_four_model_comparison_and_exact_serialized_winner",
    "identity_calibration_prediction_only_uncertainty_and_capacity_evaluation",
    "single_authorized_final_test_with_frozen_gates_and_no_rerun",
    "unsupported_claims_prohibited",
)
RELEASE_OUTPUTS = (
    "summary.json",
    "release-a-report.md",
    "validation-uncertainty.json",
    "evidence-manifest.json",
)


class ReleaseContractError(RuntimeError):
    """Raised when the frozen Release A contract is invalid or altered."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())


class ArtifactReference(_FrozenModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class UncertaintySource(ArtifactReference):
    population: Literal["development_validation_only"]
    method: Literal["stratified_prediction_only_percentile_bootstrap"]
    confidence_level: float = Field(ge=0.95, le=0.95)
    resamples: Literal[500]
    random_state: Literal[42]


class ReleaseGovernance(_FrozenModel):
    full_dataset_integrity_verification_permitted: Literal[True]
    model_loading: Literal["prohibited"]
    prediction: Literal["prohibited"]
    training: Literal["prohibited"]
    refitting: Literal["prohibited"]
    parameter_tuning: Literal["prohibited"]
    cross_validation: Literal["prohibited"]
    calibration_fitting: Literal["prohibited"]
    bootstrap_generation: Literal["prohibited"]
    test_partition_selection: Literal["prohibited"]
    test_predictions_loading: Literal["prohibited"]
    final_test_reevaluation: Literal["prohibited"]
    stress_evidence: Literal["deferred_to_g4_release_b"]


class ReleaseAConfig(_FrozenModel):
    schema_version: Literal["1.0.0"]
    release_id: Literal["release_a_v1"]
    milestone: Literal["defensible_model"]
    status: Literal["frozen_for_audit_closure"]
    source_artifacts: dict[str, ArtifactReference]
    uncertainty_source: UncertaintySource
    release_criteria: tuple[str, ...]
    governance: ReleaseGovernance
    outputs: tuple[str, ...]

    @model_validator(mode="after")
    def exact_release_contract(self) -> ReleaseAConfig:
        if set(self.source_artifacts) != SOURCE_ARTIFACT_ROLES:
            raise ValueError("Release A source artifact roles differ from the reviewed allowlist")
        if self.release_criteria != RELEASE_CRITERIA:
            raise ValueError("Release A criteria differ from the reviewed milestone")
        if self.outputs != RELEASE_OUTPUTS:
            raise ValueError("Release A outputs differ from the reviewed allowlist")
        if self.uncertainty_source.path != DEFAULT_UNCERTAINTY_SOURCE.as_posix():
            raise ValueError("Release A uncertainty source differs from the reviewed runtime file")
        paths = [reference.path for reference in self.source_artifacts.values()]
        if len(paths) != len(set(paths)):
            raise ValueError("Release A source artifact paths must be unique")
        return self


def load_release_config(path: str | Path = DEFAULT_RELEASE_CONFIG_PATH) -> ReleaseAConfig:
    """Load and strictly validate the frozen Release A configuration."""

    candidate = Path(path)
    try:
        return ReleaseAConfig.model_validate_json(candidate.read_bytes(), strict=True)
    except (OSError, ValidationError, UnicodeError, ValueError) as error:
        raise ReleaseContractError(
            f"Invalid Release A configuration {candidate}: {error}"
        ) from error


def release_config_sha256(path: str | Path = DEFAULT_RELEASE_CONFIG_PATH) -> str:
    """Return the byte-level digest of a Release A configuration."""

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise ReleaseContractError(f"Unable to hash Release A configuration: {error}") from error


def _safe_relative_path(value: str) -> str:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or "\\" in value
    ):
        raise ValueError("release paths must be safe repository-relative paths")
    return value
