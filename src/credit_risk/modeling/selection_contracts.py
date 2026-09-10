"""Strict contract for the governed one-pass model-selection workflow."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from credit_risk.modeling.contracts import (
    MONETARY_COLUMNS,
    PREDICTOR_COLUMNS,
    REPAYMENT_STATUS_COLUMNS,
)

DEFAULT_SELECTION_CONFIG_PATH = Path("configs/modeling/selection_v1.json")
MODEL_ORDER = ("logistic_l2", "random_forest", "hist_gradient_boosting", "catboost_fixed")
SIMPLICITY_ORDER = (
    "logistic_l2",
    "hist_gradient_boosting",
    "random_forest",
    "catboost_fixed",
)
DEPENDENCY_VERSIONS = {
    "catboost": "1.2.5",
    "joblib": "1.5.3",
    "mlflow": "3.15.0",
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "pandera": "0.32.1",
    "pydantic": "2.7.4",
    "scikit-learn": "1.4.2",
}


class SelectionContractError(RuntimeError):
    """Raised when the frozen one-pass protocol is invalid or incompatible."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())


class GovernanceContract(_FrozenModel):
    protocol_base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    official_run_requires_clean_commit: Literal[True]
    parameter_tuning: Literal["prohibited"]
    cross_validation_iteration: Literal["prohibited"]
    winner_refit: Literal["prohibited"]
    test_access_during_selection: Literal["prohibited"]


class HistoricalEvidenceContract(_FrozenModel):
    execution_status: Literal["superseded_no_rerun"]
    baseline_config_path: str
    baseline_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_summary_path: str
    baseline_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_config_path: str
    candidate_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_summary_path: str
    candidate_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "baseline_config_path",
        "baseline_summary_path",
        "candidate_config_path",
        "candidate_summary_path",
    )
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class SelectionDataContract(_FrozenModel):
    dataset_id: Literal["uci_credit_default"]
    dataset_version: Literal["v1"]
    feature_contract_path: str
    dataset_manifest_path: str
    split_config_path: str
    feature_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_split_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_rule: Literal["development_cv_fold_r0_not_0"]
    validation_rule: Literal["development_cv_fold_r0_equals_0"]
    training_rows: Literal[19200]
    training_target_counts: dict[Literal["0", "1"], int]
    validation_rows: Literal[4800]
    validation_target_counts: dict[Literal["0", "1"], int]
    sealed_test_rows: Literal[6000]
    holdout_access: Literal["prohibited"]

    @field_validator("feature_contract_path", "dataset_manifest_path", "split_config_path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_relative_path(value)

    @model_validator(mode="after")
    def exact_counts(self) -> SelectionDataContract:
        if self.training_target_counts != {"0": 14953, "1": 4247}:
            raise ValueError("training target counts differ from the reviewed split")
        if self.validation_target_counts != {"0": 3738, "1": 1062}:
            raise ValueError("validation target counts differ from the reviewed split")
        return self


class FeaturePolicy(_FrozenModel):
    predictor_policy: Literal["nineteen_operational_features"]
    demographic_predictors: Literal["prohibited"]
    imputation: Literal["none"]
    clipping: Literal["none"]
    resampling: Literal["none"]
    target_encoding: Literal["none"]
    feature_selection: Literal["none"]


class FixedModelContract(_FrozenModel):
    model_id: Literal["logistic_l2", "random_forest", "hist_gradient_boosting", "catboost_fixed"]
    kind: str
    parameters: dict[str, Any]
    feature_handling: str


class SelectionRule(_FrozenModel):
    fit_budget: Literal[4]
    fits_per_model: Literal[1]
    winner_refit: Literal[False]
    primary_metric: Literal["average_precision"]
    brier_guardrail_relative_to_logistic: float
    lift_at_0_1_guardrail_relative_to_logistic: float
    average_precision_equivalence_band: float
    simplicity_order: tuple[str, ...]
    calibration: Literal["identity"]
    bootstrap_resamples: Literal[500]
    bootstrap_random_state: Literal[42]
    risk_band_quantiles: tuple[float, ...]

    @model_validator(mode="after")
    def exact_selection(self) -> SelectionRule:
        if (
            self.brier_guardrail_relative_to_logistic != 0.005
            or self.lift_at_0_1_guardrail_relative_to_logistic != 0.1
            or self.average_precision_equivalence_band != 0.002
        ):
            raise ValueError("selection guardrails differ from the reviewed protocol")
        if self.simplicity_order != SIMPLICITY_ORDER:
            raise ValueError("simplicity order differs from the reviewed protocol")
        if self.risk_band_quantiles != (0.8, 0.9, 0.95):
            raise ValueError("risk-band quantiles must be 0.8, 0.9, and 0.95")
        return self


class TestGateDeltas(_FrozenModel):
    minimum_average_precision_delta: float
    maximum_brier_score_delta: float
    minimum_lift_at_0_1_delta: float
    required_rows: Literal[6000]

    @model_validator(mode="after")
    def exact_deltas(self) -> TestGateDeltas:
        if (
            self.minimum_average_precision_delta != -0.03
            or self.maximum_brier_score_delta != 0.02
            or self.minimum_lift_at_0_1_delta != -0.3
        ):
            raise ValueError("test-gate deltas differ from the reviewed protocol")
        return self


class EvidenceContract(_FrozenModel):
    deterministic_outputs: tuple[str, ...]
    runtime_outputs: tuple[str, ...]
    model_bundle: Literal["versioned_local_trusted"]
    forbidden_content: tuple[str, ...]


class SelectionConfig(_FrozenModel):
    schema_version: Literal["1.0.0"]
    protocol_id: Literal["selection_v1"]
    status: Literal["frozen_pre_selection"]
    purpose: Literal["one_pass_multi_model_release_selection"]
    governance: GovernanceContract
    historical_evidence: HistoricalEvidenceContract
    data: SelectionDataContract
    features: FeaturePolicy
    models: tuple[FixedModelContract, ...]
    selection: SelectionRule
    dependencies: dict[str, str]
    test_gate_deltas: TestGateDeltas
    evidence: EvidenceContract

    @model_validator(mode="after")
    def exact_protocol(self) -> SelectionConfig:
        if self.governance.protocol_base_commit != "1024fece1baa50ca7f9d4d1bd4ad516b32f707ce":
            raise ValueError("protocol base commit differs from the reviewed lineage")
        if (
            self.data.feature_contract_path != "configs/modeling/feature_contract_v1.json"
            or self.data.dataset_manifest_path != "configs/data/uci_credit_default_v1.json"
            or self.data.split_config_path != "configs/data/split_v1.json"
        ):
            raise ValueError("governed data paths differ from the reviewed protocol")
        if tuple(model.model_id for model in self.models) != MODEL_ORDER:
            raise ValueError(f"models must be ordered exactly as {MODEL_ORDER}")
        expected_kinds = {
            "logistic_l2": "logistic_regression_pipeline",
            "random_forest": "random_forest_classifier",
            "hist_gradient_boosting": "hist_gradient_boosting_classifier",
            "catboost_fixed": "catboost_classifier",
        }
        if any(model.kind != expected_kinds[model.model_id] for model in self.models):
            raise ValueError("model kinds differ from the reviewed protocol")
        expected_handling = {
            "logistic_l2": "one_hot_status_standard_scale_monetary",
            "random_forest": "validated_raw_numeric",
            "hist_gradient_boosting": "validated_raw_numeric",
            "catboost_fixed": "native_categorical_status_raw_monetary",
        }
        if any(
            model.feature_handling != expected_handling[model.model_id] for model in self.models
        ):
            raise ValueError("model feature handling differs from the reviewed protocol")
        expected_parameters: dict[str, dict[str, Any]] = {
            "logistic_l2": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 2000,
                "tol": 1e-8,
                "class_weight": None,
                "random_state": 42,
            },
            "random_forest": {
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": None,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight": None,
                "random_state": 42,
                "n_jobs": 4,
            },
            "hist_gradient_boosting": {
                "max_iter": 100,
                "learning_rate": 0.1,
                "max_leaf_nodes": 31,
                "l2_regularization": 0.0,
                "early_stopping": False,
                "class_weight": None,
                "random_state": 42,
            },
            "catboost_fixed": {
                "iterations": 300,
                "depth": 4,
                "learning_rate": 0.03,
                "l2_leaf_reg": 12.0,
                "random_strength": 0.0,
                "bagging_temperature": 0.0,
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "bootstrap_type": "Bayesian",
                "random_seed": 42,
                "thread_count": 4,
                "allow_writing_files": False,
                "verbose": False,
                "use_best_model": False,
                "class_weights": None,
            },
        }
        if any(model.parameters != expected_parameters[model.model_id] for model in self.models):
            raise ValueError("fixed model parameters differ from the reviewed protocol")
        if self.dependencies != DEPENDENCY_VERSIONS:
            raise ValueError("dependency versions differ from the reviewed protocol")
        if self.evidence.deterministic_outputs != (
            "summary.json",
            "selection-report.md",
            "manifest.json",
        ):
            raise ValueError("deterministic evidence allowlist differs from the protocol")
        if self.evidence.runtime_outputs != (
            "validation_predictions.csv",
            "bootstrap_intervals.json",
        ):
            raise ValueError("runtime evidence allowlist differs from the protocol")
        if self.evidence.forbidden_content != (
            "holdout_rows",
            "timestamps",
            "absolute_paths",
            "mlflow_identifiers",
        ):
            raise ValueError("forbidden evidence content differs from the protocol")
        return self

    @property
    def predictor_columns(self) -> tuple[str, ...]:
        return PREDICTOR_COLUMNS

    @property
    def status_columns(self) -> tuple[str, ...]:
        return REPAYMENT_STATUS_COLUMNS

    @property
    def monetary_columns(self) -> tuple[str, ...]:
        return MONETARY_COLUMNS


def load_selection_config(path: str | Path = DEFAULT_SELECTION_CONFIG_PATH) -> SelectionConfig:
    """Load the frozen contract and verify every referenced reviewed file."""

    config_path = Path(path)
    try:
        content = config_path.read_bytes()
        config = SelectionConfig.model_validate_json(content)
    except OSError as error:
        raise SelectionContractError(
            f"Unable to read selection config {config_path}: {error}"
        ) from error
    except ValidationError as error:
        raise SelectionContractError(f"Invalid selection config {config_path}: {error}") from error

    references = {
        config.data.feature_contract_path: config.data.feature_contract_sha256,
        config.historical_evidence.baseline_config_path: (
            config.historical_evidence.baseline_config_sha256
        ),
        config.historical_evidence.baseline_summary_path: (
            config.historical_evidence.baseline_summary_sha256
        ),
        config.historical_evidence.candidate_config_path: (
            config.historical_evidence.candidate_config_sha256
        ),
        config.historical_evidence.candidate_summary_path: (
            config.historical_evidence.candidate_summary_sha256
        ),
        "configs/data/split_v1.lock.json": config.data.reviewed_split_lock_sha256,
    }
    root = _repository_root(config_path)
    for relative, expected in references.items():
        candidate = root / relative
        try:
            observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError as error:
            raise SelectionContractError(
                f"Unable to verify reviewed file {candidate}: {error}"
            ) from error
        if observed != expected:
            raise SelectionContractError(
                f"Reviewed file digest mismatch for {relative}: expected={expected}, observed={observed}"
            )
    return config


def selection_config_sha256(path: str | Path = DEFAULT_SELECTION_CONFIG_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
        raise ValueError("configuration paths must be safe repository-relative paths")
    return value


def _repository_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise SelectionContractError(f"Unable to locate repository root from {config_path}")
