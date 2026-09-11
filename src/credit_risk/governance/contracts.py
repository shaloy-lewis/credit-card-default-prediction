"""Frozen Phase 5 governance contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from credit_risk.modeling.contracts import (
    AUDIT_COLUMNS,
    FORBIDDEN_PREDICTOR_COLUMNS,
    PREDICTOR_COLUMNS,
)

DEFAULT_GOVERNANCE_CONFIG_PATH = Path("configs/governance/phase5_v1.json")
OFFICIAL_PHASE5_CONFIG_SHA256 = "1717abd20e5dad6819d2f67fc13eecfa38a8800decf9e6954ffd0c74a913f68c"


class GovernanceContractError(ValueError):
    """Raised when the Phase 5 contract is invalid or altered."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class PopulationContract(_FrozenModel):
    assignment_column: Literal["cv_fold_r0"]
    development_rows: Literal[24000]
    partition: Literal["development_validation_only"]
    rows: Literal[4800]
    target_counts: dict[str, int]
    validation_fold: Literal[0]


class TestBoundaryContract(_FrozenModel):
    full_dataset_integrity_verification: Literal["required"]
    test_explanation_generation: Literal["prohibited"]
    test_partition_return: Literal["prohibited"]
    test_partition_selection: Literal["prohibited"]
    test_prediction_generation: Literal["prohibited"]
    test_prediction_loading: Literal["prohibited"]
    test_subgroup_analysis: Literal["prohibited"]


class FeatureContract(_FrozenModel):
    predictor_columns: tuple[str, ...]
    audit_columns: tuple[str, ...]
    forbidden_predictor_columns: tuple[str, ...]
    demographic_policy: Literal["audit_only_excluded_from_estimator"]


class PredictionContract(_FrozenModel):
    bundle_id: Literal["selected_v1"]
    calibration: Literal["identity"]
    expected_validation_metrics: dict[str, float]
    metric_absolute_tolerance: float = Field(gt=0.0, le=1e-12)
    model_id: Literal["catboost_fixed"]
    q90: float = Field(ge=0.0, le=1.0)


class ExplanationContract(_FrozenModel):
    additivity_absolute_tolerance: float = Field(gt=0.0, le=1e-10)
    direction_labels: dict[str, str]
    language_policy: Literal["model_attribution_not_causal_or_adverse_action_reason"]
    method: Literal["catboost_native_shap_values"]
    probability_absolute_tolerance: float = Field(gt=0.0, le=1e-10)
    reason_categories: dict[str, tuple[str, ...]]
    sample_rows: Literal[1000]
    sampling_algorithm: Literal["StratifiedShuffleSplit"]
    sampling_seed: Literal[42]
    shap_output_columns: Literal[20]
    space: Literal["raw_log_odds"]
    stratification: tuple[str, ...]


class SupportContract(_FrozenModel):
    minimum_negative_labels: int = Field(ge=1)
    minimum_positive_labels: int = Field(ge=1)
    minimum_rows: int = Field(ge=1)
    unsupported_status: Literal["insufficient_support"]


class BootstrapContract(_FrozenModel):
    confidence_level: float = Field(ge=0.95, le=0.95)
    metrics: tuple[str, ...]
    method: Literal["within_group_stratified_percentile"]
    resamples: Literal[500]
    seed: Literal[42]


class PrevalenceIntervalContract(_FrozenModel):
    confidence_level: float = Field(ge=0.95, le=0.95)
    method: Literal["wilson_score"]
    z_value: float = Field(ge=1.959963984540054, le=1.959963984540054)


class TriggerContract(_FrozenModel):
    absolute_calibration_in_the_large: float = Field(ge=0.0)
    absolute_false_positive_rate_gap: float = Field(ge=0.0)
    absolute_true_positive_rate_gap: float = Field(ge=0.0)
    maximum_brier_degradation: float = Field(ge=0.0)
    selection_rate_ratio_lower: float = Field(gt=0.0)
    selection_rate_ratio_upper: float = Field(gt=0.0)


class AxisContract(_FrozenModel):
    source_column: str
    groups: dict[str, tuple[int, ...]] | None = None
    boundaries: tuple[int, ...] | None = None
    labels: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def one_mapping_form(self) -> AxisContract:
        grouped = self.groups is not None
        banded = self.boundaries is not None or self.labels is not None
        if grouped == banded:
            raise ValueError("an audit axis must use exactly one mapping form")
        if banded and (
            self.boundaries is None
            or self.labels is None
            or len(self.boundaries) != len(self.labels) + 1
            or tuple(sorted(self.boundaries)) != self.boundaries
        ):
            raise ValueError("age-band boundaries and labels are invalid")
        return self


class FairnessContract(_FrozenModel):
    axes: dict[str, AxisContract]
    bootstrap: BootstrapContract
    metrics: tuple[str, ...]
    policy_threshold: Literal["q90"]
    prevalence_interval: PrevalenceIntervalContract
    support: SupportContract
    triggers: TriggerContract


class ExpectedTrigger(_FrozenModel):
    axis: str
    direction: Literal["below_lower_bound", "above_upper_bound"]
    group: str
    metric: Literal["selection_rate_ratio"]


class ReviewContract(_FrozenModel):
    disposition: tuple[str, ...]
    expected_triggers: tuple[ExpectedTrigger, ...]
    g3_result: Literal["closed_with_conditions"]
    trigger_policy: Literal["human_review_not_automatic_rejection"]


class OutputContract(_FrozenModel):
    committed: tuple[str, ...]
    runtime: tuple[str, ...]


class GovernanceConfig(_FrozenModel):
    schema_version: Literal["1.0.0"]
    governance_id: Literal["phase5_v1"]
    status: Literal["frozen_before_official_evidence"]
    dependencies: dict[str, str]
    explanation: ExplanationContract
    fairness: FairnessContract
    features: FeatureContract
    lineage: dict[str, str]
    outputs: OutputContract
    population: PopulationContract
    prediction: PredictionContract
    prohibitions: tuple[str, ...]
    review: ReviewContract
    test_boundary: TestBoundaryContract

    @model_validator(mode="after")
    def frozen_semantics(self) -> GovernanceConfig:
        if self.features.predictor_columns != PREDICTOR_COLUMNS:
            raise ValueError("predictors differ from the 19-feature operational contract")
        if self.features.audit_columns != AUDIT_COLUMNS:
            raise ValueError("audit columns differ from the demographic review contract")
        if self.features.forbidden_predictor_columns != FORBIDDEN_PREDICTOR_COLUMNS:
            raise ValueError("forbidden predictor columns differ from the feature contract")
        if self.population.target_counts != {"0": 3738, "1": 1062}:
            raise ValueError("validation target counts differ from the reviewed split")
        if set(self.dependencies) != {
            "catboost",
            "joblib",
            "numpy",
            "pandas",
            "pydantic",
            "scikit-learn",
        }:
            raise ValueError("dependency boundary differs from the selected-bundle runtime")
        category_features = tuple(
            feature
            for category in self.explanation.reason_categories.values()
            for feature in category
        )
        if len(category_features) != 19 or set(category_features) != set(PREDICTOR_COLUMNS):
            raise ValueError("explanation categories must cover every predictor exactly once")
        if self.explanation.direction_labels != {
            "negative": "risk_mitigating",
            "positive": "risk_increasing",
        }:
            raise ValueError("explanation direction labels differ from the reviewed language")
        if self.explanation.stratification != ("target", "risk_band"):
            raise ValueError("explanation sampling must stratify by target and risk band")
        if set(self.fairness.axes) != {
            "sex_code",
            "education_code",
            "marital_status_code",
            "age_band",
        }:
            raise ValueError("audit axes differ from the frozen governance protocol")
        required_prohibitions = {
            "training",
            "refitting",
            "parameter_tuning",
            "cross_validation",
            "calibration_fitting",
            "test_partition_selection",
            "test_partition_return",
            "test_prediction_generation",
            "test_prediction_loading",
            "final_test_prediction_loading",
            "test_explanation_generation",
            "test_subgroup_analysis",
            "fairness_certification",
            "regulatory_compliance_claim",
        }
        if not required_prohibitions.issubset(self.prohibitions):
            raise ValueError("required no-training/no-test prohibitions are missing")
        if self.fairness.bootstrap.metrics != (
            "mean_probability",
            "calibration_in_the_large",
            "brier_score",
            "selection_rate_at_q90",
            "true_positive_rate_at_q90",
            "false_positive_rate_at_q90",
        ):
            raise ValueError("stratified bootstrap metrics differ from the reviewed contract")
        if self.outputs.committed != (
            "summary.json",
            "governance-report.md",
            "fairness-report.md",
            "model-card.md",
            "risk-register.md",
            "g3-review.md",
            "evidence-manifest.json",
        ) or self.outputs.runtime != (
            "validation_predictions.csv",
            "sampled_shap_values.csv",
            "subgroup_bootstrap.json",
        ):
            raise ValueError("evidence allowlist differs from the Phase 5 contract")
        return self


def load_governance_config(
    path: str | Path = DEFAULT_GOVERNANCE_CONFIG_PATH,
    *,
    require_official_digest: bool = True,
) -> GovernanceConfig:
    """Read and validate the frozen Phase 5 configuration."""

    source = Path(path)
    try:
        content = source.read_bytes()
    except OSError as error:
        raise GovernanceContractError(
            f"Unable to read governance config {source}: {error}"
        ) from error
    digest = hashlib.sha256(content).hexdigest()
    if require_official_digest and digest != OFFICIAL_PHASE5_CONFIG_SHA256:
        raise GovernanceContractError(
            "Governance config digest differs from the reviewed protocol: "
            f"expected={OFFICIAL_PHASE5_CONFIG_SHA256}, observed={digest}"
        )
    try:
        return GovernanceConfig.model_validate_json(content)
    except ValidationError as error:
        raise GovernanceContractError(f"Invalid governance config {source}: {error}") from error


def governance_config_sha256(path: str | Path = DEFAULT_GOVERNANCE_CONFIG_PATH) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise GovernanceContractError(
            f"Unable to hash governance config {path}: {error}"
        ) from error
