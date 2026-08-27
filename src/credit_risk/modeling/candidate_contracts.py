"""Strict contracts for the frozen Phase 3 CatBoost experiment."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from sklearn.model_selection import ParameterSampler

from credit_risk.modeling.contracts import (
    AUDIT_COLUMNS,
    ID_COLUMN,
    MONETARY_COLUMNS,
    OFFICIAL_ASSIGNMENT_SHA256,
    OFFICIAL_DATASET_ID,
    OFFICIAL_DATASET_VERSION,
    OFFICIAL_DEVELOPMENT_ROWS,
    PREDICTOR_COLUMNS,
    REPAYMENT_STATUS_COLUMNS,
    TARGET_COLUMN,
    ModelingContractError,
    parse_feature_contract,
)

DEFAULT_CANDIDATE_CONFIG_PATH = Path("configs/modeling/candidate_v1.json")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class BaselineReferenceMetrics(_FrozenModel):
    average_precision_mean: float = Field(ge=0.0, le=1.0)
    average_precision_standard_deviation: float = Field(ge=0.0)
    brier_score_mean: float = Field(ge=0.0, le=1.0)
    lift_at_0_1_mean: float = Field(gt=0.0)
    roc_auc_mean: float = Field(ge=0.0, le=1.0)


class BaselineEvidence(_FrozenModel):
    summary_path: str
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_path: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_model_id: Literal["logistic_l2"]
    reference_metrics: BaselineReferenceMetrics

    @field_validator("summary_path", "report_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _safe_relative_path(value, suffixes={".json", ".md"})


class CandidateCrossValidation(_FrozenModel):
    n_splits: Literal[5]
    n_repeats: Literal[3]
    random_state: Literal[42]
    assignment_source: Literal["sealed_phase_1_assignments"]


class CandidateDataContract(_FrozenModel):
    dataset_id: Literal["uci_credit_default"]
    dataset_version: Literal["v1"]
    partition: Literal["development"]
    expected_rows: Literal[24000]
    id_column: Literal["account_id"]
    target_column: Literal["default_next_month"]
    positive_label: Literal[1]
    feature_contract_path: str
    feature_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cross_validation: CandidateCrossValidation
    holdout_access: Literal["prohibited"]

    @field_validator("feature_contract_path")
    @classmethod
    def validate_feature_path(cls, value: str) -> str:
        return _safe_relative_path(value, suffixes={".json"})


class FeatureView(_FrozenModel):
    view_id: Literal["repayment_status_only", "monetary_only", "operational_full"]
    purpose: Literal["diagnostic_ablation", "search_and_candidate"]
    eligible_for_advancement: bool
    predictor_count: int = Field(gt=0)
    predictor_columns: tuple[str, ...]

    @model_validator(mode="after")
    def validate_columns(self) -> FeatureView:
        if len(self.predictor_columns) != self.predictor_count:
            raise ValueError("feature-view predictor count does not match its columns")
        if len(set(self.predictor_columns)) != len(self.predictor_columns):
            raise ValueError("feature-view predictors must be unique")
        if set(self.predictor_columns) & {ID_COLUMN, TARGET_COLUMN, *AUDIT_COLUMNS}:
            raise ValueError("feature view contains a forbidden predictor")
        return self


class FeatureHandling(_FrozenModel):
    native_categorical_columns: tuple[str, ...]
    categorical_value_representation: Literal["validated_integer_code_as_string"]
    numeric_columns: tuple[str, ...]
    imputation: Literal["none"]
    clipping: Literal["none"]
    scaling: Literal["none"]
    resampling: Literal["none"]
    target_encoding: Literal["none"]


class FixedCatBoostParameters(_FrozenModel):
    loss_function: Literal["Logloss"]
    eval_metric: Literal["Logloss"]
    task_type: Literal["CPU"]
    bootstrap_type: Literal["Bayesian"]
    class_weights: None
    auto_class_weights: None
    random_seed: Literal[42]
    thread_count: Literal[4]
    allow_writing_files: Literal[False]
    verbose: Literal[False]
    use_best_model: Literal[False]
    early_stopping_rounds: None


class SampledParameters(_FrozenModel):
    bagging_temperature: float = Field(ge=0.0)
    depth: int = Field(ge=1)
    iterations: int = Field(ge=1)
    l2_leaf_reg: float = Field(ge=0.0)
    learning_rate: float = Field(gt=0.0)
    random_strength: float = Field(ge=0.0)


class SampledConfiguration(_FrozenModel):
    configuration_id: str = Field(pattern=r"^cb_cfg_[0-9]{3}$")
    parameters: SampledParameters


class ParameterSpace(_FrozenModel):
    iterations: tuple[Literal[300, 600], ...]
    depth: tuple[Literal[4, 6], ...]
    learning_rate: tuple[float, ...]
    l2_leaf_reg: tuple[float, ...]
    random_strength: tuple[float, ...]
    bagging_temperature: tuple[float, ...]

    @model_validator(mode="after")
    def validate_ordered_values(self) -> ParameterSpace:
        expected = {
            "iterations": (300, 600),
            "depth": (4, 6),
            "learning_rate": (0.03, 0.05, 0.1),
            "l2_leaf_reg": (3.0, 7.0, 12.0),
            "random_strength": (0.0, 1.0),
            "bagging_temperature": (0.0, 1.0),
        }
        if self.model_dump() != expected:
            raise ValueError("candidate parameter space differs from the reviewed ordered values")
        return self


class AblationPolicy(_FrozenModel):
    role: Literal["diagnostic_only"]
    eligible_for_advancement: Literal[False]
    hyperparameters: Literal["selected_from_operational_full_search"]
    evaluation_feature_views: tuple[
        Literal["repayment_status_only", "monetary_only", "operational_full"], ...
    ]

    @model_validator(mode="after")
    def validate_view_order(self) -> AblationPolicy:
        if self.evaluation_feature_views != (
            "repayment_status_only",
            "monetary_only",
            "operational_full",
        ):
            raise ValueError("ablation views differ from the reviewed order")
        return self


class CandidateSearch(_FrozenModel):
    strategy: Literal["sklearn_parameter_sampler_without_replacement"]
    sampler_library: Literal["scikit-learn"]
    sampler_version: Literal["1.4.2"]
    random_state: Literal[42]
    n_iter: Literal[8]
    sampled_configurations: tuple[SampledConfiguration, ...]
    search_feature_view: Literal["operational_full"]
    evaluation_assignments: Literal["all_5_folds_x_3_repeats"]
    parameter_space: ParameterSpace
    ablation_policy: AblationPolicy
    maximum_fold_fits: Literal[150]

    @model_validator(mode="after")
    def validate_materialized_sample(self) -> CandidateSearch:
        expected_ids = tuple(f"cb_cfg_{index:03d}" for index in range(1, self.n_iter + 1))
        observed_ids = tuple(item.configuration_id for item in self.sampled_configurations)
        if observed_ids != expected_ids:
            raise ValueError("sampled configuration IDs must be the ordered cb_cfg_001..008")
        parameter_grid = self.parameter_space.model_dump()
        sampled = tuple(
            SampledParameters.model_validate(item)
            for item in ParameterSampler(
                parameter_grid,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        )
        observed = tuple(item.parameters for item in self.sampled_configurations)
        if observed != sampled:
            raise ValueError("materialized configurations differ from the reviewed sampler output")
        if math.prod(len(values) for values in parameter_grid.values()) != 144:
            raise ValueError("candidate parameter space must contain exactly 144 configurations")
        return self


class CatBoostCandidate(_FrozenModel):
    model_id: Literal["catboost_v1"]
    model_family: Literal["catboost"]
    library_version: Literal["1.2.5"]
    additional_challenger: None
    feature_handling: FeatureHandling
    fixed_parameters: FixedCatBoostParameters
    search: CandidateSearch


class CapacityMetric(_FrozenModel):
    metric: Literal["lift"]
    capacity: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_capacity(self) -> CapacityMetric:
        if self.capacity != 0.1:
            raise ValueError("candidate primary capacity must be lift at 10%")
        return self


class CandidateEvaluation(_FrozenModel):
    primary_metric: Literal["average_precision"]
    probability_guardrail: Literal["brier_score"]
    primary_capacity_metric: CapacityMetric
    reported_discrimination_metrics: tuple[
        Literal["average_precision", "roc_auc", "ks", "gini"], ...
    ]
    reported_probability_metrics: tuple[Literal["brier_score", "log_loss"], ...]
    reported_capacities: tuple[float, ...]
    tie_policy: Literal["fractional_expected"]
    repeat_summary: tuple[Literal["mean", "standard_deviation", "minimum", "maximum"], ...]

    @model_validator(mode="after")
    def validate_metrics(self) -> CandidateEvaluation:
        if self.reported_discrimination_metrics != ("average_precision", "roc_auc", "ks", "gini"):
            raise ValueError("candidate discrimination metrics differ from the reviewed order")
        if self.reported_probability_metrics != ("brier_score", "log_loss"):
            raise ValueError("candidate probability metrics differ from the reviewed order")
        if self.reported_capacities != (0.05, 0.1, 0.2):
            raise ValueError("candidate capacities must be exactly 5%, 10%, and 20%")
        if self.repeat_summary != ("mean", "standard_deviation", "minimum", "maximum"):
            raise ValueError("candidate repeat summaries differ from the reviewed order")
        return self


class RelativeThresholds(_FrozenModel):
    minimum_average_precision_improvement: float
    maximum_brier_score_degradation: float
    maximum_lift_at_0_1_regression: float
    maximum_average_precision_repeat_standard_deviation: float

    @model_validator(mode="after")
    def validate_thresholds(self) -> RelativeThresholds:
        if self.model_dump() != {
            "minimum_average_precision_improvement": 0.01,
            "maximum_brier_score_degradation": 0.005,
            "maximum_lift_at_0_1_regression": 0.1,
            "maximum_average_precision_repeat_standard_deviation": 0.01,
        }:
            raise ValueError("relative candidate thresholds differ from the reviewed gate")
        return self


class AbsoluteThresholds(_FrozenModel):
    minimum_average_precision_mean: float
    maximum_brier_score_mean: float
    minimum_lift_at_0_1_mean: float
    maximum_average_precision_repeat_standard_deviation: float


class AdvancementGate(_FrozenModel):
    gate_id: Literal["balanced_v1"]
    combination_rule: Literal["all_conditions_required"]
    relative_to_model_id: Literal["logistic_l2"]
    relative_thresholds: RelativeThresholds
    derived_absolute_thresholds: AbsoluteThresholds
    equivalence_band_average_precision: float = Field(ge=0.0)
    tie_break_order: tuple[
        Literal[
            "lower_depth",
            "fewer_iterations",
            "higher_l2_leaf_reg",
            "lower_learning_rate",
            "lower_random_strength",
            "lower_bagging_temperature",
        ],
        ...,
    ]
    search_selection: Literal["best_eligible_operational_full_configuration"]
    final_variant_selection: Literal["selected_eligible_operational_full_configuration_only"]
    no_eligible_search_configuration: Literal["run_ablations_for_diagnosis_but_do_not_advance"]
    fallback_model_id: Literal["logistic_l2"]

    @model_validator(mode="after")
    def validate_tie_breaks(self) -> AdvancementGate:
        expected = (
            "lower_depth",
            "fewer_iterations",
            "higher_l2_leaf_reg",
            "lower_learning_rate",
            "lower_random_strength",
            "lower_bagging_temperature",
        )
        if self.tie_break_order != expected:
            raise ValueError("candidate tie-break order differs from the reviewed protocol")
        if self.equivalence_band_average_precision != 0.002:
            raise ValueError("candidate AP equivalence band must be exactly 0.002")
        return self


class CandidateGovernance(_FrozenModel):
    demographic_predictors: Literal["prohibited"]
    ablation_interpretation: Literal["development_diagnostic_only"]
    candidate_results_available_when_frozen: Literal[False]
    deferred_to_phase_4: tuple[str, ...]

    @model_validator(mode="after")
    def validate_deferred_work(self) -> CandidateGovernance:
        expected = (
            "calibration_selection",
            "bootstrap_confidence_intervals",
            "simulated_economics",
            "operating_policy_selection",
            "one_time_holdout_evaluation",
        )
        if self.deferred_to_phase_4 != expected:
            raise ValueError("Phase 4 deferrals differ from the reviewed protocol")
        return self


class CandidateExperimentConfig(_FrozenModel):
    """Complete immutable configuration for the Phase 3 candidate experiment."""

    schema_version: Literal["1.0.0"]
    protocol_id: Literal["candidate_v1"]
    status: Literal["frozen_pre_experiment"]
    baseline_evidence: BaselineEvidence
    data_contract: CandidateDataContract
    feature_views: tuple[FeatureView, ...]
    candidate: CatBoostCandidate
    evaluation: CandidateEvaluation
    advancement_gate: AdvancementGate
    governance: CandidateGovernance

    @model_validator(mode="after")
    def validate_complete_protocol(self) -> CandidateExperimentConfig:
        if (
            self.data_contract.dataset_id != OFFICIAL_DATASET_ID
            or self.data_contract.dataset_version != OFFICIAL_DATASET_VERSION
            or self.data_contract.expected_rows != OFFICIAL_DEVELOPMENT_ROWS
            or self.data_contract.split_assignment_sha256 != OFFICIAL_ASSIGNMENT_SHA256
        ):
            raise ValueError("candidate data contract differs from the reviewed snapshot")
        expected_views = (
            ("repayment_status_only", "diagnostic_ablation", False, REPAYMENT_STATUS_COLUMNS),
            ("monetary_only", "diagnostic_ablation", False, MONETARY_COLUMNS),
            ("operational_full", "search_and_candidate", True, PREDICTOR_COLUMNS),
        )
        observed_views = tuple(
            (view.view_id, view.purpose, view.eligible_for_advancement, view.predictor_columns)
            for view in self.feature_views
        )
        if observed_views != expected_views:
            raise ValueError("candidate feature views differ from the reviewed boundaries")
        handling = self.candidate.feature_handling
        if (
            handling.native_categorical_columns != REPAYMENT_STATUS_COLUMNS
            or handling.numeric_columns != MONETARY_COLUMNS
        ):
            raise ValueError("CatBoost feature handling differs from the governed feature boundary")
        reference = self.baseline_evidence.reference_metrics
        relative = self.advancement_gate.relative_thresholds
        absolute = self.advancement_gate.derived_absolute_thresholds
        expected_thresholds = (
            reference.average_precision_mean + relative.minimum_average_precision_improvement,
            reference.brier_score_mean + relative.maximum_brier_score_degradation,
            reference.lift_at_0_1_mean - relative.maximum_lift_at_0_1_regression,
            relative.maximum_average_precision_repeat_standard_deviation,
        )
        observed_thresholds = (
            absolute.minimum_average_precision_mean,
            absolute.maximum_brier_score_mean,
            absolute.minimum_lift_at_0_1_mean,
            absolute.maximum_average_precision_repeat_standard_deviation,
        )
        if any(
            not math.isclose(observed, expected, abs_tol=1e-15)
            for observed, expected in zip(observed_thresholds, expected_thresholds, strict=True)
        ):
            raise ValueError("candidate absolute gate thresholds do not match the baseline deltas")
        return self


def parse_candidate_config(
    content: str | bytes | bytearray,
    *,
    source: str | Path = "candidate configuration bytes",
) -> CandidateExperimentConfig:
    """Validate exact candidate-configuration bytes."""

    try:
        return CandidateExperimentConfig.model_validate_json(content)
    except ValidationError as error:
        raise ModelingContractError(f"Unable to load {source}: {error}") from error


def load_candidate_config(
    path: str | Path = DEFAULT_CANDIDATE_CONFIG_PATH,
    *,
    repo_root: str | Path = ".",
) -> CandidateExperimentConfig:
    """Load the candidate contract and verify every referenced reviewed file."""

    config_path = _root_relative(Path(path), Path(repo_root))
    try:
        content = config_path.read_bytes()
    except OSError as error:
        raise ModelingContractError(f"Unable to load {config_path}: {error}") from error
    config = parse_candidate_config(content, source=config_path)
    root = Path(repo_root).resolve()
    references = (
        (
            "baseline summary",
            config.baseline_evidence.summary_path,
            config.baseline_evidence.summary_sha256,
        ),
        (
            "baseline report",
            config.baseline_evidence.report_path,
            config.baseline_evidence.report_sha256,
        ),
        (
            "feature contract",
            config.data_contract.feature_contract_path,
            config.data_contract.feature_contract_sha256,
        ),
    )
    contents: dict[str, bytes] = {}
    for label, relative_path, expected_sha256 in references:
        resolved = (root / relative_path).resolve()
        try:
            resolved.relative_to(root)
            reference_bytes = resolved.read_bytes()
        except (OSError, ValueError) as error:
            raise ModelingContractError(
                f"Unable to read candidate {label} {resolved}: {error}"
            ) from error
        observed_sha256 = hashlib.sha256(reference_bytes).hexdigest()
        if observed_sha256 != expected_sha256:
            raise ModelingContractError(
                f"Candidate {label} hash mismatch: expected={expected_sha256}, "
                f"observed={observed_sha256}, path={resolved}"
            )
        contents[label] = reference_bytes
    parse_feature_contract(
        contents["feature contract"], source=config.data_contract.feature_contract_path
    )
    return config


def _safe_relative_path(value: str, *, suffixes: set[str]) -> str:
    pure_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or pure_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in pure_path.parts
        or pure_path.suffix.lower() not in suffixes
        or "\\" in value
    ):
        raise ValueError("candidate paths must be safe repository-relative paths")
    return value


def _root_relative(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root.resolve() / path).resolve()
