"""Versioned contracts for governed baseline experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

DEFAULT_FEATURE_CONTRACT_PATH = Path("configs/modeling/feature_contract_v1.json")
DEFAULT_BASELINE_CONFIG_PATH = Path("configs/modeling/baseline_v1.json")

OFFICIAL_DATASET_ID = "uci_credit_default"
OFFICIAL_DATASET_VERSION = "v1"
OFFICIAL_SOURCE_SHA256 = "45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0"
OFFICIAL_DATASET_MANIFEST_SHA256 = (
    "4e6463e3acce879e00435f21b64cc905b11dc5fd894b20a32cddd0f775be6979"
)
OFFICIAL_CANONICAL_SHA256 = "75b2a746781a584b0456f843f1f269190b51e90983cba44c4ed6c4a8685e6c1c"
OFFICIAL_SPLIT_CONFIG_SHA256 = "36d8bb9ae6221a60dafa389a0bdf65796bd726d4a96413f810c002de43842398"
OFFICIAL_ASSIGNMENT_SHA256 = "2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e"
OFFICIAL_SPLIT_LOCK_SHA256 = "b2312380fa46924ca414acbcfef63b0435d1321083e87e4df5ec04f18736093d"

ID_COLUMN = "account_id"
TARGET_COLUMN = "default_next_month"
AUDIT_COLUMNS = (
    "sex_code",
    "education_code",
    "marital_status_code",
    "age_years",
)
REPAYMENT_STATUS_COLUMNS = tuple(f"repayment_status_lag_{lag}" for lag in range(6))
BILL_AMOUNT_COLUMNS = tuple(f"bill_amount_ntd_lag_{lag}" for lag in range(6))
PAYMENT_AMOUNT_COLUMNS = tuple(f"payment_amount_ntd_lag_{lag}" for lag in range(6))
MONETARY_COLUMNS = (
    "credit_limit_ntd",
    *BILL_AMOUNT_COLUMNS,
    *PAYMENT_AMOUNT_COLUMNS,
)
PREDICTOR_COLUMNS = (
    "credit_limit_ntd",
    *REPAYMENT_STATUS_COLUMNS,
    *BILL_AMOUNT_COLUMNS,
    *PAYMENT_AMOUNT_COLUMNS,
)
FORBIDDEN_PREDICTOR_COLUMNS = (ID_COLUMN, TARGET_COLUMN, *AUDIT_COLUMNS)

OFFICIAL_DEVELOPMENT_ROWS = 24_000
OFFICIAL_DEVELOPMENT_TARGET_COUNTS = {"0": 18_691, "1": 5_309}
OFFICIAL_N_SPLITS = 5
OFFICIAL_N_REPEATS = 3
REPAYMENT_RULE_WEIGHTS = (6, 5, 4, 3, 2, 1)
REPAYMENT_STATUS_CATEGORIES = tuple(range(-2, 10))


class ModelingContractError(ValueError):
    """Raised when a versioned modelling configuration cannot be trusted."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class DatasetReference(_FrozenModel):
    """Dataset snapshot governed by a modelling feature contract."""

    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    dataset_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")


class FeatureColumns(_FrozenModel):
    """Ordered predictor and audit boundaries for one dataset snapshot."""

    id_column: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target_column: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    predictor_columns: tuple[str, ...]
    audit_columns: tuple[str, ...]
    forbidden_predictor_columns: tuple[str, ...]

    @model_validator(mode="after")
    def validate_feature_boundary(self) -> FeatureColumns:
        groups = {
            "predictor_columns": self.predictor_columns,
            "audit_columns": self.audit_columns,
            "forbidden_predictor_columns": self.forbidden_predictor_columns,
        }
        for name, columns in groups.items():
            if not columns:
                raise ValueError(f"{name} must not be empty")
            if len(set(columns)) != len(columns):
                raise ValueError(f"{name} must contain unique column names")
            if any(not column or not column.replace("_", "a").isalnum() for column in columns):
                raise ValueError(f"{name} contains an invalid column name")

        predictor_set = set(self.predictor_columns)
        audit_set = set(self.audit_columns)
        if predictor_set & audit_set:
            raise ValueError("predictor_columns and audit_columns must be disjoint")
        if self.id_column in predictor_set or self.target_column in predictor_set:
            raise ValueError("identifier and target must not appear in predictor_columns")
        expected_forbidden = {self.id_column, self.target_column, *self.audit_columns}
        if set(self.forbidden_predictor_columns) != expected_forbidden:
            raise ValueError(
                "forbidden_predictor_columns must contain the identifier, target, "
                "and every audit column"
            )
        return self


class DevelopmentExpectation(_FrozenModel):
    """Reviewed development-partition size and class counts."""

    rows: int = Field(gt=0)
    target_counts: dict[Literal["0", "1"], int]

    @model_validator(mode="after")
    def validate_target_counts(self) -> DevelopmentExpectation:
        if set(self.target_counts) != {"0", "1"}:
            raise ValueError("target_counts must contain exactly labels 0 and 1")
        if any(count < 1 for count in self.target_counts.values()):
            raise ValueError("each development target class must contain at least one row")
        if sum(self.target_counts.values()) != self.rows:
            raise ValueError("development target_counts must sum to rows")
        return self


class CrossValidationExpectation(_FrozenModel):
    """Shape of reviewed development-only fold assignments."""

    n_splits: int = Field(ge=2)
    n_repeats: int = Field(ge=1)


class FeatureLineage(_FrozenModel):
    """Reviewed hashes required before a modelling view may be exposed."""

    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_split_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FeatureContract(_FrozenModel):
    """Governed separation between predictive and audit-only data."""

    schema_version: Literal["1.0.0"]
    contract_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    dataset: DatasetReference
    columns: FeatureColumns
    expected_development: DevelopmentExpectation
    cross_validation: CrossValidationExpectation
    lineage: FeatureLineage

    @model_validator(mode="after")
    def validate_contract(self) -> FeatureContract:
        if min(self.expected_development.target_counts.values()) < self.cross_validation.n_splits:
            raise ValueError("each target class must contain at least n_splits development rows")
        if self.dataset.dataset_id == OFFICIAL_DATASET_ID:
            expected = {
                "dataset_version": OFFICIAL_DATASET_VERSION,
                "id_column": ID_COLUMN,
                "target_column": TARGET_COLUMN,
                "predictor_columns": PREDICTOR_COLUMNS,
                "audit_columns": AUDIT_COLUMNS,
                "forbidden_predictor_columns": FORBIDDEN_PREDICTOR_COLUMNS,
                "rows": OFFICIAL_DEVELOPMENT_ROWS,
                "target_counts": OFFICIAL_DEVELOPMENT_TARGET_COUNTS,
                "n_splits": OFFICIAL_N_SPLITS,
                "n_repeats": OFFICIAL_N_REPEATS,
                "source_sha256": OFFICIAL_SOURCE_SHA256,
                "dataset_manifest_sha256": OFFICIAL_DATASET_MANIFEST_SHA256,
                "canonical_sha256": OFFICIAL_CANONICAL_SHA256,
                "split_config_sha256": OFFICIAL_SPLIT_CONFIG_SHA256,
                "assignment_sha256": OFFICIAL_ASSIGNMENT_SHA256,
                "reviewed_split_lock_sha256": OFFICIAL_SPLIT_LOCK_SHA256,
            }
            observed = {
                "dataset_version": self.dataset.dataset_version,
                "id_column": self.columns.id_column,
                "target_column": self.columns.target_column,
                "predictor_columns": self.columns.predictor_columns,
                "audit_columns": self.columns.audit_columns,
                "forbidden_predictor_columns": self.columns.forbidden_predictor_columns,
                "rows": self.expected_development.rows,
                "target_counts": self.expected_development.target_counts,
                "n_splits": self.cross_validation.n_splits,
                "n_repeats": self.cross_validation.n_repeats,
                **self.lineage.model_dump(),
            }
            differences = [name for name, value in expected.items() if observed[name] != value]
            if differences:
                raise ValueError(
                    "official feature contract differs from reviewed values: "
                    + ", ".join(differences)
                )
        return self


class PrevalenceBaselineConfig(_FrozenModel):
    model_id: Literal["fold_prevalence"]
    kind: Literal["train_fold_prevalence"]
    prediction_kind: Literal["probability"]


class RepaymentRuleConfig(_FrozenModel):
    model_id: Literal["repayment_burden_rule"]
    kind: Literal["weighted_positive_repayment_status"]
    prediction_kind: Literal["risk_score"]
    status_columns: tuple[str, ...]
    recency_weights: tuple[int, ...]
    negative_value_floor: Literal[0]
    aggregation: Literal["sum"]

    @model_validator(mode="after")
    def validate_rule(self) -> RepaymentRuleConfig:
        if self.status_columns != REPAYMENT_STATUS_COLUMNS:
            raise ValueError("repayment rule must use the six reviewed status columns in order")
        if self.recency_weights != REPAYMENT_RULE_WEIGHTS:
            raise ValueError("repayment rule weights must be exactly 6, 5, 4, 3, 2, 1")
        return self


class LogisticBaselineConfig(_FrozenModel):
    model_id: Literal["logistic_l2"]
    kind: Literal["logistic_regression"]
    prediction_kind: Literal["probability"]
    status_columns: tuple[str, ...]
    status_encoding: Literal["one_hot"]
    status_categories: tuple[int, ...]
    status_drop: Literal["first"]
    handle_unknown: Literal["error"]
    monetary_columns: tuple[str, ...]
    scaler: Literal["standard"]
    penalty: Literal["l2"]
    c: float = Field(gt=0.0)
    solver: Literal["lbfgs"]
    class_weight: None
    fit_intercept: Literal[True]
    max_iter: int = Field(ge=1)
    tolerance: float = Field(gt=0.0)
    random_state: int

    @model_validator(mode="after")
    def validate_preprocessing(self) -> LogisticBaselineConfig:
        if self.status_columns != REPAYMENT_STATUS_COLUMNS:
            raise ValueError("logistic status columns differ from the governed order")
        if self.status_categories != REPAYMENT_STATUS_CATEGORIES:
            raise ValueError("logistic status categories must be the complete range -2..9")
        if self.monetary_columns != MONETARY_COLUMNS:
            raise ValueError("logistic monetary columns differ from the governed order")
        return self


class BaselineModelsConfig(_FrozenModel):
    prevalence: PrevalenceBaselineConfig
    repayment_rule: RepaymentRuleConfig
    logistic: LogisticBaselineConfig


class PrimaryCapacityMetric(_FrozenModel):
    """Locked operating-capacity objective for baseline comparison."""

    metric: Literal["lift"]
    capacity: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_capacity(self) -> PrimaryCapacityMetric:
        if self.capacity != 0.1:
            raise ValueError("primary capacity metric must be lift at 10%")
        return self


class EvaluationConfig(_FrozenModel):
    primary_metric: Literal["average_precision"]
    probability_guardrail: Literal["brier_score"]
    primary_capacity_metric: PrimaryCapacityMetric
    discrimination_metrics: tuple[Literal["average_precision", "roc_auc", "ks", "gini"], ...]
    probability_metrics: tuple[Literal["brier_score", "log_loss"], ...]
    capacities: tuple[float, ...]
    tie_policy: Literal["fractional_expected"]
    repeat_summary: tuple[Literal["mean", "standard_deviation", "minimum", "maximum"], ...]

    @model_validator(mode="after")
    def validate_evaluation(self) -> EvaluationConfig:
        if self.discrimination_metrics != ("average_precision", "roc_auc", "ks", "gini"):
            raise ValueError("discrimination_metrics must use the reviewed metric order")
        if self.probability_metrics != ("brier_score", "log_loss"):
            raise ValueError("probability_metrics must be exactly brier_score and log_loss")
        if self.capacities != (0.05, 0.1, 0.2):
            raise ValueError("capacities must be exactly 0.05, 0.10, and 0.20")
        if self.repeat_summary != ("mean", "standard_deviation", "minimum", "maximum"):
            raise ValueError("repeat_summary must be mean, standard_deviation, minimum, maximum")
        return self


class BaselineExperimentConfig(_FrozenModel):
    """Complete, immutable scientific configuration for Week 3 baselines."""

    schema_version: Literal["1.0.0"]
    experiment_id: Literal["baseline_v1"]
    experiment_name: Literal["credit-risk-baseline-v1"]
    feature_contract_path: str
    dataset_manifest_path: str = "configs/data/uci_credit_default_v1.json"
    split_config_path: str = "configs/data/split_v1.json"
    feature_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    partition: Literal["development"]
    positive_label: Literal[1]
    random_state: Literal[42]
    baselines: BaselineModelsConfig
    evaluation: EvaluationConfig

    @field_validator("feature_contract_path", "dataset_manifest_path", "split_config_path")
    @classmethod
    def require_safe_relative_json_path(cls, value: str) -> str:
        pure_path = PurePosixPath(value)
        windows_path = PureWindowsPath(value)
        if (
            pure_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or ".." in pure_path.parts
            or pure_path.suffix.lower() != ".json"
            or "\\" in value
        ):
            raise ValueError("configuration paths must be safe relative JSON paths")
        return value

    @model_validator(mode="after")
    def validate_baseline_protocol(self) -> BaselineExperimentConfig:
        logistic = self.baselines.logistic
        if (
            logistic.c != 1.0
            or logistic.max_iter != 2_000
            or logistic.tolerance != 1e-8
            or logistic.random_state != self.random_state
        ):
            raise ValueError("logistic baseline differs from the reviewed fixed parameters")
        return self


def _parse_model[ModelT: BaseModel](
    content: str | bytes | bytearray,
    model_type: type[ModelT],
    *,
    source: str | Path,
) -> ModelT:
    try:
        return model_type.model_validate_json(content)
    except ValidationError as error:
        raise ModelingContractError(f"Unable to load {source}: {error}") from error


def _load_model[ModelT: BaseModel](path: str | Path, model_type: type[ModelT]) -> ModelT:
    config_path = Path(path)
    try:
        content = config_path.read_bytes()
    except OSError as error:
        raise ModelingContractError(f"Unable to load {config_path}: {error}") from error
    return _parse_model(content, model_type, source=config_path)


def parse_feature_contract(
    content: str | bytes | bytearray,
    *,
    source: str | Path = "feature contract bytes",
) -> FeatureContract:
    """Validate the exact feature-contract bytes used for lineage."""

    return _parse_model(content, FeatureContract, source=source)


def parse_baseline_config(
    content: str | bytes | bytearray,
    *,
    source: str | Path = "baseline configuration bytes",
) -> BaselineExperimentConfig:
    """Validate the exact baseline-configuration bytes used for lineage."""

    return _parse_model(content, BaselineExperimentConfig, source=source)


def load_feature_contract(
    path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
) -> FeatureContract:
    """Load and strictly validate a versioned predictive-feature boundary."""

    return _load_model(path, FeatureContract)


def load_baseline_config(
    path: str | Path = DEFAULT_BASELINE_CONFIG_PATH,
) -> BaselineExperimentConfig:
    """Load and strictly validate the reviewed baseline protocol."""

    config_path = Path(path)
    try:
        config_bytes = config_path.read_bytes()
    except OSError as error:
        raise ModelingContractError(f"Unable to load {config_path}: {error}") from error
    config = parse_baseline_config(config_bytes, source=config_path)
    feature_path = _resolve_feature_contract_path(config_path, config.feature_contract_path)
    try:
        feature_bytes = feature_path.read_bytes()
    except OSError as error:
        raise ModelingContractError(
            f"Unable to read feature contract {feature_path}: {error}"
        ) from error
    observed_sha256 = hashlib.sha256(feature_bytes).hexdigest()
    if observed_sha256 != config.feature_contract_sha256:
        raise ModelingContractError(
            "Baseline feature contract hash mismatch: "
            f"expected={config.feature_contract_sha256}, observed={observed_sha256}, "
            f"path={feature_path}"
        )
    parse_feature_contract(feature_bytes, source=feature_path)
    return config


def _resolve_feature_contract_path(config_path: Path, configured_path: str) -> Path:
    relative_path = Path(configured_path)
    posix_path = PurePosixPath(configured_path)
    candidates: list[Path] = [(Path.cwd() / relative_path).resolve()]
    resolved_config = config_path.resolve()
    if len(posix_path.parts) == 1:
        candidates.append((resolved_config.parent / relative_path.name).resolve())
    candidates.extend((ancestor / relative_path).resolve() for ancestor in resolved_config.parents)

    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
        if candidate.is_file():
            return candidate

    checked = ", ".join(str(candidate) for candidate in unique_candidates)
    raise ModelingContractError(
        f"Unable to read feature contract {configured_path}: no file found; checked {checked}"
    )
