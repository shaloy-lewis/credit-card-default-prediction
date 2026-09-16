"""Strict contracts for versioned online and batch inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from credit_risk.modeling.contracts import PREDICTOR_COLUMNS

DEFAULT_INFERENCE_CONFIG_PATH = Path("configs/inference/phase6_v1.json")
PHASE6_CONFIG_SHA256 = "84227bb48c7ba812bdf2a2752ed151ede2ce5daa398911af9311491a852500a8"
ACCOUNT_ID_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"
RESERVED_SNAPSHOT_IDS = (".", "..")
REQUIRED_SERVING_DEPENDENCIES = (
    "catboost",
    "joblib",
    "numpy",
    "pandas",
    "pydantic",
    "scikit-learn",
)


class InferenceContractError(ValueError):
    """Raised when a Phase 6 contract cannot be trusted."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())


class _StrictWireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class BundleContract(_FrozenModel):
    bundle_id: Literal["selected_v1"]
    model_id: Literal["catboost_fixed"]
    manifest_path: Literal["models/selected_v1/manifest.json"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_path: Literal["models/selected_v1/model.cbm"]
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration: Literal["identity"]


class InputContract(_FrozenModel):
    format: Literal["utf8_csv"]
    account_id_pattern: str = Field(pattern=r"^\^\[A-Za-z0-9\.\_\-\]\{1,64\}\$$")
    columns: tuple[str, ...]
    duplicate_account_policy: Literal["reject_every_occurrence"]
    row_error_policy: Literal["score_valid_rows"]
    file_error_policy: Literal["fail_without_scores"]


class PredictionContract(_FrozenModel):
    feature_order: tuple[str, ...]
    risk_band_thresholds: dict[str, float]
    probability_output: Literal["float64_full_precision"]
    api_probability_decimals: Literal[6]
    api_batch_absolute_tolerance: float


class ExplanationContract(_FrozenModel):
    method: Literal["catboost_native_shap_values"]
    space: Literal["raw_log_odds"]
    shap_output_columns: Literal[20]
    additivity_absolute_tolerance: float
    probability_absolute_tolerance: float
    top_reason_count: Literal[2]
    ordering: Literal["absolute_contribution_descending_then_category_name"]
    direction_labels: dict[str, str]
    language_policy: Literal["model_attribution_not_causal_or_adverse_action_reason"]
    reason_categories: dict[str, tuple[str, ...]]


class PolicyContract(_FrozenModel):
    policy_id: Literal["outreach_top_10_v1"]
    review_capacity_fraction: float
    review_capacity_rounding: Literal["floor"]
    ranking: Literal["probability_descending_then_account_id_ascending"]
    risk_band_independent_of_capacity: Literal[True]
    decision_owner: Literal["human_outreach_operations"]


class BatchContract(_FrozenModel):
    idempotency_key_fields: tuple[str, ...]
    snapshot_id_pattern: str = Field(pattern=r"^\^\[A-Za-z0-9\.\_\-\]\{1,64\}\$$")
    reserved_snapshot_ids: tuple[str, ...]
    output_layout: Literal["output_root/as_of_date/snapshot_id"]
    identical_rerun: Literal["verify_and_reuse_without_rewrite"]
    changed_or_corrupt_existing_run: Literal["fail_without_overwrite"]
    statuses: tuple[Literal["completed", "completed_with_rejections", "failed"], ...]
    exit_codes: dict[str, int]
    outputs: tuple[Literal["scores.csv", "rejections.csv", "manifest.json"], ...]


class ApiContract(_FrozenModel):
    prediction_path: Literal["/v1/predict"]
    removed_prediction_path: Literal["/predict"]
    liveness_path: Literal["/ping"]
    readiness_path: Literal["/ready"]
    request_id_header: Literal["X-Request-ID"]
    trace_id_header: Literal["X-Trace-ID"]
    request_id_pattern: str = Field(pattern=r"^\^\[A-Za-z0-9\.\_\-\]\{1,64\}\$$")
    queue_decision_in_single_record_response: Literal[False]


class LoggingContract(_FrozenModel):
    format: Literal["json_stdout"]
    allowlisted_fields: tuple[str, ...]
    prohibited_fields: tuple[str, ...]


class InferenceConfig(_FrozenModel):
    schema_version: Literal["1.0.0"]
    protocol_id: Literal["phase6_v1"]
    status: Literal["frozen_before_implementation"]
    purpose: Literal["idempotent_monthly_batch_and_versioned_api_parity"]
    governance: dict[str, object]
    bundle: BundleContract
    input: InputContract
    prediction: PredictionContract
    explanation: ExplanationContract
    policy: PolicyContract
    batch: BatchContract
    api: ApiContract
    logging: LoggingContract
    dependencies: dict[str, str]
    prohibitions: tuple[str, ...]

    @model_validator(mode="after")
    def validate_frozen_contract(self) -> InferenceConfig:
        if self.prediction.feature_order != PREDICTOR_COLUMNS:
            raise ValueError("prediction feature order differs from the selected bundle")
        if self.input.columns != ("account_id", *PREDICTOR_COLUMNS):
            raise ValueError("batch columns must be account_id followed by the 19 predictors")
        if set(self.prediction.risk_band_thresholds) != {"q80", "q90", "q95"}:
            raise ValueError("risk-band thresholds are incomplete")
        thresholds = tuple(
            self.prediction.risk_band_thresholds[name] for name in ("q80", "q90", "q95")
        )
        if tuple(sorted(thresholds)) != thresholds:
            raise ValueError("risk-band thresholds must be ordered")
        categories = self.explanation.reason_categories
        if set(categories) != {
            "billing_balance",
            "credit_capacity",
            "payment_behaviour",
            "repayment_status",
        }:
            raise ValueError("reviewed reason categories changed")
        flattened = tuple(feature for values in categories.values() for feature in values)
        if len(flattened) != len(set(flattened)) or set(flattened) != set(PREDICTOR_COLUMNS):
            raise ValueError("reason categories must partition the operational features")
        if self.policy.review_capacity_fraction != 0.1:
            raise ValueError("review capacity must remain fixed at 10 percent")
        if self.batch.exit_codes != {
            "completed": 0,
            "completed_with_rejections": 3,
            "failed": 1,
        }:
            raise ValueError("batch exit codes differ from the reviewed contract")
        if self.batch.reserved_snapshot_ids != RESERVED_SNAPSHOT_IDS:
            raise ValueError("reserved snapshot IDs differ from the reviewed path-safety contract")
        required = {
            "model_fitting",
            "parameter_tuning",
            "calibration_fitting",
            "bootstrap_generation",
            "final_test_loading",
            "sealed_test_scoring",
        }
        if not required <= set(self.prohibitions):
            raise ValueError("prediction-only prohibitions are incomplete")
        return self


class OperationalFeatures(BaseModel):
    """Exactly the operational fields accepted by batch and online inference."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    credit_limit_ntd: int = Field(gt=0)
    repayment_status_lag_0: int = Field(ge=-2, le=9)
    repayment_status_lag_1: int = Field(ge=-2, le=9)
    repayment_status_lag_2: int = Field(ge=-2, le=9)
    repayment_status_lag_3: int = Field(ge=-2, le=9)
    repayment_status_lag_4: int = Field(ge=-2, le=9)
    repayment_status_lag_5: int = Field(ge=-2, le=9)
    bill_amount_ntd_lag_0: int
    bill_amount_ntd_lag_1: int
    bill_amount_ntd_lag_2: int
    bill_amount_ntd_lag_3: int
    bill_amount_ntd_lag_4: int
    bill_amount_ntd_lag_5: int
    payment_amount_ntd_lag_0: int = Field(ge=0)
    payment_amount_ntd_lag_1: int = Field(ge=0)
    payment_amount_ntd_lag_2: int = Field(ge=0)
    payment_amount_ntd_lag_3: int = Field(ge=0)
    payment_amount_ntd_lag_4: int = Field(ge=0)
    payment_amount_ntd_lag_5: int = Field(ge=0)


class ReasonResponse(_StrictWireModel):
    category: Literal["billing_balance", "credit_capacity", "payment_behaviour", "repayment_status"]
    direction: Literal["risk_increasing", "risk_mitigating", "neutral"]
    contribution_raw_log_odds: float


class CreditRiskResponse(_StrictWireModel):
    schema_version: Literal["1.0.0"]
    trace_id: str = Field(pattern=ACCOUNT_ID_PATTERN)
    probability_of_default: float = Field(ge=0.0, le=1.0)
    risk_band: Literal["standard", "elevated", "high", "critical"]
    reasons: tuple[ReasonResponse, ReasonResponse]
    model_id: Literal["catboost_fixed"]
    bundle_id: Literal["selected_v1"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_id: Literal["outreach_top_10_v1"]


def load_inference_config(path: str | Path = DEFAULT_INFERENCE_CONFIG_PATH) -> InferenceConfig:
    """Authenticate and parse the complete frozen Phase 6 configuration."""

    config_path = Path(path)
    try:
        content = config_path.read_bytes()
    except OSError as error:
        raise InferenceContractError(f"Unable to read inference config: {error}") from error
    observed = hashlib.sha256(content).hexdigest()
    if observed != PHASE6_CONFIG_SHA256:
        raise InferenceContractError(
            f"Inference config digest mismatch: expected={PHASE6_CONFIG_SHA256}, observed={observed}"
        )
    try:
        return InferenceConfig.model_validate_json(content)
    except ValidationError as error:
        raise InferenceContractError(f"Invalid inference config: {error}") from error
