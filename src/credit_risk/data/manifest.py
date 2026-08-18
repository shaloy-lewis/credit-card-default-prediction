"""Versioned data-source and split configuration contracts."""

from pathlib import Path, PurePath, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_DATASET_MANIFEST_PATH = Path("configs/data/uci_credit_default_v1.json")
DEFAULT_SPLIT_CONFIG_PATH = Path("configs/data/split_v1.json")

OFFICIAL_COLUMN_MAPPING = (
    ("ID", "account_id"),
    ("X1", "credit_limit_ntd"),
    ("X2", "sex_code"),
    ("X3", "education_code"),
    ("X4", "marital_status_code"),
    ("X5", "age_years"),
    *((f"X{index + 6}", f"repayment_status_lag_{index}") for index in range(6)),
    *((f"X{index + 12}", f"bill_amount_ntd_lag_{index}") for index in range(6)),
    *((f"X{index + 18}", f"payment_amount_ntd_lag_{index}") for index in range(6)),
    ("Y", "default_next_month"),
)


class ManifestLoadError(ValueError):
    """Raised when a checked-in data configuration cannot be loaded safely."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceFile(_FrozenModel):
    """Pinned bytes for one downloadable source file."""

    url: str
    filename: str
    media_type: Literal["text/csv"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("url")
    @classmethod
    def require_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("source URL must be an absolute HTTPS URL without credentials")
        return value

    @field_validator("filename")
    @classmethod
    def require_safe_csv_filename(cls, value: str) -> str:
        if (
            not value
            or "/" in value
            or "\\" in value
            or PurePath(value).name != value
            or PurePosixPath(value).name != value
            or not value.lower().endswith(".csv")
        ):
            raise ValueError("source filename must be a plain CSV filename")
        return value


class DatasetExpectations(_FrozenModel):
    """Stable structural facts about the pinned source snapshot."""

    row_count: int = Field(gt=0)
    column_count: int = Field(gt=0)
    source_columns: tuple[str, ...]
    target_column: str
    target_counts: dict[Literal["0", "1"], int]

    @model_validator(mode="after")
    def validate_consistency(self) -> "DatasetExpectations":
        if len(self.source_columns) != self.column_count:
            raise ValueError("column_count must equal the number of source_columns")
        if len(set(self.source_columns)) != len(self.source_columns):
            raise ValueError("source_columns must be unique")
        if self.target_column not in self.source_columns:
            raise ValueError("target_column must be present in source_columns")
        if set(self.target_counts) != {"0", "1"}:
            raise ValueError("target_counts must contain exactly the binary labels 0 and 1")
        if any(count < 1 for count in self.target_counts.values()):
            raise ValueError("each target class must contain at least one record")
        if sum(self.target_counts.values()) != self.row_count:
            raise ValueError("target_counts must sum to row_count")
        return self


class CanonicalColumn(_FrozenModel):
    """One ordered source-to-canonical name and logical-type mapping."""

    source_name: str = Field(min_length=1)
    canonical_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    logical_dtype: Literal["integer", "number", "string", "boolean"]


class CanonicalContract(_FrozenModel):
    """Ordered canonical column contract for the source snapshot."""

    columns: tuple[CanonicalColumn, ...]

    @model_validator(mode="after")
    def require_unique_names(self) -> "CanonicalContract":
        source_names = [column.source_name for column in self.columns]
        canonical_names = [column.canonical_name for column in self.columns]
        if len(set(source_names)) != len(source_names):
            raise ValueError("canonical contract source names must be unique")
        if len(set(canonical_names)) != len(canonical_names):
            raise ValueError("canonical contract target names must be unique")
        return self


class DatasetManifest(_FrozenModel):
    """Auditable identity, attribution, and byte contract for a dataset snapshot."""

    schema_version: Literal["1.0.0"]
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    dataset_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    title: str = Field(min_length=1)
    repository_id: int = Field(gt=0)
    dataset_page_url: str
    doi: str = Field(min_length=1)
    creator: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    license_name: str = Field(min_length=1)
    license_url: str
    source: SourceFile
    expectations: DatasetExpectations
    canonical_contract: CanonicalContract

    @field_validator("dataset_page_url", "license_url")
    @classmethod
    def require_documentation_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("documentation URLs must be absolute HTTPS URLs without credentials")
        return value

    @model_validator(mode="after")
    def require_contract_parity(self) -> "DatasetManifest":
        mapped_pairs = tuple(
            (column.source_name, column.canonical_name)
            for column in self.canonical_contract.columns
        )
        mapped_source_names = tuple(source_name for source_name, _ in mapped_pairs)
        if mapped_source_names != self.expectations.source_columns:
            raise ValueError(
                "canonical contract source names must exactly match source_columns order"
            )
        if self.dataset_id == "uci_credit_default":
            if mapped_pairs != OFFICIAL_COLUMN_MAPPING:
                raise ValueError("official canonical mapping differs from the governed contract")
            if any(column.logical_dtype != "integer" for column in self.canonical_contract.columns):
                raise ValueError("official canonical columns must use integer logical dtypes")
        return self


class HoldoutConfig(_FrozenModel):
    """Sealed holdout selection policy."""

    method: Literal["stratified_shuffle_split"]
    test_fraction: float = Field(gt=0.0, lt=1.0)
    random_state: int


class CrossValidationConfig(_FrozenModel):
    """Repeated cross-validation policy for development data only."""

    method: Literal["repeated_stratified_k_fold"]
    n_splits: int = Field(ge=2)
    n_repeats: int = Field(ge=1)
    random_state: int


class PartitionCounts(_FrozenModel):
    """Expected size and binary class counts for one deterministic partition."""

    total: int = Field(gt=0)
    target_counts: dict[Literal["0", "1"], int]

    @model_validator(mode="after")
    def validate_counts(self) -> "PartitionCounts":
        if set(self.target_counts) != {"0", "1"}:
            raise ValueError("target_counts must contain exactly the binary labels 0 and 1")
        if any(count < 1 for count in self.target_counts.values()):
            raise ValueError("each partition target class must contain at least one record")
        if sum(self.target_counts.values()) != self.total:
            raise ValueError("partition target_counts must sum to total")
        return self


class SplitExpectedCounts(_FrozenModel):
    """Frozen aggregate evidence for the deterministic split protocol."""

    total: int = Field(gt=0)
    development: PartitionCounts
    test: PartitionCounts

    @model_validator(mode="after")
    def validate_partition_total(self) -> "SplitExpectedCounts":
        if self.development.total + self.test.total != self.total:
            raise ValueError("development and test counts must sum to total")
        return self


class SplitConfig(_FrozenModel):
    """Frozen split policy consumed by the deterministic split builder."""

    config_version: Literal[1]
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    dataset_version: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    id_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    sort_by: tuple[str, ...]
    holdout: HoldoutConfig
    cross_validation: CrossValidationConfig
    expected_counts: SplitExpectedCounts

    @model_validator(mode="after")
    def require_stable_identifier_sort(self) -> "SplitConfig":
        if not self.sort_by or self.id_column not in self.sort_by:
            raise ValueError("sort_by must contain the identifier column")
        if len(set(self.sort_by)) != len(self.sort_by):
            raise ValueError("sort_by columns must be unique")
        observed_fraction = self.expected_counts.test.total / self.expected_counts.total
        if abs(observed_fraction - self.holdout.test_fraction) > 1e-12:
            raise ValueError("expected holdout count must match holdout.test_fraction")
        return self


def _load_model[ModelT: BaseModel](path: str | Path, model_type: type[ModelT]) -> ModelT:
    config_path = Path(path)
    try:
        payload = config_path.read_text(encoding="utf-8")
        return model_type.model_validate_json(payload)
    except (OSError, ValidationError) as error:
        raise ManifestLoadError(f"Unable to load {config_path}: {error}") from error


def load_dataset_manifest(
    path: str | Path = DEFAULT_DATASET_MANIFEST_PATH,
) -> DatasetManifest:
    """Load and validate the pinned source-data manifest."""

    return _load_model(path, DatasetManifest)


def load_split_config(path: str | Path = DEFAULT_SPLIT_CONFIG_PATH) -> SplitConfig:
    """Load and validate the sealed holdout and cross-validation protocol."""

    return _load_model(path, SplitConfig)
