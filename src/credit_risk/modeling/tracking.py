"""MLflow tracking and reproducibility evidence for governed baseline runs.

The adapter deliberately exposes a narrow logging surface. Only aggregate
metrics, lineage parameters, and the four schema-validated evidence files can enter
MLflow; fitted estimators and row-level source or holdout data are never
accepted by this module.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import subprocess
import threading
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final

BASELINE_MODEL_NAMES: Final[tuple[str, ...]] = (
    "fold_prevalence",
    "repayment_burden_rule",
    "logistic_l2",
)
DEFAULT_EXPERIMENT_NAME: Final[str] = "credit-risk-baseline-v1"
CANDIDATE_EXPERIMENT_NAME: Final[str] = "credit-risk-candidate-v1"
SELECTION_EXPERIMENT_NAME: Final[str] = "credit-risk-selection-v1"
MLFLOW_DATABASE_FILENAME: Final[str] = "mlflow.db"
MLFLOW_ARTIFACT_DIRECTORY: Final[str] = "artifacts"
ALLOWED_TRACKING_ARTIFACTS: Final[Mapping[str, str]] = {
    "summary.json": "evidence",
    "baseline-report.md": "evidence",
    "oof_predictions.csv": "runtime",
    "logistic_fold_diagnostics.json": "runtime",
}
CANDIDATE_TRACKING_ARTIFACTS: Final[Mapping[str, str]] = {
    "summary.json": "evidence",
    "candidate-report.md": "evidence",
    "oof_predictions.csv": "runtime",
    "fold_diagnostics.json": "runtime",
}
SELECTION_MODEL_NAMES: Final[tuple[str, ...]] = (
    "logistic_l2",
    "random_forest",
    "hist_gradient_boosting",
    "catboost_fixed",
)
SELECTION_TRACKING_ARTIFACTS: Final[Mapping[str, str]] = {
    "summary.json": "evidence",
    "selection-report.md": "evidence",
    "validation_predictions.csv": "runtime",
    "bootstrap_intervals.json": "runtime",
    "manifest.json": "bundle",
}
DEFAULT_VERSION_PACKAGES: Final[tuple[str, ...]] = (
    "catboost",
    "credit-risk-early-warning",
    "joblib",
    "mlflow",
    "numpy",
    "pandas",
    "pandera",
    "pydantic",
    "scikit-learn",
    "scipy",
)
OPTIONAL_PACKAGE_EXTRAS: Final[Mapping[str, str]] = {
    "mlflow": "modeling",
    "pandera": "data",
}
_MLFLOW_TRACKING_LOCK: Final[threading.RLock] = threading.RLock()

ParameterValue = str | int | float | bool


class TrackingError(RuntimeError):
    """Raised when experiment evidence cannot be recorded safely."""


class TrackingDependencyError(TrackingError):
    """Raised when the optional MLflow dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class GitEvidence:
    """Commit and working-tree evidence captured before an experiment starts."""

    commit_sha: str
    dirty: bool
    diff_sha256: str
    repository_root: Path | None = None


@dataclass(frozen=True, slots=True)
class ModelRunPayload:
    """Parameters and metrics for one nested baseline run."""

    model_name: str
    parameters: Mapping[str, ParameterValue]
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class TrackingRunResult:
    """Operational MLflow identifiers excluded from deterministic reports."""

    tracking_uri: str
    experiment_name: str
    parent_run_id: str
    child_run_ids: tuple[tuple[str, str], ...]


def collect_git_evidence(repo_root: str | Path = ".") -> GitEvidence:
    """Collect a content-sensitive, non-disclosing working-tree fingerprint.

    The digest covers the porcelain status, the binary tracked diff, and a
    digest of every untracked file.  File contents and diff text are never
    returned, logged, or placed in the deterministic experiment report.
    """

    requested_root = Path(repo_root).resolve()
    top_level = _run_git(requested_root, "rev-parse", "--show-toplevel").decode(
        "utf-8", errors="strict"
    )
    root = Path(top_level.strip()).resolve()
    if not root.is_dir():
        raise TrackingError(f"Git returned an invalid repository root for {requested_root}.")
    commit = _run_git(root, "rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise TrackingError(f"Git returned an invalid HEAD commit for {root}.")

    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    tracked_diff = _run_git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    digest.update(b"\0tracked-diff\0")
    digest.update(tracked_diff)

    for relative_path in _untracked_paths(status):
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise TrackingError(
                f"Git reported an untracked path outside the repository: {relative_path}"
            ) from error
        if not candidate.is_file():
            raise TrackingError(f"Git reported a non-file untracked path: {relative_path}")
        try:
            content_sha256 = hashlib.sha256(candidate.read_bytes()).digest()
        except OSError as error:
            raise TrackingError(
                f"Unable to fingerprint untracked file {relative_path}: {error}"
            ) from error
        encoded_path = relative_path.as_posix().encode("utf-8", errors="surrogateescape")
        digest.update(b"\0untracked\0")
        digest.update(encoded_path)
        digest.update(b"\0")
        digest.update(content_sha256)

    return GitEvidence(
        commit_sha=commit,
        dirty=bool(status),
        diff_sha256=digest.hexdigest(),
        repository_root=root,
    )


def collect_package_versions(
    package_names: Sequence[str] = DEFAULT_VERSION_PACKAGES,
) -> dict[str, str]:
    """Return sorted installed-distribution versions for experiment lineage."""

    versions: dict[str, str] = {}
    for package_name in sorted(set(package_names)):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError as error:
            extra = OPTIONAL_PACKAGE_EXTRAS.get(package_name)
            install_hint = (
                f"install the project with the '{extra}' extra"
                if extra is not None
                else "install the project's locked dependencies"
            )
            raise TrackingDependencyError(
                f"Package metadata for {package_name!r} is unavailable; {install_hint}."
            ) from error
    return versions


def ensure_mlflow_available() -> None:
    """Fail early with the modeling-extra installation guidance when needed."""

    _load_mlflow()


def track_baseline_runs(
    *,
    tracking_root: str | Path,
    experiment_name: str,
    parent_run_name: str,
    parent_parameters: Mapping[str, ParameterValue],
    parent_tags: Mapping[str, str],
    model_runs: Sequence[ModelRunPayload],
    artifacts: Sequence[str | Path],
) -> TrackingRunResult:
    """Record one parent and exactly three nested baseline runs in SQLite MLflow."""

    payloads = tuple(model_runs)
    _validate_model_payloads(payloads)
    evidence_files = _validate_artifacts(artifacts)
    mlflow = _load_mlflow()

    root = Path(tracking_root).resolve()
    database_path = root / MLFLOW_DATABASE_FILENAME
    artifact_root = root / MLFLOW_ARTIFACT_DIRECTORY
    try:
        root.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise TrackingError(f"Unable to initialize MLflow tracking root {root}: {error}") from error

    tracking_uri = _sqlite_tracking_uri(database_path)
    with _MLFLOW_TRACKING_LOCK:
        previous_tracking_uri: str | None = None
        try:
            previous_tracking_uri = str(mlflow.get_tracking_uri())
        except Exception as error:
            raise TrackingError(
                f"Unable to read the existing MLflow tracking URI: {error}"
            ) from error
        try:
            try:
                mlflow.set_tracking_uri(tracking_uri)
                experiment_id = _resolve_experiment(
                    mlflow,
                    experiment_name=experiment_name,
                    artifact_root=artifact_root,
                    tracking_uri=tracking_uri,
                )
                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=parent_run_name,
                    tags={"run_role": "baseline_parent", **dict(parent_tags)},
                ) as parent_run:
                    parent_run_id = str(parent_run.info.run_id)
                    mlflow.log_params(_normalise_parameters(parent_parameters))
                    child_run_ids: list[tuple[str, str]] = []
                    for payload in payloads:
                        with mlflow.start_run(
                            experiment_id=experiment_id,
                            run_name=payload.model_name,
                            nested=True,
                            tags={
                                "run_role": "baseline_model",
                                "model_id": payload.model_name,
                            },
                        ) as child_run:
                            child_run_ids.append((payload.model_name, str(child_run.info.run_id)))
                            mlflow.log_params(_normalise_parameters(payload.parameters))
                            mlflow.log_metrics(_normalise_metrics(payload.metrics))

                    for artifact in evidence_files:
                        mlflow.log_artifact(
                            str(artifact),
                            artifact_path=ALLOWED_TRACKING_ARTIFACTS[artifact.name],
                        )
            except TrackingError:
                raise
            except Exception as error:
                raise TrackingError(f"MLflow baseline tracking failed: {error}") from error
        finally:
            if previous_tracking_uri is not None:
                try:
                    mlflow.set_tracking_uri(previous_tracking_uri)
                except Exception as error:
                    raise TrackingError(
                        f"Unable to restore the previous MLflow tracking URI: {error}"
                    ) from error

    return TrackingRunResult(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        parent_run_id=parent_run_id,
        child_run_ids=tuple(child_run_ids),
    )


def track_candidate_runs(
    *,
    tracking_root: str | Path,
    parent_parameters: Mapping[str, ParameterValue],
    parent_tags: Mapping[str, str],
    variant_runs: Sequence[ModelRunPayload],
    artifacts: Sequence[str | Path],
) -> TrackingRunResult:
    """Record one parent and the 10 governed Phase 3 variant runs."""

    payloads = tuple(variant_runs)
    _validate_candidate_payloads(payloads)
    evidence_files = _validate_candidate_artifacts(artifacts)
    mlflow = _load_mlflow()
    root = Path(tracking_root).resolve()
    database_path = root / MLFLOW_DATABASE_FILENAME
    artifact_root = root / MLFLOW_ARTIFACT_DIRECTORY
    try:
        root.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise TrackingError(f"Unable to initialize MLflow tracking root {root}: {error}") from error

    tracking_uri = _sqlite_tracking_uri(database_path)
    with _MLFLOW_TRACKING_LOCK:
        previous_tracking_uri: str | None = None
        try:
            previous_tracking_uri = str(mlflow.get_tracking_uri())
            mlflow.set_tracking_uri(tracking_uri)
            experiment_id = _resolve_experiment(
                mlflow,
                experiment_name=CANDIDATE_EXPERIMENT_NAME,
                artifact_root=artifact_root,
                tracking_uri=tracking_uri,
            )
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name="candidate_v1",
                tags={"run_role": "candidate_parent", **dict(parent_tags)},
            ) as parent_run:
                parent_run_id = str(parent_run.info.run_id)
                mlflow.log_params(_normalise_parameters(parent_parameters))
                child_run_ids: list[tuple[str, str]] = []
                for payload in payloads:
                    with mlflow.start_run(
                        experiment_id=experiment_id,
                        run_name=payload.model_name,
                        nested=True,
                        tags={"run_role": "candidate_variant", "variant_id": payload.model_name},
                    ) as child_run:
                        child_run_ids.append((payload.model_name, str(child_run.info.run_id)))
                        mlflow.log_params(_normalise_parameters(payload.parameters))
                        mlflow.log_metrics(_normalise_metrics(payload.metrics))
                for artifact in evidence_files:
                    mlflow.log_artifact(
                        str(artifact),
                        artifact_path=CANDIDATE_TRACKING_ARTIFACTS[artifact.name],
                    )
        except TrackingError:
            raise
        except Exception as error:
            raise TrackingError(f"MLflow candidate tracking failed: {error}") from error
        finally:
            if previous_tracking_uri is not None:
                try:
                    mlflow.set_tracking_uri(previous_tracking_uri)
                except Exception as error:
                    raise TrackingError(
                        f"Unable to restore the previous MLflow tracking URI: {error}"
                    ) from error
    return TrackingRunResult(
        tracking_uri=tracking_uri,
        experiment_name=CANDIDATE_EXPERIMENT_NAME,
        parent_run_id=parent_run_id,
        child_run_ids=tuple(child_run_ids),
    )


def track_selection_runs(
    *,
    tracking_root: str | Path,
    parent_parameters: Mapping[str, ParameterValue],
    parent_tags: Mapping[str, str],
    model_runs: Sequence[ModelRunPayload],
    artifacts: Sequence[str | Path],
) -> TrackingRunResult:
    """Record one parent and exactly four nested one-pass model runs."""

    payloads = tuple(model_runs)
    names = tuple(payload.model_name for payload in payloads)
    if names != SELECTION_MODEL_NAMES:
        raise TrackingError(
            f"Selection tracking requires exactly the ordered model runs {SELECTION_MODEL_NAMES}; "
            f"received {names}."
        )
    evidence_files = tuple(Path(path).resolve() for path in artifacts)
    artifact_names = tuple(path.name for path in evidence_files)
    expected_names = {*SELECTION_TRACKING_ARTIFACTS, "model.joblib", "model.cbm"}
    if len(set(artifact_names)) != len(artifact_names):
        raise TrackingError("Selection tracking artifact filenames must be unique.")
    if not all(path.is_file() for path in evidence_files):
        raise TrackingError("Every selection tracking artifact must be a regular file.")
    models = set(artifact_names) & {"model.joblib", "model.cbm"}
    required = set(SELECTION_TRACKING_ARTIFACTS)
    if (
        set(artifact_names) - expected_names
        or not required.issubset(artifact_names)
        or len(models) != 1
    ):
        raise TrackingError(
            "Selection tracking artifact allowlist mismatch; require aggregate evidence, runtime "
            "evidence, manifest, and exactly one model binary."
        )
    mlflow = _load_mlflow()
    root = Path(tracking_root).resolve()
    database_path = root / MLFLOW_DATABASE_FILENAME
    artifact_root = root / MLFLOW_ARTIFACT_DIRECTORY
    try:
        root.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise TrackingError(f"Unable to initialize MLflow tracking root {root}: {error}") from error

    tracking_uri = _sqlite_tracking_uri(database_path)
    with _MLFLOW_TRACKING_LOCK:
        previous_tracking_uri: str | None = None
        try:
            previous_tracking_uri = str(mlflow.get_tracking_uri())
            mlflow.set_tracking_uri(tracking_uri)
            experiment_id = _resolve_experiment(
                mlflow,
                experiment_name=SELECTION_EXPERIMENT_NAME,
                artifact_root=artifact_root,
                tracking_uri=tracking_uri,
            )
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name="selection_v1",
                tags={"run_role": "selection_parent", **dict(parent_tags)},
            ) as parent_run:
                parent_run_id = str(parent_run.info.run_id)
                mlflow.log_params(_normalise_parameters(parent_parameters))
                child_run_ids: list[tuple[str, str]] = []
                for payload in payloads:
                    with mlflow.start_run(
                        experiment_id=experiment_id,
                        run_name=payload.model_name,
                        nested=True,
                        tags={"run_role": "selection_model", "model_id": payload.model_name},
                    ) as child_run:
                        child_run_ids.append((payload.model_name, str(child_run.info.run_id)))
                        mlflow.log_params(_normalise_parameters(payload.parameters))
                        mlflow.log_metrics(_normalise_metrics(payload.metrics))
                for artifact in evidence_files:
                    artifact_path = (
                        "bundle"
                        if artifact.name in {"model.joblib", "model.cbm"}
                        else SELECTION_TRACKING_ARTIFACTS[artifact.name]
                    )
                    mlflow.log_artifact(str(artifact), artifact_path=artifact_path)
        except TrackingError:
            raise
        except Exception as error:
            raise TrackingError(f"MLflow selection tracking failed: {error}") from error
        finally:
            if previous_tracking_uri is not None:
                try:
                    mlflow.set_tracking_uri(previous_tracking_uri)
                except Exception as error:
                    raise TrackingError(
                        f"Unable to restore the previous MLflow tracking URI: {error}"
                    ) from error
    return TrackingRunResult(
        tracking_uri=tracking_uri,
        experiment_name=SELECTION_EXPERIMENT_NAME,
        parent_run_id=parent_run_id,
        child_run_ids=tuple(child_run_ids),
    )


def mark_tracking_run_failed(
    result: TrackingRunResult,
    *,
    failure_stage: str,
) -> None:
    """Correct the parent status when a post-tracking workflow stage fails."""

    if not failure_stage.strip():
        raise TrackingError("MLflow failure stage must not be blank.")
    mlflow = _load_mlflow()
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=result.tracking_uri)
        client.set_tag(result.parent_run_id, "workflow_failure_stage", failure_stage)
        client.set_terminated(result.parent_run_id, status="FAILED")
    except Exception as error:
        raise TrackingError(
            f"Unable to mark MLflow parent run {result.parent_run_id} failed: {error}"
        ) from error


def _resolve_experiment(
    mlflow: ModuleType,
    *,
    experiment_name: str,
    artifact_root: Path,
    tracking_uri: str,
) -> str:
    if not experiment_name.strip():
        raise TrackingError("MLflow experiment name must not be blank.")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    expected_location = artifact_root.resolve().as_uri().rstrip("/")
    if experiment is None:
        return str(
            client.create_experiment(
                experiment_name,
                artifact_location=expected_location,
            )
        )

    actual_location = str(experiment.artifact_location).rstrip("/")
    if actual_location != expected_location:
        raise TrackingError(
            "Existing MLflow experiment uses an unexpected artifact location: "
            f"expected {expected_location}, found {actual_location}."
        )
    return str(experiment.experiment_id)


def _validate_model_payloads(payloads: tuple[ModelRunPayload, ...]) -> None:
    names = tuple(payload.model_name for payload in payloads)
    if names != BASELINE_MODEL_NAMES:
        raise TrackingError(
            "Baseline tracking requires exactly the ordered model runs "
            f"{BASELINE_MODEL_NAMES}; received {names}."
        )


def _validate_artifacts(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    artifacts = tuple(Path(path).resolve() for path in paths)
    names = tuple(artifact.name for artifact in artifacts)
    if len(set(names)) != len(names):
        raise TrackingError("MLflow evidence artifact filenames must be unique.")
    unexpected = sorted(set(names) - set(ALLOWED_TRACKING_ARTIFACTS))
    missing = sorted(set(ALLOWED_TRACKING_ARTIFACTS) - set(names))
    if unexpected or missing:
        raise TrackingError(
            "MLflow evidence artifact allowlist mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )
    for artifact in artifacts:
        if not artifact.is_file():
            raise TrackingError(f"MLflow evidence artifact is missing or not a file: {artifact}")
    _validate_evidence_content({artifact.name: artifact for artifact in artifacts})
    return artifacts


def _validate_candidate_payloads(payloads: tuple[ModelRunPayload, ...]) -> None:
    names = tuple(payload.model_name for payload in payloads)
    if len(names) != 10 or len(set(names)) != 10:
        raise TrackingError("Candidate tracking requires exactly 10 unique variant runs.")
    expected_search = tuple(f"operational_full__cb_cfg_{index:03d}" for index in range(1, 9))
    if names[:8] != expected_search:
        raise TrackingError(
            "Candidate tracking requires the ordered cb_cfg_001..cb_cfg_008 full-view variants."
        )
    if not names[-2].startswith("repayment_status_only__") or not names[-1].startswith(
        "monetary_only__"
    ):
        raise TrackingError("Candidate tracking requires two trailing diagnostic variants.")
    diagnostic_configurations = tuple(name.rpartition("__")[2] for name in names[-2:])
    if len(set(diagnostic_configurations)) != 1 or diagnostic_configurations[0] not in {
        f"cb_cfg_{index:03d}" for index in range(1, 9)
    }:
        raise TrackingError(
            "Candidate diagnostic variants must reuse one reviewed full-view configuration."
        )


def _validate_candidate_artifacts(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    artifacts = tuple(Path(path).resolve() for path in paths)
    names = tuple(artifact.name for artifact in artifacts)
    if len(set(names)) != len(names):
        raise TrackingError("Candidate MLflow artifact filenames must be unique.")
    unexpected = sorted(set(names) - set(CANDIDATE_TRACKING_ARTIFACTS))
    missing = sorted(set(CANDIDATE_TRACKING_ARTIFACTS) - set(names))
    if unexpected or missing:
        raise TrackingError(
            "Candidate MLflow artifact allowlist mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )
    for artifact in artifacts:
        if not artifact.is_file():
            raise TrackingError(f"Candidate MLflow artifact is missing or not a file: {artifact}")
    _validate_candidate_evidence_content({artifact.name: artifact for artifact in artifacts})
    return artifacts


def _validate_candidate_evidence_content(artifacts: Mapping[str, Path]) -> None:
    try:
        summary_bytes = artifacts["summary.json"].read_bytes()
        report_bytes = artifacts["candidate-report.md"].read_bytes()
        diagnostics_bytes = artifacts["fold_diagnostics.json"].read_bytes()
        summary = json.loads(summary_bytes)
        diagnostics = json.loads(diagnostics_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrackingError(f"Unable to validate candidate MLflow artifacts: {error}") from error
    if not isinstance(summary, dict) or not isinstance(diagnostics, dict):
        raise TrackingError("Candidate summary and diagnostics must be JSON objects.")
    try:
        data = summary["data"]
        evidence_policy = summary["evidence_policy"]
        runtime = summary["runtime_artifacts"]
        fit_budget = summary["fit_budget"]
        summary_hash = hashlib.sha256(summary_bytes).hexdigest()
        oof_hash = _file_sha256(artifacts["oof_predictions.csv"])
        diagnostics_hash = hashlib.sha256(diagnostics_bytes).hexdigest()
    except (KeyError, TypeError, OSError) as error:
        raise TrackingError(f"Candidate evidence is missing required fields: {error}") from error
    if data != {
        "development_rows": 24_000,
        "holdout_evaluated": False,
        "n_repeats": 3,
        "n_splits": 5,
        "partition": "development",
    }:
        raise TrackingError("Candidate evidence violates the development-only data boundary.")
    if fit_budget.get("completed_fold_fits") != 150 or fit_budget.get("maximum_fold_fits") != 150:
        raise TrackingError("Candidate evidence does not record exactly 150 fold fits.")
    if (
        fit_budget.get("search_fold_fits") != 120
        or fit_budget.get("diagnostic_fold_fits") != 30
        or fit_budget.get("evaluated_variants") != 10
    ):
        raise TrackingError("Candidate evidence has an invalid optimized fit-budget breakdown.")
    if evidence_policy != {
        "independent_executions_required": 2,
        "required_byte_identical_artifacts": [
            "summary.json",
            "candidate-report.md",
            "oof_predictions.csv",
            "fold_diagnostics.json",
        ],
        "tracking_roots_must_be_independent": True,
        "third_fit_pass_for_publication": "prohibited",
    }:
        raise TrackingError("Candidate evidence has an invalid independent-execution policy.")
    if (
        runtime.get("contains_holdout_rows") is not False
        or runtime.get("contains_fitted_models") is not False
    ):
        raise TrackingError("Candidate runtime evidence violates the artifact boundary.")
    if runtime.get("oof_predictions_sha256") != oof_hash:
        raise TrackingError("Candidate summary and OOF hashes do not match.")
    if runtime.get("fold_diagnostics_sha256") != diagnostics_hash:
        raise TrackingError("Candidate summary and diagnostics hashes do not match.")
    if diagnostics.get("fit_count") != 150 or len(diagnostics.get("fits", ())) != 150:
        raise TrackingError("Candidate fold diagnostics do not contain exactly 150 fits.")
    try:
        report = report_bytes.decode("utf-8")
    except UnicodeError as error:
        raise TrackingError("Candidate report must be UTF-8 text.") from error
    if not report.startswith("# Governed CatBoost candidate report\n") or (
        f"Deterministic summary SHA-256:** `{summary_hash}`" not in report
    ):
        raise TrackingError("Candidate report is not bound to the deterministic summary.")
    _validate_candidate_oof(artifacts["oof_predictions.csv"], runtime)
    forbidden = {"artifact_uri", "run_id", "timestamp", "tracking_uri", "start_time", "end_time"}
    if _nested_keys(summary) & forbidden:
        raise TrackingError("Candidate deterministic evidence contains operational identifiers.")


def _validate_candidate_oof(path: Path, runtime: Mapping[str, object]) -> None:
    expected_header = [
        "account_id",
        "variant_id",
        "feature_view",
        "configuration_id",
        "repeat_index",
        "fold_index",
        "probability",
    ]
    rows = 0
    variants: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_header:
                raise TrackingError("Candidate OOF evidence has an unexpected schema.")
            for row in reader:
                probability = float(row["probability"])
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise TrackingError("Candidate OOF evidence contains an invalid probability.")
                if int(row["repeat_index"]) not in range(3) or int(row["fold_index"]) not in range(
                    5
                ):
                    raise TrackingError(
                        "Candidate OOF evidence contains an invalid fold assignment."
                    )
                variants.add(row["variant_id"])
                rows += 1
    except (OSError, UnicodeError, ValueError, KeyError) as error:
        raise TrackingError(f"Candidate OOF evidence contains an invalid row: {error}") from error
    if rows != runtime.get("oof_prediction_rows") or rows != 720_000:
        raise TrackingError("Candidate OOF evidence has incomplete row coverage.")
    if len(variants) != 10:
        raise TrackingError("Candidate OOF evidence does not contain exactly 10 variants.")


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _nested_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _nested_keys(item)}
    return set()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_evidence_content(artifacts: Mapping[str, Path]) -> None:
    try:
        summary_bytes = artifacts["summary.json"].read_bytes()
        report_bytes = artifacts["baseline-report.md"].read_bytes()
        oof_bytes = artifacts["oof_predictions.csv"].read_bytes()
        diagnostics_bytes = artifacts["logistic_fold_diagnostics.json"].read_bytes()
        summary = json.loads(summary_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrackingError(f"Unable to validate MLflow evidence artifacts: {error}") from error
    if not isinstance(summary, dict):
        raise TrackingError("MLflow summary evidence must be a JSON object.")
    try:
        data = summary["data"]
        runtime = summary["runtime_artifacts"]
        models = summary["models"]
        development_rows = int(data["development_rows"])
        n_repeats = int(data["n_repeats"])
        n_splits = int(data["n_splits"])
    except (KeyError, TypeError, ValueError) as error:
        raise TrackingError(
            "MLflow summary evidence is missing its governed data contract."
        ) from error
    if (
        development_rows < 1
        or n_repeats != 3
        or n_splits != 5
        or data.get("holdout_evaluated") is not False
        or runtime.get("contains_holdout_rows") is not False
        or runtime.get("contains_fitted_models") is not False
        or set(models) != set(BASELINE_MODEL_NAMES)
    ):
        raise TrackingError("MLflow summary evidence violates the baseline governance boundary.")
    observed_oof_sha256 = hashlib.sha256(oof_bytes).hexdigest()
    if runtime.get("oof_predictions_sha256") != observed_oof_sha256:
        raise TrackingError("MLflow summary and OOF artifact hashes do not match.")
    observed_diagnostics_sha256 = hashlib.sha256(diagnostics_bytes).hexdigest()
    if runtime.get("logistic_diagnostics_sha256") != observed_diagnostics_sha256:
        raise TrackingError("MLflow summary and logistic-diagnostics hashes do not match.")

    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    try:
        report = report_bytes.decode("utf-8")
    except UnicodeError as error:
        raise TrackingError("MLflow baseline report must be UTF-8 text.") from error
    if not report.startswith("# Governed baseline experiment report\n") or (
        f"**Deterministic summary SHA-256:** `{summary_sha256}`" not in report
    ):
        raise TrackingError("MLflow report is not bound to the deterministic summary artifact.")

    _validate_oof_content(
        oof_bytes,
        development_rows=development_rows,
        n_repeats=n_repeats,
        n_splits=n_splits,
    )
    _validate_logistic_diagnostics(
        diagnostics_bytes,
        n_repeats=n_repeats,
        n_splits=n_splits,
    )


def _validate_oof_content(
    content: bytes,
    *,
    development_rows: int,
    n_repeats: int,
    n_splits: int,
) -> None:
    expected_header = (
        "account_id",
        "model_id",
        "repeat_index",
        "fold_index",
        "prediction_kind",
        "score",
    )
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
    except UnicodeError as error:
        raise TrackingError("MLflow OOF evidence must be UTF-8 CSV.") from error
    if tuple(reader.fieldnames or ()) != expected_header:
        raise TrackingError("MLflow OOF evidence has an unexpected schema.")

    expected_kinds = {
        "fold_prevalence": "probability",
        "repayment_burden_rule": "risk_score",
        "logistic_l2": "probability",
    }
    keys: set[tuple[int, str, int]] = set()
    account_fold_assignments: dict[tuple[int, int], int] = {}
    account_sets: dict[tuple[str, int], set[int]] = {
        (model_id, repeat_index): set()
        for model_id in BASELINE_MODEL_NAMES
        for repeat_index in range(n_repeats)
    }
    fold_sets: dict[tuple[str, int], set[int]] = {key: set() for key in account_sets}
    try:
        for row in reader:
            if None in row:
                raise ValueError("row contains fields outside the governed OOF schema")
            account_id = int(row["account_id"])
            model_id = row["model_id"]
            repeat_index = int(row["repeat_index"])
            fold_index = int(row["fold_index"])
            score = float(row["score"])
            if (
                account_id < 1
                or model_id not in expected_kinds
                or repeat_index not in range(n_repeats)
                or fold_index not in range(n_splits)
                or row["prediction_kind"] != expected_kinds[model_id]
                or not math.isfinite(score)
                or (row["prediction_kind"] == "probability" and not 0.0 <= score <= 1.0)
                or (row["prediction_kind"] == "risk_score" and score < 0.0)
            ):
                raise ValueError("row violates the governed OOF domain")
            key = (account_id, model_id, repeat_index)
            if key in keys:
                raise ValueError("duplicate account/model/repeat OOF row")
            keys.add(key)
            assignment_key = (account_id, repeat_index)
            expected_fold = account_fold_assignments.setdefault(assignment_key, fold_index)
            if fold_index != expected_fold:
                raise ValueError(
                    "account uses inconsistent fold assignments across baseline models"
                )
            account_sets[(model_id, repeat_index)].add(account_id)
            fold_sets[(model_id, repeat_index)].add(fold_index)
    except (csv.Error, KeyError, TypeError, ValueError) as error:
        raise TrackingError(f"MLflow OOF evidence contains an invalid row: {error}") from error

    expected_rows = development_rows * n_repeats * len(BASELINE_MODEL_NAMES)
    if len(keys) != expected_rows or any(
        len(account_ids) != development_rows for account_ids in account_sets.values()
    ):
        raise TrackingError("MLflow OOF evidence has incomplete account/model/repeat coverage.")
    reference_ids = next(iter(account_sets.values()))
    if any(account_ids != reference_ids for account_ids in account_sets.values()):
        raise TrackingError("MLflow OOF evidence does not use one common development population.")
    if any(folds != set(range(n_splits)) for folds in fold_sets.values()):
        raise TrackingError("MLflow OOF evidence does not cover every reviewed fold.")


def _validate_logistic_diagnostics(
    content: bytes,
    *,
    n_repeats: int,
    n_splits: int,
) -> None:
    try:
        payload = json.loads(content)
        raw_feature_names = payload["transformed_feature_names"]
        folds = payload["folds"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise TrackingError("MLflow logistic diagnostics are not valid governed JSON.") from error
    if (
        payload.get("schema_version") != "1.0.0"
        or payload.get("model_id") != "logistic_l2"
        or not isinstance(raw_feature_names, list)
    ):
        raise TrackingError("MLflow logistic diagnostics violate the fold evidence contract.")
    feature_names = tuple(raw_feature_names)
    if (
        not feature_names
        or not all(isinstance(name, str) and name for name in feature_names)
        or len(set(feature_names)) != len(feature_names)
        or not isinstance(folds, list)
        or len(folds) != n_repeats * n_splits
    ):
        raise TrackingError("MLflow logistic diagnostics violate the fold evidence contract.")
    observed_folds: set[tuple[int, int]] = set()
    try:
        for fold in folds:
            repeat_index = int(fold["repeat_index"])
            fold_index = int(fold["fold_index"])
            iterations = int(fold["iterations"])
            intercept = float(fold["intercept"])
            coefficients = tuple(float(value) for value in fold["coefficients"])
            if (
                repeat_index not in range(n_repeats)
                or fold_index not in range(n_splits)
                or (repeat_index, fold_index) in observed_folds
                or iterations < 1
                or len(coefficients) != len(feature_names)
                or not math.isfinite(intercept)
                or not all(math.isfinite(value) for value in coefficients)
            ):
                raise ValueError("fold diagnostic violates the governed domain")
            observed_folds.add((repeat_index, fold_index))
    except (KeyError, TypeError, ValueError) as error:
        raise TrackingError(
            f"MLflow logistic diagnostics contain an invalid fold: {error}"
        ) from error


def _normalise_parameters(
    parameters: Mapping[str, ParameterValue],
) -> dict[str, ParameterValue]:
    normalised: dict[str, ParameterValue] = {}
    for key, value in sorted(parameters.items()):
        if not isinstance(key, str) or not key.strip():
            raise TrackingError("MLflow parameter names must be non-blank strings.")
        if isinstance(value, (str, int, float, bool)):
            normalised[key] = value
        else:  # pragma: no cover - protected by the public dataclass annotation
            raise TrackingError(f"MLflow parameter {key!r} has an unsupported value type.")
    return normalised


def _normalise_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    normalised: dict[str, float] = {}
    for key, value in sorted(metrics.items()):
        if not isinstance(key, str) or not key.strip():
            raise TrackingError("MLflow metric names must be non-blank strings.")
        numeric_value = float(value)
        if numeric_value != numeric_value or numeric_value in (float("inf"), float("-inf")):
            raise TrackingError(f"MLflow metric {key!r} must be finite.")
        normalised[key] = numeric_value
    return normalised


def _sqlite_tracking_uri(database_path: Path) -> str:
    return f"sqlite:///{database_path.resolve().as_posix()}"


def _load_mlflow() -> ModuleType:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r'Field "model_name" has conflict with protected namespace "model_"\.',
                category=UserWarning,
                module=r"pydantic\._internal\._fields",
            )
            import mlflow
    except ModuleNotFoundError as error:
        missing = error.name or "an MLflow dependency"
        raise TrackingDependencyError(
            f"MLflow dependency {missing!r} is unavailable; "
            "install the project with the 'modeling' extra."
        ) from error
    return mlflow


def _run_git(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise TrackingError(
            f"Unable to execute git for reproducibility evidence: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise TrackingError(f"Unable to collect git reproducibility evidence: {detail}")
    return completed.stdout


def _untracked_paths(status: bytes) -> tuple[Path, ...]:
    paths: list[Path] = []
    for entry in status.split(b"\0"):
        if not entry.startswith(b"?? "):
            continue
        value = entry[3:].decode("utf-8", errors="surrogateescape")
        paths.append(Path(value))
    return tuple(sorted(paths, key=lambda path: path.as_posix()))
