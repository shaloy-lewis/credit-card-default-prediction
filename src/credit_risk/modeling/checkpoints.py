"""Validated, non-executable fold checkpoints for Phase 3 candidate runs."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final

import numpy as np

from credit_risk.modeling.candidates import CandidateFoldDiagnostics, CandidateFoldResult
from credit_risk.modeling.tracking import GitEvidence

CHECKPOINT_SCHEMA_VERSION: Final[str] = "1.0.0"
CHECKPOINT_DIRECTORY: Final[str] = "candidate-checkpoints"
MAX_CHECKPOINT_BYTES: Final[int] = 2_000_000


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be stored, quarantined, or validated safely."""


@dataclass(frozen=True, slots=True)
class CheckpointTask:
    """Content-sensitive identity for one governed candidate fold."""

    task_hash: str


@dataclass(frozen=True, slots=True)
class CheckpointExpectation:
    """Expected validation population and deterministic fold diagnostics."""

    account_ids: np.ndarray
    target: np.ndarray
    diagnostics: CandidateFoldDiagnostics


@dataclass(frozen=True, slots=True)
class CheckpointLoad:
    """Validated reuse result plus any invalid files moved aside."""

    result: CandidateFoldResult | None
    quarantined_paths: tuple[Path, ...]


def build_checkpoint_task(
    *,
    candidate_config_sha256: str,
    data_lineage: Mapping[str, str],
    git: GitEvidence,
    feature_view: str,
    configuration_id: str,
    parameters: Mapping[str, Any],
    predictor_columns: Sequence[str],
    categorical_columns: Sequence[str],
    repeat_index: int,
    fold_index: int,
    train_account_ids: Sequence[int] | np.ndarray,
    validation_account_ids: Sequence[int] | np.ndarray,
    train_target: Sequence[int] | np.ndarray,
    validation_target: Sequence[int] | np.ndarray,
) -> CheckpointTask:
    """Build a stable hash covering code, protocol, data, view, and exact fold identity."""

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "candidate_config_sha256": candidate_config_sha256,
        "data_lineage": dict(sorted(data_lineage.items())),
        "git": {
            "commit_sha": git.commit_sha,
            "dirty": git.dirty,
            "diff_sha256": git.diff_sha256,
        },
        "feature_view": feature_view,
        "configuration_id": configuration_id,
        "parameters": dict(sorted(parameters.items())),
        "predictor_columns": list(predictor_columns),
        "categorical_columns": list(categorical_columns),
        "repeat_index": repeat_index,
        "fold_index": fold_index,
        "train_account_ids_sha256": _array_sha256(train_account_ids, np.int64),
        "validation_account_ids_sha256": _array_sha256(validation_account_ids, np.int64),
        "train_target_sha256": _array_sha256(train_target, np.int8),
        "validation_target_sha256": _array_sha256(validation_target, np.int8),
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return CheckpointTask(task_hash=hashlib.sha256(content).hexdigest())


def checkpoint_path(tracking_root: str | Path, task: CheckpointTask) -> Path:
    """Return the hash-addressed checkpoint path below one execution root."""

    _validate_digest(task.task_hash, "checkpoint task")
    root = Path(tracking_root).resolve() / CHECKPOINT_DIRECTORY
    return root / task.task_hash[:2] / f"{task.task_hash}.npz"


def load_fold_checkpoint(
    tracking_root: str | Path,
    task: CheckpointTask,
    expectation: CheckpointExpectation,
) -> CheckpointLoad:
    """Reuse a valid checkpoint; quarantine invalid and partial files for refitting."""

    path = checkpoint_path(tracking_root, task)
    quarantined: list[Path] = []
    if path.parent.is_dir():
        for partial in sorted(path.parent.glob(f".{path.name}.*.partial")):
            quarantined.append(_quarantine(partial, tracking_root))
    if not path.exists():
        return CheckpointLoad(result=None, quarantined_paths=tuple(quarantined))
    if not path.is_file():
        raise CheckpointError(f"Checkpoint destination is not a regular file: {path}")
    try:
        result = _read_checkpoint(path, task, expectation)
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile, CheckpointError):
        quarantined.append(_quarantine(path, tracking_root))
        return CheckpointLoad(result=None, quarantined_paths=tuple(quarantined))
    return CheckpointLoad(result=result, quarantined_paths=tuple(quarantined))


def save_fold_checkpoint(
    tracking_root: str | Path,
    task: CheckpointTask,
    expectation: CheckpointExpectation,
    result: CandidateFoldResult,
) -> Path:
    """Atomically store a validated fold result without pickle serialization."""

    _validate_result(result, expectation)
    path = checkpoint_path(tracking_root, task)
    if path.exists():
        raise CheckpointError(f"Refusing to overwrite an existing checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".partial",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            diagnostics = result.diagnostics
            np.savez_compressed(
                handle,
                schema_version=np.asarray(CHECKPOINT_SCHEMA_VERSION),
                task_hash=np.asarray(task.task_hash),
                account_ids=np.asarray(expectation.account_ids, dtype=np.int64),
                target=np.asarray(expectation.target, dtype=np.int8),
                probabilities=np.asarray(result.probabilities, dtype=np.float64),
                train_rows=np.asarray(diagnostics.train_rows, dtype=np.int64),
                validation_rows=np.asarray(diagnostics.validation_rows, dtype=np.int64),
                train_class_counts=np.asarray(diagnostics.train_class_counts, dtype=np.int64),
                validation_class_counts=np.asarray(
                    diagnostics.validation_class_counts, dtype=np.int64
                ),
                predictor_count=np.asarray(diagnostics.predictor_count, dtype=np.int64),
                categorical_columns=np.asarray(diagnostics.categorical_columns, dtype=np.str_),
                tree_count=np.asarray(diagnostics.tree_count, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():  # pragma: no cover - concurrent writers are prohibited operationally
            raise CheckpointError(f"Checkpoint appeared during publication: {path}")
        os.replace(temporary_name, path)
        temporary_name = None
    except CheckpointError:
        raise
    except OSError as error:
        raise CheckpointError(f"Unable to publish checkpoint {path}: {error}") from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return path


def checkpoint_expectation(
    *,
    account_ids: Sequence[int] | np.ndarray,
    target: Sequence[int] | np.ndarray,
    train_target: Sequence[int] | np.ndarray,
    predictor_count: int,
    categorical_columns: tuple[str, ...],
    tree_count: int,
) -> CheckpointExpectation:
    """Construct the exact diagnostics expected for one fold."""

    ids = np.asarray(account_ids, dtype=np.int64)
    validation_target = np.asarray(target, dtype=np.int8)
    training_target = np.asarray(train_target, dtype=np.int8)
    return CheckpointExpectation(
        account_ids=ids,
        target=validation_target,
        diagnostics=CandidateFoldDiagnostics(
            train_rows=len(training_target),
            validation_rows=len(validation_target),
            train_class_counts=_class_counts(training_target),
            validation_class_counts=_class_counts(validation_target),
            predictor_count=predictor_count,
            categorical_columns=categorical_columns,
            tree_count=tree_count,
        ),
    )


def _read_checkpoint(
    path: Path,
    task: CheckpointTask,
    expectation: CheckpointExpectation,
) -> CandidateFoldResult:
    _validate_archive_envelope(path)
    with np.load(path, allow_pickle=False) as checkpoint:
        expected_members = {
            "schema_version",
            "task_hash",
            "account_ids",
            "target",
            "probabilities",
            "train_rows",
            "validation_rows",
            "train_class_counts",
            "validation_class_counts",
            "predictor_count",
            "categorical_columns",
            "tree_count",
        }
        if set(checkpoint.files) != expected_members:
            raise CheckpointError("Checkpoint schema members differ from the reviewed contract.")
        if _text_scalar(checkpoint["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("Checkpoint schema version is unsupported.")
        if _text_scalar(checkpoint["task_hash"]) != task.task_hash:
            raise CheckpointError("Checkpoint belongs to a different governed fold task.")
        account_ids = _typed_vector(checkpoint["account_ids"], np.dtype(np.int64), "account IDs")
        target = _typed_vector(checkpoint["target"], np.dtype(np.int8), "labels")
        probabilities = _typed_vector(
            checkpoint["probabilities"], np.dtype(np.float64), "probabilities"
        )
        categorical_values = checkpoint["categorical_columns"]
        if categorical_values.ndim != 1 or categorical_values.dtype.kind != "U":
            raise CheckpointError("Checkpoint categorical-column metadata must be a string vector.")
        diagnostics = CandidateFoldDiagnostics(
            train_rows=_integer_scalar(checkpoint["train_rows"]),
            validation_rows=_integer_scalar(checkpoint["validation_rows"]),
            train_class_counts=_integer_pair(checkpoint["train_class_counts"]),
            validation_class_counts=_integer_pair(checkpoint["validation_class_counts"]),
            predictor_count=_integer_scalar(checkpoint["predictor_count"]),
            categorical_columns=tuple(str(value) for value in categorical_values.tolist()),
            tree_count=_integer_scalar(checkpoint["tree_count"]),
        )
    if not np.array_equal(account_ids, expectation.account_ids):
        raise CheckpointError("Checkpoint validation account IDs do not match the fold.")
    if not np.array_equal(target, expectation.target):
        raise CheckpointError("Checkpoint validation labels do not match the fold.")
    result = CandidateFoldResult(probabilities=probabilities, diagnostics=diagnostics)
    _validate_result(result, expectation)
    return result


def _validate_result(
    result: CandidateFoldResult,
    expectation: CheckpointExpectation,
) -> None:
    ids = np.asarray(expectation.account_ids)
    target = np.asarray(expectation.target)
    probabilities = np.asarray(result.probabilities)
    if ids.ndim != 1 or target.shape != ids.shape or probabilities.shape != ids.shape:
        raise CheckpointError("Checkpoint arrays are empty or misaligned.")
    if len(ids) < 1 or len(np.unique(ids)) != len(ids):
        raise CheckpointError("Checkpoint validation account IDs must be non-empty and unique.")
    if not np.array_equal(np.unique(target), np.asarray([0, 1], dtype=np.int8)):
        raise CheckpointError("Checkpoint validation labels must contain both binary classes.")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise CheckpointError("Checkpoint probabilities must be finite and within [0, 1].")
    if result.diagnostics != expectation.diagnostics:
        raise CheckpointError("Checkpoint diagnostics differ from the governed fold contract.")


def _validate_archive_envelope(path: Path) -> None:
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise CheckpointError("Checkpoint archive exceeds the maximum expected size.")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if any(member.flag_bits & 0x1 for member in members):
            raise CheckpointError("Encrypted checkpoint members are prohibited.")
        if sum(member.file_size for member in members) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("Checkpoint expands beyond the maximum expected size.")


def _quarantine(path: Path, tracking_root: str | Path) -> Path:
    try:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        quarantine_suffix = ".partial" if path.suffix == ".partial" else ".npz"
        destination = (
            Path(tracking_root).resolve()
            / CHECKPOINT_DIRECTORY
            / "quarantine"
            / digest[:2]
            / f"{digest}{quarantine_suffix}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != content:
                raise CheckpointError(f"Checkpoint quarantine destination conflicts: {destination}")
            path.unlink()
        else:
            os.replace(path, destination)
        return destination
    except CheckpointError:
        raise
    except OSError as error:
        raise CheckpointError(f"Unable to quarantine invalid checkpoint {path}: {error}") from error


def _array_sha256(values: Sequence[int] | np.ndarray, dtype: type[np.generic]) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _class_counts(target: np.ndarray) -> tuple[int, int]:
    return int(np.count_nonzero(target == 0)), int(np.count_nonzero(target == 1))


def _text_scalar(value: np.ndarray) -> str:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise CheckpointError("Checkpoint text metadata must be a scalar string.")
    return str(value.item())


def _integer_scalar(value: np.ndarray) -> int:
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise CheckpointError("Checkpoint integer metadata must be a scalar integer.")
    return int(value.item())


def _integer_pair(value: np.ndarray) -> tuple[int, int]:
    if value.shape != (2,) or value.dtype.kind not in {"i", "u"}:
        raise CheckpointError("Checkpoint class counts must contain two integers.")
    return int(value[0]), int(value[1])


def _typed_vector(value: np.ndarray, dtype: np.dtype[Any], label: str) -> np.ndarray:
    if value.ndim != 1 or value.dtype != dtype:
        raise CheckpointError(f"Checkpoint {label} must be a one-dimensional {dtype} array.")
    return np.asarray(value)


def _validate_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CheckpointError(f"{label.capitalize()} hash must be lowercase SHA-256.")
