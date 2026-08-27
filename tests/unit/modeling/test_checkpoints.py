"""Tests for validated, hash-addressed candidate fold checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from credit_risk.modeling import checkpoints
from credit_risk.modeling.candidates import CandidateFoldResult
from credit_risk.modeling.tracking import GitEvidence


def test_checkpoint_round_trip_is_non_pickle_and_exact(tmp_path: Path) -> None:
    task = _task()
    expectation = _expectation()
    result = _result(expectation)

    path = checkpoints.save_fold_checkpoint(tmp_path, task, expectation, result)
    loaded = checkpoints.load_fold_checkpoint(tmp_path, task, expectation)

    assert path.suffix == ".npz"
    assert loaded.quarantined_paths == ()
    assert loaded.result is not None
    assert np.array_equal(loaded.result.probabilities, result.probabilities)
    assert loaded.result.diagnostics == expectation.diagnostics
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)


@pytest.mark.parametrize("content", (b"not-a-zip", b"PK\x03\x04partial"))
def test_corrupt_checkpoint_is_quarantined_and_refitted(
    content: bytes,
    tmp_path: Path,
) -> None:
    task = _task()
    path = checkpoints.checkpoint_path(tmp_path, task)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    loaded = checkpoints.load_fold_checkpoint(tmp_path, task, _expectation())

    assert loaded.result is None
    assert len(loaded.quarantined_paths) == 1
    assert loaded.quarantined_paths[0].read_bytes() == content
    assert not path.exists()


def test_foreign_lineage_checkpoint_is_quarantined(tmp_path: Path) -> None:
    expectation = _expectation()
    original_task = _task(lineage_hash="a" * 64)
    original_path = checkpoints.save_fold_checkpoint(
        tmp_path, original_task, expectation, _result(expectation)
    )
    foreign_task = _task(lineage_hash="b" * 64)
    foreign_path = checkpoints.checkpoint_path(tmp_path, foreign_task)
    foreign_path.parent.mkdir(parents=True)
    foreign_path.write_bytes(original_path.read_bytes())

    loaded = checkpoints.load_fold_checkpoint(tmp_path, foreign_task, expectation)

    assert loaded.result is None
    assert len(loaded.quarantined_paths) == 1
    assert not foreign_path.exists()


@pytest.mark.parametrize(
    ("member", "replacement"),
    (
        ("schema_version", np.asarray("2.0.0")),
        ("task_hash", np.asarray("0" * 64)),
        ("account_ids", np.asarray([1.0, 2.0], dtype=np.float64)),
        ("account_ids", np.asarray([2, 1], dtype=np.int64)),
        ("target", np.asarray([0.0, 1.0], dtype=np.float64)),
        ("target", np.asarray([1, 0], dtype=np.int8)),
        ("probabilities", np.asarray([0.2, 0.8], dtype=np.float32)),
        ("probabilities", np.asarray([0.2, 1.2], dtype=np.float64)),
        ("categorical_columns", np.asarray([b"status"], dtype=np.bytes_)),
        ("tree_count", np.asarray(301, dtype=np.int64)),
    ),
)
def test_schema_population_probability_and_tree_drift_are_quarantined(
    member: str,
    replacement: np.ndarray,
    tmp_path: Path,
) -> None:
    task = _task()
    expectation = _expectation()
    path = checkpoints.save_fold_checkpoint(tmp_path, task, expectation, _result(expectation))
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload[member] = replacement
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)

    loaded = checkpoints.load_fold_checkpoint(tmp_path, task, expectation)

    assert loaded.result is None
    assert len(loaded.quarantined_paths) == 1
    assert not path.exists()


def test_partial_write_is_quarantined_before_a_refit(tmp_path: Path) -> None:
    task = _task()
    path = checkpoints.checkpoint_path(tmp_path, task)
    path.parent.mkdir(parents=True)
    partial = path.parent / f".{path.name}.interrupted.partial"
    partial.write_bytes(b"incomplete")

    loaded = checkpoints.load_fold_checkpoint(tmp_path, task, _expectation())

    assert loaded.result is None
    assert len(loaded.quarantined_paths) == 1
    assert loaded.quarantined_paths[0].read_bytes() == b"incomplete"
    assert not partial.exists()


def test_repeated_identical_corruption_reuses_hash_addressed_quarantine(
    tmp_path: Path,
) -> None:
    task = _task()
    path = checkpoints.checkpoint_path(tmp_path, task)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"same-corruption")
    first = checkpoints.load_fold_checkpoint(tmp_path, task, _expectation())
    path.write_bytes(b"same-corruption")
    second = checkpoints.load_fold_checkpoint(tmp_path, task, _expectation())

    assert first.quarantined_paths == second.quarantined_paths
    assert first.quarantined_paths[0].read_bytes() == b"same-corruption"
    assert not path.exists()


def test_checkpoint_rejects_probability_or_diagnostic_drift(tmp_path: Path) -> None:
    task = _task()
    expectation = _expectation()
    invalid_probability = CandidateFoldResult(
        probabilities=np.asarray([0.2, np.nan]),
        diagnostics=expectation.diagnostics,
    )
    with pytest.raises(checkpoints.CheckpointError, match="finite"):
        checkpoints.save_fold_checkpoint(tmp_path, task, expectation, invalid_probability)

    changed = checkpoints.checkpoint_expectation(
        account_ids=[1, 2],
        target=[0, 1],
        train_target=[0, 1, 0, 1],
        predictor_count=2,
        categorical_columns=("status",),
        tree_count=600,
    )
    with pytest.raises(checkpoints.CheckpointError, match="diagnostics"):
        checkpoints.save_fold_checkpoint(tmp_path, task, changed, _result(expectation))


def test_checkpoint_never_overwrites_and_cleans_failed_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task()
    expectation = _expectation()
    result = _result(expectation)
    path = checkpoints.save_fold_checkpoint(tmp_path, task, expectation, result)
    with pytest.raises(checkpoints.CheckpointError, match="overwrite"):
        checkpoints.save_fold_checkpoint(tmp_path, task, expectation, result)

    second_task = _task(fold_index=1)
    monkeypatch.setattr(
        checkpoints.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(checkpoints.CheckpointError, match="Unable to publish"):
        checkpoints.save_fold_checkpoint(tmp_path, second_task, expectation, result)
    second_path = checkpoints.checkpoint_path(tmp_path, second_task)
    assert not second_path.exists()
    assert not list(second_path.parent.glob("*.partial"))
    assert path.is_file()


def test_task_hash_changes_with_every_governed_identity_boundary() -> None:
    original = _task()
    assert _task(lineage_hash="b" * 64) != original
    assert _task(commit_sha="2" * 40) != original
    assert _task(configuration_id="cb_cfg_002") != original
    assert _task(fold_index=1) != original


def _task(
    *,
    lineage_hash: str = "a" * 64,
    commit_sha: str = "1" * 40,
    configuration_id: str = "cb_cfg_001",
    fold_index: int = 0,
) -> checkpoints.CheckpointTask:
    return checkpoints.build_checkpoint_task(
        candidate_config_sha256="c" * 64,
        data_lineage={"canonical_sha256": lineage_hash, "assignment_sha256": "d" * 64},
        git=GitEvidence(
            commit_sha=commit_sha,
            dirty=False,
            diff_sha256="e" * 64,
        ),
        feature_view="operational_full",
        configuration_id=configuration_id,
        parameters={"depth": 4, "iterations": 300},
        predictor_columns=("amount", "status"),
        categorical_columns=("status",),
        repeat_index=0,
        fold_index=fold_index,
        train_account_ids=[3, 4, 5, 6],
        validation_account_ids=[1, 2],
        train_target=[0, 1, 0, 1],
        validation_target=[0, 1],
    )


def _expectation() -> checkpoints.CheckpointExpectation:
    return checkpoints.checkpoint_expectation(
        account_ids=[1, 2],
        target=[0, 1],
        train_target=[0, 1, 0, 1],
        predictor_count=2,
        categorical_columns=("status",),
        tree_count=300,
    )


def _result(expectation: checkpoints.CheckpointExpectation) -> CandidateFoldResult:
    return CandidateFoldResult(
        probabilities=np.asarray([0.2, 0.8], dtype=np.float64),
        diagnostics=expectation.diagnostics,
    )
