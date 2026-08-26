"""Tests for governed Phase 3 orchestration and deterministic evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from credit_risk.modeling import candidate_workflow as workflow
from credit_risk.modeling.candidate_contracts import load_candidate_config
from credit_risk.modeling.candidates import CandidateFoldDiagnostics, CandidateFoldResult
from credit_risk.modeling.contracts import AUDIT_COLUMNS, PREDICTOR_COLUMNS
from credit_risk.modeling.dataset import GovernedDevelopmentData, ModelingLineage
from credit_risk.modeling.tracking import GitEvidence, TrackingError, TrackingRunResult

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_candidate_workflow_runs_exact_budget_and_is_byte_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governed = _governed_data()
    fit_calls: list[tuple[tuple[str, ...], int]] = []
    tracked: list[dict[str, Any]] = []

    def fake_fit(X_train, y_train, X_validation, y_validation, **kwargs):
        del y_train
        fit_calls.append((tuple(X_train.columns), kwargs["sampled_parameters"].iterations))
        status = pd.to_numeric(X_validation.iloc[:, 0]).to_numpy(dtype=float)
        probabilities = 0.15 + 0.7 * (status - status.min()) / max(status.max() - status.min(), 1)
        return CandidateFoldResult(
            probabilities=probabilities,
            diagnostics=CandidateFoldDiagnostics(
                train_rows=len(X_train),
                validation_rows=len(X_validation),
                train_class_counts=(40, 40),
                validation_class_counts=(10, 10),
                predictor_count=X_train.shape[1],
                categorical_columns=kwargs["categorical_columns"],
                tree_count=kwargs["sampled_parameters"].iterations,
            ),
        )

    def fake_track(**kwargs):
        tracked.append(kwargs)
        assert len(kwargs["variant_runs"]) == 14
        assert {Path(path).name for path in kwargs["artifacts"]} == {
            "summary.json",
            "candidate-report.md",
            "oof_predictions.csv",
            "fold_diagnostics.json",
        }
        return TrackingRunResult(
            tracking_uri="sqlite:///candidate.db",
            experiment_name="credit-risk-candidate-v1",
            parent_run_id="parent",
            child_run_ids=tuple(
                (payload.model_name, f"child-{index}")
                for index, payload in enumerate(kwargs["variant_runs"])
            ),
        )

    monkeypatch.setattr(workflow, "ensure_mlflow_available", lambda: None)
    monkeypatch.setattr(
        workflow,
        "collect_package_versions",
        lambda *_args: {"catboost": "1.2.5", "scikit-learn": "1.4.2"},
    )
    monkeypatch.setattr(
        workflow,
        "collect_git_evidence",
        lambda _root: GitEvidence(
            commit_sha="1" * 40,
            dirty=False,
            diff_sha256="2" * 64,
            repository_root=REPOSITORY_ROOT,
        ),
    )
    monkeypatch.setattr(workflow, "load_governed_development_data", lambda **_kwargs: governed)
    monkeypatch.setattr(workflow, "_validate_runtime_contract", lambda *_args: None)
    monkeypatch.setattr(workflow, "fit_candidate_fold", fake_fit)
    monkeypatch.setattr(workflow, "track_candidate_runs", fake_track)

    kwargs = {
        "repo_root": REPOSITORY_ROOT,
        "tracking_root": tmp_path / "tracking",
        "output_root": tmp_path / "reports",
    }
    first = workflow.run_candidate_experiment(**kwargs)
    first_summary = first.summary_path.read_bytes()
    first_report = first.report_path.read_bytes()
    first_oof = first.oof_predictions_path.read_bytes()
    first_diagnostics = first.fold_diagnostics_path.read_bytes()
    second = workflow.run_candidate_experiment(**kwargs)

    assert len(fit_calls) == 420
    assert first.summary_sha256 == second.summary_sha256
    assert first.report_sha256 == second.report_sha256
    assert first.oof_predictions_sha256 == second.oof_predictions_sha256
    assert first.fold_diagnostics_sha256 == second.fold_diagnostics_sha256
    assert second.summary_path.read_bytes() == first_summary
    assert second.report_path.read_bytes() == first_report
    assert second.oof_predictions_path.read_bytes() == first_oof
    assert second.fold_diagnostics_path.read_bytes() == first_diagnostics
    assert len(tracked) == 2

    summary = json.loads(first_summary)
    assert summary["fit_budget"]["completed_fold_fits"] == 210
    assert summary["fit_budget"]["evaluated_variants"] == 14
    assert summary["data"]["holdout_evaluated"] is False
    assert summary["selection"]["diagnostic_views_eligible_for_advancement"] is False
    assert len(summary["variants"]) == 14
    diagnostics = json.loads(first_diagnostics)
    assert diagnostics["fit_count"] == len(diagnostics["fits"]) == 210
    with first.oof_predictions_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14 * 3 * len(governed.account_ids)
    assert len({row["variant_id"] for row in rows}) == 14


def test_dirty_candidate_runs_require_the_exact_provisional_root(tmp_path: Path) -> None:
    git = GitEvidence(commit_sha="1" * 40, dirty=True, diff_sha256="2" * 64)

    with pytest.raises(workflow.CandidateWorkflowError, match="dirty"):
        workflow._enforce_output_policy(
            git,
            allow_dirty=False,
            output_root=tmp_path,
            repo_root=REPOSITORY_ROOT,
        )
    with pytest.raises(workflow.CandidateWorkflowError, match="provisional"):
        workflow._enforce_output_policy(
            git,
            allow_dirty=True,
            output_root=tmp_path,
            repo_root=REPOSITORY_ROOT,
        )
    workflow._enforce_output_policy(
        git,
        allow_dirty=True,
        output_root=REPOSITORY_ROOT / workflow.PROVISIONAL_OUTPUT_ROOT,
        repo_root=REPOSITORY_ROOT,
    )


def test_runtime_rejects_dependency_drift() -> None:
    config = load_candidate_config()

    with pytest.raises(workflow.CandidateWorkflowError, match="version mismatch"):
        workflow._validate_package_versions(
            config,
            {"catboost": "1.2.4", "scikit-learn": "1.4.2"},
        )


@pytest.mark.parametrize(
    ("error", "message"),
    (
        (workflow.CandidateWorkflowError("governed"), "governed"),
        (TrackingError("tracking"), "tracking"),
        (workflow.CandidateModelError("model"), "validation failed: model"),
        (OSError("disk"), "experiment failed: disk"),
    ),
)
def test_public_workflow_normalises_expected_failures(
    error: Exception,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow,
        "_run_candidate_experiment",
        lambda **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(workflow.CandidateWorkflowError, match=message):
        workflow.run_candidate_experiment()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("rows", "row count"),
        ("folds", "5 folds"),
        ("assignment", "split-assignment"),
        ("feature_hash", "feature-contract lineage"),
        ("columns", "predictors"),
        ("partition", "non-development"),
    ),
)
def test_runtime_contract_rejects_lineage_or_boundary_drift(
    mutation: str,
    message: str,
) -> None:
    config = load_candidate_config()
    runtime = _runtime_contract_stub()
    if mutation == "rows":
        runtime.account_ids = range(23_999)
    elif mutation == "folds":
        runtime.n_splits = 4
    elif mutation == "assignment":
        runtime.lineage.assignment_sha256 = "0" * 64
    elif mutation == "feature_hash":
        runtime.lineage.feature_contract_sha256 = "0" * 64
    elif mutation == "columns":
        runtime.predictors.columns = [*PREDICTOR_COLUMNS[:-1], "unexpected"]
    elif mutation == "partition":
        runtime.assignments = pd.DataFrame({"partition": ["development", "test"]})

    with pytest.raises(workflow.CandidateWorkflowError, match=message):
        workflow._validate_runtime_contract(
            config,
            runtime,
            config.data_contract.feature_contract_sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("holdout", "holdout"),
        ("schema", "invalid schema"),
        ("metrics", "reference metrics"),
    ),
)
def test_reference_validation_rejects_invalid_baseline_evidence(
    mutation: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_candidate_config()
    baseline = json.loads(
        (REPOSITORY_ROOT / config.baseline_evidence.summary_path).read_text(encoding="utf-8")
    )
    if mutation == "holdout":
        baseline["data"]["holdout_evaluated"] = True
    elif mutation == "schema":
        del baseline["models"]
    elif mutation == "metrics":
        baseline["models"]["logistic_l2"]["repeat_summaries"]["average_precision"]["mean"] = 0.1
    feature_bytes = (REPOSITORY_ROOT / config.data_contract.feature_contract_path).read_bytes()

    def fake_reference(_root, _path, _digest, label):
        if label == "baseline summary":
            return json.dumps(baseline).encode()
        if label == "feature contract":
            return feature_bytes
        return b"report"

    monkeypatch.setattr(workflow, "_verified_reference", fake_reference)
    with pytest.raises(workflow.CandidateWorkflowError, match=message):
        workflow._validate_references(config, REPOSITORY_ROOT)


def test_reference_and_file_helpers_reject_unsafe_or_changed_inputs(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.json"
    outside.write_bytes(b"{}")
    with pytest.raises(workflow.CandidateWorkflowError, match="escapes"):
        workflow._verified_reference(tmp_path, "../outside.json", "0" * 64, "test")
    inside = tmp_path / "inside.json"
    inside.write_bytes(b"{}")
    with pytest.raises(workflow.CandidateWorkflowError, match="hash mismatch"):
        workflow._verified_reference(tmp_path, "inside.json", "0" * 64, "test")
    with pytest.raises(workflow.CandidateWorkflowError, match="missing"):
        workflow._resolve_file(Path("missing.json"), tmp_path, "candidate configuration")
    with pytest.raises(workflow.CandidateWorkflowError, match="missing"):
        workflow._read_file(tmp_path / "missing.json", "input")


def test_fold_and_coverage_validation_reject_invalid_outputs() -> None:
    governed = _governed_data()
    with pytest.raises(workflow.CandidateWorkflowError, match="invalid probabilities"):
        workflow._validate_block(
            np.asarray([1, 2]),
            np.asarray([0, 1], dtype=np.int8),
            np.asarray([0.2, 1.2]),
        )
    with pytest.raises(workflow.CandidateWorkflowError, match="unique variants"):
        workflow._validate_oof_coverage(governed, (), expected_variants=14)
    with pytest.raises(workflow.CandidateWorkflowError, match="misaligned"):
        workflow._validate_block(np.asarray([]), np.asarray([]), np.asarray([]))
    with pytest.raises(workflow.CandidateWorkflowError, match="duplicate"):
        workflow._validate_block(
            np.asarray([1, 1]),
            np.asarray([0, 1], dtype=np.int8),
            np.asarray([0.2, 0.8]),
        )
    with pytest.raises(workflow.CandidateWorkflowError, match="both binary"):
        workflow._validate_block(
            np.asarray([1, 2]),
            np.asarray([0, 0], dtype=np.int8),
            np.asarray([0.2, 0.8]),
        )


def test_publication_rolls_back_when_the_second_output_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "summary.json"
    second = tmp_path / "candidate-report.md"
    first.write_bytes(b"old-summary")
    second.write_bytes(b"old-report")
    real_write = workflow._write_atomic
    calls = 0

    def fail_second(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        real_write(path, content)

    monkeypatch.setattr(workflow, "_write_atomic", fail_second)
    with pytest.raises(workflow.CandidateWorkflowError, match="Unable to publish"):
        workflow._promote_outputs({first: b"new-summary", second: b"new-report"})

    assert first.read_bytes() == b"old-summary"
    assert second.read_bytes() == b"old-report"


def test_evidence_helpers_reject_conflicting_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(workflow.CandidateWorkflowError, match="not a regular file"):
        workflow._promote_outputs({directory: b"content"})
    with pytest.raises(workflow.CandidateWorkflowError, match="not a file"):
        workflow._publish_immutable(directory, b"content")
    immutable = tmp_path / "immutable.json"
    immutable.write_bytes(b"old")
    with pytest.raises(workflow.CandidateWorkflowError, match="different bytes"):
        workflow._publish_immutable(immutable, b"new")
    monkeypatch.setattr(
        workflow,
        "_write_atomic",
        lambda *_args: (_ for _ in ()).throw(OSError("disk")),
    )
    with pytest.raises(workflow.CandidateWorkflowError, match="runtime evidence"):
        workflow._publish_immutable(tmp_path / "new.json", b"new")


def test_missing_feature_view_is_rejected() -> None:
    with pytest.raises(workflow.CandidateWorkflowError, match="missing"):
        workflow._view(load_candidate_config(), "unknown")


def _governed_data(rows: int = 100) -> GovernedDevelopmentData:
    account_ids = pd.Index(range(1, rows + 1), name="account_id")
    row_numbers = np.arange(rows)
    predictors = pd.DataFrame(index=account_ids)
    for offset, column in enumerate(PREDICTOR_COLUMNS):
        predictors[column] = ((row_numbers + offset) % 7) - 2
    target = pd.Series(row_numbers % 2, index=account_ids, name="default_next_month").astype("int8")
    audit = pd.DataFrame(
        {
            "account_id": account_ids,
            "default_next_month": target.to_numpy(),
            **{column: np.ones(rows, dtype=int) for column in AUDIT_COLUMNS},
        },
        index=account_ids,
    )
    assignments = pd.DataFrame(
        {
            "partition": ["development"] * rows,
            **{f"cv_fold_r{repeat}": (row_numbers + repeat) % 5 for repeat in range(3)},
        },
        index=account_ids,
    )
    lineage = ModelingLineage(
        dataset_id="uci_credit_default",
        dataset_version="v1",
        source_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        canonical_sha256="c" * 64,
        quality_report_sha256="d" * 64,
        split_config_sha256="e" * 64,
        assignment_sha256="2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e",
        split_manifest_sha256="f" * 64,
        reviewed_split_lock_sha256="0" * 64,
        feature_contract_sha256="8978277ae1c92b6f0b8daed94cccf3cd51d8e6cae0aa9c0620d8cfb813384a4b",
    )
    return GovernedDevelopmentData(
        account_ids=account_ids,
        predictors=predictors,
        target=target,
        audit=audit,
        assignments=assignments,
        lineage=lineage,
        n_splits=5,
        n_repeats=3,
    )


def _runtime_contract_stub() -> SimpleNamespace:
    return SimpleNamespace(
        account_ids=range(24_000),
        n_splits=5,
        n_repeats=3,
        lineage=SimpleNamespace(
            assignment_sha256=("2f6e2cdd0b29617a48ab6fcbdabd6859822c8ad2b6b5d77665967852cb4a034e"),
            feature_contract_sha256=(
                "8978277ae1c92b6f0b8daed94cccf3cd51d8e6cae0aa9c0620d8cfb813384a4b"
            ),
        ),
        predictors=SimpleNamespace(columns=list(PREDICTOR_COLUMNS)),
        assignments=pd.DataFrame({"partition": ["development"]}),
    )
