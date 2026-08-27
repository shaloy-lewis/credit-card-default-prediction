"""Tests for two-run candidate reproducibility verification and promotion."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from credit_risk.modeling import candidate_evidence as evidence
from credit_risk.modeling.candidate_workflow import (
    CandidateExperimentResult,
    CandidateWorkflowError,
)
from credit_risk.modeling.tracking import GitEvidence, TrackingRunResult


def test_evidence_runs_exactly_twice_in_isolated_roots_and_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> CandidateExperimentResult:
        calls.append(kwargs)
        return _result(Path(kwargs["output_root"]), Path(kwargs["tracking_root"]))

    _patch_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(evidence, "run_candidate_experiment", fake_run)
    result = evidence.run_candidate_evidence(
        repo_root=tmp_path,
        data_root="data",
        tracking_root="experiment/primary",
        verification_root="experiment/verification",
        output_root="reports/modeling/candidate_v1",
    )

    assert len(calls) == 2
    assert Path(calls[0]["tracking_root"]) != Path(calls[1]["tracking_root"])
    assert calls[0]["allow_dirty"] is False
    assert calls[1]["allow_dirty"] is False
    assert result.summary_path.read_bytes() == b"summary\n"
    assert result.report_path.read_bytes() == b"report\n"
    assert result.primary.summary_path != result.verification.summary_path
    assert result.summary_sha256 == hashlib.sha256(b"summary\n").hexdigest()


def test_digest_mismatch_marks_both_runs_failed_and_preserves_official_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    marked: list[tuple[str, str]] = []
    official = tmp_path / "reports" / "modeling" / "candidate_v1"
    official.mkdir(parents=True)
    (official / "summary.json").write_bytes(b"old-summary")
    (official / "candidate-report.md").write_bytes(b"old-report")

    def fake_run(**kwargs: Any) -> CandidateExperimentResult:
        nonlocal calls
        calls += 1
        report = b"report\n" if calls == 1 else b"different\n"
        return _result(
            Path(kwargs["output_root"]),
            Path(kwargs["tracking_root"]),
            report=report,
            run_id=f"run-{calls}",
        )

    _patch_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(evidence, "run_candidate_experiment", fake_run)
    monkeypatch.setattr(
        evidence,
        "mark_tracking_run_failed",
        lambda result, *, failure_stage: marked.append((result.parent_run_id, failure_stage)),
    )

    with pytest.raises(evidence.CandidateEvidenceError, match="different"):
        evidence.run_candidate_evidence(
            repo_root=tmp_path,
            tracking_root="experiment/primary",
            verification_root="experiment/verification",
            output_root="reports/modeling/candidate_v1",
        )

    assert calls == 2
    assert marked == [
        ("run-1", "reproducibility_mismatch"),
        ("run-2", "reproducibility_mismatch"),
    ]
    assert (official / "summary.json").read_bytes() == b"old-summary"
    assert (official / "candidate-report.md").read_bytes() == b"old-report"


def test_publication_failure_marks_both_runs_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    marked: list[str] = []

    def fake_run(**kwargs: Any) -> CandidateExperimentResult:
        nonlocal calls
        calls += 1
        return _result(
            Path(kwargs["output_root"]),
            Path(kwargs["tracking_root"]),
            run_id=f"run-{calls}",
        )

    _patch_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(evidence, "run_candidate_experiment", fake_run)
    monkeypatch.setattr(
        evidence,
        "_promote_outputs",
        lambda _payloads: (_ for _ in ()).throw(CandidateWorkflowError("disk full")),
    )
    monkeypatch.setattr(
        evidence,
        "mark_tracking_run_failed",
        lambda result, *, failure_stage: marked.append(f"{result.parent_run_id}:{failure_stage}"),
    )

    with pytest.raises(evidence.CandidateEvidenceError, match="disk full"):
        evidence.run_candidate_evidence(
            repo_root=tmp_path,
            tracking_root="experiment/primary",
            verification_root="experiment/verification",
            output_root="reports/modeling/candidate_v1",
        )

    assert calls == 2
    assert marked == [
        "run-1:official_evidence_publication",
        "run-2:official_evidence_publication",
    ]


def test_second_execution_failure_marks_the_completed_primary_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    marked: list[str] = []

    def fake_run(**kwargs: Any) -> CandidateExperimentResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CandidateWorkflowError("verification fit failed")
        return _result(Path(kwargs["output_root"]), Path(kwargs["tracking_root"]))

    _patch_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(evidence, "run_candidate_experiment", fake_run)
    monkeypatch.setattr(
        evidence,
        "mark_tracking_run_failed",
        lambda result, *, failure_stage: marked.append(failure_stage),
    )

    with pytest.raises(evidence.CandidateEvidenceError, match="verification execution failed"):
        evidence.run_candidate_evidence(
            repo_root=tmp_path,
            tracking_root="experiment/primary",
            verification_root="experiment/verification",
            output_root="reports/modeling/candidate_v1",
        )

    assert calls == 2
    assert marked == ["independent_execution_failed"]


def test_result_hash_mismatch_marks_both_completed_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    marked: list[str] = []

    def fake_run(**kwargs: Any) -> CandidateExperimentResult:
        nonlocal calls
        calls += 1
        result = _result(
            Path(kwargs["output_root"]),
            Path(kwargs["tracking_root"]),
            run_id=f"run-{calls}",
        )
        return replace(result, summary_sha256="0" * 64) if calls == 1 else result

    _patch_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(evidence, "run_candidate_experiment", fake_run)
    monkeypatch.setattr(
        evidence,
        "mark_tracking_run_failed",
        lambda _result, *, failure_stage: marked.append(failure_stage),
    )

    with pytest.raises(evidence.CandidateEvidenceError, match="hash does not match"):
        evidence.run_candidate_evidence(
            repo_root=tmp_path,
            tracking_root="experiment/primary",
            verification_root="experiment/verification",
            output_root="reports/modeling/candidate_v1",
        )

    assert calls == 2
    assert marked == [
        "reproducibility_artifact_validation",
        "reproducibility_artifact_validation",
    ]


def test_lineage_change_before_promotion_marks_both_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_calls = 0
    marked: list[str] = []

    def changing_git(_root: Path) -> GitEvidence:
        nonlocal git_calls
        git_calls += 1
        return GitEvidence(
            commit_sha="1" * 40,
            dirty=git_calls > 1,
            diff_sha256=("2" if git_calls == 1 else "3") * 64,
            repository_root=tmp_path,
        )

    monkeypatch.setattr(evidence, "collect_git_evidence", changing_git)
    monkeypatch.setattr(evidence, "verify_dataset", lambda **_kwargs: None)
    monkeypatch.setattr(
        evidence,
        "run_candidate_experiment",
        lambda **kwargs: _result(Path(kwargs["output_root"]), Path(kwargs["tracking_root"])),
    )
    monkeypatch.setattr(
        evidence,
        "mark_tracking_run_failed",
        lambda _result, *, failure_stage: marked.append(failure_stage),
    )

    with pytest.raises(evidence.CandidateEvidenceError, match="Git lineage changed"):
        evidence.run_candidate_evidence(
            repo_root=tmp_path,
            tracking_root="experiment/primary",
            verification_root="experiment/verification",
            output_root="reports/modeling/candidate_v1",
        )

    assert marked == ["implementation_lineage_changed", "implementation_lineage_changed"]


def test_evidence_rejects_dirty_worktree_or_overlapping_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence,
        "collect_git_evidence",
        lambda _root: GitEvidence(
            commit_sha="1" * 40,
            dirty=True,
            diff_sha256="2" * 64,
            repository_root=tmp_path,
        ),
    )
    with pytest.raises(evidence.CandidateEvidenceError, match="clean worktree"):
        evidence.run_candidate_evidence(repo_root=tmp_path)

    _patch_preflight(monkeypatch, tmp_path)
    with pytest.raises(evidence.CandidateEvidenceError, match="must be distinct"):
        evidence.run_candidate_evidence(
            repo_root=tmp_path,
            tracking_root="same",
            verification_root="different",
            output_root="same",
        )


def _patch_preflight(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(
        evidence,
        "collect_git_evidence",
        lambda _repo_root: GitEvidence(
            commit_sha="1" * 40,
            dirty=False,
            diff_sha256="2" * 64,
            repository_root=root,
        ),
    )
    monkeypatch.setattr(evidence, "verify_dataset", lambda **_kwargs: None)


def _result(
    output_root: Path,
    tracking_root: Path,
    *,
    report: bytes = b"report\n",
    run_id: str = "run",
) -> CandidateExperimentResult:
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = tracking_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "summary": (output_root / "summary.json", b"summary\n"),
        "report": (output_root / "candidate-report.md", report),
        "oof": (runtime / "oof_predictions.csv", b"oof\n"),
        "diagnostics": (runtime / "fold_diagnostics.json", b"diagnostics\n"),
    }
    for path, content in artifacts.values():
        path.write_bytes(content)
    return CandidateExperimentResult(
        summary_path=artifacts["summary"][0],
        report_path=artifacts["report"][0],
        oof_predictions_path=artifacts["oof"][0],
        fold_diagnostics_path=artifacts["diagnostics"][0],
        summary_sha256=hashlib.sha256(artifacts["summary"][1]).hexdigest(),
        report_sha256=hashlib.sha256(artifacts["report"][1]).hexdigest(),
        oof_predictions_sha256=hashlib.sha256(artifacts["oof"][1]).hexdigest(),
        fold_diagnostics_sha256=hashlib.sha256(artifacts["diagnostics"][1]).hexdigest(),
        selected_model_id="catboost_v1",
        selected_configuration_id="cb_cfg_004",
        catboost_advances=True,
        tracking=TrackingRunResult(
            tracking_uri=f"sqlite:///{tracking_root}/mlflow.db",
            experiment_name="credit-risk-candidate-v1",
            parent_run_id=run_id,
            child_run_ids=(),
        ),
    )
