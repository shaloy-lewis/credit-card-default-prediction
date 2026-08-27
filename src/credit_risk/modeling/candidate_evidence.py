"""Two-run reproducibility gate and governed Phase 3 evidence promotion."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from credit_risk.data.manifest import DEFAULT_DATASET_MANIFEST_PATH, DEFAULT_SPLIT_CONFIG_PATH
from credit_risk.data.workflow import DataWorkflowError, verify_dataset
from credit_risk.modeling.candidate_contracts import DEFAULT_CANDIDATE_CONFIG_PATH
from credit_risk.modeling.candidate_workflow import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRACKING_ROOT,
    REPORT_FILENAME,
    SUMMARY_FILENAME,
    CandidateExperimentResult,
    CandidateProgress,
    CandidateWorkflowError,
    _promote_outputs,
    run_candidate_experiment,
)
from credit_risk.modeling.tracking import (
    TrackingError,
    collect_git_evidence,
    mark_tracking_run_failed,
)

DEFAULT_VERIFICATION_ROOT: Final[Path] = Path("experiment/phase3-verification")


class CandidateEvidenceError(RuntimeError):
    """Raised when independent Phase 3 evidence cannot be verified and promoted."""


@dataclass(frozen=True, slots=True)
class CandidateEvidenceResult:
    """Official evidence plus operational identities for two independent executions."""

    summary_path: Path
    report_path: Path
    summary_sha256: str
    report_sha256: str
    oof_predictions_sha256: str
    fold_diagnostics_sha256: str
    selected_model_id: str
    selected_configuration_id: str
    catboost_advances: bool
    primary: CandidateExperimentResult
    verification: CandidateExperimentResult


EvidenceProgressCallback = Callable[[str, CandidateProgress], None]


def run_candidate_evidence(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config_path: str | Path = DEFAULT_CANDIDATE_CONFIG_PATH,
    tracking_root: str | Path = DEFAULT_TRACKING_ROOT,
    verification_root: str | Path = DEFAULT_VERIFICATION_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    repo_root: str | Path = ".",
    progress_callback: EvidenceProgressCallback | None = None,
) -> CandidateEvidenceResult:
    """Run two isolated executions, compare all evidence, and promote without a third fit pass."""

    try:
        return _run_candidate_evidence(
            data_root=Path(data_root),
            config_path=Path(config_path),
            tracking_root=Path(tracking_root),
            verification_root=Path(verification_root),
            output_root=Path(output_root),
            repo_root=Path(repo_root),
            progress_callback=progress_callback,
        )
    except CandidateEvidenceError:
        raise
    except (CandidateWorkflowError, DataWorkflowError, TrackingError) as error:
        raise CandidateEvidenceError(str(error)) from error
    except (OSError, ValueError) as error:
        raise CandidateEvidenceError(f"Candidate evidence verification failed: {error}") from error


def _run_candidate_evidence(
    *,
    data_root: Path,
    config_path: Path,
    tracking_root: Path,
    verification_root: Path,
    output_root: Path,
    repo_root: Path,
    progress_callback: EvidenceProgressCallback | None,
) -> CandidateEvidenceResult:
    git = collect_git_evidence(repo_root)
    if git.dirty:
        raise CandidateEvidenceError(
            "Candidate evidence requires a clean worktree so implementation lineage is fixed."
        )
    root = git.repository_root or repo_root.resolve()
    data_root = _root_relative(data_root, root)
    config_path = _root_relative(config_path, root)
    tracking_root = _root_relative(tracking_root, root)
    verification_root = _root_relative(verification_root, root)
    output_root = _root_relative(output_root, root)
    _validate_roots(tracking_root, verification_root, output_root)

    verify_dataset(
        data_root=data_root,
        manifest_path=root / DEFAULT_DATASET_MANIFEST_PATH,
        split_config_path=root / DEFAULT_SPLIT_CONFIG_PATH,
    )

    primary_output = verification_root / "primary" / "reports"
    secondary_tracking = verification_root / "verification" / "mlflow"
    secondary_output = verification_root / "verification" / "reports"

    primary = run_candidate_experiment(
        data_root=data_root,
        config_path=config_path,
        tracking_root=tracking_root,
        output_root=primary_output,
        allow_dirty=False,
        repo_root=root,
        progress_callback=_run_callback("primary", progress_callback),
    )
    try:
        verification = run_candidate_experiment(
            data_root=data_root,
            config_path=config_path,
            tracking_root=secondary_tracking,
            output_root=secondary_output,
            allow_dirty=False,
            repo_root=root,
            progress_callback=_run_callback("verification", progress_callback),
        )
    except CandidateWorkflowError as error:
        _mark_failed((primary,), failure_stage="independent_execution_failed", cause=error)
        raise CandidateEvidenceError(
            f"Independent candidate verification execution failed: {error}"
        ) from error

    try:
        primary_artifacts = _artifact_bytes(primary)
        verification_artifacts = _artifact_bytes(verification)
    except CandidateEvidenceError as error:
        _mark_failed(
            (primary, verification),
            failure_stage="reproducibility_artifact_validation",
            cause=error,
        )
        raise
    primary_hashes = _artifact_hashes(primary_artifacts)
    verification_hashes = _artifact_hashes(verification_artifacts)
    if primary_hashes != verification_hashes or primary_artifacts != verification_artifacts:
        mismatch = CandidateEvidenceError(
            "Independent candidate executions produced different summary, report, OOF, or "
            "fold-diagnostic bytes; official evidence was not changed."
        )
        _mark_failed(
            (primary, verification),
            failure_stage="reproducibility_mismatch",
            cause=mismatch,
        )
        raise mismatch

    final_git = collect_git_evidence(root)
    if (
        final_git.dirty
        or final_git.commit_sha != git.commit_sha
        or final_git.diff_sha256 != git.diff_sha256
    ):
        changed = CandidateEvidenceError(
            "Git lineage changed during independent candidate execution; official evidence "
            "was not changed."
        )
        _mark_failed(
            (primary, verification),
            failure_stage="implementation_lineage_changed",
            cause=changed,
        )
        raise changed

    official_summary = output_root / SUMMARY_FILENAME
    official_report = output_root / REPORT_FILENAME
    try:
        _promote_outputs(
            {
                official_summary: primary_artifacts[SUMMARY_FILENAME],
                official_report: primary_artifacts[REPORT_FILENAME],
            }
        )
    except CandidateWorkflowError as error:
        _mark_failed(
            (primary, verification),
            failure_stage="official_evidence_publication",
            cause=error,
        )
        raise CandidateEvidenceError(str(error)) from error

    return CandidateEvidenceResult(
        summary_path=official_summary,
        report_path=official_report,
        summary_sha256=primary_hashes[SUMMARY_FILENAME],
        report_sha256=primary_hashes[REPORT_FILENAME],
        oof_predictions_sha256=primary_hashes["oof_predictions.csv"],
        fold_diagnostics_sha256=primary_hashes["fold_diagnostics.json"],
        selected_model_id=primary.selected_model_id,
        selected_configuration_id=primary.selected_configuration_id,
        catboost_advances=primary.catboost_advances,
        primary=primary,
        verification=verification,
    )


def _artifact_bytes(result: CandidateExperimentResult) -> dict[str, bytes]:
    paths = {
        SUMMARY_FILENAME: result.summary_path,
        REPORT_FILENAME: result.report_path,
        "oof_predictions.csv": result.oof_predictions_path,
        "fold_diagnostics.json": result.fold_diagnostics_path,
    }
    expected = {
        SUMMARY_FILENAME: result.summary_sha256,
        REPORT_FILENAME: result.report_sha256,
        "oof_predictions.csv": result.oof_predictions_sha256,
        "fold_diagnostics.json": result.fold_diagnostics_sha256,
    }
    artifacts: dict[str, bytes] = {}
    for name, path in paths.items():
        try:
            content = path.read_bytes()
        except OSError as error:
            raise CandidateEvidenceError(
                f"Unable to read {name} from independent candidate execution: {error}"
            ) from error
        observed = hashlib.sha256(content).hexdigest()
        if observed != expected[name]:
            raise CandidateEvidenceError(
                f"Candidate execution result hash does not match {name}: "
                f"expected={expected[name]}, observed={observed}."
            )
        artifacts[name] = content
    return artifacts


def _artifact_hashes(artifacts: Mapping[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()}


def _mark_failed(
    results: tuple[CandidateExperimentResult, ...],
    *,
    failure_stage: str,
    cause: Exception,
) -> None:
    failures: list[str] = []
    for result in results:
        try:
            mark_tracking_run_failed(result.tracking, failure_stage=failure_stage)
        except TrackingError as error:
            failures.append(str(error))
    if failures:
        raise CandidateEvidenceError(
            f"{cause} MLflow failure-state correction also failed: {'; '.join(failures)}"
        ) from cause


def _run_callback(
    run_name: str,
    callback: EvidenceProgressCallback | None,
) -> Callable[[CandidateProgress], None] | None:
    if callback is None:
        return None
    return lambda progress: callback(run_name, progress)


def _validate_roots(tracking_root: Path, verification_root: Path, output_root: Path) -> None:
    roots = (tracking_root.resolve(), verification_root.resolve(), output_root.resolve())
    if any(
        _paths_overlap(first, second)
        for index, first in enumerate(roots)
        for second in roots[index + 1 :]
    ):
        raise CandidateEvidenceError(
            "Tracking, verification, and official output roots must be distinct and non-overlapping."
        )
    secondary_tracking = (verification_root / "verification" / "mlflow").resolve()
    if secondary_tracking == tracking_root.resolve():
        raise CandidateEvidenceError(
            "Primary and verification runs must use independent tracking roots."
        )


def _root_relative(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root.resolve() / path).resolve()


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents
