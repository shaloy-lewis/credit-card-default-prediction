"""Inference CLI tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import credit_risk.inference.cli as cli
from credit_risk.inference.batch import BatchInferenceError, BatchRunResult
from credit_risk.inference.evidence import InferenceEvidenceError, InferenceEvidenceResult

runner = CliRunner()


def test_batch_cli_returns_partial_exit_and_forwards_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_inference_config", lambda _path: SimpleNamespace())
    monkeypatch.setattr(cli, "validate_batch_identity", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "InferenceEngine", lambda **_kwargs: SimpleNamespace())

    def fake_run(**kwargs: object) -> BatchRunResult:
        captured.update(kwargs)
        return BatchRunResult(
            run_root=tmp_path / "run",
            batch_id="a" * 64,
            status="completed_with_rejections",
            exit_code=3,
            valid_rows=4,
            rejected_rows=1,
            reused=False,
        )

    monkeypatch.setattr(cli, "run_batch", fake_run)
    result = runner.invoke(
        cli.inference_app,
        [
            "batch",
            "--input",
            "snapshot.csv",
            "--as-of-date",
            "2026-09-30",
            "--snapshot-id",
            "monthly-v1",
        ],
    )

    assert result.exit_code == 3
    assert "completed_with_rejections" in result.output
    assert captured["input_path"] == Path("snapshot.csv")
    assert captured["output_root"] == Path("experiment/inference/batches")


def test_batch_cli_returns_actionable_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_inference_config", lambda _path: SimpleNamespace())
    monkeypatch.setattr(cli, "validate_batch_identity", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "InferenceEngine", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        cli,
        "run_batch",
        lambda **_kwargs: (_ for _ in ()).throw(BatchInferenceError("invalid snapshot")),
    )
    result = runner.invoke(
        cli.inference_app,
        [
            "batch",
            "--input",
            "snapshot.csv",
            "--as-of-date",
            "2026-09-30",
            "--snapshot-id",
            "monthly-v1",
        ],
    )

    assert result.exit_code == 1
    assert "invalid snapshot" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("as_of_date", "snapshot_id", "message"),
    (
        ("2026-09-30", ".", "cannot be '.' or '..'"),
        ("2026-09-30", "..", "cannot be '.' or '..'"),
        ("09/30/2026", "safe", "ISO date"),
    ),
)
def test_batch_cli_rejects_invalid_identity_before_engine_or_workflow_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    as_of_date: str,
    snapshot_id: str,
    message: str,
) -> None:
    config = SimpleNamespace(batch=SimpleNamespace(reserved_snapshot_ids=(".", "..")))
    monkeypatch.setattr(cli, "load_inference_config", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "InferenceEngine",
        lambda **_kwargs: pytest.fail("engine must not be constructed during identity preflight"),
    )
    monkeypatch.setattr(
        cli,
        "run_batch",
        lambda **_kwargs: pytest.fail("batch workflow must not run after failed preflight"),
    )
    output_root = tmp_path / "must-not-exist"

    result = runner.invoke(
        cli.inference_app,
        [
            "batch",
            "--input",
            str(tmp_path / "missing.csv"),
            "--as-of-date",
            as_of_date,
            "--snapshot-id",
            snapshot_id,
            "--output-root",
            str(output_root),
        ],
    )

    assert result.exit_code == 1
    assert message in result.output
    assert "Traceback" not in result.output
    assert not output_root.exists()


def test_verify_cli_reports_verified_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_inference_config", lambda _path: SimpleNamespace())
    monkeypatch.setattr(cli, "InferenceEngine", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        cli,
        "verify_batch_run",
        lambda *_args, **_kwargs: {"batch_id": "b" * 64, "status": "completed"},
    )
    result = runner.invoke(
        cli.inference_app,
        ["verify", "--run-root", "experiment/inference/batches/run"],
    )

    assert result.exit_code == 0
    assert "Batch verified" in result.output


def test_verify_cli_returns_actionable_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_inference_config", lambda _path: SimpleNamespace())
    monkeypatch.setattr(cli, "InferenceEngine", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        cli,
        "verify_batch_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(BatchInferenceError("digest mismatch")),
    )
    result = runner.invoke(
        cli.inference_app,
        ["verify", "--run-root", "experiment/inference/batches/run"],
    )

    assert result.exit_code == 1
    assert "digest mismatch" in result.output
    assert "Traceback" not in result.output


def test_evidence_cli_reports_authenticated_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "build_inference_evidence",
        lambda **_kwargs: InferenceEvidenceResult(
            evidence_root=tmp_path / "evidence",
            runtime_root=tmp_path / "runtime",
            summary_sha256="a" * 64,
            evidence_manifest_sha256="b" * 64,
            batch_id="c" * 64,
        ),
    )
    result = runner.invoke(cli.inference_app, ["evidence"])

    assert result.exit_code == 0
    assert "Phase 6 evidence published" in result.output
    assert "manifest_sha256=" + "b" * 64 in result.output


def test_evidence_cli_hides_expected_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "build_inference_evidence",
        lambda **_kwargs: (_ for _ in ()).throw(InferenceEvidenceError("dirty worktree")),
    )
    result = runner.invoke(cli.inference_app, ["evidence"])

    assert result.exit_code == 1
    assert "dirty worktree" in result.output
    assert "Traceback" not in result.output


def test_verify_evidence_cli_forwards_external_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_verify(**kwargs: object) -> InferenceEvidenceResult:
        captured.update(kwargs)
        return InferenceEvidenceResult(
            evidence_root=tmp_path / "evidence",
            runtime_root=None,
            summary_sha256="a" * 64,
            evidence_manifest_sha256="b" * 64,
            batch_id="c" * 64,
        )

    monkeypatch.setattr(cli, "verify_inference_evidence", fake_verify)
    result = runner.invoke(
        cli.inference_app,
        ["verify-evidence", "--expected-manifest-sha256", "b" * 64],
    )

    assert result.exit_code == 0
    assert captured["expected_manifest_sha256"] == "b" * 64
    assert "Phase 6 evidence verified" in result.output


def test_verify_evidence_cli_returns_actionable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "verify_inference_evidence",
        lambda **_kwargs: (_ for _ in ()).throw(InferenceEvidenceError("bad manifest")),
    )
    result = runner.invoke(
        cli.inference_app,
        ["verify-evidence", "--expected-manifest-sha256", "b" * 64],
    )

    assert result.exit_code == 1
    assert "bad manifest" in result.output
    assert "Traceback" not in result.output
