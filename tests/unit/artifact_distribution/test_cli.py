from pathlib import Path

from typer.testing import CliRunner

from credit_risk.artifact_distribution.cli import artifact_app
from credit_risk.artifact_distribution.workflow import (
    ArtifactDistributionError,
    ArtifactOperationResult,
)

runner = CliRunner()


def test_pull_defaults_to_selected_and_reports_reuse(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_pull(**kwargs) -> ArtifactOperationResult:
        observed.update(kwargs)
        return ArtifactOperationResult(
            group="selected",
            materialized=(),
            reused=(Path("models/selected_v1/model.cbm"),),
            revision="a" * 40,
        )

    monkeypatch.setattr("credit_risk.artifact_distribution.cli.pull_artifacts", fake_pull)

    result = runner.invoke(artifact_app, ["pull", "--offline"])

    assert result.exit_code == 0
    assert observed["group"] == "selected"
    assert observed["offline"] is True
    assert "reused=1" in result.stdout


def test_legacy_pull_prints_pickle_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        "credit_risk.artifact_distribution.cli.pull_artifacts",
        lambda **_: ArtifactOperationResult("legacy", (), (), "a" * 40),
    )

    result = runner.invoke(artifact_app, ["pull", "--group", "legacy"])

    assert result.exit_code == 0
    assert "can execute code" in result.output


def test_pull_normalizes_expected_failure(monkeypatch) -> None:
    def fail(**_) -> ArtifactOperationResult:
        raise ArtifactDistributionError("offline cache miss")

    monkeypatch.setattr("credit_risk.artifact_distribution.cli.pull_artifacts", fail)

    result = runner.invoke(artifact_app, ["pull", "--offline"])

    assert result.exit_code == 1
    assert "offline cache miss" in result.output
    assert "Traceback" not in result.output


def test_verify_normalizes_expected_failure(monkeypatch) -> None:
    def fail(**_) -> ArtifactOperationResult:
        raise ArtifactDistributionError("reviewed bytes are missing")

    monkeypatch.setattr("credit_risk.artifact_distribution.cli.verify_artifacts", fail)

    result = runner.invoke(artifact_app, ["verify"])

    assert result.exit_code == 1
    assert "reviewed bytes are missing" in result.output
    assert "Traceback" not in result.output


def test_verify_reports_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "credit_risk.artifact_distribution.cli.verify_artifacts",
        lambda **_: ArtifactOperationResult(
            "selected", (), (Path("models/selected_v1/model.cbm"),)
        ),
    )

    result = runner.invoke(artifact_app, ["verify"])

    assert result.exit_code == 0
    assert "files=1" in result.stdout


def test_publish_passes_maintainer_options(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_publish(**kwargs) -> ArtifactOperationResult:
        observed.update(kwargs)
        return ArtifactOperationResult(
            "all", (Path("experiment/artifacts/candidate.json"),), (), "b" * 40
        )

    monkeypatch.setattr("credit_risk.artifact_distribution.cli.publish_artifacts", fake_publish)

    result = runner.invoke(
        artifact_app,
        [
            "publish",
            "--repo-id",
            "owner/repository",
            "--source-root",
            str(tmp_path),
            "--include-legacy",
        ],
    )

    assert result.exit_code == 0
    assert observed["repo_id"] == "owner/repository"
    assert observed["include_legacy"] is True
    assert "b" * 40 in result.stdout


def test_publish_normalizes_expected_failure(monkeypatch) -> None:
    def fail(**_) -> ArtifactOperationResult:
        raise ArtifactDistributionError("conflicting remote path")

    monkeypatch.setattr("credit_risk.artifact_distribution.cli.publish_artifacts", fail)

    result = runner.invoke(artifact_app, ["publish", "--repo-id", "owner/repository"])

    assert result.exit_code == 1
    assert "conflicting remote path" in result.output
    assert "Traceback" not in result.output
