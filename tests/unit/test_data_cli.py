"""CLI contracts for governed data acquisition, build, and verification."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from credit_risk.cli import app
from credit_risk.data import cli as data_cli
from credit_risk.data.acquisition import AcquisitionResult, SourceIntegrityError, SourceVerification
from credit_risk.data.manifest import (
    DEFAULT_DATASET_MANIFEST_PATH,
    DEFAULT_SPLIT_CONFIG_PATH,
    ManifestLoadError,
)
from credit_risk.data.workflow import DataWorkflowError

runner = CliRunner()


def _workflow_result(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        canonical_sha256="a" * 64,
        assignment_sha256="b" * 64,
        source_sha256="c" * 64,
        paths=SimpleNamespace(
            canonical=root / "processed" / "canonical.csv",
            split_manifest=root / "splits" / "split_manifest.json",
            reviewed_split_lock=root / "configs" / "split.lock.json",
        ),
        reviewed_lock_verified=False,
    )


def test_data_group_help_lists_the_three_governed_commands() -> None:
    result = runner.invoke(app, ["data", "--help"])

    assert result.exit_code == 0
    assert "fetch" in result.output
    assert "build" in result.output
    assert "verify" in result.output


def test_fetch_uses_defaults_and_reports_downloaded_source(tmp_path: Path, monkeypatch) -> None:
    manifest = object()
    source = tmp_path / "data.csv"
    observed: dict[str, Any] = {}
    monkeypatch.setattr(data_cli, "load_dataset_manifest", lambda path: manifest)

    def fake_fetch(received_manifest: object, data_root: Path) -> AcquisitionResult:
        observed.update(manifest=received_manifest, data_root=data_root)
        return AcquisitionResult(
            path=source,
            verification=SourceVerification(source, 4, "a" * 64),
            downloaded=True,
            attempts=2,
        )

    monkeypatch.setattr(data_cli, "fetch_source", fake_fetch)

    result = runner.invoke(app, ["data", "fetch"])

    assert result.exit_code == 0
    assert observed == {"manifest": manifest, "data_root": Path("data")}
    assert "Source downloaded" in result.output
    assert "attempts=2" in result.output


def test_fetch_honours_custom_paths_and_reports_offline_reuse(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "custom-data"
    manifest_path = tmp_path / "manifest.json"
    source = data_root / "raw" / "data.csv"
    observed: dict[str, Any] = {}

    def fake_load(path: Path) -> object:
        observed["manifest_path"] = path
        return "manifest"

    def fake_fetch(manifest: object, root: Path) -> AcquisitionResult:
        observed.update(manifest=manifest, data_root=root)
        return AcquisitionResult(
            path=source,
            verification=SourceVerification(source, 4, "b" * 64),
            downloaded=False,
            attempts=0,
        )

    monkeypatch.setattr(data_cli, "load_dataset_manifest", fake_load)
    monkeypatch.setattr(data_cli, "fetch_source", fake_fetch)

    result = runner.invoke(
        app,
        [
            "data",
            "fetch",
            "--data-root",
            str(data_root),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0
    assert observed == {
        "manifest_path": manifest_path,
        "manifest": "manifest",
        "data_root": data_root,
    }
    assert "verified existing" in result.output


@pytest.mark.parametrize(
    "error",
    [
        SourceIntegrityError("checksum mismatch"),
        ManifestLoadError("invalid manifest"),
        OSError("disk unavailable"),
    ],
)
def test_fetch_failures_exit_one_with_actionable_message_and_no_traceback(
    monkeypatch, error: Exception
) -> None:
    monkeypatch.setattr(data_cli, "load_dataset_manifest", lambda _: (_ for _ in ()).throw(error))

    result = runner.invoke(app, ["data", "fetch"])

    assert result.exit_code == 1
    assert f"Data fetch failed: {error}" in result.output
    assert "Traceback" not in result.output


def test_build_passes_defaults_to_workflow_and_reports_hashes(tmp_path: Path, monkeypatch) -> None:
    from credit_risk.data import workflow

    observed: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> SimpleNamespace:
        observed.update(kwargs)
        return _workflow_result(tmp_path)

    monkeypatch.setattr(workflow, "build_dataset", fake_build)

    result = runner.invoke(app, ["data", "build"])

    assert result.exit_code == 0
    assert observed == {
        "data_root": Path("data"),
        "manifest_path": DEFAULT_DATASET_MANIFEST_PATH,
        "split_config_path": DEFAULT_SPLIT_CONFIG_PATH,
        "offline": False,
    }
    assert f"canonical_sha256={'a' * 64}" in result.output
    assert f"assignment_sha256={'b' * 64}" in result.output
    assert "Canonical data:" in result.output
    assert "Runtime split manifest:" in result.output
    assert "Reviewed split lock (not yet reviewed):" in result.output


def test_build_honours_custom_paths_and_offline_flag(tmp_path: Path, monkeypatch) -> None:
    from credit_risk.data import workflow

    data_root = tmp_path / "data"
    manifest = tmp_path / "manifest.json"
    split_config = tmp_path / "split.json"
    observed: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> SimpleNamespace:
        observed.update(kwargs)
        result = _workflow_result(data_root)
        result.reviewed_lock_verified = True
        return result

    monkeypatch.setattr(workflow, "build_dataset", fake_build)

    result = runner.invoke(
        app,
        [
            "data",
            "build",
            "--data-root",
            str(data_root),
            "--manifest",
            str(manifest),
            "--split-config",
            str(split_config),
            "--offline",
        ],
    )

    assert result.exit_code == 0
    assert observed == {
        "data_root": data_root,
        "manifest_path": manifest,
        "split_config_path": split_config,
        "offline": True,
    }
    assert "Reviewed split lock (verified):" in result.output


def test_verify_honours_custom_paths_and_reports_lineage(tmp_path: Path, monkeypatch) -> None:
    from credit_risk.data import workflow

    data_root = tmp_path / "data"
    manifest = tmp_path / "manifest.json"
    split_config = tmp_path / "split.json"
    observed: dict[str, Any] = {}

    def fake_verify(**kwargs: Any) -> SimpleNamespace:
        observed.update(kwargs)
        return _workflow_result(data_root)

    monkeypatch.setattr(workflow, "verify_dataset", fake_verify)

    result = runner.invoke(
        app,
        [
            "data",
            "verify",
            "--data-root",
            str(data_root),
            "--manifest",
            str(manifest),
            "--split-config",
            str(split_config),
        ],
    )

    assert result.exit_code == 0
    assert observed == {
        "data_root": data_root,
        "manifest_path": manifest,
        "split_config_path": split_config,
    }
    assert "Offline verification passed" in result.output
    assert f"source_sha256={'c' * 64}" in result.output


@pytest.mark.parametrize(
    ("command", "function", "prefix"),
    [
        ("build", "build_dataset", "Data build failed"),
        ("verify", "verify_dataset", "Data verification failed"),
    ],
)
def test_workflow_failures_exit_one_without_a_traceback(
    monkeypatch, command: str, function: str, prefix: str
) -> None:
    from credit_risk.data import workflow

    def fail(**_: Any) -> None:
        raise DataWorkflowError("governed state is missing or corrupt")

    monkeypatch.setattr(workflow, function, fail)

    result = runner.invoke(app, ["data", command])

    assert result.exit_code == 1
    assert f"{prefix}: governed state is missing or corrupt" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("command", "function", "prefix"),
    [
        ("build", "build_dataset", "Data build failed"),
        ("verify", "verify_dataset", "Data verification failed"),
    ],
)
def test_missing_pandera_extra_has_install_guidance(
    monkeypatch, command: str, function: str, prefix: str
) -> None:
    from credit_risk.data import workflow

    def fail(**_: Any) -> None:
        raise ModuleNotFoundError("No module named 'pandera'", name="pandera")

    monkeypatch.setattr(workflow, function, fail)

    result = runner.invoke(app, ["data", command])

    assert result.exit_code == 1
    assert f"{prefix}: install the project with the 'data' extra" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["build", "verify"])
def test_unrelated_missing_module_is_not_misreported_as_pandera(monkeypatch, command: str) -> None:
    from credit_risk.data import workflow

    function = "build_dataset" if command == "build" else "verify_dataset"

    def fail(**_: Any) -> None:
        raise ModuleNotFoundError("No module named 'other'", name="other")

    monkeypatch.setattr(workflow, function, fail)

    result = runner.invoke(app, ["data", command])

    assert result.exit_code == 1
    assert isinstance(result.exception, ModuleNotFoundError)
    assert result.exception.name == "other"
