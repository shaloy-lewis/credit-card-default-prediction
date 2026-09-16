"""Tests for authenticated, prediction-only Phase 6 parity evidence."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from catboost import CatBoostClassifier

import credit_risk.inference.evidence as evidence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture()
def evidence_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repository = tmp_path / "repository"
    (repository / "configs" / "inference").mkdir(parents=True)
    (repository / "models" / "selected_v1").mkdir(parents=True)
    (repository / "tests" / "fixtures").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    for source, destination in (
        (
            REPOSITORY_ROOT / "configs" / "inference" / "phase6_v1.json",
            repository / "configs" / "inference" / "phase6_v1.json",
        ),
        (
            REPOSITORY_ROOT / "models" / "selected_v1" / "manifest.json",
            repository / "models" / "selected_v1" / "manifest.json",
        ),
        (
            REPOSITORY_ROOT / "models" / "selected_v1" / "model.cbm",
            repository / "models" / "selected_v1" / "model.cbm",
        ),
        (
            REPOSITORY_ROOT / "tests" / "fixtures" / "inference_batch_v1.csv",
            repository / "tests" / "fixtures" / "inference_batch_v1.csv",
        ),
    ):
        shutil.copyfile(source, destination)
    monkeypatch.setattr(
        evidence,
        "collect_git_evidence",
        lambda _path: SimpleNamespace(
            repository_root=repository,
            dirty=False,
            commit_sha="a" * 40,
        ),
    )
    monkeypatch.setattr(
        CatBoostClassifier,
        "fit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fit is prohibited")),
    )
    return repository


def test_build_and_verify_phase6_evidence(evidence_repository: Path) -> None:
    result = evidence.build_inference_evidence(
        config_path=evidence_repository / "configs/inference/phase6_v1.json",
        bundle_root=evidence_repository / "models/selected_v1",
        fixture_path=evidence_repository / "tests/fixtures/inference_batch_v1.csv",
        runtime_root="experiment/inference/test-runtime",
        output_root="reports/inference/test-evidence",
    )

    assert result.runtime_root == evidence_repository / "experiment/inference/test-runtime"
    assert result.evidence_root == evidence_repository / "reports/inference/test-evidence"
    assert set(path.name for path in result.evidence_root.iterdir()) == evidence.EVIDENCE_FILES
    summary = json.loads((result.evidence_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["population"]["valid_rows"] == 20
    assert summary["batch"]["selected_rows"] == 2
    assert summary["parity"]["offline_to_batch"]["maximum_offline_batch_probability_error"] == 0.0
    assert summary["parity"]["offline_to_api"]["maximum_offline_api_probability_error"] <= 5e-7
    assert summary["boundaries"]["model_fitting_performed"] is False

    verified = evidence.verify_inference_evidence(
        evidence_root="reports/inference/test-evidence",
        expected_manifest_sha256=result.evidence_manifest_sha256,
        config_path=evidence_repository / "configs/inference/phase6_v1.json",
        bundle_root=evidence_repository / "models/selected_v1",
        fixture_path=evidence_repository / "tests/fixtures/inference_batch_v1.csv",
    )
    assert verified.summary_sha256 == result.summary_sha256
    assert verified.batch_id == result.batch_id


def test_build_rejects_dirty_existing_or_altered_inputs(
    evidence_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = {
        "config_path": evidence_repository / "configs/inference/phase6_v1.json",
        "bundle_root": evidence_repository / "models/selected_v1",
        "fixture_path": evidence_repository / "tests/fixtures/inference_batch_v1.csv",
        "runtime_root": "experiment/inference/test-runtime",
        "output_root": "reports/inference/test-evidence",
    }
    monkeypatch.setattr(
        evidence,
        "collect_git_evidence",
        lambda _path: SimpleNamespace(
            repository_root=evidence_repository,
            dirty=True,
            commit_sha="a" * 40,
        ),
    )
    with pytest.raises(evidence.InferenceEvidenceError, match="clean committed"):
        evidence.build_inference_evidence(**common)

    monkeypatch.setattr(
        evidence,
        "collect_git_evidence",
        lambda _path: SimpleNamespace(
            repository_root=evidence_repository,
            dirty=False,
            commit_sha="a" * 40,
        ),
    )
    fixture = evidence_repository / "tests/fixtures/inference_batch_v1.csv"
    fixture.write_bytes(fixture.read_bytes() + b"\n")
    with pytest.raises(evidence.InferenceEvidenceError, match="differs"):
        evidence.build_inference_evidence(**common)


@pytest.mark.parametrize(
    ("runtime_root", "output_root", "message"),
    [
        ("../escape", "reports/inference/evidence", "repository-relative"),
        ("experiment/inference/same", "experiment/inference/same", "reports/inference"),
        ("experiment/inference/runtime", "reports/outside", "reports/inference"),
    ],
)
def test_build_rejects_unsafe_destinations(
    evidence_repository: Path,
    runtime_root: str,
    output_root: str,
    message: str,
) -> None:
    with pytest.raises(evidence.InferenceEvidenceError, match=message):
        evidence.build_inference_evidence(
            config_path=evidence_repository / "configs/inference/phase6_v1.json",
            bundle_root=evidence_repository / "models/selected_v1",
            fixture_path=evidence_repository / "tests/fixtures/inference_batch_v1.csv",
            runtime_root=runtime_root,
            output_root=output_root,
        )


def test_verifier_rejects_bad_external_digest_and_modified_artifact(
    evidence_repository: Path,
) -> None:
    result = evidence.build_inference_evidence(
        config_path=evidence_repository / "configs/inference/phase6_v1.json",
        bundle_root=evidence_repository / "models/selected_v1",
        fixture_path=evidence_repository / "tests/fixtures/inference_batch_v1.csv",
        runtime_root="experiment/inference/test-runtime",
        output_root="reports/inference/test-evidence",
    )
    common = {
        "evidence_root": "reports/inference/test-evidence",
        "config_path": evidence_repository / "configs/inference/phase6_v1.json",
        "bundle_root": evidence_repository / "models/selected_v1",
        "fixture_path": evidence_repository / "tests/fixtures/inference_batch_v1.csv",
    }
    with pytest.raises(evidence.InferenceEvidenceError, match="must be SHA-256"):
        evidence.verify_inference_evidence(expected_manifest_sha256="bad", **common)
    with pytest.raises(evidence.InferenceEvidenceError, match="externally reviewed"):
        evidence.verify_inference_evidence(expected_manifest_sha256="0" * 64, **common)

    summary = result.evidence_root / "summary.json"
    summary.write_bytes(summary.read_bytes() + b" ")
    with pytest.raises(evidence.InferenceEvidenceError, match="digest mismatch"):
        evidence.verify_inference_evidence(
            expected_manifest_sha256=result.evidence_manifest_sha256, **common
        )


def test_build_refuses_overwrite(evidence_repository: Path) -> None:
    common = {
        "config_path": evidence_repository / "configs/inference/phase6_v1.json",
        "bundle_root": evidence_repository / "models/selected_v1",
        "fixture_path": evidence_repository / "tests/fixtures/inference_batch_v1.csv",
        "runtime_root": "experiment/inference/test-runtime",
        "output_root": "reports/inference/test-evidence",
    }
    evidence.build_inference_evidence(**common)
    with pytest.raises(evidence.InferenceEvidenceError, match="Refusing to overwrite"):
        evidence.build_inference_evidence(**common)


def test_evidence_validation_helpers_reject_semantic_drift(
    evidence_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = evidence.build_inference_evidence(
        config_path=evidence_repository / "configs/inference/phase6_v1.json",
        bundle_root=evidence_repository / "models/selected_v1",
        fixture_path=evidence_repository / "tests/fixtures/inference_batch_v1.csv",
        runtime_root="experiment/inference/test-runtime",
        output_root="reports/inference/test-evidence",
    )
    root = result.evidence_root
    manifest = json.loads((root / "evidence-manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

    for mutation, message in (
        (lambda value: value.update(status="draft"), "identity or lineage"),
        (lambda value: value["population"].update(valid_rows=19), "population"),
        (lambda value: value["batch"].update(selected_rows=3), "batch result"),
        (
            lambda value: value["parity"]["offline_to_api"].update(risk_band_mismatches=1),
            "parity",
        ),
        (lambda value: value.update(g4_status="closed"), "boundary or G4"),
    ):
        altered = json.loads(json.dumps(summary))
        mutation(altered)
        with pytest.raises(evidence.InferenceEvidenceError, match=message):
            evidence._validate_summary(altered, manifest)

    source_kwargs = {
        "fixture": evidence_repository / "tests/fixtures/inference_batch_v1.csv",
        "config_path": evidence_repository / "configs/inference/phase6_v1.json",
        "bundle_root": evidence_repository / "models/selected_v1",
    }
    for mutation, message in (
        (lambda value: value.update(evidence_id="wrong"), "identity or allowlist"),
        (lambda value: value.update(source_artifacts={}), "source allowlist"),
        (
            lambda value: value["source_artifacts"]["selected_model"].update(sha256="0" * 64),
            "source digest mismatch",
        ),
        (lambda value: value.update(boundaries_verified={}), "misstates"),
    ):
        altered = json.loads(json.dumps(manifest))
        mutation(altered)
        with pytest.raises(evidence.InferenceEvidenceError, match=message):
            evidence._validate_manifest_sources(manifest=altered, **source_kwargs)

    monkeypatch.setattr(evidence, "FIXTURE_SHA256", "0" * 64)
    with pytest.raises(evidence.InferenceEvidenceError, match="unreviewed fixture"):
        evidence._validate_manifest_sources(manifest=manifest, **source_kwargs)


def test_evidence_io_path_and_atomic_failure_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("", encoding="utf-8")

    with pytest.raises(evidence.InferenceEvidenceError, match="missing"):
        evidence._safe_repository_file(repository, "missing.json", "source")
    with pytest.raises(evidence.InferenceEvidenceError, match="missing"):
        evidence._safe_repository_directory(repository, "missing", "directory")
    with pytest.raises(evidence.InferenceEvidenceError, match="within the repository"):
        evidence._safe_repository_path(repository, repository.parent, "source")
    with pytest.raises(evidence.InferenceEvidenceError, match="payload"):
        evidence._publish_directory(repository / "reports/inference/bad", {})
    with pytest.raises(evidence.InferenceEvidenceError, match="parse invalid"):
        evidence._read_json_bytes(b"not-json", "invalid")
    with pytest.raises(evidence.InferenceEvidenceError, match="JSON object"):
        evidence._read_json_bytes(b"[]", "invalid")
    with pytest.raises(evidence.InferenceEvidenceError, match="Unable to read"):
        evidence._read_bytes(repository / "missing", "missing source")
    with pytest.raises(evidence.InferenceEvidenceError, match="Unable to read"):
        evidence._read_text(repository / "missing", "missing source")

    source = repository / "source"
    source.mkdir()
    destination = repository / "destination"
    calls = 0
    real_replace = os.replace

    def transient_replace(first: Path, second: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("transient")
        real_replace(first, second)

    monkeypatch.setattr(evidence.os, "replace", transient_replace)
    monkeypatch.setattr(evidence.time, "sleep", lambda _seconds: None)
    evidence._replace_directory_with_retry(source, destination)
    assert calls == 2

    always_fail = repository / "always-fail"
    always_fail.mkdir()
    monkeypatch.setattr(
        evidence.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(PermissionError("locked")),
    )
    with pytest.raises(PermissionError, match="locked"):
        evidence._replace_directory_with_retry(always_fail, repository / "never")
