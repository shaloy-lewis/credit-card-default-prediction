from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from credit_risk.artifact_distribution.contracts import (
    ArtifactContractError,
    ArtifactRecord,
    DigestReference,
    DistributionLock,
    LegacyManifest,
    RepositoryContract,
    load_distribution_lock,
    load_legacy_manifest,
    resolve_digest_reference,
    safe_repository_path,
    sha256_file,
)


def _record(**updates: object) -> ArtifactRecord:
    values: dict[str, object] = {
        "artifact_id": "selected_model",
        "group": "selected",
        "remote_path": "selected_v1/model.cbm",
        "local_path": "models/selected_v1/model.cbm",
        "size_bytes": 10,
        "serialization": "catboost_cbm",
        "trust_classification": "digest_authenticated",
        "digest_reference": DigestReference(
            manifest_path="models/selected_v1/manifest.json",
            json_pointer="/model_sha256",
        ),
    }
    values.update(updates)
    return ArtifactRecord.model_validate(values)


def test_distribution_requires_full_commit_and_approved_inventory() -> None:
    repository = RepositoryContract(
        provider="huggingface_hub",
        repo_id="owner/repository",
        repo_type="model",
        revision="a" * 40,
        public=True,
    )
    lock = DistributionLock(
        schema_version="1.0.0",
        distribution_id="hf_distribution_v1",
        repository=repository,
        artifacts=(_record(),),
    )
    assert lock.repository.revision == "a" * 40
    with pytest.raises(ValidationError):
        RepositoryContract(
            provider="huggingface_hub",
            repo_id="owner/repository",
            repo_type="model",
            revision="a" * 7,
            public=True,
        )
    with pytest.raises(ValidationError, match="approved selected-and-legacy inventory"):
        DistributionLock(
            schema_version="1.0.0",
            distribution_id="hf_distribution_v1",
            repository=repository,
            artifacts=(
                _record(),
                _record(
                    artifact_id="legacy_model",
                    group="legacy",
                    local_path="artifacts/model.pkl",
                    remote_path="legacy_v1/model.pkl",
                    serialization="python_pickle",
                    trust_classification="trusted_pickle_explicit_only",
                    digest_reference=DigestReference(
                        manifest_path="configs/artifacts/legacy_v1.json",
                        json_pointer="/files/model.pkl/sha256",
                    ),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("local_path", "../model.cbm"),
        ("local_path", "/tmp/model.cbm"),
        ("remote_path", "../model.cbm"),
        ("remote_path", "selected_v1\\model.cbm"),
    ],
)
def test_artifact_record_rejects_unsafe_and_changed_paths(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        _record(**{field: value})


def test_artifact_record_rejects_changed_approved_mapping() -> None:
    with pytest.raises(ValidationError, match="approved mapping"):
        _record(local_path="models/other/model.cbm")


@pytest.mark.parametrize(
    "updates",
    [
        {
            "digest_reference": DigestReference(
                manifest_path="configs/artifacts/other.json",
                json_pointer="/model_sha256",
            )
        },
        {
            "digest_reference": DigestReference(
                manifest_path="models/selected_v1/manifest.json",
                json_pointer="/other_sha256",
            )
        },
        {"artifact_id": "legacy_preprocessor"},
        {"group": "legacy"},
    ],
)
def test_artifact_record_binds_identity_group_and_digest_authority(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="approved mapping"):
        _record(**updates)


def test_distribution_rejects_duplicate_destinations() -> None:
    repository = RepositoryContract(
        provider="huggingface_hub",
        repo_id="owner/repository",
        repo_type="model",
        revision="a" * 40,
        public=True,
    )
    with pytest.raises(ValidationError, match="inventory"):
        DistributionLock(
            schema_version="1.0.0",
            distribution_id="hf_distribution_v1",
            repository=repository,
            artifacts=(_record(), _record()),
        )


@pytest.mark.parametrize(
    "files",
    [
        {
            "model.pkl": {
                "size_bytes": 1,
                "sha256": "a" * 64,
                "serialization": "python_pickle",
            }
        },
        {
            "model.pkl": {
                "size_bytes": 1,
                "sha256": "a" * 64,
                "serialization": "json",
            },
            "preprocessor.pkl": {
                "size_bytes": 1,
                "sha256": "b" * 64,
                "serialization": "python_pickle",
            },
            "outlier_threshold.json": {
                "size_bytes": 1,
                "sha256": "c" * 64,
                "serialization": "json",
            },
        },
        {
            "model.pkl": {
                "size_bytes": 1,
                "sha256": "a" * 64,
                "serialization": "python_pickle",
            },
            "preprocessor.pkl": {
                "size_bytes": 1,
                "sha256": "b" * 64,
                "serialization": "python_pickle",
            },
            "outlier_threshold.json": {
                "size_bytes": 1,
                "sha256": "c" * 64,
                "serialization": "python_pickle",
            },
        },
    ],
)
def test_legacy_manifest_requires_exact_inventory_and_serialization(
    files: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        LegacyManifest(
            schema_version="1.0.0",
            bundle_id="legacy_v1",
            trust_classification="trusted_pickle_explicit_only",
            warning="Only digest-authenticated pickle files may be loaded.",
            files=files,
        )


def test_digest_reference_reads_git_authority(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"nested": {"sha": "a" * 64}}), encoding="utf-8")
    reference = DigestReference(manifest_path="manifest.json", json_pointer="/nested/sha")
    assert resolve_digest_reference(tmp_path, reference) == "a" * 64
    missing = reference.model_copy(update={"json_pointer": "/nested/missing"})
    with pytest.raises(ArtifactContractError, match="does not exist"):
        resolve_digest_reference(tmp_path, missing)
    bad = reference.model_copy(update={"json_pointer": "/nested"})
    with pytest.raises(ArtifactContractError, match="lowercase SHA-256"):
        resolve_digest_reference(tmp_path, bad)


def test_contract_loaders_and_hashing_normalize_file_errors(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(ArtifactContractError, match="Invalid artifact distribution lock"):
        load_distribution_lock(invalid)
    with pytest.raises(ArtifactContractError, match="Unable to read legacy"):
        load_legacy_manifest(tmp_path / "missing.json")
    with pytest.raises(ArtifactContractError, match="Unable to hash"):
        sha256_file(tmp_path / "missing.bin")


def test_safe_repository_path_rejects_missing_and_traversal(tmp_path: Path) -> None:
    with pytest.raises(ArtifactContractError, match="Unable to resolve"):
        safe_repository_path(tmp_path, "missing.bin", must_exist=True)
    with pytest.raises(ValueError, match="repository-relative"):
        safe_repository_path(tmp_path, "../outside.bin", must_exist=False)


def test_safe_repository_path_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ArtifactContractError, match="symlink"):
        safe_repository_path(tmp_path, "link/model.cbm", must_exist=False)
