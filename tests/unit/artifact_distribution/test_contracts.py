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
    RepositoryContract,
    load_distribution_lock,
    resolve_digest_reference,
    safe_repository_path,
    sha256_file,
)


def _record(**updates: object) -> ArtifactRecord:
    values: dict[str, object] = {
        "artifact_id": "selected_model",
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
        schema_version="2.0.0",
        distribution_id="hf_distribution_v2",
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
    with pytest.raises(ValidationError, match="exactly the selected model"):
        DistributionLock(
            schema_version="2.0.0",
            distribution_id="hf_distribution_v2",
            repository=repository,
            artifacts=(),
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
    with pytest.raises(ValidationError, match="approved mapping|selected_model"):
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
        {"artifact_id": "other_model"},
    ],
)
def test_artifact_record_binds_identity_and_digest_authority(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="approved mapping|selected_model"):
        _record(**updates)


def test_distribution_rejects_duplicate_destinations() -> None:
    repository = RepositoryContract(
        provider="huggingface_hub",
        repo_id="owner/repository",
        repo_type="model",
        revision="a" * 40,
        public=True,
    )
    with pytest.raises(ValidationError, match="exactly the selected model"):
        DistributionLock(
            schema_version="2.0.0",
            distribution_id="hf_distribution_v2",
            repository=repository,
            artifacts=(_record(), _record()),
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
