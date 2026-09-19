"""Integrity and recovery checks for the reviewed external distribution lock."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from credit_risk.artifact_distribution.contracts import (
    DEFAULT_DISTRIBUTION_LOCK,
    EXPECTED_DISTRIBUTION_LOCK_SHA256,
    load_distribution_lock,
)
from credit_risk.artifact_distribution.workflow import verify_artifacts
from credit_risk.modeling.selected_bundle import load_selected_bundle

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_distribution_lock_is_complete_and_byte_identical() -> None:
    lock_path = REPOSITORY_ROOT / DEFAULT_DISTRIBUTION_LOCK
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == (EXPECTED_DISTRIBUTION_LOCK_SHA256)

    lock = load_distribution_lock(lock_path)
    assert lock.repository.repo_id == "ShaloyL/credit-card-default-prediction"
    assert lock.repository.revision == "f73ca4ee7a2c2d2ea51741e75fccf66ae7a4a640"
    assert [artifact.artifact_id for artifact in lock.artifacts] == [
        "selected_model",
    ]
    assert lock.distribution_id == "hf_distribution_v2"
    assert lock.artifacts[0].digest_reference.json_pointer == "/model_sha256"


def test_publishable_repository_card_describes_only_the_supported_model() -> None:
    card = (REPOSITORY_ROOT / "docs/artifacts/hugging-face-repository-card.md").read_text(
        encoding="utf-8"
    )

    assert "selected_v1/model.cbm" in card
    assert "legacy_v1" not in card
    assert "pickle" not in card.lower()


@pytest.mark.artifact
def test_materialized_distribution_is_compatible_with_existing_loaders() -> None:
    result = verify_artifacts(
        config_path=DEFAULT_DISTRIBUTION_LOCK,
        repository_root=REPOSITORY_ROOT,
    )
    assert len(result.reused) == 1

    manifest, model = load_selected_bundle(REPOSITORY_ROOT / "models/selected_v1")
    assert manifest.model_sha256 == (
        "844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c"
    )
    assert model.model_id == "catboost_fixed"
