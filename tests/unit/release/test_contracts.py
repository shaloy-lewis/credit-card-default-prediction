"""Tests for the frozen Release A contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from credit_risk.release.contracts import (
    RELEASE_CRITERIA,
    RELEASE_OUTPUTS,
    SOURCE_ARTIFACT_ROLES,
    ReleaseContractError,
    load_release_config,
    release_config_sha256,
)
from tests.unit.release.helpers import (
    COMMITTED_UNCERTAINTY,
    SOURCE_CONFIG,
    config_payload,
    write_config,
)


def test_load_release_config_accepts_exact_reviewed_contract() -> None:
    config = load_release_config(SOURCE_CONFIG)

    assert set(config.source_artifacts) == SOURCE_ARTIFACT_ROLES
    assert config.release_criteria == RELEASE_CRITERIA
    assert config.outputs == RELEASE_OUTPUTS
    assert len(release_config_sha256(SOURCE_CONFIG)) == 64


def test_committed_uncertainty_fixture_matches_the_reviewed_runtime_digest() -> None:
    config = load_release_config(SOURCE_CONFIG)

    assert hashlib.sha256(COMMITTED_UNCERTAINTY.read_bytes()).hexdigest() == (
        config.uncertainty_source.sha256
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["source_artifacts"].pop("split_lock"), "source artifact roles"),
        (lambda value: value["release_criteria"].reverse(), "criteria"),
        (lambda value: value["outputs"].append("extra.json"), "outputs"),
        (
            lambda value: value["uncertainty_source"].update({"path": "experiment/other.json"}),
            "uncertainty source",
        ),
        (
            lambda value: value["source_artifacts"]["data_manifest"].update(
                {"path": "../outside.json"}
            ),
            "safe repository-relative",
        ),
        (
            lambda value: value["source_artifacts"]["data_manifest"].update(
                {"sha256": "not-a-digest"}
            ),
            "string_pattern_mismatch",
        ),
        (
            lambda value: value["source_artifacts"]["data_manifest"].update(
                {"path": value["source_artifacts"]["split_lock"]["path"]}
            ),
            "paths must be unique",
        ),
    ],
)
def test_contract_rejects_altered_release_protocol(
    tmp_path: Path, mutation: object, message: str
) -> None:
    payload = config_payload()
    mutation(payload)  # type: ignore[operator]
    path = tmp_path / "release.json"
    write_config(path, payload)

    with pytest.raises(ReleaseContractError, match=message):
        load_release_config(path)


def test_contract_reports_missing_malformed_and_unhashable_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ReleaseContractError, match="Invalid Release A configuration"):
        load_release_config(missing)
    with pytest.raises(ReleaseContractError, match="Unable to hash"):
        release_config_sha256(missing)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(ReleaseContractError, match="Invalid Release A configuration"):
        load_release_config(malformed)
