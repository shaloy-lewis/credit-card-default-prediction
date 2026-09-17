from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from credit_risk.registry.contracts import (
    EXPECTED_BUNDLE_MANIFEST_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_MODEL_SHA256,
)
from credit_risk.registry.deployment import (
    ActiveDeployment,
    DeploymentResolutionError,
    load_active_deployment,
    resolve_active_bundle,
)

REPOSITORY_ROOT = Path(__file__).parents[3]


def _deployment(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "deployment"
    bundle = root / "releases/phase7_rev_001/bundle"
    bundle.mkdir(parents=True)
    shutil.copyfile(REPOSITORY_ROOT / "models/selected_v1/manifest.json", bundle / "manifest.json")
    shutil.copyfile(REPOSITORY_ROOT / "models/selected_v1/model.cbm", bundle / "model.cbm")
    pointer = ActiveDeployment(
        schema_version="1.0.0",
        protocol_id="phase7_v1",
        registered_model_name="credit-risk-default",
        alias="champion",
        registry_version=1,
        release_revision="phase7_rev_001",
        bundle_relative_path="releases/phase7_rev_001/bundle",
        config_sha256=EXPECTED_CONFIG_SHA256,
        bundle_manifest_sha256=EXPECTED_BUNDLE_MANIFEST_SHA256,
        model_sha256=EXPECTED_MODEL_SHA256,
        approval_sha256="2" * 64,
        event_receipt_sha256="3" * 64,
    )
    (root / "active.json").write_text(pointer.model_dump_json(), encoding="utf-8")
    return root, bundle


def test_resolve_active_bundle_authenticates_pointer_and_files(tmp_path: Path) -> None:
    root, bundle = _deployment(tmp_path)
    assert resolve_active_bundle(root) == bundle.resolve()
    assert load_active_deployment(root).release_revision == "phase7_rev_001"


@pytest.mark.parametrize("target", ["manifest.json", "model.cbm"])
def test_resolve_active_bundle_rejects_tampered_files(tmp_path: Path, target: str) -> None:
    root, bundle = _deployment(tmp_path)
    (bundle / target).write_bytes(b"tampered")
    with pytest.raises(DeploymentResolutionError, match="digest mismatch"):
        resolve_active_bundle(root)


def test_resolve_active_bundle_rejects_extra_file_and_invalid_pointer(tmp_path: Path) -> None:
    root, bundle = _deployment(tmp_path)
    (bundle / "extra.bin").write_bytes(b"x")
    with pytest.raises(DeploymentResolutionError, match="file allowlist"):
        resolve_active_bundle(root)

    (bundle / "extra.bin").unlink()
    pointer = json.loads((root / "active.json").read_bytes())
    pointer["bundle_relative_path"] = "../outside"
    (root / "active.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(DeploymentResolutionError, match="Invalid active deployment pointer"):
        resolve_active_bundle(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("registry_version", 2, "version differs"),
        ("config_sha256", "0" * 64, "configuration digest"),
        ("bundle_manifest_sha256", "0" * 64, "manifest digest"),
        ("model_sha256", "0" * 64, "model digest"),
    ],
)
def test_resolve_active_bundle_rejects_unreviewed_pointer_identity(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    root, _ = _deployment(tmp_path)
    pointer = json.loads((root / "active.json").read_bytes())
    pointer[field] = value
    (root / "active.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(DeploymentResolutionError, match=message):
        resolve_active_bundle(root)


def test_resolve_active_bundle_rejects_foreign_deployment_entries(tmp_path: Path) -> None:
    root, _ = _deployment(tmp_path)
    (root / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DeploymentResolutionError, match="root violates"):
        resolve_active_bundle(root)

    (root / "extra.json").unlink()
    extra_release = root / "releases/foreign/bundle"
    extra_release.mkdir(parents=True)
    with pytest.raises(DeploymentResolutionError, match="releases violate"):
        resolve_active_bundle(root)

    extra_release.rmdir()
    extra_release.parent.rmdir()
    (root / "releases/phase7_rev_002").mkdir()
    with pytest.raises(DeploymentResolutionError, match="directory allowlist"):
        resolve_active_bundle(root)


def test_resolve_active_bundle_rejects_missing_pointer(tmp_path: Path) -> None:
    with pytest.raises(DeploymentResolutionError, match="pointer"):
        resolve_active_bundle(tmp_path)


def test_resolve_active_bundle_rejects_symlinked_file(tmp_path: Path) -> None:
    root, bundle = _deployment(tmp_path)
    target = bundle / "model.cbm"
    copy = tmp_path / "model.cbm"
    shutil.copyfile(target, copy)
    target.unlink()
    try:
        target.symlink_to(copy)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")
    with pytest.raises(DeploymentResolutionError, match="file allowlist"):
        resolve_active_bundle(root)
