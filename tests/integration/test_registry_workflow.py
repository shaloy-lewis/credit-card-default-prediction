from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from credit_risk.registry import workflow
from credit_risk.registry.contracts import EXPECTED_CONFIG_SHA256, REQUIRED_CHECKS

REPOSITORY_ROOT = Path(__file__).parents[2]


def _copy_repository_contract(tmp_path: Path) -> Path:
    for relative in (
        "configs/registry/phase7_v1.json",
        "models/selected_v1/manifest.json",
        "models/selected_v1/model.cbm",
        "reports/releases/release_a_v1/evidence-manifest.json",
        "reports/governance/phase5_v1/evidence-manifest.json",
        "configs/inference/phase6_v1.json",
        "reports/inference/phase6_v1/evidence-manifest.json",
        "tests/fixtures/prediction_request.json",
    ):
        source = REPOSITORY_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return tmp_path


def _approval(repository: Path, action: str) -> tuple[Path, str]:
    promotion = action == "promote"
    payload = {
        "schema_version": "1.0.0",
        "protocol_id": "phase7_v1",
        "decision_id": ("phase7_promotion_approval" if promotion else "phase7_rollback_approval"),
        "action": action,
        "decision": "approved",
        "scope": "local_portfolio_demo",
        "approved_by_role": "project_owner",
        "implementation_git_commit": "a" * 40,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "registered_model_name": "credit-risk-default",
        "source_revision": "phase7_rev_001" if promotion else "phase7_rev_002",
        "target_revision": "phase7_rev_002" if promotion else "phase7_rev_001",
        "checks": [{"name": name, "conclusion": "success"} for name in REQUIRED_CHECKS],
        "model_bytes_unchanged": True,
        "no_training_or_test_access": True,
        "limitations_acknowledged": True,
    }
    path = repository / "configs/registry" / f"{action}.json"
    content = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(content)
    return path.relative_to(repository), hashlib.sha256(content).hexdigest()


def test_complete_registry_promotion_deployment_rollback_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _copy_repository_contract(tmp_path)
    monkeypatch.setattr(workflow, "_repository_root", lambda: repository)
    monkeypatch.setattr(workflow, "_git_is_ancestor", lambda *_args: True)
    monkeypatch.setattr(workflow, "_git_dirty", lambda *_args: False)
    monkeypatch.setattr(workflow, "_git_commit", lambda *_args: "b" * 40)
    promotion_path, promotion_sha = _approval(repository, "promote")
    rollback_path, rollback_sha = _approval(repository, "rollback")

    registered = workflow.register_release_revisions()
    assert registered.aliases == {"candidate": "2", "champion": "1"}
    assert workflow.register_release_revisions().status == "verified_existing"

    registry_root = repository / "experiment/registry/phase7_v1"
    registration = registry_root / "events/registration.json"
    registration_bytes = registration.read_bytes()
    registration.write_text("{}\n", encoding="utf-8")
    with pytest.raises(workflow.RegistryWorkflowError, match="registration receipt"):
        workflow.register_release_revisions()
    registration.write_bytes(registration_bytes)

    foreign = registry_root / "foreign.txt"
    foreign.write_text("foreign", encoding="utf-8")
    with pytest.raises(workflow.RegistryWorkflowError, match="file allowlist"):
        workflow.register_release_revisions()
    foreign.unlink()

    with pytest.raises(RuntimeError, match="SHA-256|digest mismatch"):
        workflow.promote_candidate(
            approval_path=promotion_path,
            expected_approval_sha256="0" * 64,
        )

    original_inspect = workflow._inspect_registry
    inspect_calls = 0

    def fail_post_transition_inspection(*args, **kwargs):
        nonlocal inspect_calls
        inspect_calls += 1
        if inspect_calls == 2:
            raise RuntimeError("post-transition inspection failed")
        return original_inspect(*args, **kwargs)

    monkeypatch.setattr(workflow, "_inspect_registry", fail_post_transition_inspection)
    with pytest.raises(workflow.RegistryWorkflowError, match="was reverted"):
        workflow.promote_candidate(
            approval_path=promotion_path,
            expected_approval_sha256=promotion_sha,
        )
    monkeypatch.setattr(workflow, "_inspect_registry", original_inspect)
    assert not (registry_root / "events/promotion.json").exists()
    assert original_inspect(
        workflow.load_registry_config(repository / "configs/registry/phase7_v1.json"),
        registry_root,
    )["aliases"] == {"candidate": "2", "champion": "1"}

    promoted = workflow.promote_candidate(
        approval_path=promotion_path,
        expected_approval_sha256=promotion_sha,
    )
    assert promoted.aliases == {"champion": "2", "rollback": "1"}
    assert (
        workflow.promote_candidate(
            approval_path=promotion_path, expected_approval_sha256=promotion_sha
        ).status
        == "verified_existing"
    )

    promotion = registry_root / "events/promotion.json"
    promotion_bytes = promotion.read_bytes()
    promotion.write_text("{}\n", encoding="utf-8")
    with pytest.raises(workflow.RegistryWorkflowError, match="promotion receipt"):
        workflow.promote_candidate(
            approval_path=promotion_path, expected_approval_sha256=promotion_sha
        )
    promotion.write_bytes(promotion_bytes)

    deployed = workflow.deploy_champion()
    assert deployed.active_revision == "phase7_rev_002"
    assert workflow.registry_status()["active_registry_version"] == "2"
    deployment_root = repository / "experiment/deployments/phase7_v1"
    deployment_extra = deployment_root / "foreign.txt"
    deployment_extra.write_text("foreign", encoding="utf-8")
    with pytest.raises(RuntimeError, match="root violates"):
        workflow.registry_status()
    deployment_extra.unlink()

    original_activate = workflow._activate_version
    monkeypatch.setattr(
        workflow,
        "_activate_version",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("activation failed")),
    )
    with pytest.raises(workflow.RegistryWorkflowError, match="reverted"):
        workflow.rollback_release(
            approval_path=rollback_path,
            expected_approval_sha256=rollback_sha,
        )
    assert workflow.registry_status()["active_revision"] == "phase7_rev_002"

    monkeypatch.setattr(workflow, "_activate_version", original_activate)
    rolled_back = workflow.rollback_release(
        approval_path=rollback_path,
        expected_approval_sha256=rollback_sha,
    )
    assert rolled_back.aliases == {"champion": "1", "rollback": "2"}
    assert rolled_back.active_revision == "phase7_rev_001"
    assert (
        workflow.rollback_release(
            approval_path=rollback_path, expected_approval_sha256=rollback_sha
        ).status
        == "verified_existing"
    )

    rollback = registry_root / "events/rollback.json"
    rollback_bytes = rollback.read_bytes()
    rollback.write_text("{}\n", encoding="utf-8")
    with pytest.raises(workflow.RegistryWorkflowError, match="rollback receipt"):
        workflow.rollback_release(
            approval_path=rollback_path, expected_approval_sha256=rollback_sha
        )
    rollback.write_bytes(rollback_bytes)

    evidence = workflow.publish_registry_evidence()
    verified = workflow.verify_registry_evidence(
        expected_manifest_sha256=evidence.evidence_manifest_sha256
    )
    assert verified.status == "registry_release_control_complete"
    assert json.loads((repository / "reports/registry/phase7_v1/summary.json").read_bytes())[
        "boundaries"
    ] == {
        "fit_count": 0,
        "local_paths_published": False,
        "model_changed": False,
        "row_level_data_published": False,
        "runtime_mlflow_in_api": False,
        "sealed_test_accessed": False,
        "timestamps_published": False,
    }


def test_registry_rejects_unsafe_paths_and_concurrent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _copy_repository_contract(tmp_path)
    monkeypatch.setattr(workflow, "_repository_root", lambda: repository)
    with pytest.raises(workflow.RegistryWorkflowError, match="repository-relative"):
        workflow.register_release_revisions(registry_root="../outside")

    workflow.register_release_revisions()
    lock = repository / "experiment/registry/phase7_v1/.phase7.lock"
    lock.write_text("locked", encoding="utf-8")
    approval_path, approval_sha = _approval(repository, "promote")
    monkeypatch.setattr(workflow, "_git_is_ancestor", lambda *_args: True)
    with pytest.raises(workflow.RegistryWorkflowError, match="already in progress"):
        workflow.promote_candidate(
            approval_path=approval_path,
            expected_approval_sha256=approval_sha,
        )
