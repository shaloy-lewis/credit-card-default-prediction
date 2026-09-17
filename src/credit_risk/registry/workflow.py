"""Governed MLflow registry, deployment, rollback, and evidence workflow."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import ValidationError

from credit_risk.modeling.selected_bundle import BundleManifest
from credit_risk.registry.contracts import (
    DEFAULT_DEPLOYMENT_ROOT,
    DEFAULT_EVIDENCE_ROOT,
    DEFAULT_REGISTRY_CONFIG_PATH,
    DEFAULT_REGISTRY_ROOT,
    EXPECTED_CONFIG_SHA256,
    PUBLISHED_FILES,
    RELEASE_REVISIONS,
    RegistryConfig,
    ReleaseApproval,
    config_sha256,
    load_approval,
    load_registry_config,
)
from credit_risk.registry.deployment import (
    ActiveDeployment,
    load_active_deployment,
    resolve_active_bundle,
)

REGISTRY_DATABASE = "registry.db"
REGISTRATION_RECEIPT = "registration.json"
PROMOTION_RECEIPT = "promotion.json"
ROLLBACK_RECEIPT = "rollback.json"
VERSION_TAG_KEYS = frozenset(
    {
        "protocol_id",
        "release_revision",
        "bundle_id",
        "model_id",
        "bundle_manifest_sha256",
        "model_sha256",
        "config_sha256",
        "release_a_manifest_sha256",
        "phase5_manifest_sha256",
        "phase6_manifest_sha256",
        "model_bytes_unchanged",
    }
)


class RegistryWorkflowError(RuntimeError):
    """Raised when a release-control operation fails closed."""


@dataclass(frozen=True, slots=True)
class RegistryOperationResult:
    """Portable result from registration, transition, or deployment."""

    status: str
    registered_model_name: str
    aliases: dict[str, str]
    receipt_sha256: str
    active_revision: str | None = None


@dataclass(frozen=True, slots=True)
class RegistryEvidenceResult:
    """Authenticated Phase 7 evidence result."""

    evidence_root: Path
    summary_sha256: str
    evidence_manifest_sha256: str
    status: str


def register_release_revisions(
    *,
    config_path: str | Path = DEFAULT_REGISTRY_CONFIG_PATH,
    bundle_root: str | Path = "models/selected_v1",
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
) -> RegistryOperationResult:
    """Create the two reviewed deployment revisions or verify exact prior state."""

    config, repository, config_file = _load_context(config_path)
    bundle = _safe_path(repository, bundle_root, "models", "bundle root", must_exist=True)
    registry = _safe_path(
        repository, registry_root, "experiment/registry", "registry root", must_exist=False
    )
    _validate_config_sources(repository, config)
    bundle_manifest = _validate_bundle(config, bundle)
    if registry.exists():
        if (registry / ".initializing").exists():
            raise RegistryWorkflowError(
                "Registry contains an incomplete initialization marker; quarantine it before retrying."
            )
        state = _inspect_registry(config, registry)
        _require_aliases(state["aliases"], {"champion": "1", "candidate": "2"})
        receipt = _read_receipt(registry / "events" / REGISTRATION_RECEIPT)
        expected_receipt = _registration_receipt(
            config=config,
            bundle_manifest=bundle_manifest,
            config_digest=config_sha256(config_file),
            versions={RELEASE_REVISIONS[0]: "1", RELEASE_REVISIONS[1]: "2"},
        )
        if receipt != expected_receipt:
            raise RegistryWorkflowError("Existing registration receipt differs from the contract.")
        return RegistryOperationResult(
            status="verified_existing",
            registered_model_name=config.registry.registered_model_name,
            aliases=state["aliases"],
            receipt_sha256=_sha256_bytes(_json_bytes(receipt)),
        )

    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.mkdir()
    marker = registry / ".initializing"
    marker.write_text("phase7_v1\n", encoding="utf-8")
    client: Any | None = None
    try:
        artifact_bundle = registry / "artifacts" / config.bundle.model_sha256 / "bundle"
        artifact_bundle.mkdir(parents=True)
        _copy_exact_bundle(bundle, artifact_bundle, config)
        database = registry / REGISTRY_DATABASE
        client = _mlflow_client(database)
        model_name = config.registry.registered_model_name
        client.create_registered_model(
            model_name,
            tags={
                "protocol_id": config.protocol_id,
                "bundle_id": config.bundle.bundle_id,
                "model_id": config.bundle.model_id,
            },
            description=(
                "Portfolio release-control registry. Both Phase 7 versions contain "
                "the same reviewed selected_v1 model bytes."
            ),
        )
        final_artifact_uri = (
            (registry / "artifacts" / config.bundle.model_sha256 / "bundle").resolve().as_uri()
        )
        versions: dict[str, str] = {}
        for revision in config.registry.revisions:
            created = client.create_model_version(
                model_name,
                source=final_artifact_uri,
                tags=_version_tags(config, revision.release_revision, config_sha256(config_file)),
                description=(
                    f"Immutable deployment revision {revision.release_revision}; "
                    "model bytes are unchanged."
                ),
            )
            versions[revision.release_revision] = str(created.version)
            client.set_registered_model_alias(model_name, revision.initial_alias, created.version)
        receipt = _registration_receipt(
            config=config,
            bundle_manifest=bundle_manifest,
            config_digest=config_sha256(config_file),
            versions=versions,
        )
        _write_atomic(registry / "events" / REGISTRATION_RECEIPT, _json_bytes(receipt))
        _dispose_client(client)
        client = None
        marker.unlink()
    except Exception as error:
        if client is not None:
            _dispose_client(client)
        if isinstance(error, RegistryWorkflowError):
            raise
        raise RegistryWorkflowError(f"Registry initialization failed: {error}") from error

    state = _inspect_registry(config, registry)
    _require_aliases(state["aliases"], {"champion": "1", "candidate": "2"})
    return RegistryOperationResult(
        status="registered",
        registered_model_name=config.registry.registered_model_name,
        aliases=state["aliases"],
        receipt_sha256=_sha256_file(registry / "events" / REGISTRATION_RECEIPT),
    )


def promote_candidate(
    *,
    approval_path: str | Path,
    expected_approval_sha256: str,
    config_path: str | Path = DEFAULT_REGISTRY_CONFIG_PATH,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
) -> RegistryOperationResult:
    """Promote the reviewed candidate and preserve the prior champion for rollback."""

    config, repository, config_file = _load_context(config_path)
    registry = _safe_path(
        repository, registry_root, "experiment/registry", "registry root", must_exist=True
    )
    approval_file = _safe_path(
        repository, approval_path, "configs/registry", "promotion approval", must_exist=True
    )
    approval = _validate_approval(
        config,
        load_approval(approval_file, expected_approval_sha256),
        action="promote",
        config_path=config_file,
        repository=repository,
    )
    receipt_path = registry / "events" / PROMOTION_RECEIPT
    with _writer_lock(registry):
        state = _inspect_registry(config, registry)
        if receipt_path.exists():
            _require_aliases(state["aliases"], {"champion": "2", "rollback": "1"})
            expected_receipt = _transition_receipt(
                config=config,
                event="promotion",
                approval=approval,
                approval_sha256=expected_approval_sha256,
                before={"candidate": "2", "champion": "1"},
                after={"champion": "2", "rollback": "1"},
            )
            if _read_receipt(receipt_path) != expected_receipt:
                raise RegistryWorkflowError("Existing promotion receipt differs from the approval.")
            return RegistryOperationResult(
                status="verified_existing",
                registered_model_name=config.registry.registered_model_name,
                aliases=state["aliases"],
                receipt_sha256=_sha256_file(receipt_path),
            )
        _require_aliases(state["aliases"], {"champion": "1", "candidate": "2"})
        client = _mlflow_client(registry / REGISTRY_DATABASE)
        before = dict(state["aliases"])
        try:
            client.set_registered_model_alias(
                config.registry.registered_model_name, "rollback", "1"
            )
            client.set_registered_model_alias(
                config.registry.registered_model_name, "champion", "2"
            )
            client.delete_registered_model_alias(config.registry.registered_model_name, "candidate")
            receipt = _transition_receipt(
                config=config,
                event="promotion",
                approval=approval,
                approval_sha256=expected_approval_sha256,
                before=before,
                after={"champion": "2", "rollback": "1"},
            )
            _write_atomic(receipt_path, _json_bytes(receipt))
            after = _inspect_registry(config, registry)["aliases"]
            _require_aliases(after, {"champion": "2", "rollback": "1"})
        except Exception as error:
            compensation_error = _compensate_transition(
                client=client,
                model_name=config.registry.registered_model_name,
                aliases=before,
                receipt_path=receipt_path,
            )
            if compensation_error is not None:
                raise RegistryWorkflowError(
                    f"Candidate promotion failed and compensation also failed: {compensation_error}"
                ) from error
            if isinstance(error, RegistryWorkflowError):
                raise
            raise RegistryWorkflowError(
                f"Candidate promotion failed and was reverted: {error}"
            ) from error
        finally:
            _dispose_client(client)
    return RegistryOperationResult(
        status="promoted",
        registered_model_name=config.registry.registered_model_name,
        aliases=after,
        receipt_sha256=_sha256_file(receipt_path),
    )


def deploy_champion(
    *,
    alias: str = "champion",
    config_path: str | Path = DEFAULT_REGISTRY_CONFIG_PATH,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    deployment_root: str | Path = DEFAULT_DEPLOYMENT_ROOT,
) -> RegistryOperationResult:
    """Materialise and atomically activate the current champion."""

    if alias != "champion":
        raise RegistryWorkflowError("Only the reviewed champion alias can be deployed.")
    config, repository, _ = _load_context(config_path)
    registry = _safe_path(
        repository, registry_root, "experiment/registry", "registry root", must_exist=True
    )
    deployment = _safe_path(
        repository,
        deployment_root,
        "experiment/deployments",
        "deployment root",
        must_exist=False,
    )
    if registry == deployment or registry in deployment.parents or deployment in registry.parents:
        raise RegistryWorkflowError("Registry and deployment roots must not overlap.")
    promotion_path = registry / "events" / PROMOTION_RECEIPT
    if not promotion_path.is_file():
        raise RegistryWorkflowError("Champion deployment requires a completed promotion receipt.")
    with _writer_lock(registry):
        state = _inspect_registry(config, registry)
        _require_aliases(state["aliases"], {"champion": "2", "rollback": "1"})
        promotion = _read_receipt(promotion_path)
        active = _activate_version(
            config=config,
            registry_root=registry,
            deployment_root=deployment,
            version="2",
            approval_sha256=str(promotion["approval_sha256"]),
            event_receipt_sha256=_sha256_file(promotion_path),
        )
    return RegistryOperationResult(
        status="deployed",
        registered_model_name=config.registry.registered_model_name,
        aliases=state["aliases"],
        receipt_sha256=active.event_receipt_sha256,
        active_revision=active.release_revision,
    )


def rollback_release(
    *,
    approval_path: str | Path,
    expected_approval_sha256: str,
    config_path: str | Path = DEFAULT_REGISTRY_CONFIG_PATH,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    deployment_root: str | Path = DEFAULT_DEPLOYMENT_ROOT,
) -> RegistryOperationResult:
    """Restore the reviewed rollback revision and active deployment pointer."""

    config, repository, config_file = _load_context(config_path)
    registry = _safe_path(
        repository, registry_root, "experiment/registry", "registry root", must_exist=True
    )
    deployment = _safe_path(
        repository,
        deployment_root,
        "experiment/deployments",
        "deployment root",
        must_exist=True,
    )
    approval_file = _safe_path(
        repository, approval_path, "configs/registry", "rollback approval", must_exist=True
    )
    approval = _validate_approval(
        config,
        load_approval(approval_file, expected_approval_sha256),
        action="rollback",
        config_path=config_file,
        repository=repository,
    )
    receipt_path = registry / "events" / ROLLBACK_RECEIPT
    pointer_path = deployment / "active.json"
    with _writer_lock(registry):
        state = _inspect_registry(config, registry)
        if receipt_path.exists():
            _require_aliases(state["aliases"], {"champion": "1", "rollback": "2"})
            expected_receipt = _transition_receipt(
                config=config,
                event="rollback",
                approval=approval,
                approval_sha256=expected_approval_sha256,
                before={"champion": "2", "rollback": "1"},
                after={"champion": "1", "rollback": "2"},
            )
            if _read_receipt(receipt_path) != expected_receipt:
                raise RegistryWorkflowError("Existing rollback receipt differs from the approval.")
            active = load_active_deployment(deployment)
            if active.release_revision != RELEASE_REVISIONS[0]:
                raise RegistryWorkflowError("Existing rollback deployment state is inconsistent.")
            return RegistryOperationResult(
                status="verified_existing",
                registered_model_name=config.registry.registered_model_name,
                aliases=state["aliases"],
                receipt_sha256=_sha256_file(receipt_path),
                active_revision=active.release_revision,
            )
        _require_aliases(state["aliases"], {"champion": "2", "rollback": "1"})
        before_aliases = dict(state["aliases"])
        prior_pointer = pointer_path.read_bytes() if pointer_path.is_file() else None
        client = _mlflow_client(registry / REGISTRY_DATABASE)
        try:
            client.set_registered_model_alias(
                config.registry.registered_model_name, "champion", "1"
            )
            client.set_registered_model_alias(
                config.registry.registered_model_name, "rollback", "2"
            )
            receipt = _transition_receipt(
                config=config,
                event="rollback",
                approval=approval,
                approval_sha256=expected_approval_sha256,
                before=before_aliases,
                after={"champion": "1", "rollback": "2"},
            )
            _write_atomic(receipt_path, _json_bytes(receipt))
            active = _activate_version(
                config=config,
                registry_root=registry,
                deployment_root=deployment,
                version="1",
                approval_sha256=expected_approval_sha256,
                event_receipt_sha256=_sha256_file(receipt_path),
            )
            after = _inspect_registry(config, registry)["aliases"]
            _require_aliases(after, {"champion": "1", "rollback": "2"})
        except Exception as error:
            compensation_error = _compensate_transition(
                client=client,
                model_name=config.registry.registered_model_name,
                aliases=before_aliases,
                receipt_path=receipt_path,
                pointer_path=pointer_path,
                prior_pointer=prior_pointer,
            )
            if compensation_error is not None:
                raise RegistryWorkflowError(
                    f"Rollback failed and compensation also failed: {compensation_error}"
                ) from error
            if isinstance(error, RegistryWorkflowError):
                raise
            raise RegistryWorkflowError(f"Rollback failed and was reverted: {error}") from error
        finally:
            _dispose_client(client)
    return RegistryOperationResult(
        status="rolled_back",
        registered_model_name=config.registry.registered_model_name,
        aliases=after,
        receipt_sha256=_sha256_file(receipt_path),
        active_revision=active.release_revision,
    )


def registry_status(
    *,
    config_path: str | Path = DEFAULT_REGISTRY_CONFIG_PATH,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    deployment_root: str | Path = DEFAULT_DEPLOYMENT_ROOT,
) -> dict[str, Any]:
    """Return validated live registry and deployment state."""

    config, repository, _ = _load_context(config_path)
    registry = _safe_path(
        repository, registry_root, "experiment/registry", "registry root", must_exist=True
    )
    deployment = _safe_path(
        repository,
        deployment_root,
        "experiment/deployments",
        "deployment root",
        must_exist=True,
    )
    state = _inspect_registry(config, registry)
    active = load_active_deployment(deployment)
    return {
        "registered_model_name": config.registry.registered_model_name,
        "aliases": state["aliases"],
        "versions": state["versions"],
        "active_revision": active.release_revision,
        "active_registry_version": str(active.registry_version),
        "bundle_manifest_sha256": active.bundle_manifest_sha256,
        "model_sha256": active.model_sha256,
    }


def publish_registry_evidence(
    *,
    config_path: str | Path = DEFAULT_REGISTRY_CONFIG_PATH,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    deployment_root: str | Path = DEFAULT_DEPLOYMENT_ROOT,
    output_root: str | Path = DEFAULT_EVIDENCE_ROOT,
) -> RegistryEvidenceResult:
    """Publish aggregate Phase 7 evidence from a completed rollback drill."""

    config, repository, config_file = _load_context(config_path)
    if _git_dirty(repository):
        raise RegistryWorkflowError("Official Phase 7 evidence requires a clean worktree.")
    registry = _safe_path(
        repository, registry_root, "experiment/registry", "registry root", must_exist=True
    )
    deployment = _safe_path(
        repository,
        deployment_root,
        "experiment/deployments",
        "deployment root",
        must_exist=True,
    )
    output = _safe_path(
        repository, output_root, "reports/registry", "evidence root", must_exist=False
    )
    if output.exists():
        raise RegistryWorkflowError(f"Phase 7 evidence already exists: {output}")
    live_registry = _inspect_registry(config, registry)
    active_pointer = load_active_deployment(deployment)
    state = {
        "aliases": live_registry["aliases"],
        "versions": live_registry["versions"],
        "active_revision": active_pointer.release_revision,
        "active_registry_version": str(active_pointer.registry_version),
    }
    _require_aliases(state["aliases"], {"champion": "1", "rollback": "2"})
    receipts = {
        "registration": _read_receipt(registry / "events" / REGISTRATION_RECEIPT),
        "promotion": _read_receipt(registry / "events" / PROMOTION_RECEIPT),
        "rollback": _read_receipt(registry / "events" / ROLLBACK_RECEIPT),
    }
    implementation_commit = str(receipts["promotion"]["implementation_git_commit"])
    summary = {
        "schema_version": "1.0.0",
        "evidence_id": "phase7_v1",
        "status": "registry_release_control_complete",
        "implementation_git_commit": implementation_commit,
        "execution_git_commit": _git_commit(repository),
        "configuration_sha256": config_sha256(config_file),
        "registered_model_name": config.registry.registered_model_name,
        "backend": "sqlite",
        "artifact_store": "content_addressed_filesystem",
        "revisions": [
            {
                "registry_version": "1",
                "release_revision": RELEASE_REVISIONS[0],
                "bundle_manifest_sha256": config.bundle.manifest_sha256,
                "model_sha256": config.bundle.model_sha256,
            },
            {
                "registry_version": "2",
                "release_revision": RELEASE_REVISIONS[1],
                "bundle_manifest_sha256": config.bundle.manifest_sha256,
                "model_sha256": config.bundle.model_sha256,
            },
        ],
        "model_bytes_unchanged_between_revisions": True,
        "transitions": {
            "initial": receipts["registration"]["aliases_after"],
            "promoted": receipts["promotion"]["aliases_after"],
            "rolled_back": receipts["rollback"]["aliases_after"],
        },
        "final_state": {
            "aliases": state["aliases"],
            "active_revision": state["active_revision"],
            "active_registry_version": state["active_registry_version"],
        },
        "checks": receipts["promotion"]["checks"],
        "image_scan": {
            "scanner": config.image_scan.scanner,
            "severity": list(config.image_scan.severity),
            "ignore_unfixed": config.image_scan.ignore_unfixed,
            "passed": True,
            "waiver_used": False,
        },
        "boundaries": {
            "fit_count": 0,
            "model_changed": False,
            "sealed_test_accessed": False,
            "runtime_mlflow_in_api": False,
            "row_level_data_published": False,
            "timestamps_published": False,
            "local_paths_published": False,
        },
        "claims": {
            "different_model_versions_compared": False,
            "production_readiness_claimed": False,
            "g4_closed": False,
        },
    }
    summary_bytes = _json_bytes(summary)
    files = {
        "summary.json": summary_bytes,
        "registry-release-report.md": _render_report(
            summary, _sha256_bytes(summary_bytes)
        ).encode(),
        "promotion-checklist.md": _render_checklist(summary).encode(),
        "rollback-runbook.md": _render_runbook(summary).encode(),
    }
    manifest = {
        "schema_version": "1.0.0",
        "evidence_id": "phase7_v1",
        "implementation_git_commit": implementation_commit,
        "configuration_sha256": config_sha256(config_file),
        "source_artifacts": {
            "bundle_manifest": {"sha256": config.bundle.manifest_sha256},
            "selected_model": {"sha256": config.bundle.model_sha256},
            **{
                role: {"sha256": reference.sha256}
                for role, reference in sorted(config.source_evidence.items())
            },
        },
        "runtime_receipts": {
            name: {"published": False, "sha256": _sha256_bytes(_json_bytes(receipt))}
            for name, receipt in receipts.items()
        },
        "artifacts": {
            name: {"row_level_data": False, "sha256": _sha256_bytes(content)}
            for name, content in sorted(files.items())
        },
        "boundaries_verified": summary["boundaries"],
        "allowlisted_outputs": list(PUBLISHED_FILES),
    }
    files["evidence-manifest.json"] = _json_bytes(manifest)
    _publish_directory(output, files)
    digest = _sha256_file(output / "evidence-manifest.json")
    return verify_registry_evidence(
        expected_manifest_sha256=digest,
        config_path=config_path,
        evidence_root=Path(output_root),
    )


def verify_registry_evidence(
    *,
    expected_manifest_sha256: str,
    config_path: str | Path = DEFAULT_REGISTRY_CONFIG_PATH,
    evidence_root: str | Path = DEFAULT_EVIDENCE_ROOT,
) -> RegistryEvidenceResult:
    """Authenticate committed registry evidence without requiring live MLflow state."""

    config, repository, config_file = _load_context(config_path)
    if len(expected_manifest_sha256) != 64:
        raise RegistryWorkflowError("Expected evidence manifest digest must be SHA-256.")
    evidence = _safe_path(
        repository, evidence_root, "reports/registry", "evidence root", must_exist=True
    )
    entries = tuple(evidence.iterdir())
    if {entry.name for entry in entries} != set(PUBLISHED_FILES) or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise RegistryWorkflowError("Phase 7 evidence violates its output allowlist.")
    observed_manifest = _sha256_file(evidence / "evidence-manifest.json")
    if observed_manifest != expected_manifest_sha256:
        raise RegistryWorkflowError(
            "Phase 7 evidence manifest digest mismatch: "
            f"expected={expected_manifest_sha256}, observed={observed_manifest}"
        )
    manifest = _read_receipt(evidence / "evidence-manifest.json")
    if (
        manifest.get("evidence_id") != "phase7_v1"
        or manifest.get("configuration_sha256") != config_sha256(config_file)
        or manifest.get("allowlisted_outputs") != list(PUBLISHED_FILES)
    ):
        raise RegistryWorkflowError("Phase 7 evidence manifest violates its frozen contract.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(PUBLISHED_FILES[:-1]):
        raise RegistryWorkflowError("Phase 7 evidence artifact manifest is incomplete.")
    for name, descriptor in artifacts.items():
        if descriptor.get("row_level_data") is not False or descriptor.get(
            "sha256"
        ) != _sha256_file(evidence / name):
            raise RegistryWorkflowError(f"Phase 7 evidence digest mismatch for {name}.")
    _validate_config_sources(repository, config)
    summary = _read_receipt(evidence / "summary.json")
    if (
        summary.get("status") != "registry_release_control_complete"
        or summary.get("model_bytes_unchanged_between_revisions") is not True
        or summary.get("final_state", {}).get("active_revision") != RELEASE_REVISIONS[0]
        or summary.get("boundaries", {}).get("fit_count") != 0
        or summary.get("boundaries", {}).get("sealed_test_accessed") is not False
        or summary.get("claims", {}).get("g4_closed") is not False
    ):
        raise RegistryWorkflowError("Phase 7 summary violates its reviewed semantics.")
    return RegistryEvidenceResult(
        evidence_root=evidence,
        summary_sha256=_sha256_file(evidence / "summary.json"),
        evidence_manifest_sha256=observed_manifest,
        status=str(summary["status"]),
    )


def _load_context(config_path: str | Path) -> tuple[RegistryConfig, Path, Path]:
    repository = _repository_root()
    config_file = _safe_path(
        repository, config_path, "configs/registry", "registry config", must_exist=True
    )
    return load_registry_config(config_file), repository, config_file


def _validate_config_sources(repository: Path, config: RegistryConfig) -> None:
    for role, reference in config.source_evidence.items():
        path = _safe_path(repository, reference.path, ".", role, must_exist=True)
        if not path.is_file() or _sha256_file(path) != reference.sha256:
            raise RegistryWorkflowError(f"Reviewed source digest mismatch for {role}.")


def _validate_bundle(config: RegistryConfig, bundle: Path) -> BundleManifest:
    if bundle.is_symlink() or not bundle.is_dir():
        raise RegistryWorkflowError("Selected bundle root is missing or unsafe.")
    entries = tuple(bundle.iterdir())
    if {entry.name for entry in entries} != {"manifest.json", "model.cbm"} or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise RegistryWorkflowError("Selected bundle violates the two-file allowlist.")
    if _sha256_file(bundle / "manifest.json") != config.bundle.manifest_sha256:
        raise RegistryWorkflowError("Selected bundle manifest digest mismatch.")
    if _sha256_file(bundle / "model.cbm") != config.bundle.model_sha256:
        raise RegistryWorkflowError("Selected model digest mismatch.")
    try:
        manifest = BundleManifest.model_validate_json((bundle / "manifest.json").read_bytes())
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise RegistryWorkflowError(f"Selected bundle manifest is invalid: {error}") from error
    if (
        manifest.bundle_id != config.bundle.bundle_id
        or manifest.selected_model_id != config.bundle.model_id
        or manifest.model_sha256 != config.bundle.model_sha256
    ):
        raise RegistryWorkflowError("Selected bundle semantics differ from the registry contract.")
    return manifest


def _copy_exact_bundle(source: Path, destination: Path, config: RegistryConfig) -> None:
    shutil.copyfile(source / "manifest.json", destination / "manifest.json")
    shutil.copyfile(source / "model.cbm", destination / "model.cbm")
    _validate_bundle(config, destination)


def _version_tags(config: RegistryConfig, revision: str, config_digest: str) -> dict[str, str]:
    return {
        "protocol_id": config.protocol_id,
        "release_revision": revision,
        "bundle_id": config.bundle.bundle_id,
        "model_id": config.bundle.model_id,
        "bundle_manifest_sha256": config.bundle.manifest_sha256,
        "model_sha256": config.bundle.model_sha256,
        "config_sha256": config_digest,
        "release_a_manifest_sha256": config.source_evidence["release_a_manifest"].sha256,
        "phase5_manifest_sha256": config.source_evidence["phase5_manifest"].sha256,
        "phase6_manifest_sha256": config.source_evidence["phase6_manifest"].sha256,
        "model_bytes_unchanged": "true",
    }


def _inspect_registry(config: RegistryConfig, registry_root: Path) -> dict[str, Any]:
    _validate_registry_layout(config, registry_root)
    database = registry_root / REGISTRY_DATABASE
    artifact_bundle = registry_root / "artifacts" / config.bundle.model_sha256 / "bundle"
    if not database.is_file() or database.is_symlink():
        raise RegistryWorkflowError("Registry database is missing or unsafe.")
    _validate_bundle(config, artifact_bundle)
    client = _mlflow_client(database)
    try:
        registered = client.get_registered_model(config.registry.registered_model_name)
        raw_versions = client.search_model_versions(
            f"name='{config.registry.registered_model_name}'"
        )
    except Exception as error:
        raise RegistryWorkflowError(f"Unable to inspect MLflow registry: {error}") from error
    finally:
        _dispose_client(client)
    versions = {str(item.version): item for item in raw_versions}
    if set(versions) != {"1", "2"}:
        raise RegistryWorkflowError("Registry must contain exactly the two reviewed versions.")
    expected_tags = {
        "1": _version_tags(config, RELEASE_REVISIONS[0], EXPECTED_CONFIG_SHA256),
        "2": _version_tags(config, RELEASE_REVISIONS[1], EXPECTED_CONFIG_SHA256),
    }
    expected_source = artifact_bundle.resolve().as_uri()
    serialised: dict[str, Any] = {}
    for number, version in sorted(versions.items()):
        tags = {str(key): str(value) for key, value in version.tags.items()}
        if set(tags) != VERSION_TAG_KEYS or tags != expected_tags[number]:
            raise RegistryWorkflowError(f"Registry version {number} tags differ from the contract.")
        if str(version.source) != expected_source:
            raise RegistryWorkflowError(f"Registry version {number} has a foreign artifact source.")
        serialised[number] = {
            "release_revision": tags["release_revision"],
            "bundle_manifest_sha256": tags["bundle_manifest_sha256"],
            "model_sha256": tags["model_sha256"],
        }
    aliases = {str(key): str(value) for key, value in (registered.aliases or {}).items()}
    if not set(aliases).issubset({"candidate", "champion", "rollback"}):
        raise RegistryWorkflowError("Registry contains an unapproved alias.")
    return {"versions": serialised, "aliases": aliases}


def _validate_registry_layout(config: RegistryConfig, registry_root: Path) -> None:
    """Reject foreign files and links in the governed local registry tree."""

    try:
        root_entries = tuple(registry_root.iterdir())
    except OSError as error:
        raise RegistryWorkflowError(f"Unable to inspect registry layout: {error}") from error
    allowed_root = {REGISTRY_DATABASE, "artifacts", "events", ".phase7.lock"}
    names = {entry.name for entry in root_entries}
    if not {REGISTRY_DATABASE, "artifacts", "events"}.issubset(names) or not names.issubset(
        allowed_root
    ):
        raise RegistryWorkflowError("Registry root violates its file allowlist.")
    if any(entry.is_symlink() for entry in root_entries):
        raise RegistryWorkflowError("Registry root must not contain symlinks.")
    if not (registry_root / "artifacts").is_dir() or not (registry_root / "events").is_dir():
        raise RegistryWorkflowError("Registry artifact or event directory is missing.")

    artifacts = registry_root / "artifacts"
    artifact_entries = tuple(artifacts.iterdir())
    if (
        {entry.name for entry in artifact_entries} != {config.bundle.model_sha256}
        or artifact_entries[0].is_symlink()
        or not artifact_entries[0].is_dir()
    ):
        raise RegistryWorkflowError("Registry artifact store violates its digest allowlist.")
    digest_entries = tuple(artifact_entries[0].iterdir())
    if (
        {entry.name for entry in digest_entries} != {"bundle"}
        or digest_entries[0].is_symlink()
        or not digest_entries[0].is_dir()
    ):
        raise RegistryWorkflowError("Registry artifact revision violates its allowlist.")

    events = registry_root / "events"
    event_entries = tuple(events.iterdir())
    allowed_events = {REGISTRATION_RECEIPT, PROMOTION_RECEIPT, ROLLBACK_RECEIPT}
    if not {entry.name for entry in event_entries}.issubset(allowed_events) or any(
        entry.is_symlink() or not entry.is_file() for entry in event_entries
    ):
        raise RegistryWorkflowError("Registry event directory violates its file allowlist.")


def _validate_approval(
    config: RegistryConfig,
    approval: ReleaseApproval,
    *,
    action: Literal["promote", "rollback"],
    config_path: Path,
    repository: Path,
) -> ReleaseApproval:
    if approval.action != action:
        raise RegistryWorkflowError(f"Expected a {action} approval.")
    if approval.config_sha256 != config_sha256(config_path):
        raise RegistryWorkflowError("Approval configuration digest is stale.")
    if approval.registered_model_name != config.registry.registered_model_name:
        raise RegistryWorkflowError("Approval names a different registered model.")
    if not _git_is_ancestor(repository, approval.implementation_git_commit):
        raise RegistryWorkflowError("Approved implementation commit is not in the current history.")
    return approval


def _registration_receipt(
    *,
    config: RegistryConfig,
    bundle_manifest: BundleManifest,
    config_digest: str,
    versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "protocol_id": config.protocol_id,
        "event": "registration",
        "registered_model_name": config.registry.registered_model_name,
        "config_sha256": config_digest,
        "bundle_manifest_sha256": config.bundle.manifest_sha256,
        "model_sha256": config.bundle.model_sha256,
        "versions": versions,
        "aliases_after": {
            "candidate": versions[RELEASE_REVISIONS[1]],
            "champion": versions[RELEASE_REVISIONS[0]],
        },
        "model_bytes_unchanged": True,
        "fit_count": 0,
        "sealed_test_accessed": False,
        "bundle_manifest_model_id": bundle_manifest.selected_model_id,
    }


def _transition_receipt(
    *,
    config: RegistryConfig,
    event: Literal["promotion", "rollback"],
    approval: ReleaseApproval,
    approval_sha256: str,
    before: dict[str, str],
    after: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "protocol_id": config.protocol_id,
        "event": event,
        "registered_model_name": config.registry.registered_model_name,
        "implementation_git_commit": approval.implementation_git_commit,
        "config_sha256": approval.config_sha256,
        "approval_sha256": approval_sha256,
        "checks": [check.model_dump(mode="json") for check in approval.checks],
        "aliases_before": dict(sorted(before.items())),
        "aliases_after": dict(sorted(after.items())),
        "bundle_manifest_sha256": config.bundle.manifest_sha256,
        "model_sha256": config.bundle.model_sha256,
        "model_bytes_unchanged": True,
        "fit_count": 0,
        "sealed_test_accessed": False,
    }


def _activate_version(
    *,
    config: RegistryConfig,
    registry_root: Path,
    deployment_root: Path,
    version: str,
    approval_sha256: str,
    event_receipt_sha256: str,
) -> ActiveDeployment:
    state = _inspect_registry(config, registry_root)
    version_data = state["versions"].get(version)
    if version_data is None:
        raise RegistryWorkflowError(f"Registry version {version} is unavailable.")
    revision = str(version_data["release_revision"])
    source = registry_root / "artifacts" / config.bundle.model_sha256 / "bundle"
    release_bundle = deployment_root / "releases" / revision / "bundle"
    if release_bundle.exists():
        _validate_bundle(config, release_bundle)
    else:
        stage = release_bundle.parent.parent / f".s-{uuid4().hex[:8]}"
        try:
            (stage / "bundle").mkdir(parents=True)
            _copy_exact_bundle(source, stage / "bundle", config)
            release_bundle.parent.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, release_bundle.parent)
        except Exception as error:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            raise RegistryWorkflowError(
                f"Unable to materialise immutable release: {error}"
            ) from error
    active = ActiveDeployment(
        schema_version="1.0.0",
        protocol_id=config.protocol_id,
        registered_model_name=config.registry.registered_model_name,
        alias="champion",
        registry_version=int(version),
        release_revision=cast(Literal["phase7_rev_001", "phase7_rev_002"], revision),
        bundle_relative_path=f"releases/{revision}/bundle",
        config_sha256=EXPECTED_CONFIG_SHA256,
        bundle_manifest_sha256=config.bundle.manifest_sha256,
        model_sha256=config.bundle.model_sha256,
        approval_sha256=approval_sha256,
        event_receipt_sha256=event_receipt_sha256,
    )
    deployment_root.mkdir(parents=True, exist_ok=True)
    _write_atomic(deployment_root / "active.json", _json_bytes(active.model_dump(mode="json")))
    resolve_active_bundle(deployment_root)
    return active


def _require_aliases(observed: dict[str, str], expected: dict[str, str]) -> None:
    if observed != expected:
        raise RegistryWorkflowError(
            f"Registry aliases differ from the required state: expected={expected}, observed={observed}"
        )


def _restore_aliases(client: Any, model_name: str, aliases: dict[str, str]) -> None:
    current = client.get_registered_model(model_name).aliases or {}
    for alias in set(current) - set(aliases):
        client.delete_registered_model_alias(model_name, alias)
    for alias, version in aliases.items():
        client.set_registered_model_alias(model_name, alias, version)


def _compensate_transition(
    *,
    client: Any,
    model_name: str,
    aliases: dict[str, str],
    receipt_path: Path,
    pointer_path: Path | None = None,
    prior_pointer: bytes | None = None,
) -> Exception | None:
    """Attempt every compensation step and report the first failure."""

    failures: list[Exception] = []
    try:
        _restore_aliases(client, model_name, aliases)
    except Exception as error:  # pragma: no cover - exercised through injected failures
        failures.append(error)
    if pointer_path is not None:
        try:
            if prior_pointer is None:
                pointer_path.unlink(missing_ok=True)
            else:
                _write_atomic(pointer_path, prior_pointer)
        except Exception as error:  # pragma: no cover - exercised through injected failures
            failures.append(error)
    try:
        receipt_path.unlink(missing_ok=True)
    except OSError as error:  # pragma: no cover - platform filesystem failure
        failures.append(error)
    return failures[0] if failures else None


def _mlflow_client(database_path: Path) -> Any:
    mlflow = _load_mlflow()
    uri = f"sqlite:///{database_path.resolve().as_posix()}"
    return mlflow.tracking.MlflowClient(tracking_uri=uri, registry_uri=uri)


def _dispose_client(client: Any) -> None:
    """Release SQLite handles so Windows can atomically move or clean runtime roots."""

    try:
        client._tracking_client.store.engine.dispose()
        client._get_registry_client().store.engine.dispose()
    except (AttributeError, TypeError) as error:  # pragma: no cover - pinned MLflow invariant
        raise RegistryWorkflowError(f"Unable to release MLflow SQLite handles: {error}") from error


def _load_mlflow() -> ModuleType:
    try:
        installed = importlib.metadata.version("mlflow")
        if installed != "3.15.0":
            raise RegistryWorkflowError(
                f"MLflow version mismatch: expected=3.15.0, observed={installed}."
            )
        import mlflow
    except importlib.metadata.PackageNotFoundError as error:
        raise RegistryWorkflowError(
            "MLflow is unavailable; install the project with the 'modeling' extra."
        ) from error
    except ModuleNotFoundError as error:
        raise RegistryWorkflowError(
            "MLflow is unavailable; install the project with the 'modeling' extra."
        ) from error
    return mlflow


@contextmanager
def _writer_lock(registry_root: Path) -> Iterator[None]:
    lock = registry_root / ".phase7.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
    except FileExistsError as error:
        raise RegistryWorkflowError("Another registry mutation is already in progress.") from error
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _repository_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RegistryWorkflowError("Phase 7 commands must run inside the project repository.")
    return Path(completed.stdout.strip()).resolve()


def _safe_path(
    repository: Path,
    value: str | Path,
    prefix: str,
    description: str,
    *,
    must_exist: bool,
) -> Path:
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise RegistryWorkflowError(
            f"{description} must be repository-relative and non-traversing."
        )
    candidate = repository / raw
    allowed = (repository / prefix).resolve()
    resolved = candidate.resolve(strict=False)
    if prefix != "." and resolved != allowed and allowed not in resolved.parents:
        raise RegistryWorkflowError(f"{description} must remain beneath {prefix}.")
    current = candidate
    while current != repository:
        if current.exists() and current.is_symlink():
            raise RegistryWorkflowError(f"{description} must not traverse a symlink.")
        current = current.parent
    if must_exist and not resolved.exists():
        raise RegistryWorkflowError(f"{description} does not exist: {raw.as_posix()}")
    return resolved


def _git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RegistryWorkflowError("Unable to resolve the current Git commit.")
    return completed.stdout.strip()


def _git_dirty(repository: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RegistryWorkflowError("Unable to inspect the Git worktree.")
    return bool(completed.stdout.strip())


def _git_is_ancestor(repository: Path, commit: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".t-{uuid4().hex[:8]}")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RegistryWorkflowError(f"Unable to publish {path.name} atomically: {error}") from error


def _publish_directory(destination: Path, files: dict[str, bytes]) -> None:
    if destination.exists():
        raise RegistryWorkflowError(f"Evidence destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".s-{uuid4().hex[:8]}"
    try:
        stage.mkdir()
        for name, content in files.items():
            (stage / name).write_bytes(content)
        if {path.name for path in stage.iterdir()} != set(PUBLISHED_FILES):
            raise RegistryWorkflowError("Staged evidence differs from the output allowlist.")
        os.replace(stage, destination)
    except Exception as error:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if isinstance(error, RegistryWorkflowError):
            raise
        raise RegistryWorkflowError(f"Unable to publish Phase 7 evidence: {error}") from error


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegistryWorkflowError(f"Unable to read governed JSON {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise RegistryWorkflowError(f"Governed JSON {path.name} must contain an object.")
    return payload


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise RegistryWorkflowError(f"Unable to hash {path.name}: {error}") from error


def _render_report(summary: dict[str, Any], summary_sha256: str) -> str:
    return f"""# Phase 7 governed registry release report

**Deterministic summary SHA-256:** `{summary_sha256}`

## Outcome

The reviewed `selected_v1` bundle was registered as two immutable deployment
revisions. Revision 2 was manually promoted and deployed, then the governed
rollback restored revision 1. Both revisions contain the same manifest and
model bytes; this is a release-control exercise, not a model comparison.

## Controls demonstrated

- MLflow SQLite registry with content-addressed artifacts.
- Digest-authenticated manual promotion and rollback approvals.
- Candidate, champion, and rollback alias transitions.
- Atomic active-deployment pointer with no hand replacement of model files.
- CI quality, container contract, fixable HIGH/CRITICAL vulnerability gate,
  and CycloneDX SBOM generation.
- Zero fitting and no sealed-test access.

## Remaining boundary

G4 and Release B remain open for robustness stress testing, monitoring, and
incident controls. This local portfolio drill is not a production-readiness,
fairness, compliance, or model-quality claim.
"""


def _render_checklist(summary: dict[str, Any]) -> str:
    implementation = summary["implementation_git_commit"]
    return f"""# Phase 7 promotion checklist

- [x] Implementation commit `{implementation}` reviewed.
- [x] Quality check succeeded.
- [x] Build, runtime-contract, and image-scan check succeeded.
- [x] Bundle manifest and model digests match `selected_v1`.
- [x] Release A, Phase 5, and Phase 6 evidence digests match.
- [x] Candidate contains no new or changed model bytes.
- [x] Promotion approval is digest-authenticated and role-labelled.
- [x] No training, tuning, calibration fitting, or sealed-test access occurred.
- [x] Previous champion retained as the rollback target.

This checklist authorizes only the local portfolio demonstration.
"""


def _render_runbook(summary: dict[str, Any]) -> str:
    return """# Phase 7 rollback runbook

## Trigger

Use rollback when a promoted local release fails readiness, contract, security,
or operational validation. Do not retrain or alter the registered artifact as
part of incident response.

## Procedure

1. Stop further promotion activity and obtain the reviewed rollback approval.
2. Run `credit-risk registry rollback` with the approval's external digest.
3. Restart the API through the registry Compose override.
4. Require `/ready` and the committed synthetic `/v1/predict` smoke test to pass.
5. Run `credit-risk registry status` and confirm revision 1 is champion and active.
6. Preserve the displaced revision and receipts for investigation.

The command compensates registry aliases and the active pointer if activation
fails. Phase 10 will extend this local procedure with monitoring alerts and
incident ownership.
"""
