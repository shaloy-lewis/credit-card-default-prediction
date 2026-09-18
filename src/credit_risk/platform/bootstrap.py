"""Idempotent Phase 8 object-store, registry, and deployment bootstrap."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from credit_risk.modeling.selected_bundle import BundleManifest
from credit_risk.platform.contracts import (
    DEFAULT_PLATFORM_CONFIG_PATH,
    PlatformConfig,
    PlatformContractError,
    config_sha256,
    load_platform_config,
)
from credit_risk.registry.deployment import (
    ActiveDeployment,
    DeploymentResolutionError,
    load_active_deployment,
)


class PlatformBootstrapError(RuntimeError):
    """Raised when persistent platform state is missing, foreign, or incomplete."""


@dataclass(frozen=True, slots=True)
class PlatformBootstrapResult:
    """Portable result of bootstrap or verification."""

    status: str
    registered_model_name: str
    aliases: dict[str, str]
    object_sha256: dict[str, str]
    active_revision: str
    fit_count: int = 0
    sealed_test_accessed: bool = False


def bootstrap_platform(
    *,
    config_path: str | Path = DEFAULT_PLATFORM_CONFIG_PATH,
    bundle_root: str | Path = "models/selected_v1",
    deployment_root: str | Path = "/deployment",
    environment: Mapping[str, str] | None = None,
    s3_client: Any | None = None,
    mlflow_client: Any | None = None,
) -> PlatformBootstrapResult:
    """Create the exact approved Phase 8 state or verify an identical prior state."""

    context = _load_context(config_path, bundle_root, deployment_root)
    runtime_environment = os.environ if environment is None else environment
    settings = _runtime_settings(context.config, runtime_environment)
    with _postgres_writer_lock(context.config, settings):
        s3 = s3_client or _new_s3_client(settings)
        client = mlflow_client or _new_mlflow_client(settings["MLFLOW_TRACKING_URI"])
        _ensure_bucket(s3, settings["MLFLOW_ARTIFACT_BUCKET"])
        object_hashes = _ensure_objects(
            s3=s3,
            bucket=settings["MLFLOW_ARTIFACT_BUCKET"],
            prefix=context.config.registry_bootstrap.artifact_prefix,
            bundle=context.bundle,
            config=context.config,
        )
        aliases = _ensure_registry(
            client=client,
            config=context.config,
            artifact_uri=(
                f"s3://{settings['MLFLOW_ARTIFACT_BUCKET']}/"
                f"{context.config.registry_bootstrap.artifact_prefix}"
            ),
            config_digest=config_sha256(context.config_path),
        )
        active = _ensure_deployment(context)
    return PlatformBootstrapResult(
        status="ready",
        registered_model_name=context.config.registry_bootstrap.registered_model_name,
        aliases=aliases,
        object_sha256=object_hashes,
        active_revision=active.release_revision,
    )


def verify_platform(
    *,
    config_path: str | Path = DEFAULT_PLATFORM_CONFIG_PATH,
    bundle_root: str | Path = "models/selected_v1",
    deployment_root: str | Path = "/deployment",
    environment: Mapping[str, str] | None = None,
    s3_client: Any | None = None,
    mlflow_client: Any | None = None,
) -> PlatformBootstrapResult:
    """Verify the complete persistent state without creating or changing it."""

    context = _load_context(config_path, bundle_root, deployment_root)
    runtime_environment = os.environ if environment is None else environment
    settings = _runtime_settings(context.config, runtime_environment)
    s3 = s3_client or _new_s3_client(settings)
    client = mlflow_client or _new_mlflow_client(settings["MLFLOW_TRACKING_URI"])
    object_hashes = _verify_objects(
        s3=s3,
        bucket=settings["MLFLOW_ARTIFACT_BUCKET"],
        prefix=context.config.registry_bootstrap.artifact_prefix,
        bundle=context.bundle,
        config=context.config,
    )
    aliases = _verify_registry(
        client=client,
        config=context.config,
        artifact_uri=(
            f"s3://{settings['MLFLOW_ARTIFACT_BUCKET']}/"
            f"{context.config.registry_bootstrap.artifact_prefix}"
        ),
        config_digest=config_sha256(context.config_path),
    )
    try:
        active = load_active_deployment(context.deployment_root)
    except DeploymentResolutionError as error:
        raise PlatformBootstrapError(f"Unable to verify the active deployment: {error}") from error
    _validate_active(context.config, active)
    return PlatformBootstrapResult(
        status="verified",
        registered_model_name=context.config.registry_bootstrap.registered_model_name,
        aliases=aliases,
        object_sha256=object_hashes,
        active_revision=active.release_revision,
    )


@dataclass(frozen=True, slots=True)
class _Context:
    config: PlatformConfig
    config_path: Path
    project_root: Path
    bundle: Path
    deployment_root: Path


def _load_context(
    config_path: str | Path, bundle_root: str | Path, deployment_root: str | Path
) -> _Context:
    try:
        config = load_platform_config(config_path)
        config_file = Path(config_path).resolve(strict=True)
    except (PlatformContractError, OSError, RuntimeError, ValueError) as error:
        raise PlatformBootstrapError(str(error)) from error
    if len(config_file.parents) < 3:
        raise PlatformBootstrapError("Phase 8 configuration is outside the project layout.")
    project = config_file.parents[2]
    _validate_sources(project, config)
    bundle = _resolve_project_path(project, bundle_root, "models", "bundle root")
    _validate_bundle(bundle, config)
    deployment = _validate_deployment_root(deployment_root)
    return _Context(config, config_file, project, bundle, deployment)


def _validate_deployment_root(value: str | Path) -> Path:
    raw = Path(value)
    if ".." in raw.parts:
        raise PlatformBootstrapError("Deployment root must be non-traversing.")
    deployment = raw.absolute()
    try:
        for component in (deployment, *deployment.parents):
            if component.is_symlink():
                raise PlatformBootstrapError("Deployment root must not traverse a symlink.")
        if deployment.exists() and not deployment.is_dir():
            raise PlatformBootstrapError("Deployment root must be a directory.")
    except OSError as error:
        raise PlatformBootstrapError(f"Unable to inspect deployment root: {error}") from error
    return deployment


def _validate_sources(project: Path, config: PlatformConfig) -> None:
    for name, reference in config.source_evidence.items():
        path = _resolve_project_path(project, reference.path, ".", name)
        if not path.is_file() or _sha256_file(path) != reference.sha256:
            raise PlatformBootstrapError(f"Reviewed source digest mismatch for {name}.")


def _resolve_project_path(project: Path, value: str | Path, prefix: str, label: str) -> Path:
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise PlatformBootstrapError(f"{label} must be repository-relative and non-traversing.")
    candidate = project / raw
    try:
        resolved = candidate.resolve(strict=True)
        allowed = project.resolve() if prefix == "." else (project / prefix).resolve()
    except (OSError, RuntimeError) as error:
        raise PlatformBootstrapError(f"Unable to resolve {label}: {error}") from error
    if resolved != allowed and allowed not in resolved.parents:
        raise PlatformBootstrapError(f"{label} must remain beneath {prefix}.")
    current = candidate
    while current != project:
        if current.is_symlink():
            raise PlatformBootstrapError(f"{label} must not traverse a symlink.")
        current = current.parent
    return resolved


def _validate_bundle(bundle: Path, config: PlatformConfig) -> BundleManifest:
    if bundle.is_symlink() or not bundle.is_dir():
        raise PlatformBootstrapError("Selected bundle is missing or unsafe.")
    try:
        entries = tuple(bundle.iterdir())
    except OSError as error:
        raise PlatformBootstrapError(f"Unable to inspect selected bundle: {error}") from error
    if {entry.name for entry in entries} != {"manifest.json", "model.cbm"} or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise PlatformBootstrapError("Selected bundle violates the two-file allowlist.")
    if _sha256_file(bundle / "manifest.json") != config.bundle.manifest_sha256:
        raise PlatformBootstrapError("Selected bundle manifest digest mismatch.")
    if _sha256_file(bundle / "model.cbm") != config.bundle.model_sha256:
        raise PlatformBootstrapError("Selected model digest mismatch.")
    try:
        manifest = BundleManifest.model_validate_json((bundle / "manifest.json").read_bytes())
    except Exception as error:
        raise PlatformBootstrapError(f"Selected bundle manifest is invalid: {error}") from error
    if (
        manifest.bundle_id != config.bundle.bundle_id
        or manifest.selected_model_id != config.bundle.model_id
        or manifest.model_sha256 != config.bundle.model_sha256
    ):
        raise PlatformBootstrapError("Selected bundle semantics differ from Phase 8.")
    return manifest


def _runtime_settings(config: PlatformConfig, environment: Mapping[str, str]) -> dict[str, str]:
    names = {
        *config.environment.required_secret_variables,
        *config.environment.required_identity_variables,
        "MLFLOW_TRACKING_URI",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    }
    values = {name: environment.get(name, "").strip() for name in names}
    missing = sorted(name for name, value in values.items() if not value)
    if missing:
        raise PlatformBootstrapError(
            f"Required Phase 8 environment variables are missing: {', '.join(missing)}."
        )
    if values["MLFLOW_ARTIFACT_BUCKET"] != config.registry_bootstrap.artifact_bucket:
        raise PlatformBootstrapError("Artifact bucket differs from the frozen Phase 8 contract.")
    if values["AWS_ACCESS_KEY_ID"] != values["MINIO_ROOT_USER"]:
        raise PlatformBootstrapError("S3 access identity must match the local MinIO identity.")
    if values["AWS_SECRET_ACCESS_KEY"] != values["MINIO_ROOT_PASSWORD"]:
        raise PlatformBootstrapError("S3 access secret must match the local MinIO secret.")
    return values


@contextmanager
def _postgres_writer_lock(
    config: PlatformConfig,
    settings: Mapping[str, str],
    *,
    connection_factory: Any | None = None,
) -> Iterator[None]:
    """Hold the frozen PostgreSQL advisory lock for all bootstrap mutations."""

    if connection_factory is None:
        try:
            import psycopg2  # type: ignore[import-untyped]
        except ModuleNotFoundError as error:
            raise PlatformBootstrapError(
                "Platform dependencies are unavailable; install the 'platform' extra."
            ) from error
        connection_factory = psycopg2.connect
    connection: Any | None = None
    cursor: Any | None = None
    try:
        connection = connection_factory(
            host="postgres",
            port=config.services.postgres.internal_port,
            dbname=settings["POSTGRES_DB"],
            user=settings["POSTGRES_USER"],
            password=settings["POSTGRES_PASSWORD"],
            connect_timeout=5,
        )
        connection.autocommit = True
        cursor = connection.cursor()
        cursor.execute(
            "SELECT pg_try_advisory_lock(%s)",
            (config.registry_bootstrap.writer_lock.key,),
        )
        row = cursor.fetchone()
        if row is None or len(row) != 1 or row[0] is not True:
            raise PlatformBootstrapError("Another Phase 8 bootstrap is already in progress.")
    except PlatformBootstrapError:
        _close_postgres_session(cursor, connection)
        raise
    except Exception as error:
        _close_postgres_session(cursor, connection)
        raise PlatformBootstrapError(
            f"Unable to acquire the Phase 8 bootstrap writer lock: {error}"
        ) from error
    try:
        yield
    finally:
        _close_postgres_session(cursor, connection)


def _close_postgres_session(cursor: Any | None, connection: Any | None) -> None:
    if cursor is not None:
        try:
            cursor.close()
        except Exception:
            pass
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass


def _new_s3_client(settings: Mapping[str, str]) -> Any:
    try:
        import boto3
    except ModuleNotFoundError as error:
        raise PlatformBootstrapError(
            "Platform dependencies are unavailable; install the 'platform' extra."
        ) from error
    return boto3.client(
        "s3",
        endpoint_url=settings["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=settings["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=settings["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _new_mlflow_client(tracking_uri: str) -> Any:
    try:
        import mlflow
    except ModuleNotFoundError as error:
        raise PlatformBootstrapError(
            "Platform dependencies are unavailable; install the 'platform' extra."
        ) from error
    try:
        return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    except Exception as error:
        raise PlatformBootstrapError(f"Unable to initialise the MLflow client: {error}") from error


def _error_code(error: Exception) -> str:
    response = getattr(error, "response", {})
    if isinstance(response, dict):
        nested = response.get("Error", {})
        if isinstance(nested, dict):
            code = str(nested.get("Code", ""))
            if code:
                return code
    return str(getattr(error, "error_code", ""))


def _ensure_bucket(s3: Any, bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception as error:
        if _error_code(error) not in {"404", "NoSuchBucket", "NotFound"}:
            raise PlatformBootstrapError(
                f"Unable to inspect the artifact bucket: {error}"
            ) from error
        try:
            s3.create_bucket(Bucket=bucket)
        except Exception as create_error:
            raise PlatformBootstrapError(
                f"Unable to create the artifact bucket: {create_error}"
            ) from create_error


def _object_contract(bundle: Path, config: PlatformConfig) -> dict[str, tuple[Path, str]]:
    return {
        "manifest.json": (bundle / "manifest.json", config.bundle.manifest_sha256),
        "model.cbm": (bundle / "model.cbm", config.bundle.model_sha256),
    }


def _ensure_objects(
    *, s3: Any, bucket: str, prefix: str, bundle: Path, config: PlatformConfig
) -> dict[str, str]:
    for filename, (path, digest) in _object_contract(bundle, config).items():
        key = f"{prefix}/{filename}"
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except Exception as error:
            if _error_code(error) not in {"404", "NoSuchKey", "NotFound"}:
                raise PlatformBootstrapError(
                    f"Unable to inspect object {filename}: {error}"
                ) from error
            try:
                s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=path.read_bytes(),
                    Metadata={"sha256": digest},
                    ContentType=(
                        "application/json"
                        if filename.endswith(".json")
                        else "application/octet-stream"
                    ),
                )
            except Exception as put_error:
                raise PlatformBootstrapError(
                    f"Unable to publish immutable object {filename}: {put_error}"
                ) from put_error
    return _verify_objects(s3=s3, bucket=bucket, prefix=prefix, bundle=bundle, config=config)


def _verify_objects(
    *, s3: Any, bucket: str, prefix: str, bundle: Path, config: PlatformConfig
) -> dict[str, str]:
    expected = _object_contract(bundle, config)
    try:
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/")
    except Exception as error:
        raise PlatformBootstrapError(f"Unable to list governed objects: {error}") from error
    keys = {str(item["Key"]) for item in listing.get("Contents", [])}
    expected_keys = {f"{prefix}/{name}" for name in expected}
    if keys != expected_keys:
        raise PlatformBootstrapError("Content-addressed object prefix violates its allowlist.")
    observed: dict[str, str] = {}
    for filename, (_, digest) in expected.items():
        try:
            response = s3.get_object(Bucket=bucket, Key=f"{prefix}/{filename}")
            content = response["Body"].read()
            metadata = response.get("Metadata", {})
        except Exception as error:
            raise PlatformBootstrapError(f"Unable to read object {filename}: {error}") from error
        actual = _sha256_bytes(content)
        if actual != digest or metadata.get("sha256") != digest:
            raise PlatformBootstrapError(f"Artifact object digest mismatch for {filename}.")
        observed[filename] = actual
    return observed


def _version_tags(config: PlatformConfig, revision: str, digest: str) -> dict[str, str]:
    return {
        "protocol_id": config.protocol_id,
        "release_revision": revision,
        "bundle_id": config.bundle.bundle_id,
        "model_id": config.bundle.model_id,
        "bundle_manifest_sha256": config.bundle.manifest_sha256,
        "model_sha256": config.bundle.model_sha256,
        "platform_config_sha256": digest,
        "phase7_evidence_manifest_sha256": config.source_evidence[
            "phase7_evidence_manifest"
        ].sha256,
        "bootstrap_not_migration": "true",
        "model_bytes_unchanged": "true",
    }


def _registered_model_tags(config: PlatformConfig) -> dict[str, str]:
    return {
        "protocol_id": config.protocol_id,
        "bundle_id": config.bundle.bundle_id,
        "model_id": config.bundle.model_id,
    }


def _missing_model(error: Exception) -> bool:
    return _error_code(error) in {"RESOURCE_DOES_NOT_EXIST", "404", "NOT_FOUND"}


def _ensure_registry(
    *, client: Any, config: PlatformConfig, artifact_uri: str, config_digest: str
) -> dict[str, str]:
    model_name = config.registry_bootstrap.registered_model_name
    try:
        client.get_registered_model(model_name)
    except Exception as error:
        if not _missing_model(error):
            raise PlatformBootstrapError(f"Unable to inspect the MLflow model: {error}") from error
        _mlflow_call(
            "create the reviewed registered model",
            client.create_registered_model,
            model_name,
            tags=_registered_model_tags(config),
            description="Phase 8 persistent re-registration of unchanged selected_v1 bytes.",
        )
    versions = list(
        _mlflow_call(
            "search the registered model versions",
            client.search_model_versions,
            f"name='{model_name}'",
        )
    )
    if not versions:
        for contract in config.registry_bootstrap.versions:
            created = _mlflow_call(
                f"create registry version {contract.registry_version}",
                client.create_model_version,
                model_name,
                source=artifact_uri,
                tags=_version_tags(config, contract.release_revision, config_digest),
                description=(
                    f"Persistent deployment revision {contract.release_revision}; "
                    "model bytes unchanged from selected_v1."
                ),
            )
            if str(created.version) != contract.registry_version:
                raise PlatformBootstrapError("MLflow allocated an unexpected registry version.")
        _mlflow_call(
            "assign the champion alias",
            client.set_registered_model_alias,
            model_name,
            "champion",
            "1",
        )
        _mlflow_call(
            "assign the rollback alias",
            client.set_registered_model_alias,
            model_name,
            "rollback",
            "2",
        )
    return _verify_registry(
        client=client,
        config=config,
        artifact_uri=artifact_uri,
        config_digest=config_digest,
    )


def _mlflow_call(action: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except Exception as error:
        raise PlatformBootstrapError(f"Unable to {action}: {error}") from error


def _verify_registry(
    *, client: Any, config: PlatformConfig, artifact_uri: str, config_digest: str
) -> dict[str, str]:
    model_name = config.registry_bootstrap.registered_model_name
    try:
        registered = client.get_registered_model(model_name)
        versions = list(client.search_model_versions(f"name='{model_name}'"))
    except Exception as error:
        raise PlatformBootstrapError(f"Unable to verify the MLflow registry: {error}") from error
    indexed = {str(item.version): item for item in versions}
    registered_tags = {
        str(key): str(value) for key, value in (getattr(registered, "tags", {}) or {}).items()
    }
    if registered_tags != _registered_model_tags(config):
        raise PlatformBootstrapError("Registered-model tags differ from Phase 8.")
    if set(indexed) != {"1", "2"}:
        raise PlatformBootstrapError("Persistent registry must contain exactly two versions.")
    for contract in config.registry_bootstrap.versions:
        version = indexed[contract.registry_version]
        tags = {str(key): str(value) for key, value in version.tags.items()}
        if tags != _version_tags(config, contract.release_revision, config_digest):
            raise PlatformBootstrapError(
                f"Registry version {contract.registry_version} tags differ from Phase 8."
            )
        if str(version.source).rstrip("/") != artifact_uri.rstrip("/"):
            raise PlatformBootstrapError(
                f"Registry version {contract.registry_version} has a foreign artifact source."
            )
    aliases = {str(key): str(value) for key, value in (registered.aliases or {}).items()}
    if aliases != config.registry_bootstrap.aliases:
        raise PlatformBootstrapError(
            f"Persistent registry aliases differ: expected={config.registry_bootstrap.aliases}, "
            f"observed={aliases}."
        )
    return aliases


def _ensure_deployment(context: _Context) -> ActiveDeployment:
    root = _validate_deployment_root(context.deployment_root)
    revision = context.config.registry_bootstrap.active_revision
    release = root / "releases" / revision
    try:
        if root.exists() and any(root.iterdir()):
            active = load_active_deployment(root)
            _validate_active(context.config, active)
            return active
        root.mkdir(parents=True, exist_ok=True)
        root = _validate_deployment_root(root)
        bundle = release / "bundle"
        _validate_release_path(root, bundle)
        bundle.mkdir(parents=True)
        root = _validate_deployment_root(root)
        _validate_release_path(root, bundle)
        shutil.copyfile(context.bundle / "manifest.json", bundle / "manifest.json")
        shutil.copyfile(context.bundle / "model.cbm", bundle / "model.cbm")
        active = ActiveDeployment(
            schema_version="1.0.0",
            protocol_id="phase7_v1",
            registered_model_name=context.config.registry_bootstrap.registered_model_name,
            alias="champion",
            registry_version=context.config.registry_bootstrap.active_registry_version,
            release_revision=revision,
            bundle_relative_path=f"releases/{revision}/bundle",
            config_sha256=context.config.source_evidence["phase7_config"].sha256,
            bundle_manifest_sha256=context.config.bundle.manifest_sha256,
            model_sha256=context.config.bundle.model_sha256,
            approval_sha256=context.config.registry_bootstrap.rollback_approval_sha256,
            event_receipt_sha256=context.config.registry_bootstrap.rollback_receipt_sha256,
        )
        _write_atomic(root / "active.json", _json_bytes(active.model_dump(mode="json")))
        load_active_deployment(root)
    except Exception as error:
        _remove_incomplete_release(root, release)
        if isinstance(error, PlatformBootstrapError):
            raise
        raise PlatformBootstrapError(
            f"Unable to materialise Phase 8 deployment: {error}"
        ) from error
    return active


def _remove_incomplete_release(root: Path, release: Path) -> None:
    try:
        safe_root = _validate_deployment_root(root)
        _validate_release_path(safe_root, release)
        if (
            release.parent.parent != safe_root
            or not release.exists()
            or (safe_root / "active.json").exists()
        ):
            return
        shutil.rmtree(release)
    except (OSError, PlatformBootstrapError):
        return


def _validate_release_path(root: Path, path: Path) -> None:
    if root != path and root not in path.parents:
        raise PlatformBootstrapError("Deployment release path escapes its root.")
    current = path
    while current != root:
        if current.is_symlink():
            raise PlatformBootstrapError("Deployment release path must not traverse a symlink.")
        current = current.parent


def _validate_active(config: PlatformConfig, active: ActiveDeployment) -> None:
    if (
        active.release_revision != config.registry_bootstrap.active_revision
        or active.registry_version != config.registry_bootstrap.active_registry_version
        or active.approval_sha256 != config.registry_bootstrap.rollback_approval_sha256
        or active.event_receipt_sha256 != config.registry_bootstrap.rollback_receipt_sha256
    ):
        raise PlatformBootstrapError("Active deployment differs from the approved Phase 7 state.")


def _write_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".t-{uuid4().hex[:8]}")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise PlatformBootstrapError(
            f"Unable to publish {path.name} atomically: {error}"
        ) from error


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise PlatformBootstrapError(f"Unable to hash {path.name}: {error}") from error
