from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from credit_risk.platform import bootstrap as workflow
from credit_risk.platform.contracts import EXPECTED_CONFIG_SHA256, load_platform_config

REAL_POSTGRES_WRITER_LOCK = workflow._postgres_writer_lock


@contextmanager
def _unlocked(*_args: Any, **_kwargs: Any) -> Iterator[None]:
    yield


@pytest.fixture(autouse=True)
def _avoid_real_postgres(monkeypatch) -> None:
    monkeypatch.setattr(workflow, "_postgres_writer_lock", _unlocked)


class MissingObject(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self) -> None:
        self.bucket = False
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}

    def head_bucket(self, *, Bucket: str) -> None:
        del Bucket
        if not self.bucket:
            raise MissingObject("404")

    def create_bucket(self, *, Bucket: str) -> None:
        del Bucket
        self.bucket = True

    def head_object(self, *, Bucket: str, Key: str) -> None:
        del Bucket
        if Key not in self.objects:
            raise MissingObject("NoSuchKey")

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        Metadata: dict[str, str],
        ContentType: str,
    ) -> None:
        del Bucket, ContentType
        self.objects[Key] = (Body, Metadata)

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> dict[str, Any]:
        del Bucket
        return {
            "Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)]
        }

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        del Bucket
        content, metadata = self.objects[Key]
        return {"Body": BytesIO(content), "Metadata": metadata}


class MissingModel(Exception):
    error_code = "RESOURCE_DOES_NOT_EXIST"


class FakeMlflow:
    def __init__(self) -> None:
        self.created = False
        self.versions: list[SimpleNamespace] = []
        self.aliases: dict[str, str] = {}
        self.registered_tags = workflow._registered_model_tags(load_platform_config())

    def get_registered_model(self, name: str) -> SimpleNamespace:
        del name
        if not self.created:
            raise MissingModel()
        return SimpleNamespace(aliases=dict(self.aliases), tags=dict(self.registered_tags))

    def create_registered_model(self, name: str, *, tags: dict[str, str], description: str) -> None:
        del name, description
        self.created = True
        self.registered_tags = dict(tags)

    def search_model_versions(self, filter_string: str) -> list[SimpleNamespace]:
        del filter_string
        return list(self.versions)

    def create_model_version(
        self, name: str, *, source: str, tags: dict[str, str], description: str
    ) -> SimpleNamespace:
        del name, description
        version = SimpleNamespace(
            version=str(len(self.versions) + 1), source=source, tags=dict(tags)
        )
        self.versions.append(version)
        return version

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        del name
        self.aliases[alias] = str(version)


def _environment() -> dict[str, str]:
    return {
        "POSTGRES_DB": "mlflow",
        "POSTGRES_USER": "mlflow",
        "POSTGRES_PASSWORD": "local-postgres",
        "MINIO_ROOT_USER": "mlflow",
        "MINIO_ROOT_PASSWORD": "local-minio",
        "MLFLOW_ARTIFACT_BUCKET": "credit-risk-mlflow",
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
        "AWS_ACCESS_KEY_ID": "mlflow",
        "AWS_SECRET_ACCESS_KEY": "local-minio",
    }


def test_bootstrap_creates_then_verifies_exact_state(tmp_path: Path) -> None:
    s3 = FakeS3()
    mlflow = FakeMlflow()
    deployment = tmp_path / "deployment"

    created = workflow.bootstrap_platform(
        deployment_root=deployment,
        environment=_environment(),
        s3_client=s3,
        mlflow_client=mlflow,
    )
    verified = workflow.bootstrap_platform(
        deployment_root=deployment,
        environment=_environment(),
        s3_client=s3,
        mlflow_client=mlflow,
    )
    status = workflow.verify_platform(
        deployment_root=deployment,
        environment=_environment(),
        s3_client=s3,
        mlflow_client=mlflow,
    )

    assert created.status == "ready"
    assert verified.status == "ready"
    assert status.status == "verified"
    assert status.aliases == {"champion": "1", "rollback": "2"}
    assert status.active_revision == "phase7_rev_001"
    assert status.fit_count == 0
    assert status.sealed_test_accessed is False
    assert set(s3.objects) == {
        f"registry/{load_platform_config().bundle.model_sha256}/bundle/manifest.json",
        f"registry/{load_platform_config().bundle.model_sha256}/bundle/model.cbm",
    }


def test_bootstrap_rejects_foreign_object_and_alias_state(tmp_path: Path) -> None:
    s3 = FakeS3()
    mlflow = FakeMlflow()
    deployment = tmp_path / "deployment"
    workflow.bootstrap_platform(
        deployment_root=deployment,
        environment=_environment(),
        s3_client=s3,
        mlflow_client=mlflow,
    )
    prefix = load_platform_config().registry_bootstrap.artifact_prefix
    s3.objects[f"{prefix}/foreign.bin"] = (b"foreign", {"sha256": "0" * 64})

    with pytest.raises(workflow.PlatformBootstrapError, match="object prefix"):
        workflow.verify_platform(
            deployment_root=deployment,
            environment=_environment(),
            s3_client=s3,
            mlflow_client=mlflow,
        )

    s3.objects.pop(f"{prefix}/foreign.bin")
    mlflow.aliases = {"champion": "2", "rollback": "1"}
    with pytest.raises(workflow.PlatformBootstrapError, match="aliases differ"):
        workflow.verify_platform(
            deployment_root=deployment,
            environment=_environment(),
            s3_client=s3,
            mlflow_client=mlflow,
        )


def test_bootstrap_rejects_missing_and_inconsistent_environment() -> None:
    config = load_platform_config()
    with pytest.raises(workflow.PlatformBootstrapError, match="variables are missing"):
        workflow._runtime_settings(config, {})

    environment = _environment()
    environment["AWS_ACCESS_KEY_ID"] = "foreign"
    with pytest.raises(workflow.PlatformBootstrapError, match="access identity"):
        workflow._runtime_settings(config, environment)


def test_explicit_empty_environment_never_reads_ambient_values(tmp_path: Path, monkeypatch) -> None:
    for name, value in _environment().items():
        monkeypatch.setenv(name, value)
    lock_entered = False

    @contextmanager
    def track_lock(*_args: Any, **_kwargs: Any) -> Iterator[None]:
        nonlocal lock_entered
        lock_entered = True
        yield

    monkeypatch.setattr(workflow, "_postgres_writer_lock", track_lock)
    with pytest.raises(workflow.PlatformBootstrapError, match="variables are missing"):
        workflow.bootstrap_platform(
            deployment_root=tmp_path / "deployment",
            environment={},
            s3_client=FakeS3(),
            mlflow_client=FakeMlflow(),
        )

    assert lock_entered is False
    assert not (tmp_path / "deployment").exists()


def test_registry_tags_bind_platform_lineage() -> None:
    config = load_platform_config()
    tags = workflow._version_tags(config, "phase7_rev_001", EXPECTED_CONFIG_SHA256)
    assert tags["model_sha256"] == config.bundle.model_sha256
    assert (
        tags["phase7_evidence_manifest_sha256"]
        == config.source_evidence["phase7_evidence_manifest"].sha256
    )
    assert tags["bootstrap_not_migration"] == "true"
    assert workflow._registered_model_tags(config) == {
        "protocol_id": "phase8_v1",
        "bundle_id": "selected_v1",
        "model_id": "catboost_fixed",
    }


class _LockCursor:
    def __init__(self, acquired: bool = True) -> None:
        self.acquired = acquired
        self.executions: list[tuple[str, tuple[int, ...]]] = []
        self.closed = False

    def execute(self, query: str, parameters: tuple[int, ...]) -> None:
        self.executions.append((query, parameters))

    def fetchone(self) -> tuple[bool]:
        return (self.acquired,)

    def close(self) -> None:
        self.closed = True


class _LockConnection:
    def __init__(self, acquired: bool = True) -> None:
        self.autocommit = False
        self.cursor_value = _LockCursor(acquired)
        self.closed = False
        self.arguments: dict[str, Any] = {}

    def cursor(self) -> _LockCursor:
        return self.cursor_value

    def close(self) -> None:
        self.closed = True


def test_postgres_writer_lock_acquires_frozen_key_and_releases() -> None:
    connection = _LockConnection()

    def connect(**kwargs: Any) -> _LockConnection:
        connection.arguments = kwargs
        return connection

    with REAL_POSTGRES_WRITER_LOCK(
        load_platform_config(), _environment(), connection_factory=connect
    ):
        assert connection.closed is False

    assert connection.autocommit is True
    assert connection.cursor_value.executions == [
        (
            "SELECT pg_try_advisory_lock(%s)",
            (-4653285090134190835,),
        )
    ]
    assert connection.arguments["host"] == "postgres"
    assert connection.cursor_value.closed is True
    assert connection.closed is True


def test_postgres_writer_lock_fails_fast_and_releases_after_errors() -> None:
    denied = _LockConnection(acquired=False)
    with pytest.raises(workflow.PlatformBootstrapError, match="already in progress"):
        with REAL_POSTGRES_WRITER_LOCK(
            load_platform_config(),
            _environment(),
            connection_factory=lambda **_: denied,
        ):
            pytest.fail("contended lock must not enter the mutation boundary")
    assert denied.closed is True

    acquired = _LockConnection()
    with pytest.raises(RuntimeError, match="mutation failed"):
        with REAL_POSTGRES_WRITER_LOCK(
            load_platform_config(),
            _environment(),
            connection_factory=lambda **_: acquired,
        ):
            raise RuntimeError("mutation failed")
    assert acquired.closed is True


def test_contended_bootstrap_performs_no_persistent_writes(tmp_path: Path, monkeypatch) -> None:
    @contextmanager
    def denied(*_args: Any, **_kwargs: Any) -> Iterator[None]:
        raise workflow.PlatformBootstrapError("Another Phase 8 bootstrap is already in progress.")
        yield

    monkeypatch.setattr(workflow, "_postgres_writer_lock", denied)
    s3 = FakeS3()
    mlflow = FakeMlflow()
    deployment = tmp_path / "deployment"

    with pytest.raises(workflow.PlatformBootstrapError, match="already in progress"):
        workflow.bootstrap_platform(
            deployment_root=deployment,
            environment=_environment(),
            s3_client=s3,
            mlflow_client=mlflow,
        )

    assert s3.bucket is False
    assert s3.objects == {}
    assert mlflow.created is False
    assert mlflow.versions == []
    assert not deployment.exists()


def test_runtime_settings_reject_foreign_bucket_and_secret() -> None:
    config = load_platform_config()
    environment = _environment()
    environment["MLFLOW_ARTIFACT_BUCKET"] = "foreign"
    with pytest.raises(workflow.PlatformBootstrapError, match="Artifact bucket"):
        workflow._runtime_settings(config, environment)

    environment = _environment()
    environment["AWS_SECRET_ACCESS_KEY"] = "foreign"
    with pytest.raises(workflow.PlatformBootstrapError, match="access secret"):
        workflow._runtime_settings(config, environment)


class FailingS3(FakeS3):
    def __init__(self, operation: str, code: str = "AccessDenied") -> None:
        super().__init__()
        self.operation = operation
        self.code = code

    def head_bucket(self, *, Bucket: str) -> None:
        if self.operation == "head_bucket":
            raise MissingObject(self.code)
        super().head_bucket(Bucket=Bucket)

    def create_bucket(self, *, Bucket: str) -> None:
        if self.operation == "create_bucket":
            raise RuntimeError("create failed")
        super().create_bucket(Bucket=Bucket)

    def head_object(self, *, Bucket: str, Key: str) -> None:
        if self.operation == "head_object":
            raise MissingObject(self.code)
        super().head_object(Bucket=Bucket, Key=Key)

    def put_object(self, **kwargs: Any) -> None:
        if self.operation == "put_object":
            raise RuntimeError("put failed")
        super().put_object(**kwargs)

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> dict[str, Any]:
        if self.operation == "list_objects":
            raise RuntimeError("list failed")
        return super().list_objects_v2(Bucket=Bucket, Prefix=Prefix)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if self.operation == "get_object":
            raise RuntimeError("get failed")
        return super().get_object(Bucket=Bucket, Key=Key)


@pytest.mark.parametrize(
    ("operation", "code", "message"),
    [
        ("head_bucket", "AccessDenied", "inspect the artifact bucket"),
        ("create_bucket", "404", "create the artifact bucket"),
        ("head_object", "AccessDenied", "inspect object"),
        ("put_object", "NoSuchKey", "publish immutable object"),
        ("list_objects", "AccessDenied", "list governed objects"),
        ("get_object", "AccessDenied", "read object"),
    ],
)
def test_object_store_failures_are_actionable(
    tmp_path: Path, operation: str, code: str, message: str
) -> None:
    s3 = FailingS3(operation, code)
    mlflow = FakeMlflow()
    with pytest.raises(workflow.PlatformBootstrapError, match=message):
        workflow.bootstrap_platform(
            deployment_root=tmp_path / "deployment",
            environment=_environment(),
            s3_client=s3,
            mlflow_client=mlflow,
        )


def test_object_digest_metadata_is_verified(tmp_path: Path) -> None:
    s3 = FakeS3()
    mlflow = FakeMlflow()
    deployment = tmp_path / "deployment"
    workflow.bootstrap_platform(
        deployment_root=deployment,
        environment=_environment(),
        s3_client=s3,
        mlflow_client=mlflow,
    )
    key = next(iter(s3.objects))
    content, _ = s3.objects[key]
    s3.objects[key] = (content, {"sha256": "0" * 64})

    with pytest.raises(workflow.PlatformBootstrapError, match="object digest mismatch"):
        workflow.verify_platform(
            deployment_root=deployment,
            environment=_environment(),
            s3_client=s3,
            mlflow_client=mlflow,
        )


def test_registry_rejects_unexpected_versions_tags_sources_and_failures(tmp_path: Path) -> None:
    config = load_platform_config()
    uri = f"s3://credit-risk-mlflow/{config.registry_bootstrap.artifact_prefix}"

    unexpected = FakeMlflow()
    unexpected.created = True
    unexpected.versions = [SimpleNamespace(version="9", source=uri, tags={})]
    with pytest.raises(workflow.PlatformBootstrapError, match="exactly two versions"):
        workflow._verify_registry(
            client=unexpected,
            config=config,
            artifact_uri=uri,
            config_digest=EXPECTED_CONFIG_SHA256,
        )

    valid = FakeMlflow()
    valid.created = True
    for contract in config.registry_bootstrap.versions:
        valid.versions.append(
            SimpleNamespace(
                version=contract.registry_version,
                source=uri,
                tags=workflow._version_tags(
                    config, contract.release_revision, EXPECTED_CONFIG_SHA256
                ),
            )
        )
    valid.aliases = {"champion": "1", "rollback": "2"}
    valid.versions[0].tags = {"foreign": "true"}
    with pytest.raises(workflow.PlatformBootstrapError, match="tags differ"):
        workflow._verify_registry(
            client=valid,
            config=config,
            artifact_uri=uri,
            config_digest=EXPECTED_CONFIG_SHA256,
        )

    valid.versions[0].tags = workflow._version_tags(
        config, "phase7_rev_001", EXPECTED_CONFIG_SHA256
    )
    valid.versions[0].source = "s3://foreign"
    with pytest.raises(workflow.PlatformBootstrapError, match="foreign artifact source"):
        workflow._verify_registry(
            client=valid,
            config=config,
            artifact_uri=uri,
            config_digest=EXPECTED_CONFIG_SHA256,
        )

    class BrokenMlflow(FakeMlflow):
        def get_registered_model(self, name: str) -> SimpleNamespace:
            del name
            raise RuntimeError("registry unavailable")

    with pytest.raises(workflow.PlatformBootstrapError, match="inspect the MLflow model"):
        workflow._ensure_registry(
            client=BrokenMlflow(),
            config=config,
            artifact_uri=uri,
            config_digest=EXPECTED_CONFIG_SHA256,
        )

    wrong = FakeMlflow()
    wrong.created = True

    def create_wrong_version(
        name: str, *, source: str, tags: dict[str, str], description: str
    ) -> SimpleNamespace:
        del name, source, tags, description
        return SimpleNamespace(version="9")

    wrong.create_model_version = create_wrong_version  # type: ignore[method-assign]
    with pytest.raises(workflow.PlatformBootstrapError, match="unexpected registry version"):
        workflow._ensure_registry(
            client=wrong,
            config=config,
            artifact_uri=uri,
            config_digest=EXPECTED_CONFIG_SHA256,
        )


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("create_registered_model", "create the reviewed registered model"),
        ("search_model_versions", "search the registered model versions"),
        ("create_model_version", "create registry version 1"),
        ("set_registered_model_alias", "assign the champion alias"),
    ],
)
def test_mlflow_mutation_failures_are_normalized(operation: str, message: str) -> None:
    config = load_platform_config()
    uri = f"s3://credit-risk-mlflow/{config.registry_bootstrap.artifact_prefix}"

    class FailingMlflow(FakeMlflow):
        def get_registered_model(self, name: str) -> SimpleNamespace:
            if operation == "create_registered_model":
                raise MissingModel()
            self.created = True
            return super().get_registered_model(name)

        def create_registered_model(self, *args: Any, **kwargs: Any) -> None:
            if operation == "create_registered_model":
                raise RuntimeError("service unavailable")
            super().create_registered_model(*args, **kwargs)

        def search_model_versions(self, filter_string: str) -> list[SimpleNamespace]:
            if operation == "search_model_versions":
                raise RuntimeError("service unavailable")
            return super().search_model_versions(filter_string)

        def create_model_version(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            if operation == "create_model_version":
                raise RuntimeError("service unavailable")
            return super().create_model_version(*args, **kwargs)

        def set_registered_model_alias(self, *args: Any, **kwargs: Any) -> None:
            if operation == "set_registered_model_alias":
                raise RuntimeError("service unavailable")
            super().set_registered_model_alias(*args, **kwargs)

    with pytest.raises(workflow.PlatformBootstrapError, match=message):
        workflow._ensure_registry(
            client=FailingMlflow(),
            config=config,
            artifact_uri=uri,
            config_digest=EXPECTED_CONFIG_SHA256,
        )


def test_mlflow_client_construction_failure_is_normalized(monkeypatch) -> None:
    class BrokenClient:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("connection refused")

    fake_module = SimpleNamespace(tracking=SimpleNamespace(MlflowClient=BrokenClient))
    monkeypatch.setitem(sys.modules, "mlflow", fake_module)

    with pytest.raises(workflow.PlatformBootstrapError, match="initialise the MLflow client"):
        workflow._new_mlflow_client("http://mlflow:5000")


@pytest.mark.parametrize(
    "tags",
    [
        {},
        {"protocol_id": "phase8_v1", "bundle_id": "selected_v1"},
        {
            "protocol_id": "foreign",
            "bundle_id": "selected_v1",
            "model_id": "catboost_fixed",
        },
        {
            "protocol_id": "phase8_v1",
            "bundle_id": "selected_v1",
            "model_id": "catboost_fixed",
            "unexpected": "tag",
        },
    ],
)
def test_registry_rejects_inexact_registered_model_tags(tags: dict[str, str]) -> None:
    config = load_platform_config()
    uri = f"s3://credit-risk-mlflow/{config.registry_bootstrap.artifact_prefix}"
    client = FakeMlflow()
    client.created = True
    client.registered_tags = tags
    for contract in config.registry_bootstrap.versions:
        client.versions.append(
            SimpleNamespace(
                version=contract.registry_version,
                source=uri,
                tags=workflow._version_tags(
                    config, contract.release_revision, EXPECTED_CONFIG_SHA256
                ),
            )
        )
    client.aliases = {"champion": "1", "rollback": "2"}

    with pytest.raises(workflow.PlatformBootstrapError, match="Registered-model tags"):
        workflow._verify_registry(
            client=client,
            config=config,
            artifact_uri=uri,
            config_digest=EXPECTED_CONFIG_SHA256,
        )


def test_active_deployment_and_atomic_write_fail_closed(tmp_path: Path, monkeypatch) -> None:
    config = load_platform_config()
    active = SimpleNamespace(
        release_revision="foreign",
        registry_version=1,
        approval_sha256=config.registry_bootstrap.rollback_approval_sha256,
        event_receipt_sha256=config.registry_bootstrap.rollback_receipt_sha256,
    )
    with pytest.raises(workflow.PlatformBootstrapError, match="Active deployment differs"):
        workflow._validate_active(config, active)

    monkeypatch.setattr(workflow.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(workflow.PlatformBootstrapError, match="atomically"):
        workflow._write_atomic(tmp_path / "value.json", b"{}\n")

    with pytest.raises(workflow.PlatformBootstrapError, match="Unable to hash"):
        workflow._sha256_file(tmp_path / "missing")


def test_bundle_and_path_guards_reject_unsafe_inputs(tmp_path: Path, monkeypatch) -> None:
    config = load_platform_config()
    with pytest.raises(workflow.PlatformBootstrapError, match="repository-relative"):
        workflow._resolve_project_path(Path.cwd(), Path.cwd(), "models", "bundle")

    empty = tmp_path / "bundle"
    empty.mkdir()
    with pytest.raises(workflow.PlatformBootstrapError, match="two-file allowlist"):
        workflow._validate_bundle(empty, config)

    with pytest.raises(workflow.PlatformBootstrapError, match="missing or unsafe"):
        workflow._validate_bundle(tmp_path / "missing", config)

    inaccessible = tmp_path / "inaccessible"
    inaccessible.mkdir()
    original_iterdir = Path.iterdir

    def deny_selected_bundle(path: Path) -> Iterator[Path]:
        if path == inaccessible:
            raise PermissionError("access denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", deny_selected_bundle)
    with pytest.raises(workflow.PlatformBootstrapError, match="inspect selected bundle"):
        workflow._validate_bundle(inaccessible, config)


def test_context_normalizes_missing_and_invalid_paths(tmp_path: Path) -> None:
    with pytest.raises(workflow.PlatformBootstrapError, match="Invalid Phase 8"):
        workflow._load_context(tmp_path / "missing.json", "models/selected_v1", tmp_path)

    with pytest.raises(workflow.PlatformBootstrapError, match="Unable to resolve bundle root"):
        workflow._load_context("configs/platform/phase8_v1.json", "models/missing", tmp_path)

    deployment_file = tmp_path / "deployment-file"
    deployment_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(workflow.PlatformBootstrapError, match="must be a directory"):
        workflow._load_context(
            "configs/platform/phase8_v1.json", "models/selected_v1", deployment_file
        )


def test_verify_normalizes_missing_and_corrupt_deployments(tmp_path: Path) -> None:
    s3 = FakeS3()
    mlflow = FakeMlflow()
    valid = tmp_path / "valid"
    workflow.bootstrap_platform(
        deployment_root=valid,
        environment=_environment(),
        s3_client=s3,
        mlflow_client=mlflow,
    )

    with pytest.raises(workflow.PlatformBootstrapError, match="active deployment"):
        workflow.verify_platform(
            deployment_root=tmp_path / "missing",
            environment=_environment(),
            s3_client=s3,
            mlflow_client=mlflow,
        )

    (valid / "active.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(workflow.PlatformBootstrapError, match="active deployment"):
        workflow.verify_platform(
            deployment_root=valid,
            environment=_environment(),
            s3_client=s3,
            mlflow_client=mlflow,
        )


def test_deployment_rejects_symlinked_ancestors_before_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "parent" / "deployment"
    original_is_symlink = Path.is_symlink

    def mark_parent_as_symlink(path: Path) -> bool:
        return path == root.parent or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", mark_parent_as_symlink)
    with pytest.raises(workflow.PlatformBootstrapError, match="must not traverse a symlink"):
        workflow._validate_deployment_root(root)
    assert not root.exists()


def test_deployment_cleanup_never_follows_nested_symlink(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "deployment"
    release = root / "releases" / "phase7_rev_001"
    release.mkdir(parents=True)
    original_is_symlink = Path.is_symlink

    def mark_releases_as_symlink(path: Path) -> bool:
        return path == root / "releases" or original_is_symlink(path)

    removals: list[Path] = []
    monkeypatch.setattr(Path, "is_symlink", mark_releases_as_symlink)
    monkeypatch.setattr(workflow.shutil, "rmtree", lambda path: removals.append(path))

    workflow._remove_incomplete_release(root, release)

    assert removals == []
    assert release.exists()
