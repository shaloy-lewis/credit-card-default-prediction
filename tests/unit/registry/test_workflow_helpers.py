from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from credit_risk.registry import workflow
from credit_risk.registry.contracts import (
    DEFAULT_REGISTRY_CONFIG_PATH,
    EXPECTED_CONFIG_SHA256,
    REQUIRED_CHECKS,
    ReleaseApproval,
    load_registry_config,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
pytestmark = pytest.mark.artifact


def _approval(action: str = "promote") -> ReleaseApproval:
    promotion = action == "promote"
    return ReleaseApproval.model_validate(
        {
            "schema_version": "1.0.0",
            "protocol_id": "phase7_v1",
            "decision_id": (
                "phase7_promotion_approval" if promotion else "phase7_rollback_approval"
            ),
            "action": action,
            "decision": "approved",
            "scope": "local_portfolio_demo",
            "approved_by_role": "project_owner",
            "implementation_git_commit": "a" * 40,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "registered_model_name": "credit-risk-default",
            "source_revision": "phase7_rev_001" if promotion else "phase7_rev_002",
            "target_revision": "phase7_rev_002" if promotion else "phase7_rev_001",
            "checks": tuple({"name": name, "conclusion": "success"} for name in REQUIRED_CHECKS),
            "model_bytes_unchanged": True,
            "no_training_or_test_access": True,
            "limitations_acknowledged": True,
        },
        strict=True,
    )


def test_bundle_and_source_validation_fail_closed(tmp_path: Path) -> None:
    config = load_registry_config(REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH)
    with pytest.raises(workflow.RegistryWorkflowError, match="missing or unsafe"):
        workflow._validate_bundle(config, tmp_path / "missing")

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "extra").write_bytes(b"x")
    with pytest.raises(workflow.RegistryWorkflowError, match="two-file allowlist"):
        workflow._validate_bundle(config, bundle)

    source = REPOSITORY_ROOT / "models/selected_v1"
    for name in ("manifest.json", "model.cbm"):
        (bundle / name).write_bytes((source / name).read_bytes())
    (bundle / "extra").unlink()
    (bundle / "manifest.json").write_bytes(b"changed")
    with pytest.raises(workflow.RegistryWorkflowError, match="manifest digest"):
        workflow._validate_bundle(config, bundle)
    (bundle / "manifest.json").write_bytes((source / "manifest.json").read_bytes())
    (bundle / "model.cbm").write_bytes(b"changed")
    with pytest.raises(workflow.RegistryWorkflowError, match="model digest"):
        workflow._validate_bundle(config, bundle)

    reference = config.source_evidence["phase6_config"]
    changed_config = config.model_copy(
        update={
            "source_evidence": {
                **config.source_evidence,
                "phase6_config": reference.model_copy(update={"sha256": "0" * 64}),
            }
        }
    )
    with pytest.raises(workflow.RegistryWorkflowError, match="Reviewed source digest"):
        workflow._validate_config_sources(REPOSITORY_ROOT, changed_config)


def test_approval_validation_failure_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_registry_config(REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH)
    approval = _approval()
    with pytest.raises(workflow.RegistryWorkflowError, match="Expected a rollback"):
        workflow._validate_approval(
            config,
            approval,
            action="rollback",
            config_path=REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH,
            repository=REPOSITORY_ROOT,
        )

    stale = approval.model_copy(update={"config_sha256": "0" * 64})
    with pytest.raises(workflow.RegistryWorkflowError, match="configuration digest"):
        workflow._validate_approval(
            config,
            stale,
            action="promote",
            config_path=REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH,
            repository=REPOSITORY_ROOT,
        )

    foreign = approval.model_copy(update={"registered_model_name": "foreign"})
    with pytest.raises(workflow.RegistryWorkflowError, match="different registered model"):
        workflow._validate_approval(
            config,
            foreign,
            action="promote",
            config_path=REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH,
            repository=REPOSITORY_ROOT,
        )

    monkeypatch.setattr(workflow, "_git_is_ancestor", lambda *_args: False)
    with pytest.raises(workflow.RegistryWorkflowError, match="not in the current history"):
        workflow._validate_approval(
            config,
            approval,
            action="promote",
            config_path=REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH,
            repository=REPOSITORY_ROOT,
        )


def test_registry_inspection_rejects_bad_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_registry_config(REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH)
    root = tmp_path / "registry"
    root.mkdir()
    with pytest.raises(workflow.RegistryWorkflowError, match="file allowlist"):
        workflow._inspect_registry(config, root)

    (root / workflow.REGISTRY_DATABASE).write_bytes(b"db")
    (root / "artifacts" / config.bundle.model_sha256 / "bundle").mkdir(parents=True)
    (root / "events").mkdir()
    monkeypatch.setattr(workflow, "_validate_bundle", lambda *_args: None)

    class FakeClient:
        def get_registered_model(self, _name):
            return SimpleNamespace(aliases={})

        def search_model_versions(self, _query):
            raise RuntimeError("broken")

    monkeypatch.setattr(workflow, "_mlflow_client", lambda _path: FakeClient())
    monkeypatch.setattr(workflow, "_dispose_client", lambda _client: None)
    with pytest.raises(workflow.RegistryWorkflowError, match="Unable to inspect"):
        workflow._inspect_registry(config, root)

    version = SimpleNamespace(version="1", tags={}, source="foreign")
    FakeClient.search_model_versions = lambda self, _query: [version]
    with pytest.raises(workflow.RegistryWorkflowError, match="exactly the two"):
        workflow._inspect_registry(config, root)

    tags1 = workflow._version_tags(config, "phase7_rev_001", EXPECTED_CONFIG_SHA256)
    tags2 = workflow._version_tags(config, "phase7_rev_002", EXPECTED_CONFIG_SHA256)
    source = (root / "artifacts" / config.bundle.model_sha256 / "bundle").resolve().as_uri()
    v1 = SimpleNamespace(version="1", tags={**tags1, "extra": "x"}, source=source)
    v2 = SimpleNamespace(version="2", tags=tags2, source=source)
    FakeClient.search_model_versions = lambda self, _query: [v1, v2]
    with pytest.raises(workflow.RegistryWorkflowError, match="tags differ"):
        workflow._inspect_registry(config, root)

    v1.tags = tags1
    v1.source = "foreign"
    with pytest.raises(workflow.RegistryWorkflowError, match="foreign artifact"):
        workflow._inspect_registry(config, root)

    v1.source = source
    FakeClient.get_registered_model = lambda self, _name: SimpleNamespace(aliases={"bad": "1"})
    with pytest.raises(workflow.RegistryWorkflowError, match="unapproved alias"):
        workflow._inspect_registry(config, root)


def test_helper_failure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(workflow.RegistryWorkflowError, match="required state"):
        workflow._require_aliases({"champion": "2"}, {"champion": "1"})

    class AliasClient:
        def __init__(self):
            self.deleted = []
            self.set = []

        def get_registered_model(self, _name):
            return SimpleNamespace(aliases={"extra": "2"})

        def delete_registered_model_alias(self, _name, alias):
            self.deleted.append(alias)

        def set_registered_model_alias(self, _name, alias, version):
            self.set.append((alias, version))

    client = AliasClient()
    workflow._restore_aliases(client, "model", {"champion": "1"})
    assert client.deleted == ["extra"]
    assert client.set == [("champion", "1")]

    monkeypatch.setattr(workflow.importlib.metadata, "version", lambda _name: "0.0.0")
    with pytest.raises(workflow.RegistryWorkflowError, match="version mismatch"):
        workflow._load_mlflow()

    repository = REPOSITORY_ROOT.resolve()
    with pytest.raises(workflow.RegistryWorkflowError, match="beneath experiment/registry"):
        workflow._safe_path(repository, "models", "experiment/registry", "root", must_exist=True)
    with pytest.raises(workflow.RegistryWorkflowError, match="does not exist"):
        workflow._safe_path(
            repository, "experiment/not-there", "experiment", "root", must_exist=True
        )

    target = tmp_path / "atomic.json"
    monkeypatch.setattr(workflow.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(workflow.RegistryWorkflowError, match="atomically"):
        workflow._write_atomic(target, b"value")


def test_deployment_smoke_parity_is_prediction_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_registry_config(REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH)

    class FakeEngine:
        def __init__(self, bundle_root: Path) -> None:
            self.revision = bundle_root.parents[0].name

        def score(self, frame):
            assert tuple(frame.columns) == workflow.PREDICTOR_COLUMNS
            reason = SimpleNamespace(
                category="repayment_status",
                direction="risk_increasing",
                contribution_raw_log_odds=0.25,
            )
            return SimpleNamespace(
                probabilities=np.asarray([0.1903818]),
                risk_bands=("standard",),
                reasons=((reason, reason),),
            )

    monkeypatch.setattr(workflow, "InferenceEngine", FakeEngine)
    result = workflow._verify_deployment_smoke_parity(
        config, REPOSITORY_ROOT, tmp_path / "deployments"
    )
    assert result["outputs_identical"] is True
    assert result["prediction_only"] is True
    assert result["sealed_test_fixture"] is False
    assert [item["release_revision"] for item in result["revisions"]] == list(
        config.smoke_test.revisions
    )


def test_deployment_smoke_parity_rejects_fixture_or_output_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_registry_config(REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH)
    original_hash = workflow._sha256_file
    fixture = (REPOSITORY_ROOT / config.smoke_test.fixture_path).resolve()
    monkeypatch.setattr(
        workflow,
        "_sha256_file",
        lambda path: "0" * 64 if Path(path).resolve() == fixture else original_hash(path),
    )
    with pytest.raises(workflow.RegistryWorkflowError, match="fixture digest"):
        workflow._verify_deployment_smoke_parity(config, REPOSITORY_ROOT, tmp_path / "deployments")

    monkeypatch.setattr(workflow, "_sha256_file", original_hash)

    class DifferentOutputs:
        def __init__(self, bundle_root: Path) -> None:
            self.revision = bundle_root.parents[0].name

        def score(self, _frame):
            probability = 0.1903818 if self.revision.endswith("001") else 0.1903822
            reason = SimpleNamespace(
                category="repayment_status",
                direction="risk_increasing",
                contribution_raw_log_odds=probability,
            )
            return SimpleNamespace(
                probabilities=np.asarray([probability]),
                risk_bands=("standard",),
                reasons=((reason, reason),),
            )

    monkeypatch.setattr(workflow, "InferenceEngine", DifferentOutputs)
    with pytest.raises(workflow.RegistryWorkflowError, match="failed synthetic smoke parity"):
        workflow._verify_deployment_smoke_parity(config, REPOSITORY_ROOT, tmp_path / "deployments")


def test_deployment_smoke_parity_rejects_invalid_fixture_and_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_registry_config(REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH)
    operational_features = workflow.OperationalFeatures
    monkeypatch.setattr(
        workflow,
        "OperationalFeatures",
        SimpleNamespace(
            model_validate_json=lambda _value: (_ for _ in ()).throw(ValueError("invalid"))
        ),
    )
    with pytest.raises(workflow.RegistryWorkflowError, match="fixture is invalid"):
        workflow._verify_deployment_smoke_parity(config, REPOSITORY_ROOT, tmp_path / "deployments")
    monkeypatch.setattr(workflow, "OperationalFeatures", operational_features)

    class FailingEngine:
        def __init__(self, bundle_root: Path) -> None:
            self.bundle_root = bundle_root

        def score(self, _frame):
            raise RuntimeError("failed")

    monkeypatch.setattr(workflow, "InferenceEngine", FailingEngine)
    with pytest.raises(workflow.RegistryWorkflowError, match="prediction failed"):
        workflow._verify_deployment_smoke_parity(config, REPOSITORY_ROOT, tmp_path / "deployments")

    class ChangedOutputEngine(FailingEngine):
        def score(self, _frame):
            reason = SimpleNamespace(
                category="repayment_status",
                direction="risk_increasing",
                contribution_raw_log_odds=0.25,
            )
            return SimpleNamespace(
                probabilities=np.asarray([0.5]),
                risk_bands=("high",),
                reasons=((reason, reason),),
            )

    monkeypatch.setattr(workflow, "InferenceEngine", ChangedOutputEngine)
    with pytest.raises(workflow.RegistryWorkflowError, match="differs from"):
        workflow._verify_deployment_smoke_parity(config, REPOSITORY_ROOT, tmp_path / "deployments")


def test_published_smoke_parity_requires_exact_semantics() -> None:
    config = load_registry_config(REPOSITORY_ROOT / DEFAULT_REGISTRY_CONFIG_PATH)
    digest = "a" * 64
    payload = {
        "fixture_sha256": config.smoke_test.fixture_sha256,
        "synthetic": True,
        "prediction_only": True,
        "sealed_test_fixture": False,
        "expected_probability_six_decimals": 0.190382,
        "outputs_identical": True,
        "revisions": [
            {
                "release_revision": revision,
                "probability_six_decimals": 0.190382,
                "risk_band": "standard",
                "output_sha256": digest,
            }
            for revision in config.smoke_test.revisions
        ],
    }
    workflow._validate_published_smoke_parity(config, payload)

    changed = deepcopy(payload)
    changed["revisions"][1]["output_sha256"] = "b" * 64
    with pytest.raises(workflow.RegistryWorkflowError, match="not identical"):
        workflow._validate_published_smoke_parity(config, changed)

    changed = {**payload, "unexpected": True}
    with pytest.raises(workflow.RegistryWorkflowError, match="incomplete"):
        workflow._validate_published_smoke_parity(config, changed)

    changed = deepcopy(payload)
    changed["synthetic"] = False
    with pytest.raises(workflow.RegistryWorkflowError, match="parity is invalid"):
        workflow._validate_published_smoke_parity(config, changed)

    changed = deepcopy(payload)
    changed["revisions"][0]["risk_band"] = "high"
    with pytest.raises(workflow.RegistryWorkflowError, match="revision is invalid"):
        workflow._validate_published_smoke_parity(config, changed)

    changed = deepcopy(payload)
    changed["revisions"][0]["output_sha256"] = "invalid"
    with pytest.raises(workflow.RegistryWorkflowError, match="digest is invalid"):
        workflow._validate_published_smoke_parity(config, changed)


def test_transition_compensation_attempts_every_restore_step(tmp_path: Path) -> None:
    receipt = tmp_path / "events/receipt.json"
    pointer = tmp_path / "deployment/active.json"
    receipt.parent.mkdir()
    pointer.parent.mkdir()
    receipt.write_text("receipt", encoding="utf-8")
    pointer.write_bytes(b"new")

    class BrokenAliasClient:
        def get_registered_model(self, _name):
            raise RuntimeError("alias restore failed")

    error = workflow._compensate_transition(
        client=BrokenAliasClient(),
        model_name="credit-risk-default",
        aliases={"champion": "1"},
        receipt_path=receipt,
        pointer_path=pointer,
        prior_pointer=b"old",
    )
    assert isinstance(error, RuntimeError)
    assert pointer.read_bytes() == b"old"
    assert not receipt.exists()


def test_evidence_and_json_helpers_reject_corruption(tmp_path: Path, monkeypatch) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(workflow.RegistryWorkflowError, match="already exists"):
        workflow._publish_directory(existing, {})

    destination = tmp_path / "evidence"
    with pytest.raises(workflow.RegistryWorkflowError, match="output allowlist"):
        workflow._publish_directory(destination, {"summary.json": b"{}"})

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(workflow.RegistryWorkflowError, match="Unable to read"):
        workflow._read_receipt(invalid)
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(workflow.RegistryWorkflowError, match="must contain an object"):
        workflow._read_receipt(invalid)
    with pytest.raises(workflow.RegistryWorkflowError, match="Unable to hash"):
        workflow._sha256_file(tmp_path / "missing")

    failed = SimpleNamespace(returncode=1, stdout="", stderr="bad")
    monkeypatch.setattr(workflow.subprocess, "run", lambda *_args, **_kwargs: failed)
    with pytest.raises(workflow.RegistryWorkflowError, match="inside the project"):
        workflow._repository_root()
    with pytest.raises(workflow.RegistryWorkflowError, match="current Git commit"):
        workflow._git_commit(tmp_path)
    with pytest.raises(workflow.RegistryWorkflowError, match="Git worktree"):
        workflow._git_dirty(tmp_path)


def test_git_helpers_and_missing_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    assert workflow._repository_root() == REPOSITORY_ROOT.resolve()
    assert len(workflow._git_commit(REPOSITORY_ROOT)) == 40
    assert isinstance(workflow._git_dirty(REPOSITORY_ROOT), bool)
    assert workflow._git_is_ancestor(REPOSITORY_ROOT, workflow._git_commit(REPOSITORY_ROOT))

    def missing(_name):
        raise workflow.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(workflow.importlib.metadata, "version", missing)
    with pytest.raises(workflow.RegistryWorkflowError, match="unavailable"):
        workflow._load_mlflow()
