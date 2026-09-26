"""Release B assembly never scores, self-approves or rewrites historical evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from credit_risk.assurance.evidence import (
    EvidenceError,
    clean_commit,
    encode,
    hash_file,
    publish,
    read_json,
    require_sha,
    safe_path,
    source_map,
    verify,
)

CONFIG = "configs/releases/release_b_v1.json"
OUTPUT = "reports/releases/release_b_v1"
NEW_EVIDENCE = {
    "platform": "phase8_platform_evidence_v1",
    "robustness": "release_b_robustness_v1",
    "reference": "monitor_reference_v1",
    "batch_monitoring": "monitor_batch_v1",
    "service_monitoring": "monitor_service_v1",
    "benchmark": "monitor_benchmark_v1",
    "acceptance": "monitor_acceptance_v1",
    "incidents": "release_b_incidents_v1",
}
RISK_DETAILS = {
    "education_disparities": (
        "Demographic disparity",
        "Retain both G3 education disparity conditions; audit-only demographic fields and human review.",
    ),
    "geographic_temporal_transportability": (
        "Geographic and temporal transportability",
        "Local demonstration only; no claims for current or other populations without representative evaluation.",
    ),
    "calibration_drift": (
        "Calibration or feature drift",
        "Identity calibration remains fixed; investigate drift manually, with no automatic recalibration.",
    ),
    "production_privacy": (
        "Privacy and logging",
        "Use synthetic operational fixtures; keep validation scores and account mappings in ignored local storage.",
    ),
    "phase7_phase8_boundary": (
        "Missing registry and rollback",
        "Phase 7 SQLite demonstrates promotion/rollback; Phase 8 demonstrates fixed bootstrap/persistence only.",
    ),
    "production_monitoring": (
        "Missing production monitoring",
        "CLI monitoring is a local demonstration; longitudinal evaluation and production operations remain G5 work.",
    ),
    "explanation_language": (
        "Misleading explanations",
        "Explanations describe model attribution, never causality or adverse-action reasons.",
    ),
    "human_owned_use": (
        "Unsafe automation",
        "Human-owned outreach demonstration only; prohibit lending, adverse action and automated customer decisions.",
    ),
    "artifact_integrity": (
        "Artifact integrity",
        "Authenticate model and deployment digests before use; reject missing or corrupt artifacts.",
    ),
    "consumed_test_protection": (
        "Consumed-test protection",
        "The sealed test remains retired; no fits, tuning, bootstrap regeneration or new test scoring.",
    ),
    "object_store_upstream_availability": (
        None,
        "Use the source-pinned legacy MinIO build only in isolated local resources with private S3 networking; maintained object storage and production supportability remain G5 work.",
    ),
}
RISK_IDS = tuple(RISK_DETAILS)

REQUIRED_JOBS = {
    "Lint, type-check, and test",
    "Pull and test reviewed artifacts",
    "Build, test, and scan API container",
    "Build, persist, verify, and scan local platform",
}


def historical_evidence() -> dict[str, str]:
    from credit_risk.governance.workflow import verify_governance_evidence
    from credit_risk.inference.evidence import verify_inference_evidence
    from credit_risk.registry.workflow import verify_registry_evidence
    from credit_risk.release.workflow import verify_release_evidence

    anchors = {
        "reports/releases/release_a_v1/evidence-manifest.json": "7e65c7b854de15742f05c4b8c2de891f50512518f8eb2339241f87f98754edf7",
        "reports/governance/phase5_v1/evidence-manifest.json": "6df8745f6deefcd138d7d1c821e6ad38ca7762aa7581fa0fae28042cf7f2b853",
        "reports/inference/phase6_v1/evidence-manifest.json": "919087229d20fe83c1846da65d5901ea103ac3182c9c2975c3490424a49f4df8",
        "reports/registry/phase7_v1/evidence-manifest.json": "ce36f33da60fe6470d28a76b8053d102e74731115d069c4d476d0c2abbc47da9",
    }
    digests = list(anchors.values())
    verify_release_evidence(expected_manifest_sha256=digests[0])
    verify_governance_evidence(expected_manifest_sha256=digests[1], aggregate_only=True)
    verify_inference_evidence(expected_manifest_sha256=digests[2])
    verify_registry_evidence(expected_manifest_sha256=digests[3])
    return anchors


def check_ci(receipt: dict[str, Any], commit: str) -> None:
    if (
        receipt.get("repository") != "shaloy-lewis/credit-card-default-prediction"
        or receipt.get("head_sha") != commit
        or receipt.get("conclusion") != "success"
        or not isinstance(receipt.get("run_id"), int)
    ):
        raise EvidenceError("Successful exact-commit CI evidence is required.")
    jobs = receipt.get("jobs")
    if not isinstance(jobs, dict) or not REQUIRED_JOBS <= jobs.keys():
        raise EvidenceError("CI evidence omits required jobs.")
    if any(jobs[name] != "success" for name in REQUIRED_JOBS):
        raise EvidenceError("Every mandatory CI job must succeed.")


def review_ids(evidence: dict[str, Any]) -> list[str]:
    ids = [f"robustness:{x['scenario']}" for x in evidence["robustness"]["scenarios"]]
    ids += [f"robustness:{x['scenario']}" for x in evidence["robustness"]["subsets"]]
    ids += [f"incident:{x['drill']}" for x in evidence["incidents"]["drills"]]
    ids += [f"risk:{name}" for name in RISK_IDS]
    ids += ["monitoring:batch_disposition"]
    return sorted(ids)


def build(
    config: str = CONFIG,
    output: str = OUTPUT,
    ci_receipt: str = "experiment/release_b/ci-receipt.json",
    expected_ci_sha256: str | None = None,
) -> str:
    commit = clean_commit()
    contract = read_json(safe_path(config))
    if contract.get("release_id") != "release_b_v1" or set(contract.get("evidence", {})) != set(
        NEW_EVIDENCE
    ):
        raise EvidenceError("Release B requires the complete evidence contract.")
    sources = historical_evidence()
    evidence = {}
    for name, kind in NEW_EVIDENCE.items():
        item = contract["evidence"][name]
        expected = item.get("expected_manifest_sha256")
        require_sha(expected)
        evidence[name] = verify(item["root"], expected, kind)
        manifest_path = (Path(item["root"]) / "evidence-manifest.json").as_posix()
        sources[manifest_path] = expected
        # Include all authenticated outputs, not just child manifests.
        child = read_json(safe_path(manifest_path))
        sources.update(
            {(Path(item["root"]) / p).as_posix(): sha for p, sha in child["outputs"].items()}
        )
    from credit_risk.platform.evidence import verify_evidence as verify_platform
    from credit_risk.robustness.workflow import verify_evidence as verify_robustness

    for name, function in (("platform", verify_platform), ("robustness", verify_robustness)):
        item = contract["evidence"][name]
        function(item["root"], item["expected_manifest_sha256"])
    if evidence["acceptance"].get("status") != "passed":
        raise EvidenceError("Service acceptance has not passed.")
    if evidence["service_monitoring"].get("status") != "measured":
        raise EvidenceError("Missing measured service monitoring.")
    if evidence["batch_monitoring"].get("status") == "insufficient_data":
        raise EvidenceError("Release monitoring demonstration needs at least 200 valid rows.")
    required_drills = {
        "missing_columns",
        "invalid_values",
        "duplicate_ids",
        "population_shift",
        "prediction_shift",
        "artifact_integrity",
        "phase7_sqlite_rollback",
        "service_interruption",
    }
    drills = evidence["incidents"].get("drills", [])
    if len(drills) != len(required_drills) or {d["drill"] for d in drills} != required_drills:
        raise EvidenceError("Incident exercise coverage is incomplete.")
    if any(
        d.get(control) is not True
        for d in drills
        for control in ("detected", "contained", "recovered", "verified")
    ):
        raise EvidenceError("A mandatory incident control failed.")
    ci_path = safe_path(ci_receipt, "experiment")
    require_sha(expected_ci_sha256)
    if hash_file(ci_path) != expected_ci_sha256:
        raise EvidenceError("CI receipt differs from the supplied trust anchor.")
    ci = read_json(ci_path)
    check_ci(ci, commit)
    summary = {
        "status": "pending_owner_signoff",
        "g4_status": "open",
        "scope": "local_portfolio_only",
        "g3_status": "closed_with_conditions",
        "g5_status": "open",
        "implementation_commit": commit,
        "evidence": contract["evidence"],
        "required_dispositions": review_ids(evidence),
        "ci": ci,
        "model_changes": False,
    }
    sources.update(
        source_map(
            [
                config,
                "docs/operations/release-b-runbooks.md",
                "docs/operations/delayed-label-contract.md",
                "reports/governance/phase5_v1/risk-register.md",
            ]
        )
    )
    approval_template = {
        "dossier_sha256": None,
        "decision": "pending",
        "owner": None,
        "scope": "local_portfolio_only",
        "g3_conditions_retained": True,
        "dispositions": {
            key: {"decision": "pending", "restriction": None}
            for key in summary["required_dispositions"]
        },
    }
    report = (
        "# Release B review candidate\n\n"
        "All mandatory evidence has been authenticated. G4 remains open pending an explicit "
        "owner decision bound to this dossier digest. This builder does not grant approval.\n\n"
        "The existing Phase 5 risk register remains immutable. The approval disposition covers "
        "education disparity, transportability, drift, production privacy, the separate SQLite "
        "rollback/persistent-bootstrap scopes, and deferred production monitoring.\n"
    )
    return publish(
        safe_path(output, "reports/releases"),
        kind="release_b_v1",
        summary=summary,
        sources=sources,
        commit=commit,
        extra={
            "release-b-report.md": report.encode(),
            "owner-approval-template.json": encode(approval_template),
            "current-risk-disposition.json": encode(
                {
                    "status": "pending_owner_review",
                    "historical_register": "reports/governance/phase5_v1/risk-register.md",
                    "historical_register_rewritten": False,
                    "risks": [
                        {
                            "id": name,
                            "historical_entry": historical,
                            "proposed_restriction": restriction,
                            "decision": "pending",
                        }
                        for name, (historical, restriction) in RISK_DETAILS.items()
                    ],
                }
            ),
        },
    )


def verify_b(
    root: str,
    expected: str,
    approval: str | None = None,
    approval_sha256: str | None = None,
) -> dict[str, Any]:
    summary = verify(root, expected, "release_b_v1")
    historical_evidence()
    for name, kind in NEW_EVIDENCE.items():
        item = summary["evidence"][name]
        verify(item["root"], item["expected_manifest_sha256"], kind)
    check_ci(summary["ci"], summary["implementation_commit"])
    if approval is None:
        return {
            "status": "verified_pending_owner_signoff",
            "g4_status": "open",
            "dossier_sha256": expected,
        }
    require_sha(approval_sha256)
    if hash_file(approval) != approval_sha256:
        raise EvidenceError("Owner approval digest mismatch.")
    decision = read_json(safe_path(approval))
    if (
        decision.get("dossier_sha256") != expected
        or decision.get("decision") != "approved"
        or decision.get("scope") != "local_portfolio_only"
        or not isinstance(decision.get("owner"), str)
        or not decision["owner"].strip()
        or decision.get("g3_conditions_retained") is not True
        or set(decision.get("dispositions", {})) != set(summary["required_dispositions"])
    ):
        raise EvidenceError("Missing explicit dossier-bound owner approval.")
    for item in decision["dispositions"].values():
        if (
            item.get("decision") not in {"accepted_with_restriction", "resolved"}
            or not isinstance(item.get("restriction"), str)
            or not item["restriction"].strip()
        ):
            raise EvidenceError("Every finding needs a compatible disposition and restriction.")
    return {
        "status": "approved_local_portfolio_release",
        "g4_status": "closed",
        "dossier_sha256": expected,
        "approval_sha256": approval_sha256,
    }
