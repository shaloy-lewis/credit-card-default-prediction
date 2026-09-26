"""Capture actual GitHub Actions conclusions for one exact implementation commit."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from credit_risk.assurance.evidence import EvidenceError, clean_commit, hash_file, safe_path
from credit_risk.assurance.runtime import write_new
from credit_risk.release.release_b import check_ci

REPOSITORY = "shaloy-lewis/credit-card-default-prediction"


def github(path: str) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "credit-risk-evidence",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}/{path}", headers=headers
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def capture(output: str = "experiment/release_b/ci-receipt.json") -> dict[str, Any]:
    commit = clean_commit()
    runs = github(f"actions/runs?head_sha={commit}&per_page=100")["workflow_runs"]
    runs = [run for run in runs if run["name"] == "CI" and run["head_sha"] == commit]
    if not runs:
        raise EvidenceError("No CI run exists for the exact implementation commit.")
    run = max(runs, key=lambda item: item["id"])
    jobs = github(f"actions/runs/{run['id']}/jobs?per_page=100")["jobs"]
    receipt = {
        "repository": REPOSITORY,
        "head_sha": commit,
        "run_id": run["id"],
        "run_url": run["html_url"],
        "conclusion": run["conclusion"],
        "jobs": {job["name"]: job["conclusion"] for job in jobs},
    }
    check_ci(receipt, commit)
    path = safe_path(output, "experiment/release_b")
    write_new(path, receipt)
    return {"status": "ci_verified", "sha256": hash_file(path), "run_url": run["html_url"]}
