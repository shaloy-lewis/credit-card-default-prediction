"""Immutable, externally authenticated aggregate evidence envelopes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from credit_risk.modeling.tracking import collect_git_evidence

ROOT = Path(__file__).resolve().parents[3]
HEX = re.compile(r"^[0-9a-f]{64}$")
BOUNDARY = {"fits": 0, "sealed_test_scored": False, "scope": "local_portfolio_only"}


class EvidenceError(RuntimeError):
    """Evidence is unsafe, incomplete, inconsistent or unauthenticated."""


def encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise EvidenceError("Duplicate JSON keys are prohibited.")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
        if not isinstance(value, dict):
            raise ValueError("expected object")
        encode(value)  # Reject NaN/Infinity throughout the object.
        return value
    except (OSError, ValueError, TypeError) as error:
        raise EvidenceError("Invalid or missing evidence JSON.") from error


def safe_path(path: str | Path, subtree: str | None = None) -> Path:
    """Reject traversal, links/junctions and publication outside approved roots."""
    supplied = Path(path)
    if ".." in supplied.parts:
        raise EvidenceError("Parent traversal is prohibited.")
    absolute = Path(os.path.abspath(ROOT / supplied))
    try:
        relative = absolute.relative_to(ROOT)
    except ValueError as error:
        raise EvidenceError("Evidence paths must remain inside the repository.") from error
    current = ROOT
    for component in relative.parts:
        current /= component
        if current.is_symlink() or current.is_junction():
            raise EvidenceError("Symlinked or junction evidence paths are prohibited.")
    if subtree is not None:
        allowed = ROOT / subtree
        if absolute == allowed or not absolute.is_relative_to(allowed):
            raise EvidenceError(f"Destination must be a child of {subtree}.")
    return absolute


def hash_file(path: str | Path) -> str:
    try:
        return digest(safe_path(path).read_bytes())
    except OSError as error:
        raise EvidenceError("Missing evidence source.") from error


def source_map(paths: Sequence[str | Path]) -> dict[str, str]:
    return {safe_path(p).relative_to(ROOT).as_posix(): hash_file(p) for p in paths}


def clean_commit() -> str:
    git = collect_git_evidence(ROOT)
    if git.dirty:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        allowed = (
            "reports/platform/",
            "reports/robustness/",
            "reports/monitoring/",
            "reports/incidents/",
        )
        if not status or any(
            not line.startswith("?? ") or not line[3:].startswith(allowed) for line in status
        ):
            raise EvidenceError("Official evidence requires clean committed implementation.")
        # Only complete generated packages may coexist while collecting sequential evidence.
        packages = {safe_path(line[3:]).parent for line in status}
        for folder in packages:
            manifest = read_json(folder / "evidence-manifest.json")
            verify(folder, hash_file(folder / "evidence-manifest.json"), manifest["kind"])
    return git.commit_sha


def require_sha(value: str | None) -> None:
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise EvidenceError("An external SHA-256 trust anchor is required.")


def publish(
    output: str | Path,
    *,
    kind: str,
    summary: dict[str, Any],
    sources: dict[str, str],
    commit: str,
    extra: dict[str, bytes] | None = None,
) -> str:
    """Publish a fully staged package; never replace existing evidence."""
    destination = safe_path(output, "reports")
    if destination.exists():
        raise EvidenceError("Refusing to overwrite published evidence.")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise EvidenceError("Invalid implementation commit.")
    for path, expected in sources.items():
        require_sha(expected)
        if hash_file(path) != expected:
            raise EvidenceError("Source changed during evidence generation.")
    artifacts = {"summary.json": encode(summary), **(extra or {})}
    if "evidence-manifest.json" in artifacts or any(
        Path(name).name != name or name in {".", ".."} for name in artifacts
    ):
        raise EvidenceError("Invalid output allowlist.")
    manifest = {
        "schema_version": "1.0.0",
        "kind": kind,
        "implementation_commit": commit,
        "boundary": BOUNDARY,
        "sources": sources,
        "outputs": {name: digest(content) for name, content in artifacts.items()},
    }
    manifest_bytes = encode(manifest)
    artifacts["evidence-manifest.json"] = manifest_bytes
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evidence-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        for name, content in artifacts.items():
            (stage / name).write_bytes(content)
        # mkdir makes simultaneous publishers fail before either can overwrite.
        # Rename of a directory is atomic and refuses an existing destination on Windows.
        if destination.exists():
            raise EvidenceError("Evidence destination appeared during publication.")
        stage.rename(destination)
    return digest(manifest_bytes)


def verify(root: str | Path, expected: str, kind: str) -> dict[str, Any]:
    require_sha(expected)
    folder = safe_path(root, "reports")
    if hash_file(folder / "evidence-manifest.json") != expected:
        raise EvidenceError("Evidence manifest does not match external trust anchor.")
    manifest = read_json(folder / "evidence-manifest.json")
    if (
        set(manifest)
        != {"schema_version", "kind", "implementation_commit", "boundary", "sources", "outputs"}
        or manifest["schema_version"] != "1.0.0"
        or manifest["kind"] != kind
    ):
        raise EvidenceError("Unexpected evidence contract.")
    if manifest["boundary"] != BOUNDARY or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["implementation_commit"]
    ):
        raise EvidenceError("Evidence boundary or implementation lineage is invalid.")
    outputs = manifest["outputs"]
    if not isinstance(outputs, dict) or "summary.json" not in outputs:
        raise EvidenceError("Missing summary allowlist.")
    if {p.name for p in folder.iterdir()} != set(outputs) | {"evidence-manifest.json"}:
        raise EvidenceError("Evidence contains missing or unapproved files.")
    for name, expected_output in outputs.items():
        if Path(name).name != name or name in {".", ".."}:
            raise EvidenceError("Invalid output name.")
        require_sha(expected_output)
        if hash_file(folder / name) != expected_output:
            raise EvidenceError("Published artifact digest mismatch.")
    for path, expected_source in manifest["sources"].items():
        require_sha(expected_source)
        if hash_file(path) != expected_source:
            raise EvidenceError("Source artifact digest mismatch.")
    return read_json(folder / "summary.json")


def finite_number(value: Any, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise EvidenceError("Expected numeric measurement.")
    if not math.isfinite(value) or value < minimum:
        raise EvidenceError("Invalid numeric measurement.")
    return float(value)
