"""Local-only subprocess and HTTP helpers for measured operational rehearsals."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from credit_risk.assurance.evidence import ROOT, EvidenceError, encode

COMPOSE = [
    "docker",
    "compose",
    "-p",
    "credit-risk-release-b",
    "--env-file",
    ".env",
    "-f",
    "docker-compose.platform.yml",
]


def command(arguments: list[str], timeout: float = 900) -> str:
    try:
        completed = subprocess.run(
            arguments,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        # Subprocess stderr can contain credentials or local paths.
        raise EvidenceError(f"Operational command failed: {arguments[0]}.") from error


def request(path: str, payload: dict[str, Any] | None = None, port: int = 8080) -> Any:
    data = None if payload is None else encode(payload)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        content = response.read()
    try:
        return json.loads(content)
    except ValueError:
        return content.decode()


def await_ready(timeout: float = 240) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            if request("/ready") == {"status": "ready"}:
                return time.perf_counter() - started
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise EvidenceError("API did not recover within the measured deadline.")


def platform_state(operation: str = "verify") -> dict[str, Any]:
    text = command(
        [
            *COMPOSE,
            "run",
            "--rm",
            "--no-deps",
            "bootstrap",
            "credit-risk",
            "platform",
            operation,
            "--deployment-root",
            "/deployment",
        ]
    )
    # MLflow may log progress before the final JSON result.
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and "object_sha256" in value:
                return value
        except ValueError:
            pass
    raise EvidenceError("Platform command did not return verified state.")


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encode(value))
