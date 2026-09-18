"""Credential-safe launcher for the pinned Phase 8 MLflow server."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote


class MlflowServerError(RuntimeError):
    """Raised when the Phase 8 MLflow process cannot be configured or started."""


_REQUIRED_ENVIRONMENT = (
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "MLFLOW_ARTIFACT_BUCKET",
)


def build_backend_store_uri(environment: Mapping[str, str]) -> str:
    """Build an encoded PostgreSQL URI without exposing its credentials."""

    values = {name: environment.get(name, "") for name in _REQUIRED_ENVIRONMENT}
    missing = sorted(name for name, value in values.items() if not value)
    if missing:
        raise MlflowServerError(
            f"Required MLflow server environment variables are missing: {', '.join(missing)}."
        )
    username = quote(values["POSTGRES_USER"], safe="")
    password = quote(values["POSTGRES_PASSWORD"], safe="")
    database = quote(values["POSTGRES_DB"], safe="")
    return f"postgresql+psycopg2://{username}:{password}@postgres:5432/{database}"


def launch_mlflow_server(
    *,
    environment: Mapping[str, str] | None = None,
    exec_function: Callable[[str, list[str], Mapping[str, str]], Any] = os.execvpe,
) -> int:
    """Replace this process with MLflow using a secret-free argument vector."""

    runtime_environment = os.environ if environment is None else environment
    process_environment = dict(runtime_environment)
    process_environment["MLFLOW_BACKEND_STORE_URI"] = build_backend_store_uri(runtime_environment)
    bucket = runtime_environment["MLFLOW_ARTIFACT_BUCKET"]
    arguments = [
        "mlflow",
        "server",
        "--artifacts-destination",
        f"s3://{bucket}",
        "--serve-artifacts",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--workers",
        "1",
        "--allowed-hosts",
        "mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000",
    ]
    try:
        exec_function("mlflow", arguments, process_environment)
    except OSError as error:
        raise MlflowServerError(f"Unable to start the MLflow server: {error}") from error
    return 0


def main() -> int:
    """Run the launcher with a concise expected-failure response."""

    try:
        return launch_mlflow_server()
    except MlflowServerError as error:
        print(f"MLflow platform startup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised by the container entrypoint
    raise SystemExit(main())
