from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import pytest

from credit_risk.platform import mlflow_server


def _environment() -> dict[str, str]:
    return {
        "POSTGRES_DB": "ml flow/台北",
        "POSTGRES_USER": "user@example:team",
        "POSTGRES_PASSWORD": "p@ss:/% word 密碼",
        "MLFLOW_ARTIFACT_BUCKET": "credit-risk-mlflow",
        "UNRELATED": "preserved",
    }


def test_backend_uri_percent_encodes_every_credential_component() -> None:
    environment = _environment()
    uri = mlflow_server.build_backend_store_uri(environment)

    assert uri == (
        "postgresql+psycopg2://"
        f"{quote(environment['POSTGRES_USER'], safe='')}:"
        f"{quote(environment['POSTGRES_PASSWORD'], safe='')}"
        f"@postgres:5432/{quote(environment['POSTGRES_DB'], safe='')}"
    )
    assert environment["POSTGRES_PASSWORD"] not in uri


def test_launcher_uses_environment_uri_and_secret_free_process_arguments() -> None:
    captured: dict[str, Any] = {}

    def fake_exec(executable: str, arguments: list[str], environment: Mapping[str, str]) -> None:
        captured.update(
            executable=executable,
            arguments=arguments,
            environment=dict(environment),
        )

    source = _environment()
    assert mlflow_server.launch_mlflow_server(environment=source, exec_function=fake_exec) == 0

    assert captured["executable"] == "mlflow"
    assert captured["arguments"][:2] == ["mlflow", "server"]
    assert "--backend-store-uri" not in captured["arguments"]
    assert "s3://credit-risk-mlflow" in captured["arguments"]
    assert source["POSTGRES_PASSWORD"] not in " ".join(captured["arguments"])
    assert captured["environment"]["UNRELATED"] == "preserved"
    assert captured["environment"]["MLFLOW_BACKEND_STORE_URI"] == (
        mlflow_server.build_backend_store_uri(source)
    )


def test_launcher_errors_do_not_expose_database_secrets() -> None:
    source = _environment()

    def fail_exec(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("executable unavailable")

    with pytest.raises(mlflow_server.MlflowServerError) as raised:
        mlflow_server.launch_mlflow_server(environment=source, exec_function=fail_exec)

    message = str(raised.value)
    assert "Unable to start the MLflow server" in message
    assert source["POSTGRES_PASSWORD"] not in message
    assert quote(source["POSTGRES_PASSWORD"], safe="") not in message


def test_launcher_honors_explicit_empty_environment(monkeypatch) -> None:
    for name, value in _environment().items():
        monkeypatch.setenv(name, value)

    with pytest.raises(mlflow_server.MlflowServerError, match="variables are missing"):
        mlflow_server.launch_mlflow_server(environment={}, exec_function=os.execvpe)


def test_main_returns_controlled_error_without_traceback(monkeypatch, capsys) -> None:
    for name in (
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "MLFLOW_ARTIFACT_BUCKET",
    ):
        monkeypatch.delenv(name, raising=False)

    assert mlflow_server.main() == 1
    captured = capsys.readouterr()
    assert "Required MLflow server environment variables are missing" in captured.err
    assert "Traceback" not in captured.err
