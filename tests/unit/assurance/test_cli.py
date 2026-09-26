"""Additive CLI entrypoints expose controlled failures without changing older commands."""

import pytest
import typer
from typer.testing import CliRunner

from credit_risk.assurance.cli import invoke
from credit_risk.assurance.evidence import EvidenceError
from credit_risk.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "args",
    [
        ["robustness", "build"],
        ["robustness", "verify", "--expected-manifest-sha256", "a" * 64],
        ["monitor", "reference"],
        [
            "monitor",
            "batch",
            "--input",
            "input",
            "--run-root",
            "batch",
            "--expected-reference-sha256",
            "a" * 64,
            "--output",
            "out",
        ],
        ["monitor", "service", "--log-path", "log", "--output", "out"],
        [
            "monitor",
            "verify",
            "--evidence-root",
            "root",
            "--expected-manifest-sha256",
            "a" * 64,
            "--kind",
            "reference",
        ],
        ["monitor", "benchmark"],
        ["monitor", "acceptance", "--expected-benchmark-sha256", "a" * 64],
        ["monitor", "drill", "--expected-benchmark-sha256", "a" * 64],
        ["release", "capture-ci"],
        ["platform", "rehearse"],
        ["platform", "publish-evidence"],
        ["platform", "verify-evidence", "--expected-manifest-sha256", "a" * 64],
        ["release", "build-b", "--expected-ci-sha256", "a" * 64],
        ["release", "verify-b", "--expected-manifest-sha256", "a" * 64],
    ],
)
def test_commands_forward_to_workflows(args, monkeypatch):
    import credit_risk.assurance.cli as shared
    import credit_risk.monitoring.cli as monitoring
    import credit_risk.robustness.cli as robustness

    calls = []

    def record(function, **kwargs):
        calls.append((function, kwargs))

    for module in (shared, robustness, monitoring):
        monkeypatch.setattr(module, "invoke", record)
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(calls) == 1


def test_invocation_errors_and_json(capsys):
    invoke(lambda: {"status": "ok"})
    assert '"status": "ok"' in capsys.readouterr().out
    with pytest.raises(typer.Exit):
        invoke(lambda: (_ for _ in ()).throw(EvidenceError("blocked")))
    assert "blocked" in capsys.readouterr().err
