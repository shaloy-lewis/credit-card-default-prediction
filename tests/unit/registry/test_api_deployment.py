from __future__ import annotations

from pathlib import Path

from credit_risk.inference import api


def test_startup_bundle_precedence(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    active = tmp_path / "active"
    monkeypatch.setenv("CREDIT_RISK_DEPLOYMENT_ROOT", str(tmp_path / "deployment"))
    monkeypatch.setattr(api, "resolve_active_bundle", lambda _root: active)

    assert api._resolve_startup_bundle(explicit) == explicit
    assert api._resolve_startup_bundle(None) == active

    monkeypatch.delenv("CREDIT_RISK_DEPLOYMENT_ROOT")
    assert api._resolve_startup_bundle(None) == Path("models/selected_v1")
