"""ASGI compatibility entry point for the packaged versioned inference API."""

from credit_risk.inference.api import app, create_app

__all__ = ["app", "create_app"]
