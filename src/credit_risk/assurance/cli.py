"""Consistent controlled failures for additive assurance commands."""

from collections.abc import Callable
from typing import Any

import typer

from credit_risk.assurance.evidence import encode


def invoke(function: Callable[..., Any], **kwargs: Any) -> None:
    try:
        result = function(**kwargs)
        typer.echo(encode(result).decode().strip())
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        typer.echo(f"Assurance operation failed: {error}", err=True)
        raise typer.Exit(1) from None
