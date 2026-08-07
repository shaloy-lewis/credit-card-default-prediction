FROM python:3.12.10-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM python:3.12.10-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app && \
    adduser --system --ingroup app --home /app app && \
    mkdir -p /app/logs && \
    chown app:app /app/logs

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app api.py ./api.py
COPY --chown=app:app artifacts/model.pkl artifacts/preprocessor.pkl artifacts/outlier_threshold.json ./artifacts/

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=3)"]

CMD ["uvicorn", "api:app", "--workers=1", "--host=0.0.0.0", "--port=8080", "--no-access-log"]
