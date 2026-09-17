"""Versioned HTTP inference interface backed by the shared Phase 6 engine."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, cast

import pandas as pd
from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, Response

from credit_risk.inference.contracts import (
    ACCOUNT_ID_PATTERN,
    CreditRiskResponse,
    OperationalFeatures,
    ReasonResponse,
)
from credit_risk.inference.engine import InferenceEngine
from credit_risk.inference.logging import emit_event
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.registry.deployment import resolve_active_bundle

router = APIRouter()


@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "credit risk early-warning api"}


@router.get("/ping", summary="Liveness check")
def ping() -> dict[str, str]:
    return {"message": "Health check successful!"}


def get_engine(request: Request) -> InferenceEngine:
    engine = getattr(request.app.state, "engine", None)
    if not isinstance(engine, InferenceEngine):
        raise HTTPException(status_code=503, detail="Inference service is not ready")
    return engine


@router.get("/ready", summary="Selected-model readiness check")
def ready(request: Request) -> dict[str, str]:
    get_engine(request)
    return {"status": "ready"}


class CreditRiskRequest(OperationalFeatures):
    """Canonical operational features available at the monthly scoring cutoff."""


@router.post("/v1/predict", response_model=CreditRiskResponse)
def predict_default_v1(
    data: CreditRiskRequest,
    request: Request,
    response: Response,
    request_id: Annotated[
        str | None,
        Header(alias="X-Request-ID", pattern=ACCOUNT_ID_PATTERN),
    ] = None,
) -> CreditRiskResponse:
    trace_id = request_id or uuid.uuid4().hex
    response.headers["X-Trace-ID"] = trace_id
    try:
        engine = get_engine(request)
    except HTTPException as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
            headers={"X-Trace-ID": trace_id},
        ) from None
    started = time.perf_counter()
    try:
        features = pd.DataFrame([data.model_dump()], columns=PREDICTOR_COLUMNS)
        result = engine.score(features)
        reasons = tuple(
            ReasonResponse(
                category=cast(AnyReasonCategory, reason.category),
                direction=cast(AnyReasonDirection, reason.direction),
                contribution_raw_log_odds=round(reason.contribution_raw_log_odds, 6),
            )
            for reason in result.reasons[0]
        )
        payload = CreditRiskResponse(
            schema_version="1.0.0",
            trace_id=trace_id,
            probability_of_default=round(float(result.probabilities[0]), 6),
            risk_band=result.risk_bands[0],
            reasons=cast(tuple[ReasonResponse, ReasonResponse], reasons),
            model_id="catboost_fixed",
            bundle_id="selected_v1",
            manifest_sha256=engine.config.bundle.manifest_sha256,
            policy_id="outreach_top_10_v1",
        )
    except Exception:
        emit_event(
            "api_prediction_failed",
            route="/v1/predict",
            status="error",
            trace_id=trace_id,
            model_id=engine.config.bundle.model_id,
            bundle_id=engine.config.bundle.bundle_id,
            policy_id=engine.config.policy.policy_id,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed; trace_id={trace_id}",
            headers={"X-Trace-ID": trace_id},
        ) from None
    emit_event(
        "api_prediction_completed",
        route="/v1/predict",
        status="completed",
        trace_id=trace_id,
        model_id=engine.config.bundle.model_id,
        bundle_id=engine.config.bundle.bundle_id,
        policy_id=engine.config.policy.policy_id,
        duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )
    return payload


AnyReasonCategory = Literal[
    "billing_balance", "credit_capacity", "payment_behaviour", "repayment_status"
]
AnyReasonDirection = Literal["risk_increasing", "risk_mitigating", "neutral"]


def create_app(
    bundle_root: str | Path | None = None,
    config_path: str | Path = Path("configs/inference/phase6_v1.json"),
) -> FastAPI:
    """Create an API that fails startup when the reviewed release is invalid."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        selected_bundle = _resolve_startup_bundle(bundle_root)
        application.state.engine = InferenceEngine(
            bundle_root=selected_bundle,
            config_path=config_path,
        )
        try:
            yield
        finally:
            application.state.engine = None

    application = FastAPI(
        title="Credit Risk Early-Warning API",
        version="1.0.0",
        description=(
            "Scores next-month default risk from 19 operational account-history features. "
            "This portfolio demonstration must not be used for adverse credit decisions."
        ),
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


def _resolve_startup_bundle(explicit_bundle_root: str | Path | None) -> Path:
    """Apply explicit, governed-deployment, then committed-bundle precedence."""

    if explicit_bundle_root is not None:
        return Path(explicit_bundle_root)
    deployment_root = os.environ.get("CREDIT_RISK_DEPLOYMENT_ROOT")
    if deployment_root:
        return resolve_active_bundle(deployment_root)
    return Path("models/selected_v1")


app = create_app()
