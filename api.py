from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.serving import SelectedPredictPipeline

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "credit risk early-warning api"}


@router.get("/ping", summary="Liveness check")
def ping():
    return {"message": "Health check successful!"}


def get_pipeline(request: Request) -> SelectedPredictPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if not isinstance(pipeline, SelectedPredictPipeline):
        raise HTTPException(status_code=503, detail="Inference service is not ready")
    return pipeline


@router.get("/ready", summary="Selected-model readiness check")
def ready(request: Request):
    get_pipeline(request)
    return {"status": "ready"}


class CreditRiskRequest(BaseModel):
    """Canonical operational features available at the monthly scoring cutoff."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    credit_limit_ntd: int = Field(gt=0)
    repayment_status_lag_0: int = Field(ge=-2, le=9)
    repayment_status_lag_1: int = Field(ge=-2, le=9)
    repayment_status_lag_2: int = Field(ge=-2, le=9)
    repayment_status_lag_3: int = Field(ge=-2, le=9)
    repayment_status_lag_4: int = Field(ge=-2, le=9)
    repayment_status_lag_5: int = Field(ge=-2, le=9)
    bill_amount_ntd_lag_0: int
    bill_amount_ntd_lag_1: int
    bill_amount_ntd_lag_2: int
    bill_amount_ntd_lag_3: int
    bill_amount_ntd_lag_4: int
    bill_amount_ntd_lag_5: int
    payment_amount_ntd_lag_0: int = Field(ge=0)
    payment_amount_ntd_lag_1: int = Field(ge=0)
    payment_amount_ntd_lag_2: int = Field(ge=0)
    payment_amount_ntd_lag_3: int = Field(ge=0)
    payment_amount_ntd_lag_4: int = Field(ge=0)
    payment_amount_ntd_lag_5: int = Field(ge=0)


class CreditRiskResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    probability_of_default: float
    risk_band: Literal["standard", "elevated", "high", "critical"]
    model_id: Literal["catboost_fixed"]
    bundle_id: Literal["selected_v1"]


@router.post("/predict", response_model=CreditRiskResponse)
def predict_default(data: CreditRiskRequest, request: Request) -> CreditRiskResponse:
    pipeline = get_pipeline(request)
    try:
        features = pd.DataFrame([data.model_dump()], columns=PREDICTOR_COLUMNS)
        probability, band = pipeline.predict(features)
        return CreditRiskResponse(
            probability_of_default=round(probability, 6),
            risk_band=band,
            model_id="catboost_fixed",
            bundle_id="selected_v1",
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def create_app(bundle_root: str | Path = Path("models/selected_v1")) -> FastAPI:
    """Create an API that fails startup when the reviewed bundle is invalid."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.pipeline = SelectedPredictPipeline(bundle_root=bundle_root)
        try:
            yield
        finally:
            application.state.pipeline = None

    application = FastAPI(
        title="Credit Risk Early-Warning API",
        description=(
            "Scores next-month default risk from 19 operational account-history features. "
            "This portfolio demonstration must not be used for adverse credit decisions."
        ),
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
