from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel

from credit_risk.pipeline.prediction_pipeline import CustomData, PredictPipeline

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "credit card default prediction api"}


@router.get("/ping", summary="Liveness check")
def ping():
    return {"message": "Health check successful!"}


def get_pipeline(request: Request) -> PredictPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if not isinstance(pipeline, PredictPipeline):
        raise HTTPException(status_code=503, detail="Inference service is not ready")
    return pipeline


@router.get("/ready", summary="Inference readiness check")
def ready(request: Request):
    get_pipeline(request)
    return {"status": "ready"}


class CreditData(BaseModel):
    LIMIT_BAL: int = 1000000
    AGE: int = 29
    BILL_AMT1: float = 4000
    BILL_AMT2: float = 4000
    BILL_AMT3: float = 4000
    BILL_AMT4: float = 4000
    BILL_AMT5: float = 4000
    BILL_AMT6: float = 4000
    PAY_AMT1: float = 1500
    PAY_AMT2: float = 1500
    PAY_AMT3: float = 1500
    PAY_AMT4: float = 1500
    PAY_AMT5: float = 1500
    PAY_AMT6: float = 1500
    EDUCATION: str = "graduate_school"
    MARRIAGE: str = "married"
    SEX: str = "female"
    PAY_0: str = "bill_payment_delay"
    PAY_2: str = "revolving_credit"
    PAY_3: str = "bill_paid"
    PAY_4: str = "bill_paid"
    PAY_5: str = "bill_paid"
    PAY_6: str = "bill_paid"


@router.post("/predict")
def predict_default(data: CreditData, request: Request):
    try:
        custom_data = CustomData(
            LIMIT_BAL=data.LIMIT_BAL,
            AGE=data.AGE,
            BILL_AMT1=data.BILL_AMT1,
            BILL_AMT2=data.BILL_AMT2,
            BILL_AMT3=data.BILL_AMT3,
            BILL_AMT4=data.BILL_AMT4,
            BILL_AMT5=data.BILL_AMT5,
            BILL_AMT6=data.BILL_AMT6,
            PAY_AMT1=data.PAY_AMT1,
            PAY_AMT2=data.PAY_AMT2,
            PAY_AMT3=data.PAY_AMT3,
            PAY_AMT4=data.PAY_AMT4,
            PAY_AMT5=data.PAY_AMT5,
            PAY_AMT6=data.PAY_AMT6,
            EDUCATION=data.EDUCATION,
            MARRIAGE=data.MARRIAGE,
            SEX=data.SEX,
            PAY_0=data.PAY_0,
            PAY_2=data.PAY_2,
            PAY_3=data.PAY_3,
            PAY_4=data.PAY_4,
            PAY_5=data.PAY_5,
            PAY_6=data.PAY_6,
        )

        # Convert to DataFrame
        features_df = custom_data.get_data_as_dataframe()

        # Reuse the artifact bundle validated during application startup.
        pipeline = get_pipeline(request)
        prediction = pipeline.predict(features_df)

        # Compute global feature importance
        global_importance = pipeline.get_global_feature_importance()
        global_importance_dict = {
            col: imp
            for col, imp in zip(
                pipeline.preprocessor.get_feature_names_out(),
                global_importance,
                strict=True,
            )
        }
        # global_importance_dict = {f'feature_{i}': imp for i, imp in enumerate(global_importance)}
        global_importance_dict = dict(
            sorted(global_importance_dict.items(), reverse=True, key=lambda item: item[1])
        )

        # Compute instance-specific feature importance using SHAP
        shap_values = pipeline.get_instance_feature_importance(features_df)
        # Preserve the legacy response until transformed SHAP values are mapped and
        # aggregated to reviewed raw-feature reason categories in the governance phase.
        instance_importance_dict = {
            col: imp for col, imp in zip(features_df.columns, shap_values[0], strict=False)
        }
        instance_importance_dict = dict(
            sorted(instance_importance_dict.items(), reverse=True, key=lambda item: item[1])
        )

        # Return the prediction result
        return {
            "probability_of_default": round(prediction[0][1], 6),
            "instance_feature_importance": instance_importance_dict,
            "global_feature_importance": global_importance_dict,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def create_app(artifact_dir: str | Path = Path("artifacts")) -> FastAPI:
    """Create an API that fails startup when its inference artifacts are invalid."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.pipeline = PredictPipeline(artifact_dir=artifact_dir)
        try:
            yield
        finally:
            application.state.pipeline = None

    application = FastAPI(
        title="Credit Card Default Prediction",
        description="Predicts the member's probability of defaulting on their credit card bills using their credit history and demographics",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
