import logging

import pandas as pd

from fastapi import APIRouter, Request

from heart_disease.api.schemas import (
    PatientData,
    PredictionRequest,
    PredictionResult,
    PredictionResponse,
    RetrainRequest,
    RetrainResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# === Auxiliary functions ===
def _request_to_dataframe(patient_data: list[PatientData]):
    """Convert list of PatientData to a DataFrame for model prediction."""

    data = [patient.dict(by_alias=True) for patient in patient_data]
    df = pd.DataFrame(data)
    return df

# === API endpoints ===
@router.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["Prediction"],
        summary="Predict heart disease based on patient data",)
async def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
    """Endpoint to predict heart disease based on patient data."""
    logger.info(f"Received prediction request from {request.client.host} with {len(body.patient_data)} patients")
    
    # Placeholder for actual prediction logic
    df = _request_to_dataframe(body.patient_data)
    logger.debug(f"Converted request data to DataFrame:\n{df.head()}")
    # Here you would load your model and make predictions using the DataFrame
    # For demonstration, we'll return dummy predictions
    predictions = []
    for i, patient in enumerate(body.patient_data):
        predictions.append(PredictionResult(
            patient_id=i,
            probability=0.5  # Dummy probability, replace with actual model prediction
        ))
    return PredictionResponse(predictions=predictions)

@router.post(
        "/retrain",
        response_model=RetrainResponse,
        tags=["Retraining"],
        summary="Trigger model retraining with specified configuration",)
async def retrain(body: RetrainRequest, request: Request) -> RetrainResponse:
    """Endpoint to trigger model retraining with specified configuration."""
    logger.info(f"Received retrain request from {request.client.host} with n_iter={body.n_iter} and cv_splits={body.cv_splits}")
    
    # Placeholder for actual retraining logic
    # Here you would implement the logic to retrain your model using the provided configuration
    # For demonstration, we'll return a dummy response
    return RetrainResponse(
        status="Retraining started",
        model_uri="s3://your-bucket/path/to/new/model.pkl"
    )


