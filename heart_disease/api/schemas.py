from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# Patient data schema
class PatientData(BaseModel):
    """Schema for patient data used in heart disease prediction."""
    Age: int = Field(..., ge=1, le=120, description="Age in years")
    Sex: int = Field(..., ge=0, le=1, description="Sex code: 1=male, 0=female")
    Chest_pain_type: int = Field(..., alias="Chest pain type", ge=1, le=4)
    BP: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    Cholesterol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dL)")
    FBS_over_120: bool = Field(..., alias="FBS over 120", description="Fasting blood sugar > 120 mg/dL")
    EKG_results: int = Field(..., alias="EKG results", ge=0, le=2)
    Max_HR: int = Field(..., alias="Max HR", ge=60, le=220, description="Maximum heart rate achieved")
    Exercise_angina: int = Field(..., alias="Exercise angina", ge=0, le=1, description="Exercise angina code: 1=yes, 0=no")
    ST_depression: float = Field(..., alias="ST depression", ge=0.0, le=10.0)
    Slope_of_ST: int = Field(..., alias="Slope of ST", ge=1, le=3)
    Number_of_vessels_fluro: int = Field(..., alias="Number of vessels fluro", ge=0, le=3)
    Thallium: int = Field(..., description="Allowed values: 3, 6, 7")

    @field_validator("Thallium")
    @classmethod
    def thallium_must_be_valid(cls, v: int) -> int:
        if v not in {3, 6, 7}:
            raise ValueError("Thallium must be one of: 3, 6, 7")
        return v

# Prediction request schema
class PredictionRequest(BaseModel):
    """Schema for prediction request containing patient data."""
    patient_data: list[PatientData] = Field(..., description="List of patient data for prediction")

# Prediction result schema
class PredictionResult(BaseModel):
    """Prediction for a single patient."""
    patient_id: int = Field(..., description="ID of the patient")
    prediction: str = Field(..., description="Predicted class label")
    probability: float = Field(..., description="Predicted probability of heart disease", ge=0.0, le=1.0) 

# Prediction response schema
class PredictionResponse(BaseModel):
    """Schema for prediction response containing results for multiple patients."""
    model_version: str = Field(..., description="Version of the model used for prediction")
    model_uri: str = Field(..., description="MLflow URI of the model used for prediction")
    predictions: list[PredictionResult] = Field(..., description="List of prediction results for each patient")


class PredictionHistoryEntry(BaseModel):
    """Persisted prediction event returned by the history endpoint."""
    id: int
    created_at: str
    request_id: str
    patient_index: int
    model_version: str
    model_uri: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]


class PredictionModelOption(BaseModel):
    """Model metadata available for filtering prediction history."""
    model_version: str
    model_uri: str
    prediction_count: int = 0
    latest_prediction_at: str | None = None
    is_active: bool = False


class PredictionHistoryResponse(BaseModel):
    """Response payload for prediction history listing."""
    active_model_version: str | None = None
    active_model_uri: str | None = None
    models: list[PredictionModelOption]
    predictions: list[PredictionHistoryEntry]

class RetrainRequest(BaseModel):
    """Configuration for the training pipeline."""
    n_iter: int = Field(20, ge=1, description="RandomizedSearchCV iterations")
    cv_splits: int = Field(5, ge=2, description="CV folds")
    force_replacement: bool = Field(False, description="Whether to force replacement of existing model as the \"active\" model even if the new model does not outperform it")


class RetrainResponse(BaseModel):
    """Outcome of a retrain run."""
    status: str
    model_uri: str
    cv_mean_auc: float | None = None
    message: str | None = None