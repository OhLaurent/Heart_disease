from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Patient data schema
class PatientData(BaseModel):
    """Schema for patient data used in heart disease prediction."""
    Age: int = Field(..., ge=1, le=120, description="Age in years")
    Sex: str = Field(..., description="'male' or 'female'")
    Chest_pain_type: int = Field(..., alias="Chest pain type", ge=1, le=4)
    BP: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    Cholesterol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dL)")
    FBS_over_120: bool = Field(..., alias="FBS over 120", description="Fasting blood sugar > 120 mg/dL")
    EKG_results: int = Field(..., alias="EKG results", ge=0, le=2)
    Max_HR: int = Field(..., alias="Max HR", ge=60, le=220, description="Maximum heart rate achieved")
    Exercise_angina: str = Field(..., alias="Exercise angina", description="'yes' or 'no'")
    ST_depression: float = Field(..., alias="ST depression", ge=0.0, le=10.0)
    Slope_of_ST: int = Field(..., alias="Slope of ST", ge=1, le=3)
    Number_of_vessels_fluro: int = Field(..., alias="Number of vessels fluro", ge=0, le=3)
    Thallium: int = Field(..., ge=3, le=7)

    @field_validator("Sex")
    @classmethod
    def sex_must_be_valid(cls, v: str) -> str:
        if v not in {"male", "female"}:
            raise ValueError("Sex must be 'male' or 'female'")
        return v

    @field_validator("Exercise_angina")
    @classmethod
    def exercise_angina_must_be_valid(cls, v: str) -> str:
        if v not in {"yes", "no"}:
            raise ValueError("Exercise angina must be 'yes' or 'no'")
        return v

# Prediction request schema
class PredictionRequest(BaseModel):
    """Schema for prediction request containing patient data."""
    patient_data: list[PatientData] = Field(..., description="List of patient data for prediction")

# Prediction result schema
class PredictionResult(BaseModel):
    """Prediction for a single patient."""
    patient_id: int = Field(..., description="ID of the patient")
    probability: float = Field(..., description="Predicted probability of heart disease", ge=0.0, le=1.0) 

# Prediction response schema
class PredictionResponse(BaseModel):
    """Schema for prediction response containing results for multiple patients."""
    predictions: list[PredictionResult] = Field(..., description="List of prediction results for each patient")

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