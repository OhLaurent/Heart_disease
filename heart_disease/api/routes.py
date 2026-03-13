import logging

import pandas as pd

from fastapi import APIRouter, Request, HTTPException

from heart_disease.constants import (
    DRIFT_KS_P_THRESHOLD,
    DRIFT_TV_THRESHOLD,
    HIGH_RISK_PROBABILITY_THRESHOLD,
    MIN_PREDICTIONS_FOR_DRIFT,
    POSITIVE_TARGET_LABEL,
    RISK_LEVEL_THRESHOLDS_PCT,
)
from heart_disease.api.drift_monitor import drift_report_for_model
from heart_disease.api.prediction_store import PredictionStore
from heart_disease.api.schemas import (
    AppConfigResponse,
    PredictionDriftResponse,
    PredictionHistoryEntry,
    PredictionHistoryResponse,
    PredictionModelOption,
    PatientData,
    PredictionRequest,
    PredictionResult,
    PredictionResponse,
    RetrainRequest,
    RetrainResponse,
)

from heart_disease.pipelines.predict import get_model_reference, predict_patients
from heart_disease.pipelines.train import train_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()
prediction_store = PredictionStore()

# === Auxiliary functions ===
def _request_to_dataframe(patient_data: list[PatientData]) -> pd.DataFrame:
    """Convert list of PatientData to a DataFrame for model prediction.
    
    The request payload mirrors the production inference schema.
    This function adds only the synthetic `id` column required by the pipeline.
    """
    data = [patient.model_dump(by_alias=True) for patient in patient_data]
    df = pd.DataFrame(data)
    
    # Add ID column (required by pipeline, will be dropped during transformation)
    df.insert(0, 'id', range(len(df)))
    
    return df

# === API endpoints ===
@router.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["Prediction"],
        summary="Predict heart disease based on patient data",)
async def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
    """Endpoint to predict heart disease based on patient data.
    
    Uses the currently active model from MLflow to make predictions.
    Returns probability of heart disease presence for each patient.
    """
    logger.info(f"Received prediction request from {request.client.host} with {len(body.patient_data)} patients")
    
    try:
        request_inputs = [patient.model_dump(by_alias=True) for patient in body.patient_data]

        # Convert request to DataFrame
        df = _request_to_dataframe(body.patient_data)
        logger.debug(f"Converted request data to DataFrame with shape {df.shape}")
        
        # Make predictions using active model
        prediction_run = predict_patients(df, return_proba=True, include_input=False)
        results = prediction_run.predictions
        logger.info(f"Predictions completed successfully for {len(results)} patients")
        
        # Convert pipeline results to API response format
        predictions = []
        persisted_outputs = []
        for idx, row in results.iterrows():
            prediction = PredictionResult(
                patient_id=int(idx),
                prediction=str(row['prediction']),
                probability=float(row[f'probability_{POSITIVE_TARGET_LABEL}'])
            )
            predictions.append(prediction)
            persisted_outputs.append({
                "prediction": prediction.prediction,
                "probability": prediction.probability,
            })

        prediction_store.save_prediction_run(
            inputs=request_inputs,
            outputs=persisted_outputs,
            model_version=prediction_run.model.version,
            model_uri=prediction_run.model.uri,
        )
        
        return PredictionResponse(
            model_version=prediction_run.model.version,
            model_uri=prediction_run.model.uri,
            predictions=predictions,
        )
    
    except ValueError as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@router.get(
        "/predictions/history",
        response_model=PredictionHistoryResponse,
        tags=["Prediction"],
        summary="List persisted predictions optionally filtered by model version",)
async def prediction_history(model_version: str | None = None) -> PredictionHistoryResponse:
    """Return persisted prediction history and available model filters."""
    records = prediction_store.list_predictions(model_version=model_version)
    model_rows = prediction_store.list_models()

    active_model_version: str | None = None
    active_model_uri: str | None = None
    try:
        active_model = get_model_reference()
        active_model_version = active_model.version
        active_model_uri = active_model.uri
    except ValueError:
        if model_rows:
            # Fallback to latest persisted model when MLflow active alias is unavailable.
            active_model_version = model_rows[0]["model_version"]
            active_model_uri = model_rows[0]["model_uri"]

    models = []
    seen_versions = set()
    for model_row in model_rows:
        is_active = active_model_version is not None and model_row["model_version"] == active_model_version
        models.append(PredictionModelOption(
            model_version=model_row["model_version"],
            model_uri=model_row["model_uri"],
            prediction_count=model_row["prediction_count"],
            latest_prediction_at=model_row["latest_prediction_at"],
            is_active=is_active,
        ))
        seen_versions.add(model_row["model_version"])

    if active_model_version is not None and active_model_version not in seen_versions:
        models.insert(0, PredictionModelOption(
            model_version=active_model_version,
            model_uri=active_model_uri or "",
            prediction_count=0,
            latest_prediction_at=None,
            is_active=True,
        ))

    return PredictionHistoryResponse(
        active_model_version=active_model_version,
        active_model_uri=active_model_uri,
        models=models,
        predictions=[PredictionHistoryEntry(**record) for record in records],
    )


@router.get(
        "/predictions/drift",
        response_model=PredictionDriftResponse,
        tags=["Prediction"],
        summary="Compute prediction drift for a model version",)
async def prediction_drift(model_version: str | None = None) -> PredictionDriftResponse:
    """Return feature drift report for persisted predictions of one model version."""
    selected_model_version = model_version

    if selected_model_version is None:
        try:
            selected_model_version = get_model_reference().version
        except ValueError:
            model_rows = prediction_store.list_models()
            if model_rows:
                selected_model_version = model_rows[0]["model_version"]
            else:
                raise HTTPException(status_code=404, detail="No model available for drift report")

    records = prediction_store.list_predictions(model_version=selected_model_version)

    try:
        report = drift_report_for_model(records=records, model_version=selected_model_version)
    except Exception as e:
        logger.warning("Drift report fallback for model %s: %s", selected_model_version, str(e))
        model_uri = records[0]["model_uri"] if records else f"models:/heart_disease_model/{selected_model_version}"
        report = {
            "model_version": str(selected_model_version),
            "model_uri": model_uri,
            "min_predictions_required": MIN_PREDICTIONS_FOR_DRIFT,
            "sample_size": len(records),
            "has_enough_data": len(records) >= MIN_PREDICTIONS_FOR_DRIFT,
            "overall_status": "baseline_unavailable",
            "performance_summary": {
                "total_predictions": len(records),
                "mean_probability": None,
                "high_risk_pct": None,
            },
            "features": [],
        }

    return PredictionDriftResponse(**report)


@router.get(
        "/config",
        response_model=AppConfigResponse,
        tags=["Configuration"],
        summary="Get runtime threshold configuration",)
async def app_config() -> AppConfigResponse:
    """Expose UI/server thresholds from constants in one place."""
    return AppConfigResponse(
        risk_levels_pct=RISK_LEVEL_THRESHOLDS_PCT,
        high_risk_probability_threshold=HIGH_RISK_PROBABILITY_THRESHOLD,
        min_predictions_for_drift=MIN_PREDICTIONS_FOR_DRIFT,
        drift_ks_p_threshold=DRIFT_KS_P_THRESHOLD,
        drift_tv_threshold=DRIFT_TV_THRESHOLD,
    )

@router.post(
        "/retrain",
        response_model=RetrainResponse,
        tags=["Retraining"],
        summary="Trigger model retraining with specified configuration",)
async def retrain(body: RetrainRequest, request: Request) -> RetrainResponse:
    """Endpoint to trigger model retraining with specified configuration.
    
    Trains a new model using the configured training data and hyperparameter search.
    The new model is automatically promoted to 'active' if it outperforms the current
    active model or if force_replacement=True.
    """
    logger.info(
        f"Received retrain request from {request.client.host} "
        f"with n_iter={body.n_iter}, cv_splits={body.cv_splits}, "
        f"force_replacement={body.force_replacement}"
    )
    
    try:
        # Run training pipeline
        results = train_pipeline(
            n_iter=body.n_iter,
            cv_folds=body.cv_splits,
            force_replace=body.force_replacement
        )
        
        logger.info(
            f"Training completed. Run ID: {results['run_id']}, "
            f"Promoted: {results['promoted']}"
        )
        
        # Build response
        cv_auc = results['metrics'].get('cv_roc_auc')
        test_auc = results['metrics'].get('test_roc_auc')
        
        if results['promoted']:
            status = "success"
            message = (
                f"Model trained and promoted to 'active'. "
                f"CV ROC-AUC: {cv_auc:.4f}, Test ROC-AUC: {test_auc:.4f}"
            )
        else:
            status = "success"
            message = (
                f"Model trained but NOT promoted (existing model performs better). "
                f"CV ROC-AUC: {cv_auc:.4f}, Test ROC-AUC: {test_auc:.4f}"
            )
        
        return RetrainResponse(
            status=status,
            model_uri=f"models:/{results['run_id']}",
            cv_mean_auc=cv_auc,
            message=message
        )
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {str(e)}"
        )


