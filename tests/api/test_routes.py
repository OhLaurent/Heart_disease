"""Tests for heart_disease.api.routes module."""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import pandas as pd

from heart_disease.api.app import app
from heart_disease.api.routes import _request_to_dataframe
from heart_disease.api.schemas import PatientData


class TestRequestToDataframe:
    """Tests for _request_to_dataframe helper function."""

    def test_converts_patient_data_to_dataframe(self):
        """Test conversion of PatientData list to DataFrame."""
        patients = [
            PatientData(**{
                "Age": 55,
                "Sex": 1,
                "Chest pain type": 2,
                "BP": 130,
                "Cholesterol": 240,
                "FBS over 120": True,
                "EKG results": 1,
                "Max HR": 150,
                "Exercise angina": 1,
                "ST depression": 1.5,
                "Slope of ST": 2,
                "Number of vessels fluro": 1,
                "Thallium": 6
            }),
            PatientData(**{
                "Age": 62,
                "Sex": 0,
                "Chest pain type": 3,
                "BP": 140,
                "Cholesterol": 260,
                "FBS over 120": False,
                "EKG results": 0,
                "Max HR": 135,
                "Exercise angina": 0,
                "ST depression": 2.0,
                "Slope of ST": 1,
                "Number of vessels fluro": 2,
                "Thallium": 7
            })
        ]
        
        df = _request_to_dataframe(patients)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'id' in df.columns
        assert df['id'].tolist() == [0, 1]
        assert 'Age' in df.columns
        assert 'Sex' in df.columns

    def test_preserves_fbs_bool(self):
        """Test that FBS over 120 remains boolean in request DataFrame."""
        patients = [
            PatientData(**{
                "Age": 55, "Sex": 1, "Chest pain type": 2, "BP": 130,
                "Cholesterol": 240, "FBS over 120": True, "EKG results": 1,
                "Max HR": 150, "Exercise angina": 1, "ST depression": 1.5,
                "Slope of ST": 2, "Number of vessels fluro": 1, "Thallium": 6
            })
        ]
        
        df = _request_to_dataframe(patients)
        
        assert bool(df['FBS over 120'].iloc[0]) is True

    def test_preserves_sex_and_exercise_angina_codes(self):
        """Test that Sex and Exercise angina remain as coded integers."""
        patients = [
            PatientData(**{
                "Age": 55, "Sex": 1, "Chest pain type": 2, "BP": 130,
                "Cholesterol": 240, "FBS over 120": False, "EKG results": 1,
                "Max HR": 150, "Exercise angina": 1, "ST depression": 1.5,
                "Slope of ST": 2, "Number of vessels fluro": 1, "Thallium": 6
            })
        ]
        
        df = _request_to_dataframe(patients)
        
        assert df['Sex'].iloc[0] == 1
        assert df['Exercise angina'].iloc[0] == 1


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    @patch('heart_disease.api.routes.prediction_store')
    @patch('heart_disease.api.routes.predict_patients')
    def test_predict_success(self, mock_predict, mock_store):
        """Test successful prediction."""
        # Setup mock
        mock_results = pd.DataFrame({
            'prediction': ['Presence', 'Absence'],
            'probability_Absence': [0.3, 0.7],
            'probability_Presence': [0.7, 0.3]
        })
        mock_predict.return_value = Mock(
            predictions=mock_results,
            model=Mock(version='3', uri='models:/heart_disease_model@active')
        )
        
        client = TestClient(app)
        response = client.post("/api/v1/predict", json={
            "patient_data": [
                {
                    "Age": 55, "Sex": 1, "Chest pain type": 2,
                    "BP": 130, "Cholesterol": 240, "FBS over 120": True,
                    "EKG results": 1, "Max HR": 150, "Exercise angina": 1,
                    "ST depression": 1.5, "Slope of ST": 2,
                    "Number of vessels fluro": 1, "Thallium": 6
                },
                {
                    "Age": 62, "Sex": 0, "Chest pain type": 3,
                    "BP": 140, "Cholesterol": 260, "FBS over 120": False,
                    "EKG results": 0, "Max HR": 135, "Exercise angina": 0,
                    "ST depression": 2.0, "Slope of ST": 1,
                    "Number of vessels fluro": 2, "Thallium": 7
                }
            ]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
        assert data['model_version'] == '3'
        assert data['model_uri'] == 'models:/heart_disease_model@active'
        assert len(data['predictions']) == 2
        assert data['predictions'][0]['patient_id'] == 0
        assert data['predictions'][0]['prediction'] == 'Presence'
        assert data['predictions'][0]['probability'] == 0.7
        assert data['predictions'][1]['patient_id'] == 1
        assert data['predictions'][1]['prediction'] == 'Absence'
        assert data['predictions'][1]['probability'] == 0.3
        mock_store.save_prediction_run.assert_called_once()

    @patch('heart_disease.api.routes.prediction_store')
    @patch('heart_disease.api.routes.predict_patients')
    def test_predict_success_with_numeric_probability_columns(self, mock_predict, mock_store):
        """Test successful prediction with normalized probability columns."""
        mock_results = pd.DataFrame({
            'prediction': ['Presence', 'Absence'],
            'probability_Absence': [0.3, 0.7],
            'probability_Presence': [0.7, 0.3]
        })
        mock_predict.return_value = Mock(
            predictions=mock_results,
            model=Mock(version='5', uri='models:/heart_disease_model@active')
        )

        client = TestClient(app)
        response = client.post("/api/v1/predict", json={
            "patient_data": [
                {
                    "Age": 55, "Sex": 1, "Chest pain type": 2,
                    "BP": 130, "Cholesterol": 240, "FBS over 120": True,
                    "EKG results": 1, "Max HR": 150, "Exercise angina": 1,
                    "ST depression": 1.5, "Slope of ST": 2,
                    "Number of vessels fluro": 1, "Thallium": 6
                },
                {
                    "Age": 62, "Sex": 0, "Chest pain type": 3,
                    "BP": 140, "Cholesterol": 260, "FBS over 120": False,
                    "EKG results": 0, "Max HR": 135, "Exercise angina": 0,
                    "ST depression": 2.0, "Slope of ST": 1,
                    "Number of vessels fluro": 2, "Thallium": 7
                }
            ]
        })

        assert response.status_code == 200
        data = response.json()
        assert data['model_version'] == '5'
        assert data['predictions'][0]['probability'] == 0.7
        assert data['predictions'][1]['probability'] == 0.3
        mock_store.save_prediction_run.assert_called_once()

    @patch('heart_disease.api.routes.predict_patients')
    def test_predict_with_model_error(self, mock_predict):
        """Test prediction when model is not available."""
        mock_predict.side_effect = ValueError("No model found with alias 'active'")
        
        client = TestClient(app)
        response = client.post("/api/v1/predict", json={
            "patient_data": [
                {
                    "Age": 55, "Sex": 1, "Chest pain type": 2,
                    "BP": 130, "Cholesterol": 240, "FBS over 120": True,
                    "EKG results": 1, "Max HR": 150, "Exercise angina": 1,
                    "ST depression": 1.5, "Slope of ST": 2,
                    "Number of vessels fluro": 1, "Thallium": 6
                }
            ]
        })
        
        assert response.status_code == 503
        assert "No trained model is available" in response.json()['detail']

    def test_predict_with_invalid_data(self):
        """Test prediction with invalid patient data."""
        client = TestClient(app)
        response = client.post("/api/v1/predict", json={
            "patient_data": [
                {
                    "Age": 200,  # Invalid age > 120
                    "Sex": 1,
                    "Chest pain type": 2,
                    "BP": 130,
                    "Cholesterol": 240,
                    "FBS over 120": True,
                    "EKG results": 1,
                    "Max HR": 150,
                    "Exercise angina": 1,
                    "ST depression": 1.5,
                    "Slope of ST": 2,
                    "Number of vessels fluro": 1,
                    "Thallium": 6
                }
            ]
        })
        
        assert response.status_code == 422  # Validation error

    @patch('heart_disease.api.routes.get_model_reference')
    @patch('heart_disease.api.routes.prediction_store')
    def test_prediction_history(self, mock_store, mock_get_model_reference):
        """Test prediction history endpoint with model filter metadata."""
        mock_store.list_predictions.return_value = [
            {
                'id': 1,
                'created_at': '2026-03-12T12:00:00+00:00',
                'request_id': 'req-1',
                'patient_index': 0,
                'model_version': '3',
                'model_uri': 'models:/heart_disease_model@active',
                'input_data': {'Age': 55},
                'output_data': {'prediction': 'Presence', 'probability': 0.7},
            }
        ]
        mock_store.list_models.return_value = [
            {
                'model_version': '3',
                'model_uri': 'models:/heart_disease_model@active',
                'prediction_count': 1,
                'latest_prediction_at': '2026-03-12T12:00:00+00:00',
            }
        ]
        mock_get_model_reference.return_value = Mock(version='3', uri='models:/heart_disease_model@active')

        client = TestClient(app)
        response = client.get('/api/v1/predictions/history?model_version=3')

        assert response.status_code == 200
        data = response.json()
        assert data['active_model_version'] == '3'
        assert len(data['models']) == 1
        assert data['models'][0]['is_active'] is True
        assert len(data['predictions']) == 1
        assert data['predictions'][0]['patient_index'] == 0
        assert 'patient_id' not in data['predictions'][0]['output_data']
        assert data['predictions'][0]['output_data']['prediction'] == 'Presence'
        mock_store.list_predictions.assert_called_once_with(model_version='3')

    @patch('heart_disease.api.routes.drift_report_for_model')
    @patch('heart_disease.api.routes.prediction_store')
    def test_prediction_drift_with_explicit_model(self, mock_store, mock_drift_report):
        """Test drift endpoint using explicit model_version filter."""
        mock_store.list_predictions.return_value = [{
            'id': 1,
            'created_at': '2026-03-12T12:00:00+00:00',
            'request_id': 'req-1',
            'patient_index': 0,
            'model_version': '3',
            'model_uri': 'models:/heart_disease_model/3',
            'input_data': {'Age': 55},
            'output_data': {'prediction': 'Presence', 'probability': 0.7},
        }]
        mock_drift_report.return_value = {
            'model_version': '3',
            'model_uri': 'models:/heart_disease_model/3',
            'min_predictions_required': 20,
            'sample_size': 1,
            'has_enough_data': False,
            'overall_status': 'insufficient_data',
            'performance_summary': {'total_predictions': 1, 'mean_probability': 0.7, 'high_risk_pct': 100.0},
            'features': [],
        }

        client = TestClient(app)
        response = client.get('/api/v1/predictions/drift?model_version=3')

        assert response.status_code == 200
        data = response.json()
        assert data['model_version'] == '3'
        assert data['overall_status'] == 'insufficient_data'
        mock_store.list_predictions.assert_called_once_with(model_version='3')
        mock_drift_report.assert_called_once_with(records=mock_store.list_predictions.return_value, model_version='3')

    @patch('heart_disease.api.routes.get_model_reference')
    @patch('heart_disease.api.routes.drift_report_for_model')
    @patch('heart_disease.api.routes.prediction_store')
    def test_prediction_drift_defaults_to_active_model(self, mock_store, mock_drift_report, mock_get_model_reference):
        """Test drift endpoint default model selection via active model reference."""
        mock_get_model_reference.return_value = Mock(version='5', uri='models:/heart_disease_model/5')
        mock_store.list_predictions.return_value = []
        mock_drift_report.return_value = {
            'model_version': '5',
            'model_uri': 'models:/heart_disease_model/5',
            'min_predictions_required': 20,
            'sample_size': 0,
            'has_enough_data': False,
            'overall_status': 'insufficient_data',
            'performance_summary': {'total_predictions': 0, 'mean_probability': None, 'high_risk_pct': None},
            'features': [],
        }

        client = TestClient(app)
        response = client.get('/api/v1/predictions/drift')

        assert response.status_code == 200
        assert response.json()['model_version'] == '5'
        mock_store.list_predictions.assert_called_once_with(model_version='5')

    def test_config_endpoint(self):
        """Test config endpoint returns monitoring thresholds."""
        client = TestClient(app)
        response = client.get('/api/v1/config')

        assert response.status_code == 200
        data = response.json()
        assert 'risk_levels_pct' in data
        assert data['min_predictions_for_drift'] == 20
        assert 'drift_ks_p_threshold' in data
        assert 'drift_tv_threshold' in data


class TestRetrainEndpoint:
    """Tests for /retrain endpoint."""

    @patch('heart_disease.api.routes.train_pipeline')
    def test_retrain_success_with_promotion(self, mock_train):
        """Test successful retraining with model promotion."""
        mock_train.return_value = {
            'run_id': 'test_run_123',
            'metrics': {
                'cv_roc_auc': 0.87,
                'test_roc_auc': 0.85
            },
            'best_params': {'classifier__C': 1.0},
            'promoted': True
        }
        
        client = TestClient(app)
        response = client.post("/api/v1/retrain", json={
            "n_iter": 50,
            "cv_splits": 5,
            "force_replacement": False
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert 'test_run_123' in data['model_uri']
        assert data['cv_mean_auc'] == 0.87
        assert 'promoted to' in data['message']
        
        mock_train.assert_called_once_with(
            n_iter=50,
            cv_folds=5,
            force_replace=False
        )

    @patch('heart_disease.api.routes.train_pipeline')
    def test_retrain_success_without_promotion(self, mock_train):
        """Test successful retraining without model promotion."""
        mock_train.return_value = {
            'run_id': 'test_run_456',
            'metrics': {
                'cv_roc_auc': 0.75,
                'test_roc_auc': 0.73
            },
            'best_params': {'classifier__C': 0.1},
            'promoted': False
        }
        
        client = TestClient(app)
        response = client.post("/api/v1/retrain", json={
            "n_iter": 20,
            "cv_splits": 3,
            "force_replacement": False
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert 'test_run_456' in data['model_uri']
        assert data['cv_mean_auc'] == 0.75
        assert 'NOT promoted' in data['message']

    @patch('heart_disease.api.routes.train_pipeline')
    def test_retrain_with_defaults(self, mock_train):
        """Test retraining with default parameters."""
        mock_train.return_value = {
            'run_id': 'test_run_789',
            'metrics': {'cv_roc_auc': 0.80, 'test_roc_auc': 0.78},
            'best_params': {},
            'promoted': True
        }
        
        client = TestClient(app)
        # Use schema defaults
        response = client.post("/api/v1/retrain", json={})
        
        assert response.status_code == 200
        mock_train.assert_called_once()

    @patch('heart_disease.api.routes.train_pipeline')
    def test_retrain_failure(self, mock_train):
        """Test retraining when pipeline fails."""
        mock_train.side_effect = RuntimeError("Training failed due to data error")
        
        client = TestClient(app)
        response = client.post("/api/v1/retrain", json={
            "n_iter": 10,
            "cv_splits": 5,
            "force_replacement": False
        })
        
        assert response.status_code == 500
        assert "Training failed" in response.json()['detail']

    def test_retrain_with_invalid_params(self):
        """Test retraining with invalid parameters."""
        client = TestClient(app)
        response = client.post("/api/v1/retrain", json={
            "n_iter": 0,  # Invalid: must be >= 1
            "cv_splits": 5,
            "force_replacement": False
        })
        
        assert response.status_code == 422  # Validation error

    @patch('heart_disease.api.routes.retrain_job_manager')
    def test_retrain_start_job(self, mock_job_manager):
        """Test async retrain job creation endpoint."""
        mock_job_manager.start_job.return_value = {
            'job_id': 'job-123',
            'status': 'queued',
            'message': 'Retraining started. Check job status for progress.',
        }

        client = TestClient(app)
        response = client.post('/api/v1/retrain/jobs', json={
            'n_iter': 20,
            'cv_splits': 5,
            'force_replacement': False,
        })

        assert response.status_code == 200
        payload = response.json()
        assert payload['job_id'] == 'job-123'
        assert payload['status'] == 'queued'
        mock_job_manager.start_job.assert_called_once_with(
            n_iter=20,
            cv_splits=5,
            force_replacement=False,
        )

    @patch('heart_disease.api.routes.retrain_job_manager')
    def test_retrain_job_status(self, mock_job_manager):
        """Test async retrain job status endpoint."""
        mock_job_manager.get_job.return_value = {
            'job_id': 'job-123',
            'status': 'running',
            'stage': 'hyperparameter_search',
            'progress_pct': 65,
            'message': 'Running hyperparameter search...',
            'started_at': '2026-03-13T12:00:00+00:00',
            'updated_at': '2026-03-13T12:00:01+00:00',
            'finished_at': None,
            'elapsed_seconds': 1.25,
            'model_uri': None,
            'cv_mean_auc': None,
            'error': None,
        }

        client = TestClient(app)
        response = client.get('/api/v1/retrain/jobs/job-123')

        assert response.status_code == 200
        payload = response.json()
        assert payload['status'] == 'running'
        assert payload['progress_pct'] == 65
        mock_job_manager.get_job.assert_called_once_with('job-123')

    @patch('heart_disease.api.routes.retrain_job_manager')
    def test_retrain_job_status_not_found(self, mock_job_manager):
        """Test async retrain job status endpoint when job does not exist."""
        mock_job_manager.get_job.return_value = None

        client = TestClient(app)
        response = client.get('/api/v1/retrain/jobs/missing-job')

        assert response.status_code == 404
        assert 'not found' in response.json()['detail']
