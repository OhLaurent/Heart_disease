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
                "Sex": "male",
                "Chest pain type": 2,
                "BP": 130,
                "Cholesterol": 240,
                "FBS over 120": True,
                "EKG results": 1,
                "Max HR": 150,
                "Exercise angina": "yes",
                "ST depression": 1.5,
                "Slope of ST": 2,
                "Number of vessels fluro": 1,
                "Thallium": 6
            }),
            PatientData(**{
                "Age": 62,
                "Sex": "female",
                "Chest pain type": 3,
                "BP": 140,
                "Cholesterol": 260,
                "FBS over 120": False,
                "EKG results": 0,
                "Max HR": 135,
                "Exercise angina": "no",
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

    def test_converts_fbs_bool_to_string(self):
        """Test that FBS over 120 boolean is converted to string."""
        patients = [
            PatientData(**{
                "Age": 55, "Sex": "male", "Chest pain type": 2, "BP": 130,
                "Cholesterol": 240, "FBS over 120": True, "EKG results": 1,
                "Max HR": 150, "Exercise angina": "yes", "ST depression": 1.5,
                "Slope of ST": 2, "Number of vessels fluro": 1, "Thallium": 6
            })
        ]
        
        df = _request_to_dataframe(patients)
        
        assert df['FBS over 120'].iloc[0] == 'true'
        assert isinstance(df['FBS over 120'].iloc[0], str)

    def test_preserves_sex_and_exercise_angina_strings(self):
        """Test that Sex and Exercise angina remain as strings."""
        patients = [
            PatientData(**{
                "Age": 55, "Sex": "male", "Chest pain type": 2, "BP": 130,
                "Cholesterol": 240, "FBS over 120": False, "EKG results": 1,
                "Max HR": 150, "Exercise angina": "yes", "ST depression": 1.5,
                "Slope of ST": 2, "Number of vessels fluro": 1, "Thallium": 6
            })
        ]
        
        df = _request_to_dataframe(patients)
        
        assert df['Sex'].iloc[0] == "male"
        assert df['Exercise angina'].iloc[0] == "yes"


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    @patch('heart_disease.api.routes.predict_patients')
    def test_predict_success(self, mock_predict):
        """Test successful prediction."""
        # Setup mock
        mock_results = pd.DataFrame({
            'prediction': ['Presence', 'Absence'],
            'probability_Absence': [0.3, 0.7],
            'probability_Presence': [0.7, 0.3]
        })
        mock_predict.return_value = mock_results
        
        client = TestClient(app)
        response = client.post("/api/v1/predict", json={
            "patient_data": [
                {
                    "Age": 55, "Sex": "male", "Chest pain type": 2,
                    "BP": 130, "Cholesterol": 240, "FBS over 120": True,
                    "EKG results": 1, "Max HR": 150, "Exercise angina": "yes",
                    "ST depression": 1.5, "Slope of ST": 2,
                    "Number of vessels fluro": 1, "Thallium": 6
                },
                {
                    "Age": 62, "Sex": "female", "Chest pain type": 3,
                    "BP": 140, "Cholesterol": 260, "FBS over 120": False,
                    "EKG results": 0, "Max HR": 135, "Exercise angina": "no",
                    "ST depression": 2.0, "Slope of ST": 1,
                    "Number of vessels fluro": 2, "Thallium": 7
                }
            ]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
        assert len(data['predictions']) == 2
        assert data['predictions'][0]['patient_id'] == 0
        assert data['predictions'][0]['probability'] == 0.7
        assert data['predictions'][1]['patient_id'] == 1
        assert data['predictions'][1]['probability'] == 0.3

    @patch('heart_disease.api.routes.predict_patients')
    def test_predict_with_model_error(self, mock_predict):
        """Test prediction when model is not available."""
        mock_predict.side_effect = ValueError("No model found with alias 'active'")
        
        client = TestClient(app)
        response = client.post("/api/v1/predict", json={
            "patient_data": [
                {
                    "Age": 55, "Sex": "male", "Chest pain type": 2,
                    "BP": 130, "Cholesterol": 240, "FBS over 120": True,
                    "EKG results": 1, "Max HR": 150, "Exercise angina": "yes",
                    "ST depression": 1.5, "Slope of ST": 2,
                    "Number of vessels fluro": 1, "Thallium": 6
                }
            ]
        })
        
        assert response.status_code == 400
        assert "No model found with alias 'active'" in response.json()['detail']

    def test_predict_with_invalid_data(self):
        """Test prediction with invalid patient data."""
        client = TestClient(app)
        response = client.post("/api/v1/predict", json={
            "patient_data": [
                {
                    "Age": 200,  # Invalid age > 120
                    "Sex": "male",
                    "Chest pain type": 2,
                    "BP": 130,
                    "Cholesterol": 240,
                    "FBS over 120": True,
                    "EKG results": 1,
                    "Max HR": 150,
                    "Exercise angina": "yes",
                    "ST depression": 1.5,
                    "Slope of ST": 2,
                    "Number of vessels fluro": 1,
                    "Thallium": 6
                }
            ]
        })
        
        assert response.status_code == 422  # Validation error


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
