"""Tests for API routes."""
import pytest
from fastapi import status


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_predict_single_patient(self, client, valid_patient_data):
        """Test prediction with a single patient."""
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["patient_id"] == 0
        assert 0.0 <= data["predictions"][0]["probability"] <= 1.0
    
    def test_predict_multiple_patients(self, client, multiple_patients):
        """Test prediction with multiple patients."""
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": multiple_patients}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["predictions"]) == 2
        
        for i, pred in enumerate(data["predictions"]):
            assert pred["patient_id"] == i
            assert 0.0 <= pred["probability"] <= 1.0
    
    def test_predict_invalid_sex(self, client, valid_patient_data):
        """Test prediction with invalid sex value."""
        valid_patient_data["Sex"] = "invalid"
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_age(self, client, valid_patient_data):
        """Test prediction with invalid age."""
        valid_patient_data["Age"] = 150
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_cholesterol(self, client, valid_patient_data):
        """Test prediction with invalid cholesterol value."""
        valid_patient_data["Cholesterol"] = 700
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_bp(self, client, valid_patient_data):
        """Test prediction with invalid blood pressure."""
        valid_patient_data["BP"] = 300
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_missing_required_field(self, client, valid_patient_data):
        """Test prediction with missing required field."""
        del valid_patient_data["Age"]
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_exercise_angina(self, client, valid_patient_data):
        """Test prediction with invalid exercise angina value."""
        valid_patient_data["Exercise angina"] = "maybe"
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_request_format(self, client):
        """Test prediction with invalid request format."""
        response = client.post(
            "/api/v1/predict",
            json={"invalid": "data"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_max_hr_validation(self, client, valid_patient_data):
        """Test prediction with invalid max heart rate."""
        valid_patient_data["Max HR"] = 300
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_st_depression_validation(self, client, valid_patient_data):
        """Test prediction with invalid ST depression."""
        valid_patient_data["ST depression"] = 15.0
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_chest_pain_type_validation(self, client, valid_patient_data):
        """Test prediction with invalid chest pain type."""
        valid_patient_data["Chest pain type"] = 5
        response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestRetrainEndpoint:
    """Tests for the /retrain endpoint."""
    
    def test_retrain_default_params(self, client):
        """Test retrain with default parameters."""
        response = client.post(
            "/api/v1/retrain",
            json={}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "model_uri" in data
    
    def test_retrain_custom_params(self, client):
        """Test retrain with custom parameters."""
        response = client.post(
            "/api/v1/retrain",
            json={
                "n_iter": 100,
                "cv_splits": 10,
                "force_replacement": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "model_uri" in data
    
    def test_retrain_n_iter_validation(self, client):
        """Test retrain rejects invalid n_iter."""
        response = client.post(
            "/api/v1/retrain",
            json={"n_iter": 0}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_retrain_cv_splits_validation(self, client):
        """Test retrain rejects invalid cv_splits."""
        response = client.post(
            "/api/v1/retrain",
            json={"cv_splits": 1}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_retrain_negative_n_iter(self, client):
        """Test retrain rejects negative n_iter."""
        response = client.post(
            "/api/v1/retrain",
            json={"n_iter": -5}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_retrain_response_structure(self, client):
        """Test that retrain response has correct structure."""
        response = client.post(
            "/api/v1/retrain",
            json={}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Required fields
        assert isinstance(data.get("status"), str)
        assert isinstance(data.get("model_uri"), str)
        
        # Optional fields
        if "cv_mean_auc" in data:
            assert isinstance(data["cv_mean_auc"], (float, type(None)))
        if "message" in data:
            assert isinstance(data["message"], (str, type(None)))
    
    def test_retrain_with_all_fields(self, client):
        """Test retrain with all optional fields."""
        response = client.post(
            "/api/v1/retrain",
            json={
                "n_iter": 50,
                "cv_splits": 5,
                "force_replacement": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK


class TestEndpointIntegration:
    """Integration tests for multiple endpoints."""
    
    def test_predict_after_retrain(self, client, valid_patient_data):
        """Test that predictions work after retraining."""
        # First retrain
        retrain_response = client.post(
            "/api/v1/retrain",
            json={"n_iter": 10}
        )
        assert retrain_response.status_code == status.HTTP_200_OK
        
        # Then predict
        predict_response = client.post(
            "/api/v1/predict",
            json={"patient_data": [valid_patient_data]}
        )
        assert predict_response.status_code == status.HTTP_200_OK
    
    def test_multiple_predictions_sequence(self, client, valid_patient_data):
        """Test multiple prediction requests in sequence."""
        for _ in range(3):
            response = client.post(
                "/api/v1/predict",
                json={"patient_data": [valid_patient_data]}
            )
            assert response.status_code == status.HTTP_200_OK
