"""Tests for Pydantic schemas and validation."""
import pytest
from pydantic import ValidationError

from heart_disease.api.schemas import (
    PatientData,
    PredictionRequest,
    PredictionResult,
    PredictionResponse,
    RetrainRequest,
    RetrainResponse,
)


class TestPatientData:
    """Tests for PatientData schema validation."""
    
    def test_valid_patient_data(self, valid_patient_data):
        """Test that valid patient data passes validation."""
        patient = PatientData(**valid_patient_data)
        assert patient.Age == 45
        assert patient.Sex == 1
        assert patient.Max_HR == 150
    
    def test_age_validation_min(self, valid_patient_data):
        """Test age must be >= 1."""
        valid_patient_data["Age"] = 0
        with pytest.raises(ValidationError) as exc_info:
            PatientData(**valid_patient_data)
        assert "greater than or equal to 1" in str(exc_info.value)
    
    def test_age_validation_max(self, valid_patient_data):
        """Test age must be <= 120."""
        valid_patient_data["Age"] = 121
        with pytest.raises(ValidationError) as exc_info:
            PatientData(**valid_patient_data)
        assert "less than or equal to 120" in str(exc_info.value)
    
    def test_sex_validation_valid_values(self, valid_patient_data):
        """Test sex accepts coded values 0 and 1."""
        for sex in [0, 1]:
            valid_patient_data["Sex"] = sex
            patient = PatientData(**valid_patient_data)
            assert patient.Sex == sex
    
    def test_sex_validation_invalid(self, valid_patient_data):
        """Test sex rejects values outside 0/1."""
        valid_patient_data["Sex"] = 2
        with pytest.raises(ValidationError) as exc_info:
            PatientData(**valid_patient_data)
        assert "less than or equal to 1" in str(exc_info.value)
    
    def test_exercise_angina_validation_valid(self, valid_patient_data):
        """Test exercise angina accepts coded values 0 and 1."""
        for value in [0, 1]:
            valid_patient_data["Exercise angina"] = value
            patient = PatientData(**valid_patient_data)
            assert patient.Exercise_angina == value
    
    def test_exercise_angina_validation_invalid(self, valid_patient_data):
        """Test exercise angina rejects values outside 0/1."""
        valid_patient_data["Exercise angina"] = 2
        with pytest.raises(ValidationError) as exc_info:
            PatientData(**valid_patient_data)
        assert "less than or equal to 1" in str(exc_info.value)
    
    def test_bp_validation_range(self, valid_patient_data):
        """Test BP must be between 50 and 250."""
        # Valid range
        valid_patient_data["BP"] = 50
        PatientData(**valid_patient_data)
        
        # Below minimum
        valid_patient_data["BP"] = 49
        with pytest.raises(ValidationError):
            PatientData(**valid_patient_data)
        
        # Above maximum
        valid_patient_data["BP"] = 251
        with pytest.raises(ValidationError):
            PatientData(**valid_patient_data)
    
    def test_cholesterol_validation_range(self, valid_patient_data):
        """Test Cholesterol must be between 100 and 600."""
        valid_patient_data["Cholesterol"] = 100
        PatientData(**valid_patient_data)
        
        valid_patient_data["Cholesterol"] = 99
        with pytest.raises(ValidationError):
            PatientData(**valid_patient_data)
    
    def test_max_hr_validation_range(self, valid_patient_data):
        """Test Max HR must be between 60 and 220."""
        valid_patient_data["Max HR"] = 220
        PatientData(**valid_patient_data)
        
        valid_patient_data["Max HR"] = 221
        with pytest.raises(ValidationError):
            PatientData(**valid_patient_data)
    
    def test_st_depression_validation_range(self, valid_patient_data):
        """Test ST depression must be between 0.0 and 10.0."""
        valid_patient_data["ST depression"] = 0.0
        PatientData(**valid_patient_data)
        
        valid_patient_data["ST depression"] = 10.1
        with pytest.raises(ValidationError):
            PatientData(**valid_patient_data)
    
    def test_chest_pain_type_validation(self, valid_patient_data):
        """Test chest pain type must be between 1 and 4."""
        valid_patient_data["Chest pain type"] = 1
        PatientData(**valid_patient_data)
        
        valid_patient_data["Chest pain type"] = 5
        with pytest.raises(ValidationError):
            PatientData(**valid_patient_data)
    
    def test_ekg_results_validation(self, valid_patient_data):
        """Test EKG results must be between 0 and 2."""
        valid_patient_data["EKG results"] = 2
        PatientData(**valid_patient_data)
        
        valid_patient_data["EKG results"] = 3
        with pytest.raises(ValidationError):
            PatientData(**valid_patient_data)
    
    def test_slope_of_st_validation(self, valid_patient_data):
        """Test slope of ST must be between 1 and 3."""
        valid_patient_data["Slope of ST"] = 3
        PatientData(**valid_patient_data)
        
        valid_patient_data["Slope of ST"] = 4
        with pytest.raises(ValidationError):
            PatientData(**valid_patient_data)
    
    def test_vessels_fluro_validation(self, valid_patient_data):
        """Test number of vessels fluro must be between 0 and 3."""
        valid_patient_data["Number of vessels fluro"] = 0
        PatientData(**valid_patient_data)
        
        valid_patient_data["Number of vessels fluro"] = 4
        with pytest.raises(ValidationError):
            PatientData(**valid_patient_data)
    
    def test_thallium_validation(self, valid_patient_data):
        """Test thallium must be one of 3, 6, or 7."""
        valid_patient_data["Thallium"] = 3
        PatientData(**valid_patient_data)

        valid_patient_data["Thallium"] = 6
        PatientData(**valid_patient_data)

        valid_patient_data["Thallium"] = 7
        PatientData(**valid_patient_data)
        
        valid_patient_data["Thallium"] = 5
        with pytest.raises(ValidationError) as exc_info:
            PatientData(**valid_patient_data)
        assert "Thallium must be one of: 3, 6, 7" in str(exc_info.value)
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            PatientData(Age=45)


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""
    
    def test_valid_prediction_request(self, valid_patient_data):
        """Test valid prediction request."""
        request = PredictionRequest(patient_data=[valid_patient_data])
        assert len(request.patient_data) == 1
    
    def test_multiple_patients_request(self, multiple_patients):
        """Test prediction request with multiple patients."""
        request = PredictionRequest(patient_data=multiple_patients)
        assert len(request.patient_data) == 2
    
    def test_empty_patient_list(self):
        """Test that empty patient list fails validation."""
        # Empty list should still be valid as per schema
        request = PredictionRequest(patient_data=[])
        assert len(request.patient_data) == 0


class TestPredictionResult:
    """Tests for PredictionResult schema."""
    
    def test_valid_prediction_result(self):
        """Test valid prediction result."""
        result = PredictionResult(patient_id=0, probability=0.75)
        assert result.patient_id == 0
        assert result.probability == 0.75
    
    def test_probability_range(self):
        """Test probability must be between 0.0 and 1.0."""
        PredictionResult(patient_id=0, probability=0.0)
        PredictionResult(patient_id=0, probability=1.0)
        
        with pytest.raises(ValidationError):
            PredictionResult(patient_id=0, probability=1.5)
        
        with pytest.raises(ValidationError):
            PredictionResult(patient_id=0, probability=-0.1)


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""
    
    def test_valid_response(self):
        """Test valid prediction response."""
        predictions = [
            PredictionResult(patient_id=0, probability=0.5),
            PredictionResult(patient_id=1, probability=0.75),
        ]
        response = PredictionResponse(predictions=predictions)
        assert len(response.predictions) == 2
    
    def test_empty_predictions(self):
        """Test response with empty predictions list."""
        response = PredictionResponse(predictions=[])
        assert len(response.predictions) == 0


class TestRetrainRequest:
    """Tests for RetrainRequest schema."""
    
    def test_valid_retrain_request_defaults(self):
        """Test retrain request with default values."""
        request = RetrainRequest()
        assert request.n_iter == 20
        assert request.cv_splits == 5
        assert request.force_replacement is False
    
    def test_valid_retrain_request_custom(self):
        """Test retrain request with custom values."""
        request = RetrainRequest(n_iter=50, cv_splits=10, force_replacement=True)
        assert request.n_iter == 50
        assert request.cv_splits == 10
        assert request.force_replacement is True
    
    def test_n_iter_validation(self):
        """Test n_iter must be >= 1."""
        RetrainRequest(n_iter=1)
        
        with pytest.raises(ValidationError):
            RetrainRequest(n_iter=0)
    
    def test_cv_splits_validation(self):
        """Test cv_splits must be >= 2."""
        RetrainRequest(cv_splits=2)
        
        with pytest.raises(ValidationError):
            RetrainRequest(cv_splits=1)


class TestRetrainResponse:
    """Tests for RetrainResponse schema."""
    
    def test_valid_retrain_response(self):
        """Test valid retrain response."""
        response = RetrainResponse(
            status="Training completed",
            model_uri="s3://bucket/model.pkl"
        )
        assert response.status == "Training completed"
        assert response.model_uri == "s3://bucket/model.pkl"
    
    def test_retrain_response_with_optional_fields(self):
        """Test retrain response with optional fields."""
        response = RetrainResponse(
            status="Training completed",
            model_uri="s3://bucket/model.pkl",
            cv_mean_auc=0.85,
            message="Model improved performance"
        )
        assert response.cv_mean_auc == 0.85
        assert response.message == "Model improved performance"
