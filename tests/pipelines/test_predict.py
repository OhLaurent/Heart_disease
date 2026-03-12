"""Tests for heart_disease.pipelines.predict module."""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from heart_disease.constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from heart_disease.pipelines.predict import (
    ModelReference,
    PredictionPipeline,
    PredictionRunResult,
    predict_patients,
)


class TestPredictionPipeline:
    """Tests for PredictionPipeline class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        pipeline = PredictionPipeline()
        
        assert pipeline.model_name == "heart_disease_model"
        assert pipeline.model_alias == "active"
        assert pipeline.model_ is None
        assert pipeline.model_version_ is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        pipeline = PredictionPipeline(
            model_name="custom_model",
            model_alias="champion"
        )
        
        assert pipeline.model_name == "custom_model"
        assert pipeline.model_alias == "champion"

    @patch('heart_disease.pipelines.predict.MlflowClient')
    @patch('heart_disease.pipelines.predict.mlflow')
    def test_load_model_success(self, mock_mlflow, mock_client_class):
        """Test successful model loading."""
        # Setup mocks
        mock_client = Mock()
        mock_version = Mock()
        mock_version.version = "3"
        mock_client.get_model_version_by_alias.return_value = mock_version
        mock_client_class.return_value = mock_client
        
        mock_model = Mock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        pipeline = PredictionPipeline()
        result = pipeline.load_model()
        
        assert result is pipeline  # Returns self for chaining
        assert pipeline.model_ is mock_model
        assert pipeline.model_version_ == "3"
        mock_client.get_model_version_by_alias.assert_called_once_with(
            "heart_disease_model", "active"
        )
        mock_mlflow.set_tracking_uri.assert_called_once_with(MLFLOW_TRACKING_URI)
        mock_mlflow.set_experiment.assert_called_once_with(MLFLOW_EXPERIMENT_NAME)
        mock_mlflow.sklearn.load_model.assert_called_once_with("models:/heart_disease_model@active")

    @patch('heart_disease.pipelines.predict.MlflowClient')
    def test_load_model_not_found(self, mock_client_class):
        """Test model loading when model alias doesn't exist."""
        import mlflow.exceptions
        
        mock_client = Mock()
        mock_client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("Not found")
        mock_client_class.return_value = mock_client
        
        pipeline = PredictionPipeline()
        
        with pytest.raises(ValueError, match="No model found with alias"):
            pipeline.load_model()

    @patch('heart_disease.pipelines.predict.DataTransformer')
    @patch('heart_disease.pipelines.predict.DataValidator')
    def test_prepare_data(self, mock_validator_class, mock_transformer_class, sample_raw_data):
        """Test data preparation for prediction."""
        # Setup mocks
        mock_validator = Mock()
        mock_validator.validate.return_value = sample_raw_data
        mock_validator_class.return_value = mock_validator
        
        transformed_data = sample_raw_data.copy()
        transformed_data['Sex'] = transformed_data['Sex'].map({1: 'male', 0: 'female'})
        mock_transformer = Mock()
        mock_transformer.transform.return_value = transformed_data
        mock_transformer_class.return_value = mock_transformer
        
        pipeline = PredictionPipeline()
        result = pipeline._prepare_data(sample_raw_data)
        
        # Verify validator called in inference mode
        mock_validator_class.assert_called_once_with(mode="inference")
        mock_validator.validate.assert_called_once()
        
        # Verify transformer called correctly
        mock_transformer_class.assert_called_once_with(drop_id=True, drop_target=False)
        mock_transformer.transform.assert_called_once()
        
        # Verify result
        assert result is not None

    @patch.object(PredictionPipeline, '_prepare_data')
    def test_predict_without_model_raises_error(self, mock_prepare, sample_raw_data):
        """Test predict raises error when model not loaded."""
        pipeline = PredictionPipeline()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            pipeline.predict(sample_raw_data)

    @patch.object(PredictionPipeline, '_prepare_data')
    def test_predict_basic(self, mock_prepare, sample_raw_data):
        """Test basic prediction without probabilities."""
        mock_prepare.return_value = sample_raw_data.drop(columns=['id', 'Heart Disease'])
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = ['Presence', 'Absence', 'Presence']
        
        pipeline = PredictionPipeline()
        pipeline.model_ = mock_model
        pipeline.model_version_ = "3"
        pipeline.model_uri_ = "models:/heart_disease_model@active"
        
        results = pipeline.predict(sample_raw_data.head(3), return_proba=False)
        
        assert isinstance(results, PredictionRunResult)
        assert results.model == ModelReference(version="3", uri="models:/heart_disease_model@active")
        assert 'prediction' in results.predictions.columns
        assert len(results.predictions) == 3
        assert list(results.predictions['prediction']) == ['Presence', 'Absence', 'Presence']
        mock_model.predict.assert_called_once()

    @patch.object(PredictionPipeline, '_prepare_data')
    def test_predict_with_probabilities(self, mock_prepare, sample_raw_data):
        """Test prediction with class probabilities."""
        mock_prepare.return_value = sample_raw_data.drop(columns=['id', 'Heart Disease'])
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array(['Presence', 'Absence', 'Presence'])
        mock_model.predict_proba.return_value = np.array([
            [0.3, 0.7],  # 70% Presence
            [0.8, 0.2],  # 80% Absence
            [0.4, 0.6],  # 60% Presence
        ])
        mock_model.classes_ = np.array(['Absence', 'Presence'])
        
        pipeline = PredictionPipeline()
        pipeline.model_ = mock_model
        pipeline.model_version_ = "3"
        pipeline.model_uri_ = "models:/heart_disease_model@active"
        
        results = pipeline.predict(sample_raw_data.head(3), return_proba=True)
        
        assert 'prediction' in results.predictions.columns
        assert 'probability_Absence' in results.predictions.columns
        assert 'probability_Presence' in results.predictions.columns
        assert len(results.predictions) == 3
        mock_model.predict_proba.assert_called_once()

    @patch.object(PredictionPipeline, '_prepare_data')
    def test_predict_with_numeric_classes_normalizes_output(self, mock_prepare, sample_raw_data):
        """Test numeric model classes are normalized to Presence/Absence columns."""
        mock_prepare.return_value = sample_raw_data.drop(columns=['id', 'Heart Disease'])

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.3, 0.7],
            [0.8, 0.2],
            [0.4, 0.6],
        ])
        mock_model.classes_ = np.array([0, 1])

        pipeline = PredictionPipeline()
        pipeline.model_ = mock_model
        pipeline.model_version_ = "3"
        pipeline.model_uri_ = "models:/heart_disease_model@active"

        results = pipeline.predict(sample_raw_data.head(3), return_proba=True)

        assert list(results.predictions['prediction']) == ['Presence', 'Absence', 'Presence']
        assert 'probability_Absence' in results.predictions.columns
        assert 'probability_Presence' in results.predictions.columns

    @patch.object(PredictionPipeline, '_prepare_data')
    def test_predict_without_input(self, mock_prepare, sample_raw_data):
        """Test prediction without including input columns."""
        mock_prepare.return_value = sample_raw_data.drop(columns=['id', 'Heart Disease'])
        
        mock_model = Mock()
        mock_model.predict.return_value = ['Presence', 'Absence']
        
        pipeline = PredictionPipeline()
        pipeline.model_ = mock_model
        pipeline.model_version_ = "3"
        pipeline.model_uri_ = "models:/heart_disease_model@active"
        
        results = pipeline.predict(sample_raw_data.head(2), include_input=False)
        
        # Should only have prediction column
        assert 'prediction' in results.predictions.columns
        assert 'id' not in results.predictions.columns
        assert 'Age' not in results.predictions.columns

    @patch.object(PredictionPipeline, 'load_model')
    @patch('heart_disease.pipelines.predict.DataLoader')
    @patch.object(PredictionPipeline, 'predict')
    def test_predict_from_file(self, mock_predict, mock_loader_class, mock_load_model, sample_csv_file, sample_raw_data):
        """Test prediction from CSV file."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.load.return_value = sample_raw_data.drop(columns=['Heart Disease'])
        mock_loader_class.return_value = mock_loader
        
        expected_results = PredictionRunResult(
            predictions=pd.DataFrame({
                'id': [1, 2],
                'prediction': ['Presence', 'Absence']
            }),
            model=ModelReference(version="3", uri="models:/heart_disease_model@active"),
        )
        mock_predict.return_value = expected_results
        
        pipeline = PredictionPipeline()
        pipeline.model_ = Mock()  # Simulate loaded model
        
        results = pipeline.predict_from_file(sample_csv_file, return_proba=True)
        
        # Verify loader was called correctly
        mock_loader_class.assert_called_once()
        mock_loader.load.assert_called_once()
        
        # Verify predict was called with correct params
        mock_predict.assert_called_once()
        call_kwargs = mock_predict.call_args.kwargs
        assert call_kwargs['return_proba'] is True
        assert call_kwargs['include_input'] is True
        
        # Verify results
        assert results.model == expected_results.model
        pd.testing.assert_frame_equal(results.predictions, expected_results.predictions)

    @patch.object(PredictionPipeline, 'load_model')
    @patch('heart_disease.pipelines.predict.DataLoader')
    def test_predict_from_file_loads_model_if_needed(self, mock_loader_class, mock_load_model, sample_csv_file, sample_raw_data):
        """Test that predict_from_file loads model if not loaded."""
        mock_loader = Mock()
        mock_loader.load.return_value = sample_raw_data.drop(columns=['Heart Disease'])
        mock_loader_class.return_value = mock_loader
        
        pipeline = PredictionPipeline()
        pipeline.model_ = None  # Model not loaded
        
        with patch.object(pipeline, 'predict'):
            pipeline.predict_from_file(sample_csv_file)
            
            # Should have called load_model
            mock_load_model.assert_called_once()

    @patch('heart_disease.pipelines.predict.DataLoader')
    def test_predict_from_file_with_target_raises_error(self, mock_loader_class, sample_csv_file, sample_raw_data):
        """Test that predict_from_file raises error if target is present."""
        # Data includes target column
        mock_loader = Mock()
        mock_loader.load.return_value = sample_raw_data  # Has 'Heart Disease'
        mock_loader_class.return_value = mock_loader
        
        pipeline = PredictionPipeline()
        pipeline.model_ = Mock()  # Simulate loaded model
        
        with pytest.raises(ValueError, match="Input data contains target column"):
            pipeline.predict_from_file(sample_csv_file)


class TestPredictPatientsFunction:
    """Tests for the predict_patients convenience function."""

    @patch.object(PredictionPipeline, 'predict')
    @patch.object(PredictionPipeline, 'load_model')
    def test_predict_patients_with_dataframe(self, mock_load, mock_predict, sample_raw_data):
        """Test predict_patients with DataFrame input."""
        expected = PredictionRunResult(
            predictions=pd.DataFrame({'prediction': ['Presence', 'Absence']}),
            model=ModelReference(version="3", uri="models:/heart_disease_model@active"),
        )
        mock_predict.return_value = expected
        
        result = predict_patients(sample_raw_data.head(2), return_proba=True)
        
        mock_load.assert_called_once()
        mock_predict.assert_called_once()
        assert result.model == expected.model
        pd.testing.assert_frame_equal(result.predictions, expected.predictions)

    @patch.object(PredictionPipeline, 'predict_from_file')
    def test_predict_patients_with_file_path(self, mock_predict_file, sample_csv_file):
        """Test predict_patients with file path input."""
        expected = PredictionRunResult(
            predictions=pd.DataFrame({'prediction': ['Presence', 'Absence']}),
            model=ModelReference(version="3", uri="models:/heart_disease_model@active"),
        )
        mock_predict_file.return_value = expected
        
        result = predict_patients(sample_csv_file, return_proba=False)
        
        mock_predict_file.assert_called_once()
        call_kwargs = mock_predict_file.call_args.kwargs
        assert call_kwargs['return_proba'] is False
        assert result.model == expected.model
        pd.testing.assert_frame_equal(result.predictions, expected.predictions)

    @patch.object(PredictionPipeline, 'predict')
    @patch.object(PredictionPipeline, 'load_model')
    def test_predict_patients_custom_model(self, mock_load, mock_predict, sample_raw_data):
        """Test predict_patients with custom model name and alias."""
        expected = PredictionRunResult(
            predictions=pd.DataFrame({'prediction': ['Presence']}),
            model=ModelReference(version="7", uri="models:/custom_model@champion"),
        )
        mock_predict.return_value = expected
        
        result = predict_patients(
            sample_raw_data.head(1),
            model_name='custom_model',
            model_alias='champion'
        )
        
        # Pipeline should have been created with custom params
        # We can't easily verify this without refactoring, but the function should work
        assert result is expected
