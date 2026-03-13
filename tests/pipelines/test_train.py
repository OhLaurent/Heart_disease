"""Tests for heart_disease.pipelines.train module."""
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from heart_disease.constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from heart_disease.pipelines.train import TrainingPipeline, train_pipeline
from heart_disease.constants import RANDOM_STATE


class TestTrainingPipeline:
    """Tests for TrainingPipeline class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        pipeline = TrainingPipeline()
        
        assert pipeline.n_iter == 50
        assert pipeline.cv_folds == 5
        assert pipeline.force_replace is False
        assert pipeline.model_ is None
        assert pipeline.metrics_ is None
        assert pipeline.run_id_ is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        pipeline = TrainingPipeline(
            n_iter=100,
            cv_folds=10,
            force_replace=True,
            data_path="custom/path.csv"
        )
        
        assert pipeline.n_iter == 100
        assert pipeline.cv_folds == 10
        assert pipeline.force_replace is True
        assert str(pipeline.data_path) == "custom/path.csv"

    @patch('heart_disease.pipelines.train.DataValidator')
    def test_load_and_validate_data(self, mock_validator_class, sample_csv_file, sample_raw_data):
        """Test loading and validating data."""
        # Setup mock validator
        mock_validator = Mock()
        mock_validator.validate.return_value = sample_raw_data
        mock_validator_class.return_value = mock_validator
        
        pipeline = TrainingPipeline(data_path=sample_csv_file)
        df = pipeline._load_and_validate_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Heart Disease" in df.columns
        mock_validator.validate.assert_called_once()

    @patch.object(TrainingPipeline, '_load_and_validate_data')
    def test_prepare_features(self, mock_load, sample_raw_data):
        """Test preparing features from raw data."""
        mock_load.return_value = sample_raw_data
        
        pipeline = TrainingPipeline()
        df = pipeline._load_and_validate_data()
        X, y = pipeline._prepare_features(df)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert "Heart Disease" not in X.columns
        assert "id" not in X.columns
        assert y.isin([0, 1]).all()

    @patch.object(TrainingPipeline, '_load_and_validate_data')
    def test_split_train_test(self, mock_load, sample_raw_data):
        """Test train-test split."""
        mock_load.return_value = sample_raw_data
        
        pipeline = TrainingPipeline()
        df = pipeline._load_and_validate_data()
        X, y = pipeline._prepare_features(df)
        
        X_train, X_test, y_train, y_test = pipeline._split_train_test(X, y)
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    @patch.object(TrainingPipeline, '_load_and_validate_data')
    def test_create_ml_pipeline(self, mock_load, sample_raw_data):
        """Test creating sklearn pipeline."""
        mock_load.return_value = sample_raw_data
        
        pipeline = TrainingPipeline()
        df = pipeline._load_and_validate_data()
        X, y = pipeline._prepare_features(df)
        
        ml_pipeline = pipeline._create_ml_pipeline(X)
        
        assert ml_pipeline is not None
        assert hasattr(ml_pipeline, 'fit')
        assert hasattr(ml_pipeline, 'predict')
        assert "preprocessor" in ml_pipeline.named_steps
        assert "classifier" in ml_pipeline.named_steps

    @patch.object(TrainingPipeline, '_load_and_validate_data')
    @patch('heart_disease.pipelines.train.RandomizedSearchCV')
    def test_tune_hyperparameters(self, mock_search, mock_load, sample_raw_data):
        """Test hyperparameter tuning."""
        # Setup mocks
        mock_load.return_value = sample_raw_data
        mock_search_instance = Mock()
        mock_search_instance.best_score_ = 0.85
        mock_search_instance.best_params_ = {"classifier__C": 1.0}
        mock_search_instance.best_estimator_ = Mock()
        mock_search.return_value = mock_search_instance
        
        pipeline = TrainingPipeline(n_iter=10, cv_folds=3)
        df = pipeline._load_and_validate_data()
        X, y = pipeline._prepare_features(df)
        X_train, X_test, y_train, y_test = pipeline._split_train_test(X, y)
        ml_pipeline = pipeline._create_ml_pipeline(X)
        
        search = pipeline._tune_hyperparameters(ml_pipeline, X_train, y_train)
        
        assert search is not None
        mock_search_instance.fit.assert_called_once()

    @patch.object(TrainingPipeline, '_load_and_validate_data')
    def test_evaluate_model(self, mock_load, sample_raw_data):
        """Test model evaluation."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        mock_load.return_value = sample_raw_data
        
        pipeline = TrainingPipeline()
        df = pipeline._load_and_validate_data()
        X, y = pipeline._prepare_features(df)
        X_train, X_test, y_train, y_test = pipeline._split_train_test(X, y)
        
        # Create and train a simple model
        simple_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
        ])
        
        # Get only numerical columns for this simple test
        numerical_cols = X_train.select_dtypes(include=['number']).columns
        X_train_num = X_train[numerical_cols]
        X_test_num = X_test[numerical_cols]
        
        simple_pipeline.fit(X_train_num, y_train)
        
        metrics = pipeline._evaluate_model(simple_pipeline, X_test_num, y_test, cv_score=0.8)
        
        assert isinstance(metrics, dict)
        assert "test_accuracy" in metrics
        assert "test_precision" in metrics
        assert "test_recall" in metrics
        assert "test_f1" in metrics
        assert "test_roc_auc" in metrics
        assert "cv_roc_auc" in metrics
        assert metrics["cv_roc_auc"] == 0.8

    @patch.object(TrainingPipeline, '_load_and_validate_data')
    @patch('heart_disease.pipelines.train.mlflow')
    def test_log_to_mlflow(self, mock_mlflow, mock_load, sample_raw_data):
        """Test MLflow logging."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        # Setup mocks
        mock_load.return_value = sample_raw_data
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id_123"
        mock_mlflow.active_run.return_value = mock_run
        
        pipeline = TrainingPipeline()
        df = pipeline._load_and_validate_data()
        X, y = pipeline._prepare_features(df)
        X_train, X_test, y_train, y_test = pipeline._split_train_test(X, y)
        
        # Create simple model
        simple_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=RANDOM_STATE))
        ])
        
        numerical_cols = X_train.select_dtypes(include=['number']).columns
        X_train_num = X_train[numerical_cols]
        simple_pipeline.fit(X_train_num, y_train)
        
        y_pred_proba = simple_pipeline.predict_proba(X_train_num)[:, 1]
        
        params = {"n_iter": 10, "cv_folds": 5}
        metrics = {"test_roc_auc": 0.85, "test_accuracy": 0.80}
        baseline_stats = {"performance": {}, "numerical_features": {}, "categorical_features": {}}
        
        run_id = pipeline._log_to_mlflow(
            simple_pipeline, params, metrics, X_train_num, y_pred_proba, baseline_stats
        )
        
        assert run_id == "test_run_id_123"
        mock_mlflow.log_params.assert_called_once_with(params)
        mock_mlflow.log_metrics.assert_called_once_with(metrics)
        mock_mlflow.sklearn.log_model.assert_called_once()

    @patch.object(TrainingPipeline, '_should_promote_model', return_value=False)
    @patch.object(TrainingPipeline, '_calculate_baseline_stats')
    @patch.object(TrainingPipeline, '_evaluate_model')
    @patch.object(TrainingPipeline, '_tune_hyperparameters')
    @patch.object(TrainingPipeline, '_split_train_test')
    @patch.object(TrainingPipeline, '_prepare_features')
    @patch.object(TrainingPipeline, '_load_and_validate_data')
    @patch('heart_disease.pipelines.train.mlflow')
    def test_run_configures_mlflow(self, mock_mlflow, mock_load, mock_prepare, mock_split, mock_tune, mock_evaluate, mock_calculate_baseline_stats, mock_should_promote):
        """Test run applies shared MLflow tracking configuration."""
        sample_df = pd.DataFrame({"dummy": [1, 2]})
        X_train = pd.DataFrame({"x": [1, 2, 3, 4]})
        X_test = pd.DataFrame({"x": [5, 6]})
        y_train = pd.Series([0, 1, 0, 1])
        y_test = pd.Series([0, 1])

        mock_load.return_value = sample_df
        mock_prepare.return_value = (pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}), pd.Series([0, 1, 0, 1, 0, 1]))
        mock_split.return_value = (X_train, X_test, y_train, y_test)

        best_model = Mock()
        best_model.predict_proba.return_value = np.array([[0.4, 0.6], [0.7, 0.3]])
        search = Mock()
        search.best_estimator_ = best_model
        search.best_score_ = 0.9
        search.best_params_ = {"classifier__C": 1.0}
        mock_tune.return_value = search
        mock_evaluate.return_value = {"test_roc_auc": 0.8}
        mock_calculate_baseline_stats.return_value = {"performance": {}, "numerical_features": {}, "categorical_features": {}}

        mock_run = Mock()
        mock_run.__enter__ = Mock(return_value=mock_run)
        mock_run.__exit__ = Mock(return_value=None)
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value.info.run_id = "run-123"

        pipeline = TrainingPipeline()
        pipeline.run()

        mock_mlflow.set_tracking_uri.assert_called_once_with(MLFLOW_TRACKING_URI)
        mock_mlflow.set_experiment.assert_called_once_with(MLFLOW_EXPERIMENT_NAME)

    @patch('heart_disease.pipelines.train.mlflow.tracking.MlflowClient')
    def test_get_active_model_metric_no_active_model(self, mock_client_class):
        """Test getting metric when no active model exists."""
        mock_client = Mock()
        mock_client.get_model_version_by_alias.side_effect = RuntimeError("No active model")
        mock_client_class.return_value = mock_client

        metric = TrainingPipeline._get_active_model_metric("test_roc_auc")
        
        # Should return None if no active model or error occurs
        assert metric is None

    @patch('heart_disease.pipelines.train.mlflow.tracking.MlflowClient')
    def test_promote_model_success(self, mock_client_class):
        """Test successful model promotion."""
        mock_client = Mock()
        mock_version = Mock()
        mock_version.version = "1"
        mock_client.search_model_versions.return_value = [mock_version]
        mock_client_class.return_value = mock_client
        
        result = TrainingPipeline._promote_model("test_run_id")
        
        assert result is True
        mock_client.search_model_versions.assert_called_once()
        mock_client.set_registered_model_alias.assert_called_once()

    @patch('heart_disease.pipelines.train.mlflow.tracking.MlflowClient')
    def test_promote_model_no_version_found(self, mock_client_class):
        """Test model promotion when no version is found."""
        mock_client = Mock()
        mock_client.search_model_versions.return_value = []
        mock_client_class.return_value = mock_client
        
        result = TrainingPipeline._promote_model("test_run_id")
        
        assert result is False

    def test_should_promote_model_with_force_replace(self):
        """Test should_promote_model with force_replace=True."""
        pipeline = TrainingPipeline(force_replace=True)
        
        should_promote = pipeline._should_promote_model(new_metric=0.7)
        
        assert should_promote is True

    @patch.object(TrainingPipeline, '_get_active_model_metric')
    def test_should_promote_model_better_than_active(self, mock_get_metric):
        """Test should_promote_model when new model is better."""
        mock_get_metric.return_value = 0.8
        pipeline = TrainingPipeline(force_replace=False)
        
        should_promote = pipeline._should_promote_model(new_metric=0.85)
        
        assert should_promote is True

    @patch.object(TrainingPipeline, '_get_active_model_metric')
    def test_should_promote_model_worse_than_active(self, mock_get_metric):
        """Test should_promote_model when new model is worse."""
        mock_get_metric.return_value = 0.9
        pipeline = TrainingPipeline(force_replace=False)
        
        should_promote = pipeline._should_promote_model(new_metric=0.85)
        
        assert should_promote is False

    @patch.object(TrainingPipeline, '_get_active_model_metric')
    def test_should_promote_model_no_active_model(self, mock_get_metric):
        """Test should_promote_model when no active model exists."""
        mock_get_metric.return_value = None
        pipeline = TrainingPipeline(force_replace=False)
        
        should_promote = pipeline._should_promote_model(new_metric=0.85)
        
        assert should_promote is True

    @patch.object(TrainingPipeline, '_load_and_validate_data')
    @patch('heart_disease.pipelines.train.mlflow')
    @patch.object(TrainingPipeline, '_promote_model')
    @patch.object(TrainingPipeline, '_should_promote_model')
    @patch.object(TrainingPipeline, '_calculate_baseline_stats')
    def test_run_full_pipeline(
        self, 
        mock_calculate_baseline_stats,
        mock_should_promote, 
        mock_promote, 
        mock_mlflow,
        mock_load,
        sample_raw_data
    ):
        """Test running the full pipeline with mocks."""
        # Setup mocks
        mock_load.return_value = sample_raw_data
        mock_run = Mock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run
        mock_should_promote.return_value = True
        mock_promote.return_value = True
        mock_calculate_baseline_stats.return_value = {"performance": {}, "numerical_features": {}, "categorical_features": {}}
        
        pipeline = TrainingPipeline(
            n_iter=5,  # Small for fast test
            cv_folds=2  # Small for fast test
        )
        
        with patch.object(pipeline, '_tune_hyperparameters') as mock_tune:
            mock_search = Mock()
            mock_search.best_score_ = 0.85
            mock_search.best_params_ = {"classifier__C": 1.0}
            mock_search.best_estimator_ = Mock()
            mock_search.best_estimator_.fit = Mock()
            mock_search.best_estimator_.predict = Mock(return_value=np.array([0, 1]))
            mock_search.best_estimator_.predict_proba = Mock(
                return_value=np.array([[0.3, 0.7], [0.6, 0.4]])
            )
            mock_tune.return_value = mock_search
            
            results = pipeline.run()
        
        assert isinstance(results, dict)
        assert "run_id" in results
        assert "metrics" in results
        assert "best_params" in results
        assert "promoted" in results
        
        # Check instance attributes were set
        assert pipeline.model_ is not None
        assert pipeline.metrics_ is not None
        assert pipeline.run_id_ is not None

    def test_pipeline_attributes_before_run(self):
        """Test that pipeline attributes are None before running."""
        pipeline = TrainingPipeline()
        
        assert pipeline.model_ is None
        assert pipeline.metrics_ is None
        assert pipeline.run_id_ is None

    @patch.object(TrainingPipeline, '_load_and_validate_data')
    def test_split_train_test_stratified(self, mock_load, sample_raw_data):
        """Test that train-test split maintains class distribution."""
        mock_load.return_value = sample_raw_data
        
        pipeline = TrainingPipeline()
        df = pipeline._load_and_validate_data()
        X, y = pipeline._prepare_features(df)
        
        X_train, X_test, y_train, y_test = pipeline._split_train_test(X, y)
        
        # Check that both classes are present in train and test
        assert len(y_train.unique()) > 1
        assert len(y_test.unique()) > 1


class TestTrainPipelineFunction:
    """Tests for the train_pipeline convenience function."""

    @patch.object(TrainingPipeline, 'run')
    def test_train_pipeline_function_calls_class(self, mock_run):
        """Test that train_pipeline function calls TrainingPipeline.run()."""
        mock_run.return_value = {"run_id": "test", "metrics": {}}
        
        result = train_pipeline(n_iter=100, cv_folds=10, force_replace=True)
        
        mock_run.assert_called_once()
        assert result == {"run_id": "test", "metrics": {}}

    @patch.object(TrainingPipeline, 'run')
    def test_train_pipeline_function_with_defaults(self, mock_run):
        """Test train_pipeline function with default parameters."""
        mock_run.return_value = {"run_id": "test", "metrics": {}}
        
        result = train_pipeline()
        
        mock_run.assert_called_once()
        assert result is not None
