"""Training pipeline for heart disease model.

Implements end-to-end training workflow with MLflow tracking and model registration.
Organized as a class with modular, testable methods.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import mlflow
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_disease.constants import (
    CATEGORICAL_COLUMNS,
    DEFAULT_CV_FOLDS,
    DEFAULT_N_ITER,
    HYPERPARAMETER_GRID,
    INPUT_FILE,
    LOGISTIC_MAX_ITER,
    MLFLOW_ACTIVE_ALIAS,
    MLFLOW_ARTIFACT_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    N_JOBS,
    NUMERICAL_COLUMNS,
    RANDOM_STATE,
    SCORING_METRIC,
    TEST_SIZE,
)
from heart_disease.pipelines.components.dataset import DataLoader, DataValidator
from heart_disease.pipelines.components.features import DataTransformer

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass(slots=True)
class SearchResult:
    """Minimal search result payload compatible with current training flow."""

    best_estimator_: Pipeline
    best_score_: float
    best_params_: dict[str, Any]


def _configure_mlflow() -> None:
    """Apply shared MLflow configuration for tracking and experiments."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


class TrainingPipeline:
    """Heart disease model training pipeline with MLflow integration.

    This class orchestrates the complete training workflow including data loading,
    preprocessing, hyperparameter tuning, model evaluation, and MLflow tracking.

    Parameters
    ----------
    n_iter : int, optional
        Number of iterations for RandomizedSearchCV. Default from constants.
    cv_folds : int, optional
        Number of cross-validation folds. Default from constants.
    force_replace : bool, optional
        If True, always replace the currently active model, regardless of
        performance comparison. Default is False.
    data_path : str | Path, optional
        Path to training data CSV. Default from constants.

    Attributes
    ----------
    n_iter : int
        Number of hyperparameter search iterations.
    cv_folds : int
        Number of CV folds.
    force_replace : bool
        Whether to force model replacement.
    data_path : Path
        Path to input data file.
    model_ : Pipeline or None
        Trained model (available after calling run()).
    metrics_ : dict or None
        Evaluation metrics (available after calling run()).
    run_id_ : str or None
        MLflow run ID (available after calling run()).

    Examples
    --------
    >>> from heart_disease.pipelines.train import TrainingPipeline
    >>>
    >>> # Basic usage
    >>> pipeline = TrainingPipeline()
    >>> results = pipeline.run()
    >>> print(f"ROC-AUC: {results['metrics']['test_roc_auc']:.4f}")
    >>>
    >>> # Custom parameters
    >>> pipeline = TrainingPipeline(n_iter=100, cv_folds=10, force_replace=True)
    >>> results = pipeline.run()
    >>>
    >>> # Access trained model
    >>> print(pipeline.model_)
    >>> print(pipeline.metrics_)
    """

    def __init__(
        self,
        n_iter: int = DEFAULT_N_ITER,
        cv_folds: int = DEFAULT_CV_FOLDS,
        force_replace: bool = False,
        data_path: str | None = None,
        progress_callback: Callable[[str, str, int], None] | None = None,
    ):
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.force_replace = force_replace
        self.data_path = data_path or INPUT_FILE
        self.progress_callback = progress_callback

        # Attributes populated during training
        self.model_: Pipeline | None = None
        self.metrics_: dict[str, float] | None = None
        self.run_id_: str | None = None
        self._best_params: dict[str, Any] | None = None

    def _report_progress(self, stage: str, message: str, progress_pct: int) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, message, progress_pct)

    # -----------------------------------------------------------------------
    # Data loading and preparation
    # -----------------------------------------------------------------------

    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load raw data and validate against schema.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame with all original columns including target.
        """
        loader = DataLoader(self.data_path, drop_target=False)
        df = loader.load()

        validator = DataValidator(mode="training")
        df = validator.validate(df)

        return df

    def _prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Transform raw data and split into features and target.

        Parameters
        ----------
        df : pd.DataFrame
            Raw validated DataFrame.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (id and target excluded).
        y : pd.Series
            Binary target (1 = Presence, 0 = Absence).
        """
        transformer = DataTransformer(drop_id=True, drop_target=False)
        df_transformed = transformer.transform(df)
        X, y = DataTransformer.split_features_target(df_transformed)
        return X, y

    def _split_train_test(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into stratified train and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        X_train, X_test, y_train, y_test
            Train and test splits.
        """
        return train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

    # -----------------------------------------------------------------------
    # Model building
    # -----------------------------------------------------------------------

    def _create_ml_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create sklearn pipeline with preprocessing and logistic regression.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (used to identify actual columns present).

        Returns
        -------
        Pipeline
            Sklearn pipeline with StandardScaler, OneHotEncoder, and LogisticRegression.
        """
        # Identify actual categorical and numerical columns in X
        cat_cols = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
        num_cols = [col for col in NUMERICAL_COLUMNS if col in X.columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_cols,
                ),
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(random_state=RANDOM_STATE, max_iter=LOGISTIC_MAX_ITER),
                ),
            ]
        )

        return pipeline

    def _tune_hyperparameters(
        self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RandomizedSearchCV | SearchResult:
        """Perform hyperparameter tuning with RandomizedSearchCV.

        Parameters
        ----------
        pipeline : Pipeline
            Base sklearn pipeline to optimize.
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.

        Returns
        -------
        RandomizedSearchCV
            Fitted search object with best estimator.
        """
        if self.progress_callback is None:
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=HYPERPARAMETER_GRID,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring=SCORING_METRIC,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                verbose=1,
            )

            random_search.fit(X_train, y_train)
            return random_search

        candidates = list(ParameterSampler(HYPERPARAMETER_GRID, n_iter=self.n_iter, random_state=RANDOM_STATE))
        cv_splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
        total_fits = len(candidates) * self.cv_folds
        fits_done = 0

        best_score = float("-inf")
        best_params: dict[str, Any] = {}
        best_estimator: Pipeline | None = None

        for params in candidates:
            fold_scores: list[float] = []

            for train_idx, valid_idx in cv_splitter.split(X_train, y_train):
                fold_model = clone(pipeline)
                fold_model.set_params(**params)

                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_valid = X_train.iloc[valid_idx]
                y_fold_valid = y_train.iloc[valid_idx]

                fold_model.fit(X_fold_train, y_fold_train)
                fold_proba = fold_model.predict_proba(X_fold_valid)[:, 1]
                fold_scores.append(float(roc_auc_score(y_fold_valid, fold_proba)))

                fits_done += 1
                progress = 50 + int((fits_done / total_fits) * 24)
                self._report_progress(
                    "hyperparameter_search",
                    f"Running hyperparameter search... Fits: {fits_done}/{total_fits}",
                    progress,
                )

            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        best_estimator = clone(pipeline)
        best_estimator.set_params(**best_params)

        return SearchResult(
            best_estimator_=best_estimator,
            best_score_=best_score,
            best_params_=best_params,
        )

    # -----------------------------------------------------------------------
    # Model evaluation
    # -----------------------------------------------------------------------

    def _evaluate_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, cv_score: float
    ) -> dict[str, float]:
        """Evaluate model on test set and return metrics.

        Parameters
        ----------
        model : Pipeline
            Trained sklearn pipeline.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target.
        cv_score : float
            Cross-validation score from hyperparameter tuning.

        Returns
        -------
        dict[str, float]
            Dictionary of evaluation metrics.
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_pred_proba),
            "cv_roc_auc": cv_score,
        }

        return metrics

    # -----------------------------------------------------------------------
    # MLflow operations
    # -----------------------------------------------------------------------

    def _log_to_mlflow(
        self,
        model: Pipeline,
        params: dict[str, Any],
        metrics: dict[str, float],
        X_train: pd.DataFrame,
        y_pred_proba: pd.Series,
        baseline_stats: dict[str, Any],
    ) -> str:
        """Log model, parameters, and metrics to MLflow.

        Parameters
        ----------
        model : Pipeline
            Trained sklearn pipeline.
        params : dict
            Training parameters and hyperparameters.
        metrics : dict
            Evaluation metrics.
        X_train : pd.DataFrame
            Training features (for signature inference).
        y_pred_proba : pd.Series
            Prediction probabilities (for signature inference).
        baseline_stats : dict
            Baseline statistics for the features.

        Returns
        -------
        str
            MLflow run ID.
        """
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, y_pred_proba)
        mlflow.sklearn.log_model(
            model,
            artifact_path=MLFLOW_ARTIFACT_PATH,
            signature=signature,
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        mlflow.log_dict(baseline_stats, "baseline_stats.json")

        return mlflow.active_run().info.run_id

    @staticmethod
    def _get_active_model_metric(metric_name: str = "test_roc_auc") -> float | None:
        """Get metric from currently active model.

        Parameters
        ----------
        metric_name : str, optional
            Name of metric to retrieve. Default is "test_roc_auc".

        Returns
        -------
        float or None
            Metric value, or None if no active model exists.
        """
        try:
            _configure_mlflow()
            client = mlflow.tracking.MlflowClient()
            active_version = client.get_model_version_by_alias(
                name=MLFLOW_MODEL_NAME, alias=MLFLOW_ACTIVE_ALIAS
            )

            active_run = client.get_run(active_version.run_id)
            return active_run.data.metrics.get(metric_name)

        except Exception:
            return None

    @staticmethod
    def _promote_model(run_id: str) -> bool:
        """Promote model from given run to active alias.

        Parameters
        ----------
        run_id : str
            MLflow run ID containing the model to promote.

        Returns
        -------
        bool
            True if promotion succeeded, False otherwise.
        """
        try:
            _configure_mlflow()
            client = mlflow.tracking.MlflowClient()

            # Get the model version from this run
            model_versions = client.search_model_versions(filter_string=f"run_id='{run_id}'")

            if not model_versions:
                return False

            version = model_versions[0].version

            # Remove "active" alias from previous version (if exists)
            try:
                client.delete_registered_model_alias(
                    name=MLFLOW_MODEL_NAME, alias=MLFLOW_ACTIVE_ALIAS
                )
            except Exception:
                pass  # No previous active model

            # Set "active" alias on new version
            client.set_registered_model_alias(
                name=MLFLOW_MODEL_NAME, alias=MLFLOW_ACTIVE_ALIAS, version=version
            )

            return True

        except Exception:
            return False

    def _should_promote_model(
        self, new_metric: float, comparison_metric: str = "test_roc_auc"
    ) -> bool:
        """Determine if new model should be promoted to active.

        Parameters
        ----------
        new_metric : float
            Metric value of the new model.
        comparison_metric : str, optional
            Metric name to use for comparison. Default is "test_roc_auc".

        Returns
        -------
        bool
            True if model should be promoted, False otherwise.
        """
        if self.force_replace:
            return True

        active_metric = self._get_active_model_metric(comparison_metric)

        if active_metric is None:
            # No active model exists
            return True

        # Promote if new model is better
        return new_metric > active_metric


    # ---------------------------------------------------------------------------
    # Baseline statistics helper
    # ---------------------------------------------------------------------------

    def _calculate_baseline_stats(
            self,
            X: pd.DataFrame, y: pd.Series,
            search: RandomizedSearchCV | SearchResult,
            metrics: dict[str, float] | None = None) -> dict[str, float]:
        """Calculate baseline statistics for the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        search : RandomizedSearchCV or None, optional
            Hyperparameter search object, by default None.

        Returns
        -------
        dict[str, float]
            Dictionary containing baseline statistics.
        """
        best_model = search.best_estimator_
        cv_mean_auc = search.best_score_
        y_pred_proba = best_model.predict_proba(X)[:, 1]

        baseline_stats = {
            "performance":
                {
                    "positive_class_proportion": y.mean(),
                    "mean_probability": y_pred_proba.mean(),
                    "high_risk_proportion": (y_pred_proba >= 0.5).mean(),
                    "cv_mean_auc": cv_mean_auc,
                    "test_roc_auc": metrics.get("test_roc_auc") if metrics else roc_auc_score(y, y_pred_proba),
                    "test_precision": metrics.get("test_precision") if metrics else None,
                    "test_recall": metrics.get("test_recall") if metrics else None,
                    "test_f1": metrics.get("test_f1") if metrics else None,
                    "n_samples": len(y),
                },
            "numerical_features": {},
            "categorical_features": {},
        }

        for col in NUMERICAL_COLUMNS:
            if col not in X.columns:
                continue
            series = X[col].dropna().astype(float)
            counts, bin_edges = np.histogram(series, bins=20)
            baseline_stats["numerical_features"][col] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "min": float(series.min()),
                "max": float(series.max()),
                "histogram": {
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist(),
                },
            }

        for col in CATEGORICAL_COLUMNS:
            if col not in X.columns:
                continue
            vc = X[col].value_counts(normalize=True, dropna=False)
            baseline_stats["categorical_features"][col] = {str(k): float(v) for k, v in vc.items()}

        return baseline_stats

    # -----------------------------------------------------------------------
    # Main training orchestration
    # -----------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the complete training pipeline with MLflow tracking.

        This method orchestrates all training steps: data loading, preprocessing,
        hyperparameter tuning, model evaluation, MLflow logging, and model promotion.

        Returns
        -------
        dict
            Dictionary containing:
            - run_id: MLflow run ID
            - metrics: evaluation metrics
            - best_params: best hyperparameters from search
            - promoted: whether model was promoted to active

        Examples
        --------
        >>> pipeline = TrainingPipeline(n_iter=100, cv_folds=10)
        >>> results = pipeline.run()
        >>> print(f"ROC-AUC: {results['metrics']['test_roc_auc']:.4f}")
        >>> print(f"Run ID: {results['run_id']}")
        >>> print(f"Promoted: {results['promoted']}")
        """
        self._report_progress("configuring", "Configuring MLflow...", 10)
        _configure_mlflow()

        # Load and prepare data
        self._report_progress("loading_data", "Loading and validating training data...", 20)
        df = self._load_and_validate_data()
        self._report_progress("preprocessing", "Preparing features and target...", 32)
        X, y = self._prepare_features(df)
        self._report_progress("splitting", "Creating train/test split...", 40)
        X_train, X_test, y_train, y_test = self._split_train_test(X, y)

        # Build and tune model
        self._report_progress("building_model", "Building training pipeline...", 50)
        pipeline = self._create_ml_pipeline(X)
        self._report_progress("hyperparameter_search", "Starting hyperparameter search...", 55)
        search = self._tune_hyperparameters(pipeline, X_train, y_train)
        best_model = search.best_estimator_

        # Retrain on full training set
        self._report_progress("fitting", "Fitting best model...", 78)
        best_model.fit(X_train, y_train)

        # Evaluate
        self._report_progress("evaluating", "Evaluating model performance...", 86)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = self._evaluate_model(best_model, X_test, y_test, search.best_score_)

        # Evaluate baseline stats
        self._report_progress("baseline_stats", "Computing baseline statistics...", 92)
        baseline_stats = self._calculate_baseline_stats(X, y, search, metrics)

        # MLflow logging and promotion
        self._report_progress("logging", "Logging model and metrics to MLflow...", 96)
        with mlflow.start_run() as run:
            # Prepare parameters to log
            params = {
                "n_iter": self.n_iter,
                "cv_folds": self.cv_folds,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                **{f"best_{k}": v for k, v in search.best_params_.items()},
            }

            run_id = self._log_to_mlflow(best_model, params, metrics, X_train, y_pred_proba, baseline_stats)

            # Decide on promotion
            promote = self._should_promote_model(metrics["test_roc_auc"])

            if promote:
                promoted = self._promote_model(run_id)
            else:
                promoted = False

        self._report_progress("completed", "Training completed.", 100)

        # Store results as instance attributes
        self.model_ = best_model
        self.metrics_ = metrics
        self.run_id_ = run_id
        self._best_params = search.best_params_

        return {
            "run_id": run_id,
            "metrics": metrics,
            "best_params": search.best_params_,
            "promoted": promoted,
        }


# ---------------------------------------------------------------------------
# Convenience function for backward compatibility
# ---------------------------------------------------------------------------


def train_pipeline(
    n_iter: int = DEFAULT_N_ITER,
    cv_folds: int = DEFAULT_CV_FOLDS,
    force_replace: bool = False,
    progress_callback: Callable[[str, str, int], None] | None = None,
) -> dict[str, Any]:
    """Run the complete training pipeline with MLflow tracking.

    Convenience function that instantiates TrainingPipeline and runs it.
    Provided for backward compatibility.

    Parameters
    ----------
    n_iter : int, optional
        Number of iterations for RandomizedSearchCV.
    cv_folds : int, optional
        Number of cross-validation folds.
    force_replace : bool, optional
        If True, always replace the currently active model, regardless of
        performance comparison.

    Returns
    -------
    dict
        Dictionary containing:
        - run_id: MLflow run ID
        - metrics: evaluation metrics
        - best_params: best hyperparameters from search
        - promoted: whether model was promoted to active

    Examples
    --------
    >>> from heart_disease.pipelines.train import train_pipeline
    >>> results = train_pipeline(n_iter=100, cv_folds=10)
    >>> print(f"ROC-AUC: {results['metrics']['test_roc_auc']:.4f}")
    """
    pipeline = TrainingPipeline(
        n_iter=n_iter,
        cv_folds=cv_folds,
        force_replace=force_replace,
        progress_callback=progress_callback,
    )
    return pipeline.run()
