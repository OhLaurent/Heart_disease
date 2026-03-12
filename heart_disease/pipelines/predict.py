"""Prediction pipeline for heart disease model.

Loads the active model from MLflow model registry and makes predictions
for one or more patients. Reuses data validation and feature engineering
components from the training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from heart_disease.constants import (
    MLFLOW_ACTIVE_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    POSITIVE_TARGET_LABEL,
    TARGET_COLUMN,
    TARGET_VALUE_TO_LABEL,
)
from heart_disease.pipelines.components.dataset import DataLoader, DataValidator
from heart_disease.pipelines.components.features import DataTransformer


def _configure_mlflow() -> None:
    """Apply shared MLflow configuration for tracking and experiments."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


@dataclass(frozen=True)
class ModelReference:
    """Minimal model metadata associated with a prediction run."""

    version: str
    uri: str


@dataclass(frozen=True)
class PredictionRunResult:
    """Predictions produced by a model plus the model reference used."""

    predictions: pd.DataFrame
    model: ModelReference


def get_model_reference(
    model_name: str = MLFLOW_MODEL_NAME,
    model_alias: str = MLFLOW_ACTIVE_ALIAS,
) -> ModelReference:
    """Resolve the model reference for the requested MLflow alias."""
    _configure_mlflow()
    client = MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(model_name, model_alias)
    except mlflow.exceptions.MlflowException as exc:
        raise ValueError(
            f"No model found with alias '{model_alias}' "
            f"for model '{model_name}'. "
            f"Have you trained and promoted a model yet?"
        ) from exc

    return ModelReference(
        version=str(model_version.version),
        uri=f"models:/{model_name}@{model_alias}",
    )


class PredictionPipeline:
    """Pipeline for making predictions with the active heart disease model.

    Loads the current active model from MLflow and applies it to new patient data.
    Handles data loading, validation, transformation, and prediction in a single
    unified interface.

    Parameters
    ----------
    model_name : str, optional
        Name of the model in MLflow model registry.
        Defaults to :data:`~heart_disease.constants.MLFLOW_MODEL_NAME`.
    model_alias : str, optional
        Alias of the model version to use (e.g., "active", "champion").
        Defaults to :data:`~heart_disease.constants.MLFLOW_ACTIVE_ALIAS`.

    Attributes
    ----------
    model_ : sklearn.pipeline.Pipeline or None
        The loaded ML model. Set after calling :meth:`load_model`.
    model_version_ : str or None
        Version number of the loaded model.

    Examples
    --------
    Predict from a DataFrame:

    >>> pipeline = PredictionPipeline()
    >>> pipeline.load_model()
    >>> results = pipeline.predict(patient_df)

    Predict from a CSV file:

    >>> results = pipeline.predict_from_file("data/new_patients.csv")

    Get both predictions and probabilities:

    >>> results = pipeline.predict(patient_df, return_proba=True)
    >>> results[['id', 'prediction', 'probability_Absence', 'probability_Presence']]
    """

    def __init__(
        self,
        model_name: str = MLFLOW_MODEL_NAME,
        model_alias: str = MLFLOW_ACTIVE_ALIAS,
    ) -> None:
        self.model_name = model_name
        self.model_alias = model_alias

        # Will be set during load_model()
        self.model_: Any | None = None
        self.model_version_: str | None = None
        self.model_uri_: str | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> PredictionPipeline:
        """Load the active model from MLflow model registry.

        Returns
        -------
        self
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If no model with the specified alias is found.

        Examples
        --------
        >>> pipeline = PredictionPipeline()
        >>> pipeline.load_model()
        >>> print(f"Loaded model version: {pipeline.model_version_}")
        """
        model_reference = get_model_reference(self.model_name, self.model_alias)
        self.model_version_ = model_reference.version
        self.model_uri_ = model_reference.uri
        self.model_ = mlflow.sklearn.load_model(self.model_uri_)

        print(
            f"✓ Loaded {self.model_name} (version {self.model_version_}, "
            f"alias: {self.model_alias})"
        )

        return self

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and transform input data for prediction.

        Parameters
        ----------
        data : pd.DataFrame
            Raw input data with all required features.

        Returns
        -------
        pd.DataFrame
            Transformed feature DataFrame ready for model prediction.
        """
        # Validate schema (inference mode — no target column allowed)
        validator = DataValidator(mode="inference")
        df = validator.validate(data)

        # Transform features (drop ID, no target to drop since it's inference)
        transformer = DataTransformer(drop_id=True, drop_target=False)
        df_transformed = transformer.transform(df)

        return df_transformed

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_target_label(value: Any) -> str:
        """Map model target values to canonical labels using shared constants."""
        try:
            return TARGET_VALUE_TO_LABEL[value]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported class label for heart disease target: {value!r}. "
                f"Expected one of: {list(TARGET_VALUE_TO_LABEL)}"
            ) from exc

    def predict(
        self,
        data: pd.DataFrame,
        *,
        return_proba: bool = False,
        include_input: bool = True,
    ) -> PredictionRunResult:
        """Make predictions on patient data.

        Parameters
        ----------
        data : pd.DataFrame
            Input patient data. Must contain all required features and
            must NOT contain the target column (``Heart Disease``).
        return_proba : bool, optional
            If ``True``, include prediction probabilities for each class.
            Default ``False``.
        include_input : bool, optional
            If ``True``, include original input columns in the results.
            Default ``True``.

        Returns
        -------
        PredictionRunResult
            Prediction results and the model reference used. The DataFrame always includes:
            - ``prediction`` : predicted class ("Absence" or "Presence")

            If ``return_proba=True``, also includes:
            - ``probability_Absence`` : probability of class "Absence"
            - ``probability_Presence`` : probability of class "Presence"

            If ``include_input=True``, includes all original input columns.

        Raises
        ------
        ValueError
            If model has not been loaded yet or if data validation fails.

        Examples
        --------
        >>> pipeline = PredictionPipeline().load_model()
        >>> results = pipeline.predict(patients_df, return_proba=True)
        >>> results[['id', 'prediction', f'probability_{POSITIVE_TARGET_LABEL}']]
        """
        if self.model_ is None:
            raise ValueError(
                "Model not loaded. Call load_model() first or use predict_from_file()."
            )
        if self.model_version_ is None or self.model_uri_ is None:
            raise ValueError("Model metadata not loaded. Call load_model() before predict().")

        # Prepare features
        X = self._prepare_data(data)

        # Make predictions
        raw_predictions = self.model_.predict(X)
        predictions = [self._normalize_target_label(v) for v in raw_predictions]

        # Build results DataFrame
        results = pd.DataFrame({"prediction": predictions})

        # Add probabilities if requested
        if return_proba:
            probabilities = self.model_.predict_proba(X)
            class_names = self.model_.classes_

            for i, class_name in enumerate(class_names):
                normalized_name = self._normalize_target_label(class_name)
                results[f"probability_{normalized_name}"] = probabilities[:, i]

        # Prepend input data if requested
        if include_input:
            results = pd.concat([data.reset_index(drop=True), results], axis=1)

        return PredictionRunResult(
            predictions=results,
            model=ModelReference(version=self.model_version_, uri=self.model_uri_),
        )

    def predict_from_file(
        self,
        file_path: str | Path,
        *,
        return_proba: bool = False,
        include_input: bool = True,
    ) -> PredictionRunResult:
        """Make predictions from a CSV file.

        Convenience method that loads data from disk and makes predictions.
        Automatically loads the model if not already loaded.

        Parameters
        ----------
        file_path : str | Path
            Path to CSV file containing patient data.
        return_proba : bool, optional
            If ``True``, include prediction probabilities. Default ``False``.
        include_input : bool, optional
            If ``True``, include original input columns. Default ``True``.

        Returns
        -------
        PredictionRunResult
            Prediction results (see :meth:`predict` for details).

        Examples
        --------
        >>> pipeline = PredictionPipeline()
        >>> results = pipeline.predict_from_file(
        ...     "data/new_patients.csv",
        ...     return_proba=True
        ... )
        """
        # Load model if not already loaded
        if self.model_ is None:
            self.load_model()

        # Load data (don't drop target — validator will check it's not present)
        loader = DataLoader(file_path, drop_target=False)
        data = loader.load()

        # Check that target column is not present
        if TARGET_COLUMN in data.columns:
            raise ValueError(
                f"Input data contains target column '{TARGET_COLUMN}'. "
                f"Prediction data should not include the target."
            )

        return self.predict(data, return_proba=return_proba, include_input=include_input)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def predict_patients(
    data: pd.DataFrame | str | Path,
    *,
    return_proba: bool = False,
    include_input: bool = True,
    model_name: str = MLFLOW_MODEL_NAME,
    model_alias: str = MLFLOW_ACTIVE_ALIAS,
) -> PredictionRunResult:
    """Make predictions using the active model (convenience function).

    This is a simplified interface to :class:`PredictionPipeline` for
    one-off predictions without needing to instantiate the class.

    Parameters
    ----------
    data : pd.DataFrame or str or Path
        Patient data as a DataFrame or path to a CSV file.
    return_proba : bool, optional
        Include prediction probabilities. Default ``False``.
    include_input : bool, optional
        Include original input columns in results. Default ``True``.
    model_name : str, optional
        MLflow model name. Defaults to configured model name.
    model_alias : str, optional
        Model alias to use. Defaults to "active".

    Returns
    -------
    PredictionRunResult
        Prediction results with model metadata.

    Examples
    --------
    Predict from DataFrame:

    >>> from heart_disease.pipelines.predict import predict_patients
    >>> results = predict_patients(patients_df, return_proba=True)

    Predict from file:

    >>> results = predict_patients("data/new_patients.csv")
    """
    pipeline = PredictionPipeline(model_name=model_name, model_alias=model_alias)

    if isinstance(data, (str, Path)):
        return pipeline.predict_from_file(
            data, return_proba=return_proba, include_input=include_input
        )
    else:
        pipeline.load_model()
        return pipeline.predict(
            data, return_proba=return_proba, include_input=include_input
        )
