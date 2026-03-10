"""Shared fixtures for pipeline tests."""
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from heart_disease.constants import TARGET_COLUMN, ID_COLUMN


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Age": [45, 55, 60, 38, 50, 65, 42, 59, 48, 52],
        "Sex": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        "Chest pain type": [2, 3, 1, 4, 2, 3, 1, 2, 4, 3],
        "BP": [120, 130, 140, 110, 125, 135, 115, 128, 132, 122],
        "Cholesterol": [200, 250, 220, 180, 240, 260, 190, 230, 210, 245],
        "FBS over 120": [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        "EKG results": [1, 0, 2, 1, 0, 1, 2, 0, 1, 2],
        "Max HR": [150, 140, 130, 170, 145, 135, 160, 142, 155, 148],
        "Exercise angina": [0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
        "ST depression": [1.5, 2.0, 1.0, 0.5, 1.8, 2.5, 0.8, 1.2, 1.6, 1.4],
        "Slope of ST": [2, 1, 2, 3, 1, 2, 3, 2, 1, 2],
        "Number of vessels fluro": [1, 2, 0, 0, 1, 3, 0, 2, 1, 1],
        "Thallium": [5, 6, 7, 5, 6, 7, 5, 6, 7, 5],
        "Heart Disease": ["Presence", "Absence", "Presence", "Absence", "Presence",
                         "Absence", "Absence", "Presence", "Absence", "Presence"],
    })


@pytest.fixture
def sample_csv_file(sample_raw_data, tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "test_data.csv"
    sample_raw_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_transformed_data():
    """Create sample transformed data (after feature engineering)."""
    return pd.DataFrame({
        # Categorical (already mapped to strings)
        "Sex": ["male", "female", "male", "male", "female"],
        "FBS over 120": ["false", "true", "false", "false", "true"],
        "Exercise angina": ["no", "yes", "yes", "no", "yes"],
        "Chest pain type": [2, 3, 1, 4, 2],
        "EKG results": [1, 0, 2, 1, 0],
        "Slope of ST": [2, 1, 2, 3, 1],
        "Number of vessels fluro": [1, 2, 0, 0, 1],
        "Thallium": [5, 6, 7, 5, 6],
        # Numerical
        "Age": [45, 55, 60, 38, 50],
        "BP": [120, 130, 140, 110, 125],
        "Cholesterol": [200, 250, 220, 180, 240],
        "Max HR": [150, 140, 130, 170, 145],
        "ST depression": [1.5, 2.0, 1.0, 0.5, 1.8],
        # Target
        "Heart Disease": ["Presence", "Absence", "Presence", "Absence", "Presence"],
    })


@pytest.fixture
def mock_schema_path(tmp_path):
    """Create a mock schema YAML file for testing."""
    schema_content = """
schema_type: DataFrameSchema
version: 0.1.0

columns:
  id:
    dtype: int64
    nullable: false
    checks:
      greater_than_or_equal_to: 0

  Age:
    dtype: int64
    nullable: false
    checks:
      in_range:
        min_value: 1
        max_value: 120

  Sex:
    dtype: str
    nullable: false
    checks:
      isin: ["male", "female"]

  Chest pain type:
    dtype: int64
    nullable: false
    checks:
      isin: [1, 2, 3, 4]

  BP:
    dtype: int64
    nullable: false
    checks:
      in_range:
        min_value: 50
        max_value: 250

  Cholesterol:
    dtype: int64
    nullable: false
    checks:
      in_range:
        min_value: 100
        max_value: 600

  FBS over 120:
    dtype: str
    nullable: false
    checks:
      isin: ["true", "false"]

  EKG results:
    dtype: int64
    nullable: false
    checks:
      isin: [0, 1, 2]

  Max HR:
    dtype: int64
    nullable: false
    checks:
      in_range:
        min_value: 60
        max_value: 220

  Exercise angina:
    dtype: str
    nullable: false
    checks:
      isin: ["yes", "no"]

  ST depression:
    dtype: float64
    nullable: false
    checks:
      in_range:
        min_value: 0
        max_value: 10

  Slope of ST:
    dtype: int64
    nullable: false
    checks:
      isin: [1, 2, 3]

  Number of vessels fluro:
    dtype: int64
    nullable: false
    checks:
      isin: [0, 1, 2, 3]

  Thallium:
    dtype: int64
    nullable: false
    checks:
      isin: [3, 5, 6, 7]

  Heart Disease:
    dtype: str
    nullable: false
    checks:
      isin: ["Presence", "Absence"]
"""
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(schema_content)
    return schema_path
