"""Tests for heart_disease.pipelines.components.dataset module."""
import pandas as pd
import pytest
from pathlib import Path

from heart_disease.pipelines.components.dataset import DataLoader, DataValidator
from heart_disease.constants import TARGET_COLUMN


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_existing_file(self, sample_csv_file):
        """Test loading data from an existing CSV file."""
        loader = DataLoader(sample_csv_file, drop_target=False)
        df = loader.load()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert TARGET_COLUMN in df.columns

    def test_load_with_drop_target(self, sample_csv_file):
        """Test loading data with target column dropped."""
        loader = DataLoader(sample_csv_file, drop_target=True)
        df = loader.load()

        assert isinstance(df, pd.DataFrame)
        assert TARGET_COLUMN not in df.columns

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from a non-existent file raises FileNotFoundError."""
        fake_path = tmp_path / "nonexistent.csv"
        loader = DataLoader(fake_path, drop_target=False)

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load()

        assert "not found" in str(exc_info.value).lower()

    def test_load_preserves_data_types(self, sample_csv_file):
        """Test that loading preserves expected data types."""
        loader = DataLoader(sample_csv_file, drop_target=False)
        df = loader.load()

        # Check that numeric columns are numeric
        assert pd.api.types.is_numeric_dtype(df["Age"])
        assert pd.api.types.is_numeric_dtype(df["BP"])
        assert pd.api.types.is_numeric_dtype(df["Cholesterol"])

    def test_path_as_string(self, sample_csv_file):
        """Test that path can be passed as string."""
        loader = DataLoader(str(sample_csv_file), drop_target=False)
        df = loader.load()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_path_as_path_object(self, sample_csv_file):
        """Test that path can be passed as Path object."""
        loader = DataLoader(Path(sample_csv_file), drop_target=False)
        df = loader.load()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_validate_training_mode_with_target(self, sample_raw_data, mock_schema_path):
        """Test validation in training mode with target column present."""
        # Transform binary columns first
        df = sample_raw_data.copy()
        df["Sex"] = df["Sex"].map({1: "male", 0: "female"})
        df["FBS over 120"] = df["FBS over 120"].map({1: "true", 0: "false"})
        df["Exercise angina"] = df["Exercise angina"].map({1: "yes", 0: "no"})

        validator = DataValidator(schema_path=mock_schema_path, mode="training")
        validated_df = validator.validate(df)

        assert isinstance(validated_df, pd.DataFrame)
        assert TARGET_COLUMN in validated_df.columns

    def test_validate_inference_mode_without_target(self, sample_raw_data, mock_schema_path):
        """Test validation in inference mode without target column."""
        df = sample_raw_data.copy()
        # Transform binary columns
        df["Sex"] = df["Sex"].map({1: "male", 0: "female"})
        df["FBS over 120"] = df["FBS over 120"].map({1: "true", 0: "false"})
        df["Exercise angina"] = df["Exercise angina"].map({1: "yes", 0: "no"})
        # Drop target
        df = df.drop(columns=[TARGET_COLUMN])

        validator = DataValidator(schema_path=mock_schema_path, mode="inference")
        validated_df = validator.validate(df)

        assert isinstance(validated_df, pd.DataFrame)
        assert TARGET_COLUMN not in validated_df.columns

    def test_validate_inference_mode_with_target_raises_error(self, sample_raw_data, mock_schema_path):
        """Test that inference mode raises error if target column is present."""
        df = sample_raw_data.copy()
        df["Sex"] = df["Sex"].map({1: "male", 0: "female"})
        df["FBS over 120"] = df["FBS over 120"].map({1: "true", 0: "false"})
        df["Exercise angina"] = df["Exercise angina"].map({1: "yes", 0: "no"})

        validator = DataValidator(schema_path=mock_schema_path, mode="inference")

        with pytest.raises(ValueError) as exc_info:
            validator.validate(df)

        assert TARGET_COLUMN in str(exc_info.value)
        assert "must not be present" in str(exc_info.value).lower()

    def test_validate_with_invalid_schema_path(self, sample_raw_data, tmp_path):
        """Test validation with non-existent schema file."""
        fake_schema = tmp_path / "nonexistent_schema.yaml"
        validator = DataValidator(schema_path=fake_schema, mode="training")

        with pytest.raises(FileNotFoundError):
            validator.validate(sample_raw_data)

    def test_validate_with_out_of_range_values(self, sample_raw_data, mock_schema_path):
        """Test validation fails with out-of-range values."""
        df = sample_raw_data.copy()
        df["Sex"] = df["Sex"].map({1: "male", 0: "female"})
        df["FBS over 120"] = df["FBS over 120"].map({1: "true", 0: "false"})
        df["Exercise angina"] = df["Exercise angina"].map({1: "yes", 0: "no"})
        
        # Set invalid age
        df.loc[0, "Age"] = 150  # Exceeds max of 120

        validator = DataValidator(schema_path=mock_schema_path, mode="training")

        with pytest.raises(Exception):  # Pandera will raise SchemaError
            validator.validate(df)

    def test_validate_with_invalid_categorical(self, sample_raw_data, mock_schema_path):
        """Test validation fails with invalid categorical values."""
        df = sample_raw_data.copy()
        df["Sex"] = df["Sex"].map({1: "male", 0: "female"})
        df["FBS over 120"] = df["FBS over 120"].map({1: "true", 0: "false"})
        df["Exercise angina"] = df["Exercise angina"].map({1: "yes", 0: "no"})
        
        # Set invalid value
        df.loc[0, "Sex"] = "invalid"

        validator = DataValidator(schema_path=mock_schema_path, mode="training")

        with pytest.raises(Exception):  # Pandera will raise SchemaError
            validator.validate(df)

    def test_validate_caches_schema(self, sample_raw_data, mock_schema_path):
        """Test that schema is cached after first validation."""
        df = sample_raw_data.copy()
        df["Sex"] = df["Sex"].map({1: "male", 0: "female"})
        df["FBS over 120"] = df["FBS over 120"].map({1: "true", 0: "false"})
        df["Exercise angina"] = df["Exercise angina"].map({1: "yes", 0: "no"})

        validator = DataValidator(schema_path=mock_schema_path, mode="training")
        
        # First validation
        validator.validate(df)
        assert validator._schema is not None
        
        # Second validation should use cached schema
        validator.validate(df)
        assert validator._schema is not None

    def test_default_schema_path(self, sample_raw_data):
        """Test that default schema path is used when not specified."""
        validator = DataValidator(mode="training")
        assert validator.schema_path is not None

    def test_inference_mode_excludes_target_from_schema(self, mock_schema_path):
        """Test that inference mode excludes target column from schema."""
        validator = DataValidator(schema_path=mock_schema_path, mode="inference")
        schema = validator._get_schema()
        
        # Target column should not be in the schema columns
        assert TARGET_COLUMN not in schema.columns
