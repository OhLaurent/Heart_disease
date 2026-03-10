"""Tests for heart_disease.pipelines.components.features module."""
import pandas as pd
import pytest

from heart_disease.pipelines.components.features import DataTransformer
from heart_disease.constants import (
    TARGET_COLUMN,
    ID_COLUMN,
    CATEGORICAL_COLUMNS,
    BINARY_MAPPINGS,
)


class TestDataTransformer:
    """Tests for DataTransformer class."""

    def test_transform_basic(self, sample_raw_data):
        """Test basic transformation without dropping columns."""
        transformer = DataTransformer(drop_id=False, drop_target=False)
        df_transformed = transformer.transform(sample_raw_data)

        assert isinstance(df_transformed, pd.DataFrame)
        assert len(df_transformed) == len(sample_raw_data)
        assert ID_COLUMN in df_transformed.columns
        assert TARGET_COLUMN in df_transformed.columns

    def test_transform_drops_id(self, sample_raw_data):
        """Test transformation with id column dropped."""
        transformer = DataTransformer(drop_id=True, drop_target=False)
        df_transformed = transformer.transform(sample_raw_data)

        assert ID_COLUMN not in df_transformed.columns
        assert TARGET_COLUMN in df_transformed.columns

    def test_transform_drops_target(self, sample_raw_data):
        """Test transformation with target column dropped."""
        transformer = DataTransformer(drop_id=False, drop_target=True)
        df_transformed = transformer.transform(sample_raw_data)

        assert ID_COLUMN in df_transformed.columns
        assert TARGET_COLUMN not in df_transformed.columns

    def test_transform_drops_both(self, sample_raw_data):
        """Test transformation with both id and target dropped."""
        transformer = DataTransformer(drop_id=True, drop_target=True)
        df_transformed = transformer.transform(sample_raw_data)

        assert ID_COLUMN not in df_transformed.columns
        assert TARGET_COLUMN not in df_transformed.columns

    def test_binary_columns_mapped_correctly(self, sample_raw_data):
        """Test that binary columns are mapped to string labels."""
        transformer = DataTransformer(drop_id=True, drop_target=False)
        df_transformed = transformer.transform(sample_raw_data)

        # Check Sex mapping
        assert df_transformed["Sex"].isin(["male", "female"]).all()
        assert "male" in df_transformed["Sex"].values
        assert "female" in df_transformed["Sex"].values

        # Check FBS over 120 mapping
        assert df_transformed["FBS over 120"].isin(["true", "false"]).all()

        # Check Exercise angina mapping
        assert df_transformed["Exercise angina"].isin(["yes", "no"]).all()

    def test_categorical_columns_cast_to_category(self, sample_raw_data):
        """Test that categorical columns are cast to category dtype."""
        transformer = DataTransformer(drop_id=True, drop_target=False)
        df_transformed = transformer.transform(sample_raw_data)

        for col in CATEGORICAL_COLUMNS:
            if col in df_transformed.columns:
                assert pd.api.types.is_categorical_dtype(df_transformed[col]), \
                    f"Column {col} is not categorical"

    def test_transform_is_idempotent(self, sample_raw_data):
        """Test that calling transform twice produces the same result."""
        transformer = DataTransformer(drop_id=True, drop_target=False)
        
        df_first = transformer.transform(sample_raw_data)
        df_second = transformer.transform(df_first)

        pd.testing.assert_frame_equal(df_first, df_second)

    def test_transform_does_not_mutate_original(self, sample_raw_data):
        """Test that transform does not mutate the original DataFrame."""
        original_data = sample_raw_data.copy()
        transformer = DataTransformer(drop_id=True, drop_target=False)
        
        transformer.transform(sample_raw_data)

        pd.testing.assert_frame_equal(sample_raw_data, original_data)

    def test_split_features_target(self, sample_transformed_data):
        """Test splitting features and target."""
        X, y = DataTransformer.split_features_target(sample_transformed_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert TARGET_COLUMN not in X.columns
        assert ID_COLUMN not in X.columns
        assert y.name == "target"

    def test_split_features_target_values(self, sample_transformed_data):
        """Test that target values are correctly binarized."""
        X, y = DataTransformer.split_features_target(sample_transformed_data)

        # Target should be binary (0 or 1)
        assert y.isin([0, 1]).all()
        
        # Check that Presence maps to 1 and Absence to 0
        presence_mask = sample_transformed_data[TARGET_COLUMN] == "Presence"
        assert (y[presence_mask] == 1).all()
        
        absence_mask = sample_transformed_data[TARGET_COLUMN] == "Absence"
        assert (y[absence_mask] == 0).all()

    def test_split_features_target_without_target_raises_error(self, sample_raw_data):
        """Test that splitting without target column raises KeyError."""
        df_no_target = sample_raw_data.drop(columns=[TARGET_COLUMN])
        
        with pytest.raises(KeyError) as exc_info:
            DataTransformer.split_features_target(df_no_target)

        assert TARGET_COLUMN in str(exc_info.value)

    def test_transform_with_missing_binary_columns(self, sample_raw_data):
        """Test transform handles missing binary columns gracefully."""
        df_missing = sample_raw_data.drop(columns=["Sex"])
        transformer = DataTransformer(drop_id=True, drop_target=False)
        
        df_transformed = transformer.transform(df_missing)
        
        assert "Sex" not in df_transformed.columns
        assert "FBS over 120" in df_transformed.columns

    def test_transform_with_already_mapped_binary_columns(self, sample_raw_data):
        """Test transform handles already-mapped binary columns."""
        df = sample_raw_data.copy()
        # Pre-map one column
        df["Sex"] = df["Sex"].map(BINARY_MAPPINGS["Sex"])
        
        transformer = DataTransformer(drop_id=True, drop_target=False)
        df_transformed = transformer.transform(df)
        
        # Should still have correct values
        assert df_transformed["Sex"].isin(["male", "female"]).all()

    def test_transform_preserves_numerical_columns(self, sample_raw_data):
        """Test that numerical columns are preserved correctly."""
        transformer = DataTransformer(drop_id=True, drop_target=False)
        df_transformed = transformer.transform(sample_raw_data)

        numerical_cols = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
        
        for col in numerical_cols:
            assert col in df_transformed.columns
            assert pd.api.types.is_numeric_dtype(df_transformed[col])

    def test_split_features_keeps_all_features(self, sample_transformed_data):
        """Test that split_features_target keeps all feature columns."""
        X, y = DataTransformer.split_features_target(sample_transformed_data)
        
        expected_features = set(sample_transformed_data.columns) - {TARGET_COLUMN, ID_COLUMN}
        actual_features = set(X.columns)
        
        assert expected_features == actual_features

    def test_transform_with_empty_dataframe(self):
        """Test transform with empty DataFrame."""
        df_empty = pd.DataFrame()
        transformer = DataTransformer(drop_id=True, drop_target=False)
        
        df_transformed = transformer.transform(df_empty)
        
        assert isinstance(df_transformed, pd.DataFrame)
        assert len(df_transformed) == 0

    def test_multiple_transformers_independent(self, sample_raw_data):
        """Test that multiple transformer instances are independent."""
        transformer1 = DataTransformer(drop_id=True, drop_target=False)
        transformer2 = DataTransformer(drop_id=False, drop_target=True)
        
        df1 = transformer1.transform(sample_raw_data)
        df2 = transformer2.transform(sample_raw_data)
        
        assert ID_COLUMN not in df1.columns
        assert TARGET_COLUMN in df1.columns
        assert ID_COLUMN in df2.columns
        assert TARGET_COLUMN not in df2.columns
