"""Feature engineering — cleaning, encoding, and feature creation.

Mirrors the transformations from the EDA and modelling notebooks so the
same logic can be reused in a reproducible production pipeline.
"""

from __future__ import annotations

import pandas as pd

from heart_disease.constants import (
    BINARY_MAPPINGS,
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)


class DataTransformer:
    """Clean and type-cast a raw heart-disease DataFrame.

    Steps applied by :meth:`transform`:

    1. Map binary-coded columns (``Sex``, ``FBS over 120``, ``Exercise angina``)
       to descriptive string labels.
    2. Cast categorical columns to ``category`` dtype.
    3. Optionally drop ``id`` and / or ``Heart Disease``.

    The transformer is stateless; every call to :meth:`transform` works on a
    copy of the input and does not mutate the original.

    Parameters
    ----------
    drop_id : bool, optional
        Drop the ``id`` column. Default ``True``.
    drop_target : bool, optional
        Drop the ``Heart Disease`` target column. Default ``False`` (keep it
        for training). Set to ``True`` for inference.

    Examples
    --------
    >>> from heart_disease.features import DataTransformer
    >>> transformer = DataTransformer(drop_id=True, drop_target=True)
    >>> X = transformer.transform(raw_df)
    """

    def __init__(self, *, drop_id: bool = True, drop_target: bool = False) -> None:
        self.drop_id = drop_id
        self.drop_target = drop_target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps and return a new DataFrame."""
        df = df.copy()
        df = self._map_binary_columns(df)
        df = self._cast_categoricals(df)
        df = self._drop_columns(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Replace integer codes with descriptive string labels."""
        for col, mapping in BINARY_MAPPINGS.items():
            if col not in df.columns:
                continue
            if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].map(mapping)
            # Already string/category → leave as-is (idempotent)
        return df

    @staticmethod
    def _cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        """Cast nominated columns to ``category`` dtype."""
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = []
        if self.drop_id and ID_COLUMN in df.columns:
            to_drop.append(ID_COLUMN)
        if self.drop_target and TARGET_COLUMN in df.columns:
            to_drop.append(TARGET_COLUMN)
        return df.drop(columns=to_drop) if to_drop else df

    # ------------------------------------------------------------------
    # Convenience: split features / target for training
    # ------------------------------------------------------------------

    @staticmethod
    def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Return ``(X, y)`` from a transformed training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Transformed DataFrame that **still contains** ``Heart Disease``.

        Returns
        -------
        X : pd.DataFrame
            Feature columns (``id`` and ``Heart Disease`` excluded).
        y : pd.Series
            Binary target: 1 = Presence, 0 = Absence.

        Raises
        ------
        KeyError
            If ``Heart Disease`` is not in *df*.
        """
        if TARGET_COLUMN not in df.columns:
            raise KeyError(
                f"'{TARGET_COLUMN}' not found. "
                "Did you pass drop_target=True to the transformer?"
            )
        drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = (df[TARGET_COLUMN] == "Presence").astype(int).rename("target")
        return X, y