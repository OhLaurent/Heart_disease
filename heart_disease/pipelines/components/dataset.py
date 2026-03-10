"""Dataset utilities — loading and validation.

Two classes are provided:

* :class:`DataLoader`   — reads a raw CSV from disk.
* :class:`DataValidator` — validates a DataFrame against ``config/schema.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pandera.pandas as pa
import yaml

from heart_disease.constants import TARGET_COLUMN, SCHEMA_PATH

# Columns that must be absent from inference payloads.
_INFERENCE_EXCLUDE = {TARGET_COLUMN}


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class DataLoader:
    """Load raw heart-disease CSV data into a DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file to load.
    drop_target : bool, optional
        Drop the ``Heart Disease`` target column after loading.
        Set to ``True`` for production inference payloads. Default ``False``.

    Examples
    --------
    >>> from heart_disease.dataset import DataLoader
    >>> df = DataLoader("data/raw/heart_disease.csv").load()
    """

    def __init__(self, path: str | Path, *, drop_target: bool = False) -> None:
        self.path = Path(path)
        self.drop_target = drop_target

    def load(self) -> pd.DataFrame:
        """Read the CSV and return a :class:`~pandas.DataFrame`.

        Raises
        ------
        FileNotFoundError
            If ``self.path`` does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")

        df = pd.read_csv(self.path)

        if self.drop_target and TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])

        return df


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------


class DataValidator:
    """Validate a DataFrame against the production schema defined in YAML.

    The schema is loaded from ``config/schema.yaml`` and interpreted via
    `pandera <https://pandera.readthedocs.io/>`_.

    Parameters
    ----------
    schema_path : str | Path, optional
        Override the path to the YAML schema file.
        Defaults to :data:`~heart_disease.config.SCHEMA_PATH`.
    mode : {'training', 'inference'}
        ``'training'``  — validates all columns including ``Heart Disease``.
        ``'inference'`` — ``Heart Disease`` is excluded from the schema and
        raises :exc:`ValueError` if it is present in the input.

    Examples
    --------
    >>> from heart_disease.dataset import DataValidator
    >>> df = DataValidator(mode="inference").validate(df)
    """

    def __init__(
        self,
        schema_path: str | Path | None = None,
        *,
        mode: Literal["training", "inference"] = "inference",
    ) -> None:
        self.schema_path = Path(schema_path) if schema_path else SCHEMA_PATH
        self.mode = mode
        self._schema: pa.DataFrameSchema | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate *df* and return it unchanged if valid.

        Raises
        ------
        pa.errors.SchemaError
            If any column fails a schema check.
        ValueError
            If ``mode='inference'`` and ``Heart Disease`` is present in *df*.
        """
        if self.mode == "inference" and TARGET_COLUMN in df.columns:
            raise ValueError(
                f"Column '{TARGET_COLUMN}' must not be present in inference "
                "payloads. Pass drop_target=True to DataLoader, or drop it manually."
            )

        return self._get_schema().validate(df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_schema(self) -> pa.DataFrameSchema:
        if self._schema is not None:
            return self._schema
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        self._schema = self._build_schema(self._load_yaml())
        return self._schema

    def _load_yaml(self) -> dict:
        with open(self.schema_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _build_schema(self, raw: dict) -> pa.DataFrameSchema:
        columns: dict[str, pa.Column] = {}
        for col_name, spec in raw.get("columns", {}).items():
            if self.mode == "inference" and col_name in _INFERENCE_EXCLUDE:
                continue
            columns[col_name] = pa.Column(
                dtype=spec.get("dtype"),
                checks=self._build_checks(spec.get("checks") or {}),
                nullable=bool(spec.get("nullable", True)),
                required=True,
            )
        return pa.DataFrameSchema(columns=columns, strict=False, coerce=True)

    @staticmethod
    def _build_checks(checks_yaml: dict) -> list[pa.Check]:
        checks: list[pa.Check] = []
        for name, args in checks_yaml.items():
            if name == "isin":
                checks.append(pa.Check.isin(args))
            elif name == "in_range":
                checks.append(pa.Check.in_range(args["min_value"], args["max_value"]))
            elif name == "greater_than_or_equal_to":
                checks.append(pa.Check.greater_than_or_equal_to(args))
            elif name == "less_than_or_equal_to":
                checks.append(pa.Check.less_than_or_equal_to(args))
        return checks