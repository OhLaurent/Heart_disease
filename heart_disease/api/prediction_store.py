from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from heart_disease.constants import PREDICTIONS_DB_PATH


class PredictionStore:
    """Persist API prediction requests and outputs in SQLite."""

    def __init__(self, db_path: Path = PREDICTIONS_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    patient_index INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    model_uri TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    output_data TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def count_model_predictions(self, model_version: str) -> int:
        self.initialize()
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM prediction_history WHERE model_version = ?",
                (model_version,),
            ).fetchone()
        return int(result[0]) if result else 0

    def save_prediction_run(
        self,
        *,
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        model_version: str,
        model_uri: str,
    ) -> None:
        self.initialize()
        created_at = datetime.now(UTC).isoformat()
        request_id = str(uuid.uuid4())

        predictions_count = self.count_model_predictions(model_version)
        print(f"Prediction run for model version '{model_version}' has {predictions_count} previous predictions.")
        rows = [
            (
                created_at,
                request_id,
                predictions_count + index,
                model_version,
                model_uri,
                json.dumps(input_row),
                json.dumps(output_row),
            )
            for index, (input_row, output_row) in enumerate(zip(inputs, outputs, strict=True))
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO prediction_history (
                    created_at,
                    request_id,
                    patient_index,
                    model_version,
                    model_uri,
                    input_data,
                    output_data
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def list_predictions(self, *, model_version: str | None = None) -> list[dict[str, Any]]:
        self.initialize()
        query = (
            "SELECT id, created_at, request_id, patient_index, model_version, model_uri, input_data, output_data "
            "FROM prediction_history "
        )
        params: tuple[Any, ...] = ()
        if model_version:
            query += "WHERE model_version = ? "
            params = (model_version,)
        query += "ORDER BY created_at DESC, patient_index DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "id": int(row["id"]),
                "created_at": row["created_at"],
                "request_id": row["request_id"],
                "patient_index": int(row["patient_index"]),
                "model_version": row["model_version"],
                "model_uri": row["model_uri"],
                "input_data": json.loads(row["input_data"]),
                "output_data": json.loads(row["output_data"]),
            }
            for row in rows
        ]

    def list_models(self) -> list[dict[str, Any]]:
        self.initialize()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT model_version, model_uri, COUNT(*) AS prediction_count, MAX(created_at) AS latest_prediction_at
                FROM prediction_history
                GROUP BY model_version, model_uri
                ORDER BY latest_prediction_at DESC, model_version DESC
                """
            ).fetchall()

        return [
            {
                "model_version": row["model_version"],
                "model_uri": row["model_uri"],
                "prediction_count": int(row["prediction_count"]),
                "latest_prediction_at": row["latest_prediction_at"],
            }
            for row in rows
        ]