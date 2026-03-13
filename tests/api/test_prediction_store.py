"""Tests for SQLite-backed prediction history persistence."""

from pathlib import Path

from heart_disease.api.prediction_store import PredictionStore


def test_save_and_list_predictions(tmp_path: Path):
    """Prediction store should persist inputs and outputs in SQLite."""
    store = PredictionStore(tmp_path / "predictions.db")

    store.save_prediction_run(
        inputs=[{"Age": 55, "Sex": 1}],
        outputs=[{"prediction": "Presence", "probability": 0.7}],
        model_version="3",
        model_uri="models:/heart_disease_model@active",
    )

    rows = store.list_predictions()

    assert len(rows) == 1
    assert rows[0]["model_version"] == "3"
    assert rows[0]["input_data"]["Age"] == 55
    assert rows[0]["output_data"]["prediction"] == "Presence"


def test_list_models_aggregates_prediction_counts(tmp_path: Path):
    """Prediction store should expose distinct models for filtering."""
    store = PredictionStore(tmp_path / "predictions.db")

    store.save_prediction_run(
        inputs=[{"Age": 55}, {"Age": 60}],
        outputs=[
            {"prediction": "Presence", "probability": 0.7},
            {"prediction": "Absence", "probability": 0.3},
        ],
        model_version="3",
        model_uri="models:/heart_disease_model@active",
    )

    models = store.list_models()

    assert len(models) == 1
    assert models[0]["model_version"] == "3"
    assert models[0]["prediction_count"] == 2