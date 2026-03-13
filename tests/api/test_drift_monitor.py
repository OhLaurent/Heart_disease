"""Tests for drift monitoring helpers and report composition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heart_disease.api import drift_monitor
from heart_disease.constants import MIN_PREDICTIONS_FOR_DRIFT


def test_to_recent_features_with_empty_records():
    """Empty records should return empty frame and probabilities array."""
    features, probs = drift_monitor._to_recent_features([])

    assert features.empty
    assert probs.size == 0


def test_compute_drift_marks_insufficient_data():
    """When sample size is below threshold, drift should be gated."""
    recent = pd.DataFrame({
        "Age": [50.0] * 5,
        "Sex": ["male"] * 5,
    })
    baseline = {
        "numerical_features": {
            "Age": {
                "mean": 52.0,
                "histogram": {"counts": [5, 5], "bin_edges": [40.0, 50.0, 60.0]},
            }
        },
        "categorical_features": {
            "Sex": {"male": 0.5, "female": 0.5}
        },
    }

    result = drift_monitor.compute_drift(baseline, recent)

    assert result["numerical"]["Age"]["status"] == "insufficient_data"
    assert result["numerical"]["Age"]["n"] == 5
    assert result["categorical"]["Sex"]["status"] == "insufficient_data"
    assert result["categorical"]["Sex"]["n"] == 5


def test_compute_drift_marks_baseline_unavailable_for_invalid_histogram():
    """Invalid baseline histogram should produce baseline_unavailable status."""
    recent = pd.DataFrame({
        "Age": [55.0] * MIN_PREDICTIONS_FOR_DRIFT,
    })
    baseline = {
        "numerical_features": {
            "Age": {
                "mean": 50.0,
                "histogram": {"counts": [10, 10], "bin_edges": [40.0, 50.0]},
            }
        },
        "categorical_features": {},
    }

    result = drift_monitor.compute_drift(baseline, recent)

    assert result["numerical"]["Age"]["status"] == "baseline_unavailable"
    assert result["numerical"]["Age"]["n"] == MIN_PREDICTIONS_FOR_DRIFT


def test_compute_drift_detects_categorical_drift():
    """Large TV distance should flag categorical drift."""
    recent = pd.DataFrame({
        "Sex": ["male"] * MIN_PREDICTIONS_FOR_DRIFT,
    })
    baseline = {
        "numerical_features": {},
        "categorical_features": {
            "Sex": {"male": 0.1, "female": 0.9}
        },
    }

    result = drift_monitor.compute_drift(baseline, recent)

    assert result["categorical"]["Sex"]["drifted"] is True
    assert result["categorical"]["Sex"]["tv_distance"] > 0.2


def test_performance_summary_includes_baseline_and_high_risk_pct():
    """Performance summary should include aggregate values and baseline payload."""
    probs = np.array([0.2, 0.65, 0.9], dtype=float)
    baseline = {"performance": {"cv_mean_auc": 0.81, "n_samples": 123}}

    summary = drift_monitor.performance_summary(probs, baseline)

    assert summary["total_predictions"] == 3
    assert summary["mean_probability"] == np.mean(probs)
    assert summary["high_risk_pct"] == (2 / 3) * 100
    assert summary["baseline"] == baseline["performance"]


def test_drift_report_for_model_with_empty_recent_data(monkeypatch):
    """Report should be well-formed when there are no persisted rows."""
    monkeypatch.setattr(
        drift_monitor,
        "_load_baseline_stats",
        lambda model_version: ({"performance": {}}, f"models:/heart_disease_model/{model_version}"),
    )
    monkeypatch.setattr(
        drift_monitor,
        "_to_recent_features",
        lambda records: (pd.DataFrame(), np.array([])),
    )

    report = drift_monitor.drift_report_for_model(records=[], model_version="7")

    assert report["model_version"] == "7"
    assert report["sample_size"] == 0
    assert report["has_enough_data"] is False
    assert report["overall_status"] == "insufficient_data"
    assert report["features"] == []


def test_drift_report_for_model_maps_features_and_overall_status(monkeypatch):
    """Report should map internal drift payload into API-friendly feature rows."""
    recent = pd.DataFrame({"Age": [55.0] * MIN_PREDICTIONS_FOR_DRIFT})
    probs = np.array([0.1, 0.9] * (MIN_PREDICTIONS_FOR_DRIFT // 2), dtype=float)

    monkeypatch.setattr(
        drift_monitor,
        "_load_baseline_stats",
        lambda model_version: ({"performance": {"n_samples": 200}}, "models:/heart_disease_model/9"),
    )
    monkeypatch.setattr(
        drift_monitor,
        "_to_recent_features",
        lambda records: (recent, probs),
    )
    monkeypatch.setattr(
        drift_monitor,
        "compute_drift",
        lambda baseline, recent_df: {
            "numerical": {
                "Age": {
                    "ks_statistic": 0.12,
                    "drifted": False,
                    "n": MIN_PREDICTIONS_FOR_DRIFT,
                    "recent_mean": 55.0,
                    "baseline_mean": 53.5,
                }
            },
            "categorical": {
                "Sex": {
                    "tv_distance": 0.4,
                    "drifted": True,
                    "n": MIN_PREDICTIONS_FOR_DRIFT,
                }
            },
        },
    )

    report = drift_monitor.drift_report_for_model(records=[{"dummy": 1}], model_version="9")

    assert report["model_version"] == "9"
    assert report["has_enough_data"] is True
    assert report["overall_status"] == "drifted"
    assert len(report["features"]) == 2

    by_feature = {item["feature"]: item for item in report["features"]}
    assert by_feature["Age"]["status"] == "stable"
    assert by_feature["Age"]["feature_type"] == "numerical"
    assert by_feature["Sex"]["status"] == "drifted"
    assert by_feature["Sex"]["feature_type"] == "categorical"
