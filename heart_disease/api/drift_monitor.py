from __future__ import annotations

import json
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from heart_disease.constants import (
    CATEGORICAL_COLUMNS,
    DRIFT_KS_P_THRESHOLD,
    DRIFT_TV_THRESHOLD,
    HIGH_RISK_PROBABILITY_THRESHOLD,
    MIN_PREDICTIONS_FOR_DRIFT,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    NUMERICAL_COLUMNS,
)
from heart_disease.pipelines.components.features import DataTransformer


def _configure_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _load_baseline_stats(model_version: str) -> tuple[dict[str, Any], str]:
    """Load baseline_stats.json artifact and return (baseline, model_uri)."""
    _configure_mlflow()
    client = MlflowClient()
    mv = client.get_model_version(name=MLFLOW_MODEL_NAME, version=model_version)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id,
        artifact_path="baseline_stats.json",
    )
    with open(local_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    return baseline, f"models:/{MLFLOW_MODEL_NAME}/{model_version}"


def _to_recent_features(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, np.ndarray]:
    if not records:
        return pd.DataFrame(), np.array([])

    input_rows = [r["input_data"] for r in records]
    output_rows = [r["output_data"] for r in records]

    raw_df = pd.DataFrame(input_rows)
    transformer = DataTransformer(drop_id=True, drop_target=False)
    recent_features = transformer.transform(raw_df)

    probabilities = np.array([float(out["probability"]) for out in output_rows], dtype=float)
    return recent_features, probabilities


def compute_drift(baseline: dict[str, Any], recent: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compare recent prediction features against training baseline stats."""
    from scipy.stats import ks_2samp

    result: dict[str, dict[str, Any]] = {"numerical": {}, "categorical": {}}

    num_baseline = baseline.get("numerical_features", {})
    for col in NUMERICAL_COLUMNS:
        if col not in recent.columns or col not in num_baseline:
            continue

        recent_vals = recent[col].dropna().astype(float).to_numpy()
        if len(recent_vals) < MIN_PREDICTIONS_FOR_DRIFT:
            result["numerical"][col] = {
                "status": "insufficient_data",
                "n": int(len(recent_vals)),
            }
            continue

        hist = num_baseline[col].get("histogram") or {}
        counts = np.array(hist.get("counts", []), dtype=int)
        bin_edges = np.array(hist.get("bin_edges", []), dtype=float)

        if len(counts) == 0 or len(bin_edges) != len(counts) + 1:
            result["numerical"][col] = {
                "status": "baseline_unavailable",
                "n": int(len(recent_vals)),
            }
            continue

        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        baseline_sample = np.repeat(bin_mids, counts)

        stat, p_value = ks_2samp(baseline_sample, recent_vals)
        result["numerical"][col] = {
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "drifted": bool(p_value < DRIFT_KS_P_THRESHOLD),
            "n": int(len(recent_vals)),
            "recent_mean": float(recent_vals.mean()),
            "baseline_mean": float(num_baseline[col]["mean"]),
        }

    cat_baseline = baseline.get("categorical_features", {})
    n_total = len(recent)
    for col in CATEGORICAL_COLUMNS:
        if col not in recent.columns or col not in cat_baseline:
            continue

        if n_total < MIN_PREDICTIONS_FOR_DRIFT:
            result["categorical"][col] = {
                "status": "insufficient_data",
                "n": int(n_total),
            }
            continue

        recent_freq = recent[col].astype(str).value_counts(normalize=True, dropna=False)
        baseline_freq: dict[str, float] = cat_baseline[col]
        all_cats = set(recent_freq.index) | set(baseline_freq.keys())
        tv = max(abs(float(recent_freq.get(cat, 0.0)) - float(baseline_freq.get(cat, 0.0))) for cat in all_cats)

        result["categorical"][col] = {
            "tv_distance": float(tv),
            "drifted": bool(tv > DRIFT_TV_THRESHOLD),
            "n": int(n_total),
        }

    return result


def performance_summary(probabilities: np.ndarray, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_predictions": int(len(probabilities)),
        "mean_probability": None,
        "high_risk_pct": None,
    }

    if len(probabilities):
        summary["mean_probability"] = float(np.mean(probabilities))
        summary["high_risk_pct"] = float(np.mean(probabilities >= HIGH_RISK_PROBABILITY_THRESHOLD) * 100.0)

    if baseline and "performance" in baseline:
        summary["baseline"] = baseline["performance"]

    return summary


def drift_report_for_model(
    *,
    records: list[dict[str, Any]],
    model_version: str,
) -> dict[str, Any]:
    baseline, model_uri = _load_baseline_stats(model_version)
    recent_features, probabilities = _to_recent_features(records)

    sample_size = int(len(recent_features))
    has_enough_data = sample_size >= MIN_PREDICTIONS_FOR_DRIFT
    report = compute_drift(baseline, recent_features) if sample_size else {"numerical": {}, "categorical": {}}

    features: list[dict[str, Any]] = []

    for col, payload in report["numerical"].items():
        if payload.get("status"):
            features.append(
                {
                    "feature": col,
                    "feature_type": "numerical",
                    "baseline_value": "-",
                    "current_value": "-",
                    "score": None,
                    "status": payload["status"],
                    "sample_size": payload.get("n", sample_size),
                }
            )
            continue

        features.append(
            {
                "feature": col,
                "feature_type": "numerical",
                "baseline_value": f"mean={payload['baseline_mean']:.3f}",
                "current_value": f"mean={payload['recent_mean']:.3f}",
                "score": round(payload["ks_statistic"], 4),
                "status": "drifted" if payload["drifted"] else "stable",
                "sample_size": payload["n"],
            }
        )

    for col, payload in report["categorical"].items():
        if payload.get("status"):
            features.append(
                {
                    "feature": col,
                    "feature_type": "categorical",
                    "baseline_value": "-",
                    "current_value": "-",
                    "score": None,
                    "status": payload["status"],
                    "sample_size": payload.get("n", sample_size),
                }
            )
            continue

        features.append(
            {
                "feature": col,
                "feature_type": "categorical",
                "baseline_value": "distribution",
                "current_value": "distribution",
                "score": round(payload["tv_distance"], 4),
                "status": "drifted" if payload["drifted"] else "stable",
                "sample_size": payload["n"],
            }
        )

    if not has_enough_data:
        overall_status = "insufficient_data"
    elif any(f["status"] == "drifted" for f in features):
        overall_status = "drifted"
    else:
        overall_status = "stable"

    return {
        "model_version": str(model_version),
        "model_uri": model_uri,
        "min_predictions_required": MIN_PREDICTIONS_FOR_DRIFT,
        "sample_size": sample_size,
        "has_enough_data": has_enough_data,
        "overall_status": overall_status,
        "performance_summary": performance_summary(probabilities, baseline),
        "features": sorted(features, key=lambda x: (x["feature_type"], x["feature"])),
    }
