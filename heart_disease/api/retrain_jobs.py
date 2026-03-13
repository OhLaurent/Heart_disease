from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from heart_disease.pipelines.train import train_pipeline


@dataclass(slots=True)
class RetrainJob:
    """In-memory representation of a retraining job state."""

    job_id: str
    n_iter: int
    cv_splits: int
    force_replacement: bool
    status: str
    stage: str
    progress_pct: int
    message: str
    started_at: str
    updated_at: str
    started_monotonic: float
    finished_at: str | None = None
    model_uri: str | None = None
    cv_mean_auc: float | None = None
    error: str | None = None


class RetrainJobManager:
    """Manage async retraining jobs with thread-safe status updates."""

    def __init__(self) -> None:
        self._jobs: dict[str, RetrainJob] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def start_job(self, *, n_iter: int, cv_splits: int, force_replacement: bool) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        now = self._now_iso()
        job = RetrainJob(
            job_id=job_id,
            n_iter=n_iter,
            cv_splits=cv_splits,
            force_replacement=force_replacement,
            status="queued",
            stage="queued",
            progress_pct=0,
            message="Retraining job queued.",
            started_at=now,
            updated_at=now,
            started_monotonic=monotonic(),
        )

        with self._lock:
            self._jobs[job_id] = job

        worker = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        worker.start()

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Retraining started. Check job status for progress.",
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return self._to_payload(job)

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = self._now_iso()

    def _to_payload(self, job: RetrainJob) -> dict[str, Any]:
        elapsed = max(0.0, monotonic() - job.started_monotonic)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "stage": job.stage,
            "progress_pct": int(job.progress_pct),
            "message": job.message,
            "started_at": job.started_at,
            "updated_at": job.updated_at,
            "finished_at": job.finished_at,
            "elapsed_seconds": round(elapsed, 2),
            "model_uri": job.model_uri,
            "cv_mean_auc": job.cv_mean_auc,
            "error": job.error,
        }

    def _run_job(self, job_id: str) -> None:
        def report(stage: str, message: str, progress_pct: int) -> None:
            self._update_job(
                job_id,
                status="running",
                stage=stage,
                message=message,
                progress_pct=progress_pct,
            )

        report("initializing", "Initializing training pipeline...", 5)

        try:
            with self._lock:
                job = self._jobs[job_id]
                n_iter = job.n_iter
                cv_splits = job.cv_splits
                force_replacement = job.force_replacement

            results = train_pipeline(
                n_iter=n_iter,
                cv_folds=cv_splits,
                force_replace=force_replacement,
                progress_callback=report,
            )

            cv_auc = results["metrics"].get("cv_roc_auc")
            test_auc = results["metrics"].get("test_roc_auc")
            promoted = bool(results.get("promoted"))
            model_uri = f"models:/{results['run_id']}"

            if promoted:
                message = (
                    f"Training completed and model promoted. "
                    f"CV ROC-AUC: {cv_auc:.4f}, Test ROC-AUC: {test_auc:.4f}"
                )
            else:
                message = (
                    f"Training completed but model was not promoted. "
                    f"CV ROC-AUC: {cv_auc:.4f}, Test ROC-AUC: {test_auc:.4f}"
                )

            self._update_job(
                job_id,
                status="completed",
                stage="completed",
                progress_pct=100,
                message=message,
                model_uri=model_uri,
                cv_mean_auc=cv_auc,
                finished_at=self._now_iso(),
            )

        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                stage="failed",
                progress_pct=100,
                message="Training failed.",
                error=str(exc),
                finished_at=self._now_iso(),
            )
