from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from heart_disease.api.routes import prediction_store, router

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

STATIC_DIR = Path(__file__).parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up the application...")
    prediction_store.initialize()
    try:
        from heart_disease.api.routes import get_model_reference
        model = get_model_reference()
        logging.info("Active model: version=%s uri=%s", model.version, model.uri)
    except Exception:
        logging.warning(
            "No active model found. The /predict endpoint will return 503 "
            "until a model is trained via POST /api/v1/retrain."
        )
    yield
    logging.info("Shutting down the application...")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Heart Disease Prediction API",
        description="An API for predicting heart disease based on patient data.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api/v1")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def _index():
        """Serve the API documentation as the homepage."""
        return FileResponse(STATIC_DIR / "index.html")
    
    return app

app = create_app()