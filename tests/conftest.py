"""Shared test fixtures and configuration."""
import pytest
from fastapi.testclient import TestClient

from heart_disease.api.app import create_app


@pytest.fixture
def app():
    """Create a test FastAPI application instance."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def valid_patient_data():
    """Fixture providing valid patient data."""
    return {
        "Age": 45,
        "Sex": "male",
        "Chest pain type": 2,
        "BP": 120,
        "Cholesterol": 200,
        "FBS over 120": False,
        "EKG results": 1,
        "Max HR": 150,
        "Exercise angina": "no",
        "ST depression": 1.5,
        "Slope of ST": 2,
        "Number of vessels fluro": 1,
        "Thallium": 5,
    }


@pytest.fixture
def multiple_patients():
    """Fixture providing multiple valid patient records."""
    return [
        {
            "Age": 45,
            "Sex": "male",
            "Chest pain type": 2,
            "BP": 120,
            "Cholesterol": 200,
            "FBS over 120": False,
            "EKG results": 1,
            "Max HR": 150,
            "Exercise angina": "no",
            "ST depression": 1.5,
            "Slope of ST": 2,
            "Number of vessels fluro": 1,
            "Thallium": 5,
        },
        {
            "Age": 60,
            "Sex": "female",
            "Chest pain type": 1,
            "BP": 140,
            "Cholesterol": 250,
            "FBS over 120": True,
            "EKG results": 2,
            "Max HR": 120,
            "Exercise angina": "yes",
            "ST depression": 2.5,
            "Slope of ST": 1,
            "Number of vessels fluro": 2,
            "Thallium": 6,
        },
    ]
