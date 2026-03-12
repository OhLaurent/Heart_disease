"""Tests for the FastAPI application setup."""
import pytest
from fastapi import FastAPI


def test_app_creation():
    """Test that the app is created successfully."""
    from heart_disease.api.app import create_app
    
    app = create_app()
    assert isinstance(app, FastAPI)
    assert app.title == "Heart Disease Prediction API"


def test_app_routes_registered(app):
    """Test that all required routes are registered."""
    routes = [route.path for route in app.routes]
    
    assert "/" in routes
    assert "/api/v1/predict" in routes
    assert "/api/v1/predictions/history" in routes
    assert "/api/v1/retrain" in routes


def test_app_lifespan_startup(app):
    """Test that app starts and shuts down correctly."""
    # The app should be created without errors
    assert app is not None


def test_index_endpoint(client):
    """Test the index endpoint serves static files."""
    response = client.get("/")
    assert response.status_code == 200
