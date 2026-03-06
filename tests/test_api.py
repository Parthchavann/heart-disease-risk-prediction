"""
Tests for the FastAPI endpoints.
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient


VALID_PATIENT_DATA = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 0,
    "thalach": 170,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
}


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_status(client):
    """Test GET /health/status returns 200."""
    response = client.get("/health/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_health_live(client):
    """Test GET /health/live returns 200."""
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "alive"


def test_root_endpoint(client):
    """Test GET / returns 200 with API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data or "version" in data


def test_predict_example(client):
    """Test GET /predict/example returns 200 with example request data."""
    response = client.get("/predict/example")
    assert response.status_code == 200
    data = response.json()
    assert "patient_data" in data
    # Should have 13 features
    assert len(data["patient_data"]) == 13


def test_predict_features(client):
    """Test GET /predict/features returns 200 with feature descriptions."""
    response = client.get("/predict/features")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    # Should return exactly 13 features
    assert len(data["features"]) == 13


def test_predict_simple_invalid_data(client):
    """Test POST /predict/simple with invalid data returns 422."""
    invalid_data = {
        "age": 999,       # Way out of range
        "sex": 5,         # Invalid
        "cp": 10,         # Invalid
        "trestbps": 50,   # Below min
        "chol": 50,       # Below min
        "fbs": 2,         # Invalid
        "restecg": 5,     # Invalid
        "thalach": 30,    # Below min
        "exang": 3,       # Invalid
        "oldpeak": -5.0,  # Below min
        "slope": 5,       # Invalid
        "ca": 10,         # Invalid
        "thal": 10        # Invalid
    }
    response = client.post("/predict/simple", json=invalid_data)
    assert response.status_code == 422


def test_predict_batch_empty(client):
    """Test POST /predict/batch with empty list returns 422."""
    response = client.post("/predict/batch", json={"patients": []})
    assert response.status_code == 422
