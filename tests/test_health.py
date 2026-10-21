"""Unit tests for the health check and root endpoints."""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check_success():
    """Verify that the /health endpoint returns 200 OK and expected status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "meditrace-backend"
    assert data["mock_engine_active"] is True
    assert "version" in data
    assert "timestamp" in data


def test_root_endpoint():
    """Verify that the root endpoint returns service overview."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "MediTrace" in data["service"]
    assert data["status"] == "operational"
    assert data["mock_engine_active"] is True
