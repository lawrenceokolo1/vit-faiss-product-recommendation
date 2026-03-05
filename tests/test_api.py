"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert "model_version" in data
    assert "index_size" in data


def test_categories(client):
    r = client.get("/categories")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
