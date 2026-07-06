import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEST_DB = Path("/tmp/safevision_test.db")
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-safevision-ai-32-chars")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{TEST_DB}")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:8501")

from backend.app.database import Base, engine
from backend.app.main import app


@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    user = {
        "email": "admin@safevision.ai",
        "full_name": "Safety Admin",
        "password": "ChangeMe123",
        "role": "admin",
    }
    response = client.post("/api/v1/auth/register", json=user)
    assert response.status_code == 200
    login = client.post(
        "/api/v1/auth/login",
        data={"username": user["email"], "password": user["password"]},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
