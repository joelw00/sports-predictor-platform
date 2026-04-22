from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_endpoint(monkeypatch) -> None:
    # Avoid binding the real DB.
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from app.main import create_app

    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "demo_mode" in body


def test_root_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from app.main import create_app

    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "sports-predictor-platform"
