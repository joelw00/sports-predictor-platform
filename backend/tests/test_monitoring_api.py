"""Endpoint tests for /monitoring/*."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app.db import models as m
from app.db.session import get_db
from app.main import create_app


def _make_client(db_session):
    app = create_app()

    def _override():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override
    return TestClient(app)


def test_latest_returns_404_before_any_run(db_session) -> None:
    client = _make_client(db_session)
    resp = client.get("/monitoring/latest")
    assert resp.status_code == 404


def test_latest_returns_most_recent_row(db_session) -> None:
    now = datetime.now(UTC)
    db_session.add_all(
        [
            m.MonitoringSnapshot(
                sport_code="football",
                market="1x2",
                computed_at=now - timedelta(days=2),
                model_version="v-old",
                n_recent_finished=10,
                n_predictions_evaluated=10,
                brier_live=0.60,
                max_psi=0.05,
                drift={"features": []},
                alerts=[],
            ),
            m.MonitoringSnapshot(
                sport_code="football",
                market="1x2",
                computed_at=now,
                model_version="v-new",
                n_recent_finished=50,
                n_predictions_evaluated=40,
                brier_live=0.58,
                max_psi=0.28,
                drift={"features": []},
                alerts=[{"code": "high_drift", "severity": "critical", "message": "x"}],
            ),
        ]
    )
    db_session.commit()

    client = _make_client(db_session)
    resp = client.get("/monitoring/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "v-new"
    assert body["max_psi"] == 0.28
    assert body["alerts"][0]["code"] == "high_drift"


def test_history_respects_limit_and_ordering(db_session) -> None:
    now = datetime.now(UTC)
    for i in range(5):
        db_session.add(
            m.MonitoringSnapshot(
                sport_code="football",
                market="1x2",
                computed_at=now - timedelta(days=i),
                model_version=f"v{i}",
                n_recent_finished=i,
                n_predictions_evaluated=i,
                drift={"features": []},
                alerts=[],
            )
        )
    db_session.commit()

    client = _make_client(db_session)
    resp = client.get("/monitoring/history?limit=3")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 3
    # Newest first.
    versions = [r["model_version"] for r in rows]
    assert versions == ["v0", "v1", "v2"]


def test_run_endpoint_creates_snapshot(db_session) -> None:
    sport = m.Sport(code="football", name="Football")
    db_session.add(sport)
    db_session.commit()

    client = _make_client(db_session)
    resp = client.post("/monitoring/run?market=1x2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sport"] == "football"
    assert body["market"] == "1x2"
    assert "alerts" in body
    # Row was persisted.
    rows = db_session.query(m.MonitoringSnapshot).all()
    assert len(rows) == 1
