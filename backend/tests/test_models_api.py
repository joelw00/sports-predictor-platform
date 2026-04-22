"""Tests for /models/calibration, /models/registry and train_baseline flow."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.db import models as m
from app.db.session import get_db
from app.ingestion.demo import DemoSource
from app.ingestion.orchestrator import ingest_all
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


def test_calibration_endpoint_404_when_no_active_model(db_session) -> None:
    client = _make_client(db_session)
    resp = client.get("/models/calibration")
    assert resp.status_code == 404


def test_calibration_endpoint_returns_active_row(db_session) -> None:
    row = m.ModelRegistry(
        sport_code="football",
        market="1x2",
        family="ensemble_poisson_gbm",
        version="20260101T000000Z-test",
        artifact_path="/tmp/does-not-exist.joblib",
        is_active=True,
        metrics={
            "calibrator": "isotonic",
            "brier_raw": 0.64,
            "brier_calibrated": 0.58,
            "log_loss": 1.05,
            "accuracy": 0.45,
            "n_samples": 120,
            "n_holdout": 30,
            "markets": [
                {
                    "market": "1x2",
                    "n": 30,
                    "brier": 0.58,
                    "log_loss": 1.02,
                    "reliability": [{"p_mean": 0.3, "empirical": 0.28, "n": 10}],
                }
            ],
        },
    )
    db_session.add(row)
    db_session.commit()

    client = _make_client(db_session)
    resp = client.get("/models/calibration")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "20260101T000000Z-test"
    assert body["calibrator"] == "isotonic"
    assert body["brier_calibrated"] == 0.58
    assert body["markets"][0]["market"] == "1x2"


def test_registry_endpoint_lists_rows(db_session) -> None:
    db_session.add_all(
        [
            m.ModelRegistry(
                sport_code="football",
                market="1x2",
                family="ensemble_poisson_gbm",
                version="v-old",
                artifact_path="/tmp/old.joblib",
                is_active=False,
                metrics={},
            ),
            m.ModelRegistry(
                sport_code="football",
                market="1x2",
                family="ensemble_poisson_gbm",
                version="v-new",
                artifact_path="/tmp/new.joblib",
                is_active=True,
                metrics={"calibrator": "platt"},
            ),
        ]
    )
    db_session.commit()

    client = _make_client(db_session)
    resp = client.get("/models/registry")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    versions = {r["version"] for r in rows}
    assert versions == {"v-old", "v-new"}


def test_register_and_activate_deactivates_previous(db_session, tmp_path) -> None:
    from app.ml.trainer import TrainingReport
    from app.scripts.train_baseline import _register_and_activate

    ingest_all(db_session, sources=[DemoSource(seed=17)])

    for version in ("v1", "v2"):
        artifact = tmp_path / f"{version}.joblib"
        artifact.write_bytes(b"\x00")
        report = TrainingReport(
            n_samples=100,
            log_loss=1.0,
            accuracy=0.44,
            version=version,
            calibrator_kind="isotonic",
            brier_raw=0.65,
            brier_calibrated=0.58,
            n_holdout=25,
        )
        _register_and_activate(
            db_session, version=version, artifact=artifact, report=report
        )
        db_session.commit()

    rows = (
        db_session.query(m.ModelRegistry)
        .filter_by(sport_code="football", market="1x2")
        .all()
    )
    assert {r.version for r in rows} == {"v1", "v2"}
    active = [r for r in rows if r.is_active]
    assert len(active) == 1
    assert active[0].version == "v2"
    assert active[0].metrics["calibrator"] == "isotonic"
