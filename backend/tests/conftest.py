from __future__ import annotations

import os
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db import models  # noqa: F401 — register models
from app.db.base import Base


@pytest.fixture()
def db_session() -> Session:
    # In-memory SQLite DB, isolated per test.
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    url = f"sqlite:///{tmp.name}"
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    session = SessionMaker()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
        os.unlink(tmp.name)
