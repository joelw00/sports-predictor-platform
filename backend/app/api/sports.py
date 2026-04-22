from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.db import models as m
from app.schemas import SportOut

router = APIRouter(tags=["reference"])


@router.get("/sports", response_model=list[SportOut])
def list_sports(db: Session = Depends(get_db)) -> list[SportOut]:
    return [SportOut.model_validate(s) for s in db.query(m.Sport).order_by(m.Sport.name).all()]


@router.get("/competitions")
def list_competitions(
    sport: str | None = None,
    db: Session = Depends(get_db),
) -> list[dict]:
    q = db.query(m.Competition).join(m.Sport)
    if sport:
        q = q.filter(m.Sport.code == sport)
    return [
        {
            "id": c.id,
            "code": c.code,
            "name": c.name,
            "country": c.country,
            "sport": c.sport.code if c.sport else None,
        }
        for c in q.order_by(m.Competition.name).all()
    ]
