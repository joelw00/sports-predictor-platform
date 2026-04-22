from fastapi import APIRouter

from app.api import admin, backtests, events, health, predictions, sports, value_bets

router = APIRouter()
router.include_router(health.router)
router.include_router(sports.router)
router.include_router(events.router)
router.include_router(predictions.router)
router.include_router(value_bets.router)
router.include_router(backtests.router)
router.include_router(admin.router)

__all__ = ["router"]
