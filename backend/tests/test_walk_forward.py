"""Regression tests for the leakage-free walk-forward backtester.

The key property: on demo data (where the book's margin is ~5% and the model
has no information the odds don't), the honest walk-forward ROI must be much
closer to zero (and no better than) the leaky pretrained-replay ROI.
"""

from __future__ import annotations

import numpy as np

from app.backtesting.engine import Backtester
from app.backtesting.walk_forward import WalkForwardBacktester
from app.ingestion.demo import DemoSource
from app.ingestion.orchestrator import ingest_all
from app.metrics.calibration import brier_score, log_loss_multi, reliability_bins
from app.ml.trainer import FootballTrainer
from app.value_bet.engine import ValueBetEngine


def test_walk_forward_produces_metrics_and_no_leakage(db_session):
    ingest_all(db_session, sources=[DemoSource(seed=17)])
    wf = WalkForwardBacktester(
        n_folds=5,
        min_train_folds=2,
        engine=ValueBetEngine(min_edge=0.03, min_confidence=0.0),
    )
    result = wf.run(db_session, sport_code="football", market="1x2", label="t-wf")

    assert result.n_bets >= 0
    assert result.start_date <= result.end_date
    assert result.breakdown["mode"] == "walk_forward"
    # Calibration metrics must be reported whenever at least one fold was scored.
    cal = result.breakdown.get("calibration") or {}
    assert "brier" in cal and 0.0 <= cal["brier"] <= 2.0
    assert "log_loss" in cal and cal["log_loss"] > 0.0
    assert cal["n_holdout"] > 0
    assert len(cal["reliability_home"]) > 0


def test_walk_forward_roi_is_not_inflated_vs_pretrained(db_session):
    """The honest backtester must never report a wildly better ROI than the leaky
    one on the same demo universe — that would mean we made the bug worse.

    Because the leaky version is trained on the same matches it scores, it has
    an unfair advantage; the walk-forward ROI should land below (or at worst
    near) the pretrained ROI.
    """
    ingest_all(db_session, sources=[DemoSource(seed=23)])
    predictor, _ = FootballTrainer(version="pre").train(db_session)
    pretrained = Backtester(
        predictor,
        engine=ValueBetEngine(min_edge=0.03, min_confidence=0.0),
    ).run(db_session, sport_code="football", market="1x2", label="t-pre")

    wf = WalkForwardBacktester(
        n_folds=5,
        min_train_folds=2,
        engine=ValueBetEngine(min_edge=0.03, min_confidence=0.0),
    ).run(db_session, sport_code="football", market="1x2", label="t-wf")

    # Walk-forward must never "cheat" past the leaky baseline by > 5 pp ROI.
    # Empirically on the demo it lands well below; this ceiling is a safety net.
    assert wf.roi <= pretrained.roi + 0.05, (
        f"walk-forward ROI ({wf.roi:.3f}) suspiciously exceeds "
        f"pretrained ROI ({pretrained.roi:.3f}) — leakage may have regressed."
    )


def test_calibration_metrics_roundtrip():
    probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.3, 0.4, 0.3],
            [0.6, 0.3, 0.1],
        ]
    )
    labels = np.array([0, 2, 1, 1])
    br = brier_score(probs, labels)
    assert br.n == 4
    assert 0.0 <= br.score <= 2.0
    ll = log_loss_multi(probs, labels)
    assert ll > 0
    bins = reliability_bins(probs, labels, positive_class=0, n_bins=4)
    assert all("bin" in row and "predicted" in row and "observed" in row for row in bins)
