"""Unit tests for the alert-rule evaluator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.monitoring.alerts import AlertRule, evaluate_alerts

NOW = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)


def _codes(alerts):
    return {a.code for a in alerts}


def test_no_alerts_on_clean_state() -> None:
    alerts = evaluate_alerts(
        n_recent_finished=50,
        n_predictions_evaluated=40,
        max_psi=0.02,
        model_trained_at=NOW - timedelta(days=2),
        brier_live=0.60,
        brier_training=0.60,
        now=NOW,
    )
    assert alerts == []


def test_low_data_fires_warning() -> None:
    alerts = evaluate_alerts(
        n_recent_finished=5,
        n_predictions_evaluated=0,
        max_psi=None,
        model_trained_at=None,
        brier_live=None,
        brier_training=None,
        now=NOW,
    )
    assert "low_data" in _codes(alerts)


def test_high_drift_critical_above_025() -> None:
    alerts = evaluate_alerts(
        n_recent_finished=100,
        n_predictions_evaluated=80,
        max_psi=0.31,
        model_trained_at=NOW - timedelta(days=3),
        brier_live=0.60,
        brier_training=0.60,
        now=NOW,
    )
    drift = next(a for a in alerts if a.code == "high_drift")
    assert drift.severity == "critical"


def test_high_drift_warning_between_010_and_025() -> None:
    alerts = evaluate_alerts(
        n_recent_finished=100,
        n_predictions_evaluated=80,
        max_psi=0.15,
        model_trained_at=NOW - timedelta(days=3),
        brier_live=0.60,
        brier_training=0.60,
        now=NOW,
    )
    drift = next(a for a in alerts if a.code == "high_drift")
    assert drift.severity == "warning"


def test_stale_model_uses_training_timestamp() -> None:
    alerts = evaluate_alerts(
        n_recent_finished=100,
        n_predictions_evaluated=80,
        max_psi=0.05,
        model_trained_at=NOW - timedelta(days=45),
        brier_live=0.60,
        brier_training=0.60,
        now=NOW,
    )
    stale = next(a for a in alerts if a.code == "stale_model")
    assert stale.severity == "critical"


def test_calibration_drift_requires_enough_predictions() -> None:
    """A Brier gap with tiny n must not fire — it's sampling noise."""
    alerts = evaluate_alerts(
        n_recent_finished=100,
        n_predictions_evaluated=5,
        max_psi=0.05,
        model_trained_at=NOW - timedelta(days=3),
        brier_live=0.80,
        brier_training=0.60,
        now=NOW,
    )
    assert "calibration_drift" not in _codes(alerts)


def test_calibration_drift_fires_with_enough_predictions() -> None:
    alerts = evaluate_alerts(
        n_recent_finished=100,
        n_predictions_evaluated=80,
        max_psi=0.05,
        model_trained_at=NOW - timedelta(days=3),
        brier_live=0.80,
        brier_training=0.60,
        now=NOW,
    )
    cal = next(a for a in alerts if a.code == "calibration_drift")
    assert cal.severity == "critical"


def test_alert_rule_is_tunable() -> None:
    rule = AlertRule(low_data_min=2)
    alerts = evaluate_alerts(
        n_recent_finished=5,
        n_predictions_evaluated=0,
        max_psi=None,
        model_trained_at=None,
        brier_live=None,
        brier_training=None,
        now=NOW,
        rule=rule,
    )
    assert "low_data" not in _codes(alerts)


def test_alerts_ordered_by_severity() -> None:
    alerts = evaluate_alerts(
        n_recent_finished=5,  # warning
        n_predictions_evaluated=80,
        max_psi=0.35,  # critical
        model_trained_at=NOW - timedelta(days=3),
        brier_live=0.60,
        brier_training=0.60,
        now=NOW,
    )
    severities = [a.severity for a in alerts]
    assert severities == sorted(severities, key={"critical": 0, "warning": 1}.get)
