"""Monitoring module: feature drift, live calibration and alerting.

This package is intentionally decoupled from training:

* ``drift.py`` — distribution-shift statistics (PSI, two-sample KS).
* ``performance.py`` — rolling Brier / log-loss / accuracy on realised matches.
* ``alerts.py`` — rule-based degradation detection.
* ``runner.py`` — orchestrates a full monitoring pass and persists a
  :class:`app.db.models.MonitoringSnapshot` row.

The goal is to catch the two classes of problem the Phase 1 brief calls out:
"drift detection" (feature-side) and "alert se modello degrada"
(performance-side), and surface both through badges on the dashboard.
"""

from app.monitoring.alerts import AlertRule, evaluate_alerts
from app.monitoring.drift import compute_feature_drift, ks_two_sample, psi
from app.monitoring.performance import LivePerformance, compute_live_performance
from app.monitoring.runner import MonitoringResult, run_monitoring

__all__ = [
    "AlertRule",
    "LivePerformance",
    "MonitoringResult",
    "compute_feature_drift",
    "compute_live_performance",
    "evaluate_alerts",
    "ks_two_sample",
    "psi",
    "run_monitoring",
]
