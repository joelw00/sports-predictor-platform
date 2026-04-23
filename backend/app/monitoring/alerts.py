"""Rule-based alert evaluation.

We intentionally keep the rules explicit (no learning) so the dashboard badges
are easy to explain. Each rule fires an :class:`Alert` with a severity; the
runner aggregates them into the monitoring snapshot and the frontend renders
them as colour-coded badges.

Thresholds are conservative defaults based on common industry practice:

* ``low_data`` — below this size any live-Brier comparison is noisy.
* ``high_drift`` — PSI > 0.25 is the canonical "significant shift" threshold.
* ``stale_model`` — anything older than two weeks is suspicious given a
  weekly retrain cadence.
* ``calibration_drift`` — a Brier gap > 0.05 relative to the training Brier
  usually indicates real degradation rather than sampling noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any


@dataclass(frozen=True)
class Alert:
    code: str
    severity: str  # "info" | "warning" | "critical"
    message: str
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.meta is not None:
            out["meta"] = self.meta
        return out


@dataclass(frozen=True)
class AlertRule:
    """Thresholds driving :func:`evaluate_alerts`. All fields have defaults."""

    low_data_min: int = 20
    drift_psi_warning: float = 0.1
    drift_psi_critical: float = 0.25
    stale_model_warning_days: int = 14
    stale_model_critical_days: int = 30
    calibration_gap_warning: float = 0.03
    calibration_gap_critical: float = 0.05


def evaluate_alerts(
    *,
    n_recent_finished: int,
    n_predictions_evaluated: int,
    max_psi: float | None,
    model_trained_at: datetime | None,
    brier_live: float | None,
    brier_training: float | None,
    now: datetime | None = None,
    rule: AlertRule | None = None,
) -> list[Alert]:
    """Return a list of fired alerts ordered by descending severity."""
    rule = rule or AlertRule()
    now = now or datetime.now(UTC)
    alerts: list[Alert] = []

    if n_recent_finished < rule.low_data_min:
        alerts.append(
            Alert(
                code="low_data",
                severity="warning",
                message=(
                    f"Only {n_recent_finished} finished matches in the live window "
                    f"(< {rule.low_data_min}); live metrics are noisy."
                ),
                meta={"threshold": rule.low_data_min, "observed": n_recent_finished},
            )
        )

    if max_psi is not None:
        if max_psi >= rule.drift_psi_critical:
            alerts.append(
                Alert(
                    code="high_drift",
                    severity="critical",
                    message=(
                        f"Feature drift PSI={max_psi:.3f} ≥ {rule.drift_psi_critical:.2f} "
                        "— retrain recommended."
                    ),
                    meta={"max_psi": max_psi, "threshold": rule.drift_psi_critical},
                )
            )
        elif max_psi >= rule.drift_psi_warning:
            alerts.append(
                Alert(
                    code="high_drift",
                    severity="warning",
                    message=(
                        f"Feature drift PSI={max_psi:.3f} ≥ {rule.drift_psi_warning:.2f} "
                        "— watch upstream feed."
                    ),
                    meta={"max_psi": max_psi, "threshold": rule.drift_psi_warning},
                )
            )

    if model_trained_at is not None:
        if model_trained_at.tzinfo is None:
            model_trained_at = model_trained_at.replace(tzinfo=UTC)
        age = now - model_trained_at
        if age >= timedelta(days=rule.stale_model_critical_days):
            alerts.append(
                Alert(
                    code="stale_model",
                    severity="critical",
                    message=(
                        f"Active model is {age.days} days old "
                        f"(≥ {rule.stale_model_critical_days} days)."
                    ),
                    meta={"age_days": age.days},
                )
            )
        elif age >= timedelta(days=rule.stale_model_warning_days):
            alerts.append(
                Alert(
                    code="stale_model",
                    severity="warning",
                    message=(
                        f"Active model is {age.days} days old "
                        f"(≥ {rule.stale_model_warning_days} days); consider retraining."
                    ),
                    meta={"age_days": age.days},
                )
            )

    if (
        brier_live is not None
        and brier_training is not None
        and n_predictions_evaluated >= rule.low_data_min
    ):
        gap = brier_live - brier_training
        if gap >= rule.calibration_gap_critical:
            alerts.append(
                Alert(
                    code="calibration_drift",
                    severity="critical",
                    message=(
                        f"Live Brier {brier_live:.3f} exceeds training Brier "
                        f"{brier_training:.3f} by {gap:.3f} — model is miscalibrated."
                    ),
                    meta={
                        "gap": gap,
                        "brier_live": brier_live,
                        "brier_training": brier_training,
                    },
                )
            )
        elif gap >= rule.calibration_gap_warning:
            alerts.append(
                Alert(
                    code="calibration_drift",
                    severity="warning",
                    message=(
                        f"Live Brier {brier_live:.3f} exceeds training Brier "
                        f"{brier_training:.3f} by {gap:.3f}."
                    ),
                    meta={
                        "gap": gap,
                        "brier_live": brier_live,
                        "brier_training": brier_training,
                    },
                )
            )

    sev_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: sev_order.get(a.severity, 99))
    return alerts
