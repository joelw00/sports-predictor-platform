"""backtest clv columns

Revision ID: 0004_backtest_clv
Revises: 0003_monitoring_snapshots
Create Date: 2026-04-21

Adds ``avg_clv`` / ``clv_win_rate`` / ``n_clv_tracked`` to ``backtest_runs``
so the dashboard can surface CLV alongside ROI / yield / drawdown. Columns
default to 0 for back-compat on existing rows.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_backtest_clv"
down_revision = "0003_monitoring_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "backtest_runs",
        sa.Column("avg_clv", sa.Float, nullable=False, server_default="0"),
    )
    op.add_column(
        "backtest_runs",
        sa.Column("clv_win_rate", sa.Float, nullable=False, server_default="0"),
    )
    op.add_column(
        "backtest_runs",
        sa.Column("n_clv_tracked", sa.Integer, nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("backtest_runs", "n_clv_tracked")
    op.drop_column("backtest_runs", "clv_win_rate")
    op.drop_column("backtest_runs", "avg_clv")
