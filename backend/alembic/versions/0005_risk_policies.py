"""risk policies table

Revision ID: 0005_risk_policies
Revises: 0004_backtest_clv
Create Date: 2026-04-21

Adds ``risk_policies`` for the bankroll / fractional-Kelly / exposure limits
applied to the value-bet engine and backtester. Seeds a ``"default"`` row.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_risk_policies"
down_revision = "0004_backtest_clv"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "risk_policies",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(64), nullable=False, unique=True),
        sa.Column("bankroll", sa.Float, nullable=False, server_default="1000.0"),
        sa.Column("kelly_fraction", sa.Float, nullable=False, server_default="0.25"),
        sa.Column("max_stake_pct", sa.Float, nullable=False, server_default="0.02"),
        sa.Column("max_daily_exposure_pct", sa.Float, nullable=False, server_default="0.10"),
        sa.Column("max_concurrent_positions", sa.Integer, nullable=False, server_default="10"),
        sa.Column("stop_loss_drawdown_pct", sa.Float, nullable=False, server_default="0.20"),
        sa.Column("min_edge", sa.Float, nullable=False, server_default="0.03"),
        sa.Column("min_confidence", sa.Float, nullable=False, server_default="0.55"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.execute(
        "INSERT INTO risk_policies (name) VALUES ('default') ON CONFLICT (name) DO NOTHING"
    )


def downgrade() -> None:
    op.drop_table("risk_policies")
