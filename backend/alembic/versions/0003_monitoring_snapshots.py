"""monitoring snapshots

Revision ID: 0003_monitoring_snapshots
Revises: 0002_raw_ingestion
Create Date: 2026-04-22

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_monitoring_snapshots"
down_revision = "0002_raw_ingestion"
branch_labels = None
depends_on = None


def _ts() -> list[sa.Column]:
    return [
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
    ]


def upgrade() -> None:
    op.create_table(
        "monitoring_snapshots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("sport_code", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(64)),
        sa.Column("n_recent_finished", sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_predictions_evaluated", sa.Integer, nullable=False, server_default="0"),
        sa.Column("brier_live", sa.Float),
        sa.Column("log_loss_live", sa.Float),
        sa.Column("accuracy_live", sa.Float),
        sa.Column("brier_training", sa.Float),
        sa.Column("max_psi", sa.Float),
        sa.Column("drift", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("alerts", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("meta", sa.JSON, nullable=False, server_default="{}"),
        *_ts(),
    )
    op.create_index(
        "ix_monitoring_snapshots_sport_market_computed",
        "monitoring_snapshots",
        ["sport_code", "market", "computed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_monitoring_snapshots_sport_market_computed",
        table_name="monitoring_snapshots",
    )
    op.drop_table("monitoring_snapshots")
